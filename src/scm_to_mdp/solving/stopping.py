"""When value iteration stops: tol is the distance to the fixed point, not the size of the last step.

For a gamma-contraction T with fixed point V_fix, ||V_fix - V_k|| <= gamma / (1 - gamma) *
||V_k - V_{k-1}|| in the sup norm (Kallenberg, Markov Decision Processes, lecture notes, Leiden,
2016, Thm 3.2(1), p. 45). Stopping when the step is below tol * (1 - gamma) / gamma therefore leaves
V_k within tol of the fixed point.

Within tol is not below: the promise of a robust method must be a LOWER bound. Let pi be the greedy
policy of the last update (greedy with respect to V_{k-1}), so V_k = T_pi V_{k-1}, with T_pi the
(robust) Bellman operator of pi. T_pi is itself a gamma-contraction with fixed point V^pi, so the
same bound gives
    V^pi >= V_k - gamma / (1 - gamma) * ||V_k - V_{k-1}||
at whatever step value iteration stops (for plain VI this is Kallenberg's Thm 3.7(1), p. 50; for
robust VI the argument needs only that T_pi is a gamma-contraction, which holds for a min over an
(s, a)-rectangular set of kernels). sound_lower_bound returns this, and the solvers report it as the
promise: at most tol below V_k. Comparing the step itself to tol is not enough: at gamma = 0.99 it
leaves up to 99 * tol, and made a robust promise exceed the true value by about 1e-4 in a test with
tol = 1e-6.
"""

import math

import numpy as np


def iterations_bound(scale: float, tol: float, gamma: float) -> int:
    """How many sweeps value iteration may need from V = 0 at the worst rate gamma: the first step is at
    most `scale` (the largest |R| + |r_next|), and every later one at most gamma times the one before,
    so the step falls below stop_at(tol, gamma) after at most this many sweeps."""
    if scale <= 0:
        return 1
    return 1 + max(0, math.ceil(math.log(stop_at(tol, gamma) / scale) / math.log(gamma)))


def reward_scale(reward) -> float:
    """max |R| + max |r_next|: a bound on the first step of value iteration from V = 0."""
    return float(np.abs(reward.R).max() + (np.abs(reward.r_next).max() if reward.r_next.size else 0.0))


def stop_at(tol: float, gamma: float) -> float:
    """The step size below which the iterate is within tol of the fixed point."""
    return tol * (1 - gamma) / max(gamma, 1e-12)


def value_floor(reward, gamma: float) -> float:
    """A lower bound on every discounted value: r_min / (1 - gamma), r_min = min R + min r_next the smallest
    one-step reward (with terminal states min(r_min, 0), since an episode may end)."""
    r_min = float(reward.R.min() + (reward.r_next.min() if reward.r_next.size else 0.0))
    if reward.terminal.any():
        r_min = min(r_min, 0.0)
    return r_min / (1 - gamma)


def data_only_values(reward, gamma: float, masks: dict | None, optimistic: bool = False):
    """(fixed_v [S], fixed_q [S, K]) of a data-only model (scm_learning/data_only.py), NaN where not fixed.
    V_min = min(r_min, 0) / (1 - gamma), r_min the smallest one-step reward (0: an episode may end). A state never
    seen as a source is a known state: V_min after its own arrival reward. A state holding a value never seen is
    unknown: it is entered with the smallest arrival reward, so its value is set to make
    r_next(s) + gamma V(s) = min r_next + gamma V_min. A pair with an unseen row sends all its mass to an
    unknown sink: Q = R + min r_next + gamma V_min. Every fixed value is below the true one, so the robust
    update keeps a lower bound (badings_claude, 2026-11-06).

    optimistic=True: the same sinks at the BEST value, for an upper bound (the certificate's V_opt,
    evaluation.optimistic_values): V_max = max(r_max, 0) / (1 - gamma), r_max the largest one-step reward,
    and an unknown state entered with the largest arrival reward. Every value is at most V_max: with
    c = V_max >= 0, T(c e) <= c e (an episode that ends earns 0 after it), so V* <= c (kallenberg_claude:
    Kallenberg, Thm 3.3, p. 46). Every fixed value is then above the true one, and with the true kernel in
    the set on the other rows the cooperative operator T' satisfies T'V* >= V* (not T'V >= TV for every V,
    which fails at the fixed entries); T' is a monotone gamma-contraction (Lemma 3.1, p. 46; Thm 3.4,
    pp. 48-49), so its fixed point is >= V* (Thm 3.3(2), p. 46). The unestimated mass at the best value is
    the upper bound of [Ashok et al. 2019] (U-hat, pdf 8; badings_claude, 2026-11-07), black-box form; with
    gamma < 1 no end-component handling is needed. The worst-value sinks are not sound there: with the
    terminal states removed and negative rewards, a true terminal state the model keeps paying for would
    pull V_opt below the true value. Assumes that a true terminal state is never a source in
    the data (sampling stops there)."""
    S, K = reward.R.shape
    fixed_v, fixed_q = np.full(S, np.nan), np.full((S, K), np.nan)
    if masks is None:
        return None, None
    if optimistic:
        r_next_end = float(reward.r_next.max()) if reward.r_next.size else 0.0
        v_end = max(float(reward.R.max()) + r_next_end, 0.0) / (1 - gamma)
    else:
        r_next_end = float(reward.r_next.min()) if reward.r_next.size else 0.0
        v_end = min(float(reward.R.min()) + r_next_end, 0.0) / (1 - gamma)
    fixed_v[masks["never_source"]] = v_end
    unknown = masks["unknown"]
    fixed_v[unknown] = v_end + (r_next_end - reward.r_next[unknown]) / max(gamma, 1e-12)
    doomed = masks["doomed"] & np.isnan(fixed_v)[:, None]
    fixed_q[doomed] = (reward.R + r_next_end + gamma * v_end)[doomed]
    return fixed_v, fixed_q


def greedy(
    Q: np.ndarray,
    allowed: np.ndarray | None = None,
    floor: float | None = None,
    fixed_v: np.ndarray | None = None,
    fixed_q: np.ndarray | None = None,
):
    """(V, pi) of one value-iteration step: the max of Q [S, K] over the actions. allowed [S, K] (bool,
    optional): only over the allowed actions ("seen pairs only", tool/methods.py); a state with none gets
    the constant floor (value_floor), and its action is the best of all (any action would do). fixed_q, fixed_v
    (optional, NaN where free): the data-only sinks (data_only_values), applied before / after the max.

    Soundness (adversarial nature): restricted to the allowed actions, with the floor at the states that
    have none, the robust update is still a gamma-contraction (a max over fewer actions; a constant), and
    it maps {V >= floor} into itself (an allowed backup is >= r_min + gamma floor = floor), so its fixed
    point V' is >= floor. For the greedy policy pi' and every kernel P in the set: at an allowed state
    V'(s) = R + gamma min_P P V' <= T^pi'_P V'(s); at a state with none, V'(s) = floor <= r_min + gamma
    floor <= T^pi'_P V'(s). By monotonicity V' <= V^pi'_P: a lower bound, and the stopping rule above
    still applies (the update is a contraction). Only the allowed pairs need P in [L, U] (the others may be
    [0, 1]). A max over fewer actions can only lower the value. (badings_claude, 2026-11-06: the proof
    is general knowledge, not in its references; closest: actions enabled by data, Nazeri et al. 2025,
    Sec. IV.A and Thm 1; enabled actions A(s), Suilen et al. 2024, Def. 1, p. 3 ⚠ pages not re-checked.)"""
    if fixed_q is not None:  # data-only: the pairs sent to a sink (data_only_values)
        Q = np.where(np.isnan(fixed_q), Q, fixed_q)
    if allowed is None:
        V, pi = Q.max(1), Q.argmax(1)
    else:
        masked = np.where(allowed, Q, -np.inf)
        empty = ~allowed.any(1)
        V = np.where(empty, floor, masked.max(1))
        pi = np.where(empty, Q.argmax(1), masked.argmax(1))
    if fixed_v is not None:  # data-only: the sinks
        V = np.where(np.isnan(fixed_v), V, fixed_v)
    return V, pi


def sound_lower_bound(V_new, step: float, gamma: float):
    """A lower bound on the robust value of the greedy policy of the last update (see above)."""
    return V_new - gamma / (1 - gamma) * step
