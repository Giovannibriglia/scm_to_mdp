"""Robust value iteration on an interval MDP (step S8).

At every (s, a), nature picks the kernel in {P : L <= P <= U, sum P = 1} that is worst for the agent:
it gives every next state its lower bound, then hands the remaining mass 1 - sum L to the next states
in ascending order of value, each up to its upper bound (Nilim & El Ghaoui, Oper. Res. 2005;
Iyengar, Math. Oper. Res. 2005; Suilen et al., arXiv:2411.11451, 2024, Alg. 1, p. 14). The set is
non-empty iff L <= U and sum L <= 1 <= sum U. Interval sets are (s, a)-rectangular, so this local
minimum is exact, and the fixed point V is a lower bound on the value of the returned policy under
every kernel in the set, also kernels that change over time (up to the stopping rule, stopping.py).

nature="cooperative" (the uncertainty resolution mode of that name in Storm) is the mirror image:
nature hands the free mass to the BEST next states, so the agent maximises over policies and kernels.
Its V is then a sound UPPER bound on that optimistic fixed point (V_k + gamma / (1 - gamma) * step;
stopping.py), e.g. V_opt of the near-optimality certificate (evaluation.optimistic_values). It promises
nothing about the true value of its policy.

    solve       the best policy against nature, and its value (a bound, as above)
    evaluate    the value of a GIVEN policy (deterministic or stochastic) against nature: a lower
                bound (adversarial) or an upper bound (cooperative) on its value under every kernel in
                the set (robustness.py: the value bracket of a promise)
    on_policy   the set restricted to the actions a policy plays (exact for that policy: nature's
                choice at (s, a) does not depend on the other actions, by rectangularity). A
                deterministic policy keeps one action per state, so its evaluation is K times cheaper
    Nature      nature's inner problem alone: min or max over the set of E_P[W] at every (s, a)

Both iterations take a start V0 (default 0): the operators are gamma-contractions, so they converge
from any start and the stopping rule's bound is sound from any start (stopping.py); a start close to the
answer (e.g. the solver's own V) only saves sweeps.
"""

import dataclasses
import time

import numpy as np

from scm_to_mdp.core import IntervalMDP, Solution
from scm_to_mdp.solving.stopping import sound_lower_bound, stop_at

NATURES = ("adversarial", "cooperative")


class Nature:
    """Nature's choice on an interval MDP: for a vector W over next states, the smallest
    ("adversarial") or largest ("cooperative") E_P[W] over the set, at every (s, a) [S, K].

    The part of the lower bounds, L @ W, and the free mass 1 - sum L do not depend on the order of the
    next states, so only the room U - L is sorted. Memory: besides L and U, three arrays of their size
    (U - L and two buffers reused at every call)."""

    def __init__(self, mdp: IntervalMDP, nature: str = "adversarial"):
        if nature not in NATURES:
            raise ValueError(f"nature must be one of {NATURES}, not {nature!r}")
        self.adversarial = nature == "adversarial"
        self.L = mdp.L
        self.room = mdp.U - mdp.L  # [S, K, S'] what nature may add above the lower bounds
        self.remaining = (1.0 - mdp.L.sum(-1))[..., None]  # [S, K, 1] mass left after the lower bounds
        self._sorted, self._given = np.empty_like(self.room), np.empty_like(self.room)  # buffers

    def expectation(self, W: np.ndarray) -> np.ndarray:
        """[S, K] nature's E_P[W] at every (s, a)."""
        order = np.argsort(W if self.adversarial else -W)  # the next states nature fills first
        np.take(self.room, order, axis=-1, out=self._sorted)
        given = self._given
        np.cumsum(self._sorted, axis=-1, out=given)
        given -= self._sorted  # mass the states filled before already took
        np.subtract(self.remaining, given, out=given)  # what is left for this state ...
        np.maximum(given, 0.0, out=given)
        np.minimum(given, self._sorted, out=given)  # ... up to its room: the extra mass it gets
        return self.L @ W + given @ W[order]


def _bound(V_new: np.ndarray, step: float, gamma: float, adversarial: bool) -> np.ndarray:
    """The stopping rule's sound bound on the fixed point: below it (adversarial) or above it."""
    return sound_lower_bound(V_new, step, gamma) if adversarial else V_new + gamma / (1 - gamma) * step


def on_policy(mdp: IntervalMDP, policy: np.ndarray) -> tuple[IntervalMDP, np.ndarray]:
    """(the set on the actions the policy plays [S, m, S'], their probabilities [S, m]); m = the most
    actions any state plays with positive probability (1 for a deterministic policy). Row k of state s
    is the policy's k-th most likely action there (rows of zero probability pad states playing fewer)."""
    m = max(int((policy > 0).sum(1).max()), 1)
    actions = np.argsort(-policy, axis=1, kind="stable")[:, :m]  # [S, m]
    rows = np.arange(policy.shape[0])[:, None]
    reward = dataclasses.replace(mdp.reward, R=mdp.reward.R[rows, actions])
    restricted = IntervalMDP(mdp.L[rows, actions], mdp.U[rows, actions], reward, mdp.info)
    return restricted, policy[rows, actions]


def solve(
    mdp: IntervalMDP,
    gamma: float,
    tol: float = 1e-7,
    nature: str = "adversarial",
    V0: np.ndarray | None = None,
) -> Solution:
    """The best policy against nature. nature: "adversarial" (the promise: a lower bound) or
    "cooperative" (an upper bound on the best value over policies and kernels). V0: the start."""
    inner = Nature(mdp, nature)
    V = np.zeros(mdp.L.shape[0]) if V0 is None else np.asarray(V0, dtype=float)
    start, sweeps = time.perf_counter(), 0
    while True:
        sweeps += 1
        Q = mdp.reward.R + inner.expectation(mdp.reward.backup(V, gamma))  # [S, K]
        V_new = Q.max(1)
        step = np.abs(V_new - V).max()
        if step < stop_at(tol, gamma):
            seconds = (time.perf_counter() - start) / sweeps
            bound = _bound(V_new, step, gamma, inner.adversarial)
            return Solution(bound, Q.argmax(1), iterations=sweeps, iteration_s=seconds)
        V = V_new


def evaluate(
    mdp: IntervalMDP,
    policy: np.ndarray,
    gamma: float,
    tol: float = 1e-7,
    nature: str = "adversarial",
    V0: np.ndarray | None = None,
) -> np.ndarray:
    """[S] the value of a fixed policy against nature: policy [S, K] action probabilities (a
    deterministic policy is one-hot). Adversarial: a sound lower bound on the policy's value under
    every kernel in the set; cooperative: a sound upper bound. The fixed-policy operator is a
    gamma-contraction, so the same stopping rule applies. Runs on the actions the policy plays
    (on_policy); V0: the start."""
    mdp, policy = on_policy(mdp, policy)
    inner = Nature(mdp, nature)
    V = np.zeros(mdp.L.shape[0]) if V0 is None else np.asarray(V0, dtype=float)
    while True:
        Q = mdp.reward.R + inner.expectation(mdp.reward.backup(V, gamma))
        V_new = (policy * Q).sum(1)
        step = np.abs(V_new - V).max()
        if step < stop_at(tol, gamma):
            return _bound(V_new, step, gamma, inner.adversarial)
        V = V_new
