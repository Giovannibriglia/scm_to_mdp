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
"""

import time

import numpy as np

from scm_to_mdp.core import IntervalMDP, Solution
from scm_to_mdp.solving.stopping import sound_lower_bound, stop_at

NATURES = ("adversarial", "cooperative")


def solve(mdp: IntervalMDP, gamma: float, tol: float = 1e-7, nature: str = "adversarial") -> Solution:
    """Memory: besides L and U, three arrays of their size (U - L and two buffers reused at every
    iteration). The lower bounds' part L @ W and the free mass 1 - sum L do not depend on the order of
    the next states, so only the room U - L is sorted. nature: "adversarial" (the promise: a lower
    bound) or "cooperative" (an upper bound on the best value over policies and kernels)."""
    if nature not in NATURES:
        raise ValueError(f"nature must be one of {NATURES}, not {nature!r}")
    adversarial = nature == "adversarial"
    L = mdp.L
    V = np.zeros(L.shape[0])
    room = mdp.U - L  # [S, K, S'] what nature may add above the lower bounds
    remaining = (1.0 - L.sum(-1))[..., None]  # [S, K, 1] mass left after the lower bounds
    sorted_room, given = np.empty_like(room), np.empty_like(room)  # buffers, reused
    start, sweeps = time.perf_counter(), 0
    while True:
        sweeps += 1
        W = mdp.reward.backup(V, gamma)  # [S'] what arriving in s' is worth
        order = np.argsort(W if adversarial else -W)  # the next states nature fills first
        np.take(room, order, axis=-1, out=sorted_room)
        np.cumsum(sorted_room, axis=-1, out=given)
        given -= sorted_room  # mass the states filled before already took
        np.subtract(remaining, given, out=given)  # what is left for this state ...
        np.maximum(given, 0.0, out=given)
        np.minimum(given, sorted_room, out=given)  # ... up to its room: the extra mass it gets
        Q = mdp.reward.R + L @ W + given @ W[order]  # [S, K]
        V_new = Q.max(1)
        step = np.abs(V_new - V).max()
        if step < stop_at(tol, gamma):
            seconds = (time.perf_counter() - start) / sweeps
            bound = (
                sound_lower_bound(V_new, step, gamma) if adversarial else V_new + gamma / (1 - gamma) * step
            )
            return Solution(bound, Q.argmax(1), iterations=sweeps, iteration_s=seconds)
        V = V_new
