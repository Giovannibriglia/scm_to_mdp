"""Robust value iteration on an interval MDP (step S8).

At every (s, a), nature picks the kernel in {P : L <= P <= U, sum P = 1} that is worst for the agent:
it gives every next state its lower bound, then hands the remaining mass 1 - sum L to the next states
in ascending order of value, each up to its upper bound (Nilim & El Ghaoui, Oper. Res. 2005;
Iyengar, Math. Oper. Res. 2005; Suilen et al., arXiv:2411.11451, 2024, Alg. 1, p. 14). The set is
non-empty iff L <= U and sum L <= 1 <= sum U. Interval sets are (s, a)-rectangular, so this local
minimum is exact, and the fixed point V is a lower bound on the value of the returned policy under
every kernel in the set, also kernels that change over time (up to the stopping rule, stopping.py).
"""

import numpy as np

from scm_to_mdp.core import IntervalMDP, Solution
from scm_to_mdp.solving.stopping import sound_lower_bound, stop_at


def solve(mdp: IntervalMDP, gamma: float, tol: float = 1e-7) -> Solution:
    L, U = mdp.L, mdp.U
    V = np.zeros(L.shape[0])
    while True:
        W = mdp.reward.backup(V, gamma)  # [S'] what arriving in s' is worth
        order = np.argsort(W)  # next states from worst to best
        lower, room, w = L[..., order], (U - L)[..., order], W[order]  # [S, K, S']
        remaining = 1.0 - lower.sum(-1, keepdims=True)  # mass left after the lower bounds
        given_before = np.cumsum(room, -1) - room  # mass the worse states already took
        extra = np.clip(remaining - given_before, 0.0, room)
        Q = mdp.reward.R + ((lower + extra) * w).sum(-1)  # [S, K]
        V_new = Q.max(1)
        step = np.abs(V_new - V).max()
        if step < stop_at(tol, gamma):
            return Solution(sound_lower_bound(V_new, step, gamma), Q.argmax(1))
        V = V_new
