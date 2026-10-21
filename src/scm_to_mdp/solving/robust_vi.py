"""Robust value iteration on an interval MDP (step S8).

At every (s, a), nature picks the kernel in [L, U] that is worst for the agent: it gives every next
state its lower bound, then hands the remaining mass to the next states with the lowest value first,
each up to its upper bound (Nilim & El Ghaoui, Oper. Res. 2005; Iyengar, Math. Oper. Res. 2005).
The result V is a lower bound on the value of the returned policy under every kernel in the set.
"""

import numpy as np

from scm_to_mdp.core import IntervalMDP, Solution


def solve(mdp: IntervalMDP, gamma: float, tol: float = 1e-6) -> Solution:
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
        if np.abs(V_new - V).max() < tol:
            return Solution(V_new, Q.argmax(1))
        V = V_new
