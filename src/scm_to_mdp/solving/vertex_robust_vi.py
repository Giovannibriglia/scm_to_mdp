"""Exact robust value iteration over a product of factor intervals (step S8), for 0/1 variables.

Schnitzer, Abate, Parker, "Efficient Solution and Learning of Robust Factored MDPs" (AAAI 2026,
arXiv:2508.00707), Thm 1: nature chooses q_i = P(x_i' = 1) in [q_lo_i, q_hi_i] for every factor,
and the kernel is the product of the factors. The expected next value is multilinear in
(q_1..q_n), so its minimum is at a vertex of the box: we enumerate the 2^n vertices of every (s, a).

Tighter than the interval relaxation (compilation/product_bounds.py), but the kernels need
S * K * 2^n * 2^n float64 numbers: with K = n + 1 actions (SysAdmin), 1.2 GB at n = 8, 10.7 GB at
n = 9.
"""

import time

import numpy as np

from scm_to_mdp.core import FactorBounds, Solution, all_states
from scm_to_mdp.solving.stopping import sound_lower_bound, stop_at


def solve(bounds: FactorBounds, gamma: float, tol: float = 1e-7) -> Solution:
    if bounds.conditional:
        raise ValueError("vertex_robust_vi needs factors without same-step parents (use box_robust_vi)")
    if any(c != 2 for c in bounds.cards):
        raise ValueError("vertex_robust_vi needs 0/1 state variables")
    B, n = bounds.B, bounds.B.shape[1]
    q_lo, q_hi = bounds.lo[..., 1], bounds.hi[..., 1]  # [S, K, n] bounds of P(x_i' = 1)
    corners = all_states([2] * n)  # [C, n]: 0 = lower bound, 1 = upper bound of each factor
    q = np.where(corners[None, None] == 1, q_hi[:, :, None, :], q_lo[:, :, None, :])  # [S, K, C, n]
    # kernels[s, k, c, s'] = prod_i P(x_i' = s'_i) at vertex c, built factor by factor in place
    kernels = np.ones(q.shape[:3] + (B.shape[0],))
    for i in range(n):
        q_i = q[:, :, :, i, None]  # [S, K, C, 1]
        kernels *= np.where(B[:, i][None, None, None, :] == 1, q_i, 1 - q_i)
    V, start, sweeps = np.zeros(B.shape[0]), time.perf_counter(), 0
    while True:
        sweeps += 1
        Q = bounds.reward.R + (kernels @ bounds.reward.backup(V, gamma)).min(-1)  # nature: the worst vertex
        V_new = Q.max(1)
        step = np.abs(V_new - V).max()
        if step < stop_at(tol, gamma):
            seconds = (time.perf_counter() - start) / sweeps
            return Solution(
                sound_lower_bound(V_new, step, gamma), Q.argmax(1), iterations=sweeps, iteration_s=seconds
            )
        V = V_new
