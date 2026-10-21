"""Factor intervals -> interval MDP by products of bounds (step S7).

The state variables move independently given (s, a), so P(s' | s, a) = prod_i P(x_i' = s'_i | s, a).
Each factor lies in its interval, hence

    L(s' | s, a) = prod_i lo_i(s'_i),   U(s' | s, a) = min(1, prod_i hi_i(s'_i)).

First every factor's box is tightened with the fact that its probabilities sum to one:
lo_v >= 1 - sum_{w != v} hi_w and hi_v <= 1 - sum_{w != v} lo_w (as NBN's joint_row_bounds; no
change for 0/1 variables). This is the interval-arithmetic (IA) relaxation of Schnitzer, Abate,
Parker (AAAI 2026, arXiv:2508.00707). It forgets that the joint must be a product, so its worst case
can be worse than any product kernel (solving/vertex_robust_vi.py keeps the product).
"""

import numpy as np

from scm_to_mdp.core import FactorBounds, IntervalMDP


def tighten(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Box of a probability vector (last axis) intersected with the simplex, entry-wise."""
    lo_t = np.maximum(lo, 1.0 - (hi.sum(-1, keepdims=True) - hi))
    hi_t = np.minimum(hi, 1.0 - (lo.sum(-1, keepdims=True) - lo))
    return np.clip(lo_t, 0.0, 1.0), np.clip(hi_t, 0.0, 1.0)


def compile(bounds: FactorBounds) -> IntervalMDP:  # noqa: A001 - the name of the step
    S, K, n, _ = bounds.lo.shape
    L, U = np.ones((S, K, S)), np.ones((S, K, S))
    for i in range(n):
        card = bounds.cards[i]
        lo, hi = tighten(bounds.lo[:, :, i, :card], bounds.hi[:, :, i, :card])  # [S, K, card]
        L *= lo[:, :, bounds.B[:, i]]  # factor i at the value of x_i in every next state s'
        U *= hi[:, :, bounds.B[:, i]]
    return IntervalMDP(L, np.minimum(U, 1.0), bounds.reward)
