"""Factor intervals -> interval MDP by products of bounds (step S7).

The state variables move independently given (s, a), so P(s' | s, a) = prod_i P(x_i' = s'_i | s, a).
Each factor lies in its interval, hence

    L(s' | s, a) = prod_i lo_i(s'_i),   U(s' | s, a) = min(1, prod_i hi_i(s'_i)).

First every factor's box is tightened with the fact that its probabilities sum to one:
lo_v >= 1 - sum_{w != v} hi_w and hi_v <= 1 - sum_{w != v} lo_w (as NBN's joint_row_bounds; no
change for 0/1 variables). This is the interval-arithmetic (IA) relaxation of Schnitzer, Abate,
Parker (AAAI 2026, arXiv:2508.00707). It is sound (every product of factors in their intervals lies
in the box) but forgets that the joint must be a product, so its worst case can be strictly worse
than that of any product kernel (solving/vertex_robust_vi.py keeps the product, for 0/1 variables).

A factor with same-step parents (FactorBounds.conditional) enters with its conditional row at the next
state's values: P(s' | s, a) = prod_i P(x_i' = s'_i | s, a, x_js' = s'_js). Each conditional row is a
probability vector, tightened the same way, and the product of the lower (upper) ends bounds the
product from below (above): the box is still sound.
"""

import numpy as np

from scm_to_mdp.core import FactorBounds, IntervalMDP, encode


def tighten(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Box of a probability vector (last axis) intersected with the simplex, entry-wise."""
    lo_t = np.maximum(lo, 1.0 - (hi.sum(-1, keepdims=True) - hi))
    hi_t = np.minimum(hi, 1.0 - (lo.sum(-1, keepdims=True) - lo))
    return np.clip(lo_t, 0.0, 1.0), np.clip(hi_t, 0.0, 1.0)


def factor_at_next(bounds: FactorBounds, i: int, table: np.ndarray) -> np.ndarray:
    """[S, K, S'] factor i's entry for every next state s': table [S, K, card] (no same-step parents) or
    [S, K, R, card] (a conditional: the row of the same-step parents' values in s')."""
    B = bounds.B
    if i not in bounds.conditional:
        return table[:, :, B[:, i]]
    js = bounds.conditional[i][0]
    r = encode(B[:, list(js)], [bounds.cards[j] for j in js])  # [S'] the parents' row in s'
    return table[:, :, r, B[:, i]]


def compile(bounds: FactorBounds) -> IntervalMDP:  # noqa: A001 - the name of the step
    S, K, n, _ = bounds.lo.shape
    L, U = np.ones((S, K, S)), np.ones((S, K, S))
    for i in range(n):
        card = bounds.cards[i]
        if i in bounds.conditional:
            _, lo_i, hi_i = bounds.conditional[i]
            lo, hi = tighten(lo_i[..., :card], hi_i[..., :card])  # [S, K, R, card]
        else:
            lo, hi = tighten(bounds.lo[:, :, i, :card], bounds.hi[:, :, i, :card])  # [S, K, card]
        L *= factor_at_next(bounds, i, lo)  # factor i at the value of x_i in every next state s'
        U *= factor_at_next(bounds, i, hi)
    return IntervalMDP(L, np.minimum(U, 1.0), bounds.reward, seen=bounds.seen, data_only=bounds.data_only)


def exact(bounds: FactorBounds) -> IntervalMDP:
    """A point model (lo == hi, scm_learning/nbn_point.py) -> its exact MDP, P = prod_i P_i (L == U).
    No tightening: each factor's estimate is already a distribution."""
    point = [np.array_equal(lo_i, hi_i) for _, lo_i, hi_i in bounds.conditional.values()]
    plain = [i for i in range(bounds.lo.shape[2]) if i not in bounds.conditional]
    if not all(point) or not np.array_equal(bounds.lo[:, :, plain], bounds.hi[:, :, plain]):
        raise ValueError("exact() needs point estimates (lo == hi); use compile() for interval sets")
    S, K, n, _ = bounds.lo.shape
    P = np.ones((S, K, S))
    for i in range(n):
        card = bounds.cards[i]
        table = bounds.conditional[i][1][..., :card] if i in bounds.conditional else bounds.lo[:, :, i, :card]
        P *= factor_at_next(bounds, i, table)
    return IntervalMDP(P, P.copy(), bounds.reward, seen=bounds.seen, data_only=bounds.data_only)
