"""Robust value iteration on an (s, a)-rectangular interval MDP (pipeline step S8).

Each row ``(s, a)`` has an interval box ``L <= p <= U`` with ``sum L <= 1 <= sum U``; nature picks any
distribution in the box, adversarially and afresh at every visit (dynamic nature, rectangular sets).
The robust Bellman operator is

    (T V)(s) = R(s) + gamma * max_a min_{L(s,a) <= p <= U(s,a), sum p = 1} p . V,

(Suilen et al. 2024, §3.2, Eq. 4, p. 11, discounted form p. 5). It is a gamma-contraction in the
sup norm (|min_p p.V - min_p p.V'| <= ||V - V'||), so value iteration converges to the robust value.
The inner minimum is solved exactly by the greedy method: start from ``p = L`` and give the remaining
mass ``1 - sum L`` to the next states with the lowest values, each up to ``U`` (Suilen et al. 2024,
Algorithm 1, p. 14, after Nilim & El Ghaoui 2005 and Katoen et al. 2012; it is a fractional knapsack,
checked against a linear program in the tests). Stationary deterministic policies suffice, and for
(s, a)-rectangular sets static and dynamic nature give the same value (*ibid.*, Lemma 1, p. 10).

The *promise* of a policy is its own robust value, ``robust_policy_value`` (not the stopped iterate of
value iteration). If the true kernel lies in every row's box, the promise is a lower bound on the
policy's true value: the true kernel is one choice of nature (sandwich lemma, Badings 2025, Lemma 3.32,
p. 44). With PAC sets the bound holds on the event that every row holds, with the sets' joint level.
Per-factor sets combined into a box (``lower = prod lower_i``, ``upper = prod upper_i``, clipped at 1)
form an outer relaxation of the exact factored set (interval closure, *ibid.*, §13.4, Thm 13.17,
p. 247), so the bound still holds.
"""
from __future__ import annotations

import numpy as np


def check_boxes(L: np.ndarray, U: np.ndarray, tol: float = 1e-9) -> None:
    """Raise unless ``0 <= L <= U <= 1`` and every row admits a distribution."""
    if L.shape != U.shape or L.ndim != 3:
        raise ValueError(f"L and U must both be [S, A, S']; got {L.shape} and {U.shape}")
    if np.any(L < -tol) or np.any(U > 1 + tol) or np.any(L > U + tol):
        raise ValueError("boxes must satisfy 0 <= L <= U <= 1")
    if np.any(L.sum(-1) > 1 + tol) or np.any(U.sum(-1) < 1 - tol):
        raise ValueError("every row needs sum L <= 1 <= sum U (an empty box has no distribution)")


def worst_case_distributions(L: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``argmin_{p in box} p . V`` for every row of ``[..., S']`` boxes (exact greedy solution)."""
    order = np.argsort(V, kind="stable")                    # lowest value first
    Ls, Us = L[..., order], U[..., order]
    free = 1.0 - Ls.sum(-1, keepdims=True)                   # mass still to place
    room = Us - Ls
    before = np.cumsum(room, axis=-1) - room                 # room of the lower-valued states
    extra = np.clip(free - before, 0.0, room)
    p_sorted = Ls + extra
    p = np.empty_like(p_sorted)
    p[..., order] = p_sorted
    return p


def robust_q(L: np.ndarray, U: np.ndarray, R: np.ndarray, gamma: float, V: np.ndarray) -> np.ndarray:
    """``Q(s, a) = R(s) + gamma * min_{p in box(s, a)} p . V``."""
    p = worst_case_distributions(L, U, V)
    return R[:, None] + gamma * (p @ V)


def robust_value_iteration(L: np.ndarray, U: np.ndarray, R: np.ndarray, gamma: float,
                           tol: float = 1e-10, max_iter: int = 100_000,
                           allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust ``V [S]``, ``Q [S, A]`` and a greedy deterministic policy ``[S]``.

    ``allowed`` (``[S, A]`` bool) masks actions that may not be chosen (all allowed by default).
    Stops when the sup-norm change is below ``tol * (1 - gamma)``, so ``|V - V_rob| <= tol``.
    """
    check_boxes(L, U)
    S, A, _ = L.shape
    mask = np.ones((S, A), dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = np.where(mask, robust_q(L, U, R, gamma, V), -np.inf)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol * (1.0 - gamma):
            Q = np.where(mask, robust_q(L, U, R, gamma, V_new), -np.inf)
            return V_new, Q, Q.argmax(axis=1)
        V = V_new
    raise RuntimeError("robust value iteration did not converge")


def robust_policy_value(L: np.ndarray, U: np.ndarray, R: np.ndarray, gamma: float, policy: np.ndarray,
                        tol: float = 1e-10, max_iter: int = 100_000) -> np.ndarray:
    """Robust value ``[S]`` of a fixed deterministic policy: the worst case over the boxes (its promise)."""
    check_boxes(L, U)
    rows = np.arange(L.shape[0])
    Lp, Up = L[rows, policy], U[rows, policy]
    V = np.zeros(L.shape[0])
    for _ in range(max_iter):
        V_new = R + gamma * (worst_case_distributions(Lp, Up, V) * V).sum(-1)
        if np.max(np.abs(V_new - V)) < tol * (1.0 - gamma):
            return V_new
        V = V_new
    raise RuntimeError("robust policy evaluation did not converge")
