"""Exact tabular MDP routines (S8 for known models, S9 checks): value iteration and policy evaluation."""
from __future__ import annotations

import numpy as np


def value_iteration(P: np.ndarray, R: np.ndarray, gamma: float, tol: float = 1e-10,
                    max_iter: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """Optimal ``V [S]`` and ``Q [S, A]`` of ``P [S, A, S']`` with state reward ``R [S]``.

    Convention: ``Q(s, a) = R(s) + gamma * sum_s' P(s' | s, a) V(s')``.
    """
    V = np.zeros(P.shape[0])
    for _ in range(max_iter):
        Q = R[:, None] + gamma * P @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol * (1.0 - gamma):
            return V_new, R[:, None] + gamma * P @ V_new
        V = V_new
    raise RuntimeError("value iteration did not converge")


def policy_evaluation(P: np.ndarray, R: np.ndarray, gamma: float, policy: np.ndarray) -> np.ndarray:
    """Exact ``V [S]`` of a stochastic policy ``[S, A]`` (or a deterministic one ``[S]``)."""
    S, A = P.shape[:2]
    pi = np.asarray(policy)
    if pi.ndim == 1:
        pi = np.eye(A)[pi]
    P_pi = np.einsum("sa,sat->st", pi, P)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R)


def greedy(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy ``[S]`` (lowest action index on ties)."""
    return Q.argmax(axis=1)
