"""Value iteration on an exact MDP (step S8; also gives V* on the true model)."""

import numpy as np

from scm_to_mdp.core import Reward, Solution
from scm_to_mdp.solving.stopping import stop_at


def solve(P: np.ndarray, reward: Reward, gamma: float, tol: float = 1e-8) -> Solution:
    """P [S, A, S']. Q(s, a) = R(s, a) + sum_s' P(s' | s, a) (r_next(s') + gamma V(s'))."""
    V = np.zeros(P.shape[0])
    while True:
        Q = reward.R + P @ reward.backup(V, gamma)  # [S, A]
        V_new = Q.max(1)
        if np.abs(V_new - V).max() < stop_at(tol, gamma):
            return Solution(V_new, Q.argmax(1))
        V = V_new


def policy_value(P: np.ndarray, reward: Reward, policy: np.ndarray, gamma: float) -> np.ndarray:
    """Exact value of a policy: V = R_pi + P_pi (r_next + gamma * c * V), with c = 0 on terminal
    states, i.e. solve (I - gamma P_pi diag(c)) V = R_pi + P_pi r_next. policy: [S] action indices
    (deterministic) or [S, A] action probabilities (stochastic)."""
    states = np.arange(P.shape[0])
    if policy.ndim == 2:
        P_pi, R_pi = np.einsum("sa,sat->st", policy, P), (policy * reward.R).sum(1)
    else:
        P_pi, R_pi = P[states, policy], reward.R[states, policy]
    goes_on = np.where(reward.terminal, 0.0, 1.0)
    return np.linalg.solve(np.eye(len(states)) - gamma * P_pi * goes_on[None, :], R_pi + P_pi @ reward.r_next)
