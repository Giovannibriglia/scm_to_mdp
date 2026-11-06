"""Value iteration on an exact MDP (step S8; also gives V* on the true model)."""

import time

import numpy as np

from scm_to_mdp.core import Reward, Solution
from scm_to_mdp.solving.stopping import greedy, stop_at, value_floor


def solve(
    P: np.ndarray,
    reward: Reward,
    gamma: float,
    tol: float = 1e-8,
    allowed: np.ndarray | None = None,
    fixed_v: np.ndarray | None = None,
    fixed_q: np.ndarray | None = None,
) -> Solution:
    """P [S, A, S']. Q(s, a) = R(s, a) + sum_s' P(s' | s, a) (r_next(s') + gamma V(s')). allowed [S, A]
    (optional): the max only over these actions, a state with none at the value floor (stopping.greedy).
    fixed_v, fixed_q (optional): the data-only sinks (stopping.data_only_values)."""
    V, start, sweeps = np.zeros(P.shape[0]), time.perf_counter(), 0
    floor = value_floor(reward, gamma) if allowed is not None else None
    while True:
        Q = reward.R + P @ reward.backup(V, gamma)  # [S, A]
        V_new, pi = greedy(Q, allowed, floor, fixed_v, fixed_q)
        sweeps += 1
        if np.abs(V_new - V).max() < stop_at(tol, gamma):
            seconds = (time.perf_counter() - start) / sweeps
            return Solution(V_new, pi, iterations=sweeps, iteration_s=seconds)
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
