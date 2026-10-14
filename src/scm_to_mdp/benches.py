"""Building blocks for the per-step benches of E1: vary one pipeline step, keep the others fixed.

- **S4 (sets):** ``factor_bounds`` builds per-factor bounds with one of NBN's count-based builders
  (L1 balls with the Weissman radius; KL, Hoeffding or Clopper–Pearson intervals; a Dirichlet
  posterior box, which has no frequentist guarantee).
- **S7 (compilation):** ``boxes_from_factors`` turns per-factor bounds into joint-row boxes, by the
  product of the bounds (machines independent given ``(s, a)``) or by Fréchet bounds (any dependence).
- **S8 (solving):** ``factored_robust_vi`` solves the *exact* factored inner problem. With binary
  machines a factor's set is an interval on ``q_i = P(m_i' = 1)``; the joint probability of every next
  state is multilinear in ``(q_1, .., q_N)``, so ``p . V`` is multilinear too and its minimum over the
  product of intervals is attained at one of the ``2^N`` corners. Comparing it with robust value
  iteration on the box (``robust.robust_value_iteration``) measures how much the box relaxation loses.
- **S6 (criterion):** ``criterion_vi`` solves the same boxes under other declared criteria: optimistic
  (nature cooperates), Hurwicz (a mix of worst and best case), and the nominal model (the box's
  centre is replaced by the maximum-likelihood kernel by the caller). Only the worst case comes with
  a guarantee (promise <= true value); the others trade it for regret.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping

import numpy as np
import torch
from nbn.uncertainty.builders import (
    dirichlet_set_from_counts,
    interval_set_from_counts,
    l1_ball_set_from_counts,
)

from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.robust import worst_case_distributions
from scm_to_mdp.structure import Candidate, family_counts

BUILDERS = ("l1", "kl", "hoeffding", "clopper_pearson", "dirichlet")
GUARANTEED = {"l1", "kl", "hoeffding", "clopper_pearson"}


def factor_bounds(env: SysAdminRing, cols: Mapping[str, np.ndarray], parents: Mapping[str, Candidate],
                  builder: str, delta: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per node: ``(a, b)`` with ``a <= q <= b`` for ``q = P(node = 1 | parent row)``, rows in C order.

    The budget ``delta`` is split over the nodes. Unseen rows get ``[0, 1]``.
    """
    out = {}
    for node, pa in parents.items():
        counts = torch.as_tensor(family_counts(cols, node, tuple(pa)), dtype=torch.float64)
        d = delta / len(parents)
        if builder == "l1":
            ps = l1_ball_set_from_counts(counts, delta=d, sampling="trajectory")
        elif builder in ("kl", "hoeffding", "clopper_pearson"):
            ps = interval_set_from_counts(counts, delta=d, method=builder, sampling="trajectory")
        elif builder == "dirichlet":
            torch.manual_seed(0)
            ps = dirichlet_set_from_counts(counts, delta=d, prior=1.0, n_samples=20_000)
        else:
            raise ValueError(f"unknown builder {builder!r}; known: {BUILDERS}")
        lo, hi = (t.detach().cpu().numpy().astype(float) for t in ps.bounds())
        a, b = lo[:, 1].copy(), hi[:, 1].copy()
        unseen = counts.sum(-1).numpy() == 0
        a[unseen], b[unseen] = 0.0, 1.0
        out[node] = (np.clip(a, 0.0, 1.0), np.clip(b, 0.0, 1.0))
    return out


def q_intervals(env: SysAdminRing, parents: Mapping[str, Candidate],
                bounds: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """``[S, A, N]`` lower and upper ``q_i`` for every (s, a), read through each factor's parent row."""
    S, A = env.n_states, env.n_actions
    flags = env.flags(np.arange(A))
    values = {**{m: env.states[:, None, k] for k, m in enumerate(env.state_vars)},
              **{r: flags[None, :, k] for k, r in enumerate(env.action_flags)}}
    qa, qb = np.zeros((S, A, env.n)), np.zeros((S, A, env.n))
    for i, v in enumerate(env.state_vars):
        node, pa = f"{v}'", tuple(parents[f"{v}'"])
        idx = (np.ravel_multi_index([np.broadcast_to(values[p], (S, A)) for p in pa], [2] * len(pa))
               if pa else np.zeros((S, A), dtype=np.int64))
        a, b = bounds[node]
        qa[:, :, i], qb[:, :, i] = a[idx], b[idx]
    return qa, qb


def _joint_from_q(env: SysAdminRing, q: np.ndarray) -> np.ndarray:
    """``[..., S']`` joint next-state distribution of independent machines with ``P(m_i' = 1) = q_i``."""
    nxt = env.states                                                  # [S', N]
    return np.where(nxt == 1, q[..., None, :], 1.0 - q[..., None, :]).prod(-1)


def boxes_from_factors(env: SysAdminRing, qa: np.ndarray, qb: np.ndarray,
                       joint: str = "product") -> tuple[np.ndarray, np.ndarray]:
    """S7: joint-row boxes ``L, U [S, A, S']`` from per-factor intervals on ``q``."""
    nxt = env.states[None, None, :, :]                                # [1, 1, S', N]
    lo_i = np.where(nxt == 1, qa[:, :, None, :], 1.0 - qb[:, :, None, :])
    hi_i = np.where(nxt == 1, qb[:, :, None, :], 1.0 - qa[:, :, None, :])
    if joint == "product":
        return lo_i.prod(-1), np.minimum(hi_i.prod(-1), 1.0)
    if joint == "frechet":
        return np.maximum(0.0, lo_i.sum(-1) - (env.n - 1)), np.minimum(hi_i.min(-1), 1.0)
    raise ValueError(f"joint must be 'product' or 'frechet'; got {joint!r}")


def _corner_distributions(env: SysAdminRing, qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """``[S, A, 2^N, S']``: the joint distribution at every corner of each row's factor box."""
    corners = np.array(list(itertools.product((0, 1), repeat=env.n)), dtype=bool)   # [C, N]
    q = np.where(corners[None, None], qb[:, :, None, :], qa[:, :, None, :])         # [S, A, C, N]
    return _joint_from_q(env, q)


def factored_robust_vi(env: SysAdminRing, qa: np.ndarray, qb: np.ndarray, R: np.ndarray, gamma: float,
                       tol: float = 1e-10, max_iter: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """S8 exact factored: robust ``V [S]`` and a greedy policy, nature picking ``q_i`` in each interval."""
    P = _corner_distributions(env, qa, qb)                            # [S, A, C, S']
    V = np.zeros(env.n_states)
    for _ in range(max_iter):
        Q = R[:, None] + gamma * (P @ V).min(-1)
        V_new = Q.max(1)
        if np.max(np.abs(V_new - V)) < tol * (1.0 - gamma):
            Q = R[:, None] + gamma * (P @ V_new).min(-1)
            return V_new, Q.argmax(1)
        V = V_new
    raise RuntimeError("factored robust value iteration did not converge")


def factored_policy_value(env: SysAdminRing, qa: np.ndarray, qb: np.ndarray, R: np.ndarray, gamma: float,
                          policy: np.ndarray, tol: float = 1e-10, max_iter: int = 100_000) -> np.ndarray:
    """Exact factored robust value of a fixed deterministic policy (its promise under the factored set)."""
    rows = np.arange(env.n_states)
    P = _corner_distributions(env, qa, qb)[rows, policy]              # [S, C, S']
    V = np.zeros(env.n_states)
    for _ in range(max_iter):
        V_new = R + gamma * (P @ V).min(-1)
        if np.max(np.abs(V_new - V)) < tol * (1.0 - gamma):
            return V_new
        V = V_new
    raise RuntimeError("factored robust policy evaluation did not converge")


def criterion_vi(L: np.ndarray, U: np.ndarray, R: np.ndarray, gamma: float, pessimism: float,
                 tol: float = 1e-10, max_iter: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    """S6: value iteration under the Hurwicz criterion on the boxes.

    ``pessimism = 1``: worst case (robust); ``0``: optimistic (nature cooperates); in between, a fixed
    mix of the two inner values. Returns ``V [S]`` and a greedy policy.
    """
    if not 0.0 <= pessimism <= 1.0:
        raise ValueError("pessimism must lie in [0, 1]")
    V = np.zeros(L.shape[0])
    for _ in range(max_iter):
        worst = (worst_case_distributions(L, U, V) * V).sum(-1)
        best = -(worst_case_distributions(L, U, -V) * -V).sum(-1)
        Q = R[:, None] + gamma * (pessimism * worst + (1 - pessimism) * best)
        V_new = Q.max(1)
        if np.max(np.abs(V_new - V)) < tol * (1.0 - gamma):
            return V_new, Q.argmax(1)
        V = V_new
    raise RuntimeError("criterion value iteration did not converge")
