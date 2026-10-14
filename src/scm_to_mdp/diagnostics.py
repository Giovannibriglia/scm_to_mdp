"""What a dataset can tell (S1 profiling): coverage, overlap between loggers, aliasing, concentrability.

These numbers characterise the loggers independently of the learning arms, so that results on the
SysAdmin ring can be related to other domains.
"""
from __future__ import annotations

import numpy as np

from scm_to_mdp.data import Transitions
from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.mdp import value_iteration


def row_counts(env: SysAdminRing, data: Transitions) -> np.ndarray:
    """``[S, A]`` visits of every state-action row."""
    counts = np.zeros((env.n_states, env.n_actions), dtype=np.int64)
    np.add.at(counts, (env.index(data.state), data.action), 1)
    return counts


def factor_counts(env: SysAdminRing, data: Transitions, i: int) -> np.ndarray:
    """``[2, 2, 2]`` visits of the true parent values ``(m_i, m_{i-1}, rep_i)`` of machine ``i``."""
    counts = np.zeros((2, 2, 2), dtype=np.int64)
    rep = env.flags(data.action)[:, i]
    np.add.at(counts, (data.state[:, i], data.state[:, (i - 1) % env.n], rep), 1)
    return counts


def factor_covered_rows(env: SysAdminRing, data: Transitions) -> np.ndarray:
    """``[S, A]`` rows whose true parent values are visited for every machine."""
    ok = np.ones((env.n_states, env.n_actions), dtype=bool)
    st = env.states
    for i in range(env.n):
        seen = factor_counts(env, data, i) > 0
        rep = env.flags(np.arange(env.n_actions))[:, i]                 # [A]
        ok &= seen[st[:, i][:, None], st[:, (i - 1) % env.n][:, None], rep[None, :]]
    return ok


def reachable_rows(env: SysAdminRing) -> np.ndarray:
    """``[S, A]``: every row is reachable from all-up on the ring (all states are reachable)."""
    return np.ones((env.n_states, env.n_actions), dtype=bool)


def aliasing_index(env: SysAdminRing, data: Transitions, i: int) -> float:
    """How well ``rep_i`` determines ``m_{i-1}`` in the data: normalised mutual information in [0, 1].

    ``I(rep_i; m_{i-1}) / min(H(rep_i), H(m_{i-1}))``: 1 when one is a function of the other, 0 when
    independent (``0 <= I <= H``, with equality for a deterministic function: Koller & Friedman,
    *PGM*, Prop. A.4, p. 1138; the normalisation is ours). When ``rep_i`` is a one-to-one function of
    ``m_{i-1}``, the parent sets with either of them have the same likelihood and BIC score (the score
    depends on the data only through ``I(m_i'; Pa)``, *ibid.*, Prop. 18.1, pp. 791-793), so the data
    cannot tell them apart. See also :func:`aliasing_empty_cells`.
    """
    x = env.flags(data.action)[:, i].astype(np.int64)
    y = data.state[:, (i - 1) % env.n].astype(np.int64)
    joint = np.zeros((2, 2))
    np.add.at(joint, (x, y), 1)
    joint /= max(joint.sum(), 1)
    px, py = joint.sum(1), joint.sum(0)

    def h(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / np.outer(px, py)[nz])).sum())
    denom = min(h(px), h(py))
    return mi / denom if denom > 0 else 1.0


def aliasing_empty_cells(env: SysAdminRing, data: Transitions) -> int:
    """Largest number of empty cells in the 2x2 table of ``(rep_i, m_{i-1})`` over machines.

    An empty cell is a parent configuration never seen, so its conditional probabilities are not
    estimable from the data (the likelihood does not depend on them; Koller & Friedman, *PGM*,
    Eq. 17.4, p. 726): only a prior or the structure fills them.
    """
    worst = 0
    flags = env.flags(data.action)
    for i in range(env.n):
        table = np.zeros((2, 2), dtype=np.int64)
        np.add.at(table, (flags[:, i], data.state[:, (i - 1) % env.n]), 1)
        worst = max(worst, int((table == 0).sum()))
    return worst


def single_policy_concentrability(env: SysAdminRing, data: Transitions) -> float:
    """Single-policy concentrability: ``max_{s,a} d^{pi*}(s, a) / mu_D(s, a)`` over rows ``pi*`` visits.

    ``d^{pi*}`` is the normalised discounted occupancy of the optimal policy from all-up, ``mu_D`` the
    empirical row frequency of the dataset (inf if a row ``pi*`` visits never appears). This is the
    ratio of Agarwal et al., *RL: Theory and Algorithms*, Assumption 4.1 (p. 54) restricted to
    ``pi = pi*``; the book's coefficient takes the sup over *all* policies and assumes i.i.d. samples
    from a fixed ``nu``. Here ``mu_D`` is a plug-in estimate of the loggers' mixture over trajectories,
    so the number is a descriptive diagnostic, not the constant of a theorem.
    """
    P, R = env.kernel(), env.reward
    _, Q = value_iteration(P, R, env.gamma)
    pi = Q.argmax(1)
    S = env.n_states
    P_pi = P[np.arange(S), pi]                                          # [S, S']
    mu = np.zeros(S)
    mu[env.index(env.initial_state)] = 1.0
    d_s = (1 - env.gamma) * np.linalg.solve(np.eye(S) - env.gamma * P_pi.T, mu)
    d_opt = np.zeros((S, env.n_actions))
    d_opt[np.arange(S), pi] = d_s
    counts = row_counts(env, data)
    d_data = counts / max(counts.sum(), 1)
    mask = d_opt > 1e-12
    if np.any(d_data[mask] == 0):
        return float("inf")
    return float(np.max(d_opt[mask] / d_data[mask]))


def profile(env: SysAdminRing, data: Transitions) -> dict[str, float]:
    """One line of numbers that describe what a dataset covers and can separate."""
    counts = row_counts(env, data)
    joint = counts > 0
    factor = factor_covered_rows(env, data)
    rows = env.n_states * env.n_actions
    return {
        "transitions": len(data),
        "joint_rows": int(joint.sum()) / rows,
        "factor_rows": int(factor.sum()) / rows,
        "factor_only_rows": int((factor & ~joint).sum()) / rows,
        "min_count_visited": int(counts[joint].min()) if joint.any() else 0,
        "aliasing_max": max(aliasing_index(env, data, i) for i in range(env.n)),
        "aliasing_empty_cells": aliasing_empty_cells(env, data),
        "concentrability_pi_star": single_policy_concentrability(env, data),
    }


def overlap(env: SysAdminRing, a: Transitions, b: Transitions) -> float:
    """Jaccard overlap of the visited rows of two datasets."""
    ra, rb = row_counts(env, a) > 0, row_counts(env, b) > 0
    union = (ra | rb).sum()
    return float((ra & rb).sum() / union) if union else 0.0


def observational_gap(env: SysAdminRing, data: Transitions, z: float = 4.0, min_rows: int = 30) -> float:
    """Largest excess of |P_data(m_i' = up | parents) - P_do(m_i' = up | parents)| over ``z`` standard
    errors, over parent values with at least ``min_rows`` rows (the two differ under unobserved
    confounding: Bareinboim, *Causal AI* v0.27, Eq. 9.137, p. 623).

    About 0 when no logger is confounded (the gap is sampling noise); positive when a logger reads the
    hidden load that also moves the dynamics (the data then show P(s' | s, a), not P(s' | s, do(a))).
    """
    excess = 0.0
    rep_all = env.flags(data.action)
    for i in range(env.n):
        up, left, rep = data.state[:, i], data.state[:, (i - 1) % env.n], rep_all[:, i]
        for u in (0, 1):
            for lft in (0, 1):
                for r in (0, 1):
                    m = (up == u) & (left == lft) & (rep == r)
                    n = int(m.sum())
                    if n < min_rows:
                        continue
                    p_obs = float(data.next_state[m, i].mean())
                    p_do = float(sum(pr * env.p_up(u, lft, r, wv) for wv, pr in env._load_values()))
                    se = (max(p_do * (1 - p_do), 1e-12) / n) ** 0.5
                    excess = max(excess, abs(p_obs - p_do) - z * se)
    return excess
