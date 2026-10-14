"""Structure learning per target under a known time order (pipeline steps S2 and S3).

In a 2-slice model with edges only from slice t to slice t + 1, choosing the parents of a next-state
variable is a separate variable-selection problem per target (Peters, Janzing & Schölkopf, *ECI*,
§7.2.4, p. 152; Koller & Friedman, *PGM*, Prop. 18.4, p. 810). With few inputs every candidate parent
set of size <= d can be scored or tested exactly. This module returns, per target:

- ``best_bic``: the BIC-optimal parent set (one learned graph; consistent, K&F Thm 18.2, p. 803, but
  it drops weak parents at finite data, p. 802);
- ``posterior`` / ``credible_family``: the exact posterior over parent sets (BDeu marginal likelihood,
  uniform structure prior; K&F Prop. 18.7, p. 828) and its smallest credible family;
- ``icp_family``: invariant causal prediction with the loggers as environments; the accepted sets of
  size <= d contain the true parent set with probability >= 1 - alpha (ECI, §7.2.5, eq. 7.8,
  pp. 153-154). ICP's intersection is *not* returned: it controls false parents only and can miss true ones.

Inputs and targets are binary columns (``data.columns(env)``); a candidate is a tuple of input names.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.special import gammaln
from scipy.stats import chi2

Candidate = tuple[str, ...]


def candidate_sets(inputs: Sequence[str], d: int) -> list[Candidate]:
    """Every subset of ``inputs`` of size ``0 .. d`` (sorted by size, then input order)."""
    return [c for k in range(d + 1) for c in itertools.combinations(inputs, k)]


def _config_index(cols: Mapping[str, np.ndarray], parents: Candidate, cards, n_rows: int):
    """Row index of each configuration of ``parents`` (C order) and the number of configurations."""
    pc = [int((cards or {}).get(p, 2)) for p in parents]
    if not parents:
        return np.zeros(n_rows, dtype=np.int64), 1
    return np.ravel_multi_index([cols[p].astype(np.int64) for p in parents], pc), math.prod(pc)


def family_counts(cols: Mapping[str, np.ndarray], target: str, parents: Candidate,
                  cards: Mapping[str, int] | None = None) -> np.ndarray:
    """``[R, K]`` counts of ``target`` per configuration of ``parents`` (C order, last parent fastest)."""
    k = int((cards or {}).get(target, 2))
    rows, n_conf = _config_index(cols, parents, cards, len(cols[target]))
    counts = np.zeros((n_conf, k), dtype=np.int64)
    np.add.at(counts, (rows, cols[target].astype(np.int64)), 1)
    return counts


def bic_score(counts: np.ndarray) -> float:
    """Log-likelihood at the MLE minus ``(log M / 2) * (number of free parameters)`` (K&F p. 802)."""
    m = counts.sum()
    n_row = counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.where(counts > 0, counts * np.log(counts / n_row), 0.0).sum()
    dim = counts.shape[0] * (counts.shape[1] - 1)
    return float(ll - 0.5 * math.log(max(m, 1)) * dim)


def bdeu_score(counts: np.ndarray, ess: float = 1.0) -> float:
    """Log marginal likelihood of one family under the BDeu prior (equivalent sample size ``ess``)."""
    r, k = counts.shape
    a_row, a_cell = ess / r, ess / (r * k)
    n_row = counts.sum(1)
    return float((gammaln(a_row) - gammaln(a_row + n_row)).sum()
                 + (gammaln(a_cell + counts) - gammaln(a_cell)).sum())


def best_bic(cols, target: str, candidates: Sequence[Candidate], cards=None) -> Candidate:
    scores = [bic_score(family_counts(cols, target, c, cards)) for c in candidates]
    return candidates[int(np.argmax(scores))]


def posterior(cols, target: str, candidates: Sequence[Candidate], ess: float = 1.0,
              cards=None) -> dict[Candidate, float]:
    """Exact posterior over ``candidates`` (uniform prior over them), K&F Prop. 18.7, p. 828."""
    log_ml = np.array([bdeu_score(family_counts(cols, target, c, cards), ess) for c in candidates])
    w = np.exp(log_ml - log_ml.max())
    w /= w.sum()
    return dict(zip(candidates, w.tolist(), strict=True))


def credible_family(post: Mapping[Candidate, float], mass: float) -> list[Candidate]:
    """Smallest family of candidates whose posterior mass reaches ``mass``."""
    ranked = sorted(post.items(), key=lambda kv: -kv[1])
    out, total = [], 0.0
    for c, p in ranked:
        out.append(c)
        total += p
        if total >= mass - 1e-12:
            break
    return out


def _g_stat(tables: np.ndarray) -> np.ndarray:
    """G statistic of ``[..., R, K]`` contingency tables (independence of rows and columns)."""
    tot = tables.sum((-2, -1), keepdims=True)
    exp = tables.sum(-1, keepdims=True) * tables.sum(-2, keepdims=True) / np.maximum(tot, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2 * np.where(tables > 0, tables * np.log(tables / exp), 0.0).sum((-2, -1))


def invariance_pvalue(cols, target: str, parents: Candidate, regime: np.ndarray, cards=None,
                      min_rows: int = 5, reference: str = "permutation", n_perm: int = 999,
                      seed: int = 0) -> float:
    """p-value of "``P(target | parents)`` is the same in every group of ``regime``".

    For each configuration of ``parents`` with at least two groups of ``min_rows`` rows, a
    group-by-target G statistic; the statistics are summed. Given the true parents (or a superset)
    the next states within a configuration are independent draws whatever their group, so the group
    labels are exchangeable within each configuration. ``reference="permutation"`` (default) uses that:
    each configuration's table is redrawn with its margins fixed (multivariate hypergeometric), which
    gives a valid level at any sample size for independent rows (approximate on trajectories, where
    later configurations depend on earlier outcomes). ``reference="chi2"`` uses the chi-square approximation,
    which is anti-conservative when cells are sparse. With no testable configuration the p-value is 1.
    """
    if reference not in ("permutation", "chi2"):
        raise ValueError("reference must be 'permutation' or 'chi2'")
    groups = np.unique(regime)
    if groups.size < 2:
        return 1.0
    k = int((cards or {}).get(target, 2))
    rows, n_conf = _config_index(cols, parents, cards, len(regime))
    gidx = np.searchsorted(groups, regime)
    table = np.zeros((n_conf, groups.size, k), dtype=np.int64)
    np.add.at(table, (rows, gidx, cols[target].astype(np.int64)), 1)
    tables = []
    for t in table:
        t = t[t.sum(1) >= min_rows]
        t = t[:, t.sum(0) > 0]
        if t.shape[0] >= 2 and t.shape[1] >= 2:
            tables.append(t)
    if not tables:
        return 1.0
    g_obs = float(sum(_g_stat(t.astype(float)) for t in tables))
    if reference == "chi2":
        df = sum((t.shape[0] - 1) * (t.shape[1] - 1) for t in tables)
        return float(chi2.sf(g_obs, df))
    rng = np.random.default_rng(seed)
    g_sim = np.zeros(n_perm)
    for t in tables:
        sim = np.empty((n_perm,) + t.shape)
        remaining = np.broadcast_to(t.sum(0), (n_perm, t.shape[1])).astype(np.int64).copy()
        for r, size in enumerate(t.sum(1)):                 # draw each group's classes in turn
            left = np.full(n_perm, int(size))
            for c in range(t.shape[1]):                     # multivariate hypergeometric, one class at a time
                rest = remaining[:, c + 1:].sum(1)
                draw = left if c == t.shape[1] - 1 else rng.hypergeometric(remaining[:, c], rest, left)
                sim[:, r, c] = draw
                remaining[:, c] -= draw
                left = left - draw
        g_sim += _g_stat(sim)
    return float((1 + np.sum(g_sim >= g_obs - 1e-9)) / (n_perm + 1))


def icp_family(cols, target: str, candidates: Sequence[Candidate], regime: np.ndarray,
               alpha: float, cards=None) -> list[Candidate]:
    """Candidates not rejected at level ``alpha``: contains the true parent set w.p. >= 1 - alpha."""
    return [c for c in candidates if invariance_pvalue(cols, target, c, regime, cards) > alpha]


def contains_superset(family: Sequence[Candidate], true_parents: Candidate) -> bool:
    """Whether some candidate contains every true parent (the superset rule: valid conditional)."""
    tp = set(true_parents)
    return any(tp <= set(c) for c in family)
