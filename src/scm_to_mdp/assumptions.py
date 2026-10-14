"""Checks of the pipeline's assumptions from the data alone (pipeline step S1: the dataset profile).

Unlike :mod:`scm_to_mdp.diagnostics`, nothing here uses the true model: these are the tests the
pipeline itself runs before discovery. Each test compares ``P(m_i' | all inputs)`` across groups of
rows with the invariance test of :mod:`scm_to_mdp.structure` (G statistic within each input
configuration). Conditioning on *all* slice-t inputs is a superset of every target's parents, so under
the null the next states in one configuration are draws from the same law. On trajectories later
configurations depend on earlier outcomes, so the permutation level is approximate rather than exact;
it stays close to nominal in simulations (the E1 false-alarm rates are about nominal).

- **Confounding gate (policy as context).** Groups = loggers. Under no unobserved confounding the
  kernel is the same under every logger (Bareinboim, *Causal AI* v0.27, Def. 8.2.1, p. 547), so a
  rejection means confounding or a changed mechanism (JCI: Mooij, Magliacane & Claassen 2020). A
  non-rejection certifies nothing, and loggers confounded in the same way are invisible to it.
- **Markov (first order).** Groups = the previous step's values of the machine and its repair flag.
  The 2-slice model assumes the next state depends on the past only through ``(s_t, a_t)``
  (Koller & Friedman, *PGM*, Def. 6.1, p. 201).
- **Stationarity.** Groups = early vs late half of the episode: the same transition model at every
  step (*ibid.*, Def. 6.2, p. 202).

p-values are per target; a family-wise decision uses a Bonferroni split over the targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scm_to_mdp.data import Transitions
from scm_to_mdp.envs.sysadmin import SysAdminRing
from scm_to_mdp.structure import invariance_pvalue


@dataclass
class AssumptionReport:
    gate: dict[str, float] = field(default_factory=dict)            # target -> p (loggers)
    markov: dict[str, float] = field(default_factory=dict)          # target -> p (previous step)
    stationarity: dict[str, float] = field(default_factory=dict)    # target -> p (episode halves)
    episodes: int = 0                                               # effective sample size unit
    episodes_per_logger: dict[int, int] = field(default_factory=dict)
    joint_coverage: float = 0.0                                     # share of (s, a) rows seen
    alpha: float = 0.05

    def rejected(self, test: str) -> bool:
        """Family-wise decision at ``alpha`` (Bonferroni over targets)."""
        p = getattr(self, test)
        return bool(p) and min(p.values()) <= self.alpha / len(p)


def _inputs(env: SysAdminRing) -> tuple[str, ...]:
    return tuple(env.state_vars + env.action_flags)


def confounding_gate(env: SysAdminRing, data: Transitions, cols=None) -> dict[str, float]:
    cols = data.columns(env) if cols is None else cols
    return {f"m{i}'": invariance_pvalue(cols, f"m{i}'", _inputs(env), data.regime) for i in range(env.n)}


def _previous_step(env: SysAdminRing, data: Transitions) -> tuple[np.ndarray, np.ndarray]:
    """Index of the previous transition of the same episode (``-1`` at ``t = 0``)."""
    ep = data.episode_ids()
    order = np.lexsort((data.t, ep))                       # by episode, then time
    same = (np.diff(ep[order]) == 0) & (np.diff(data.t[order]) == 1)
    prev = np.full(len(data), -1)
    prev[order[1:][same]] = order[:-1][same]
    return prev, prev >= 0


def markov_test(env: SysAdminRing, data: Transitions, cols=None) -> dict[str, float]:
    prev, has = _previous_step(env, data)
    cols = {k: v[has] for k, v in (data.columns(env) if cols is None else cols).items()}
    flags = env.flags(data.action)
    out = {}
    for i in range(env.n):
        group = 2 * data.state[prev[has], i] + flags[prev[has], i]   # (m_i, rep_i) one step earlier
        out[f"m{i}'"] = invariance_pvalue(cols, f"m{i}'", _inputs(env), group)
    return out


def stationarity_test(env: SysAdminRing, data: Transitions, cols=None) -> dict[str, float]:
    half = (data.t >= (data.t.max() + 1) / 2).astype(np.int64)
    cols = data.columns(env) if cols is None else cols
    return {f"m{i}'": invariance_pvalue(cols, f"m{i}'", _inputs(env), half) for i in range(env.n)}


def profile_assumptions(env: SysAdminRing, data: Transitions, alpha: float = 0.05) -> AssumptionReport:
    from scm_to_mdp.diagnostics import row_counts

    cols = data.columns(env)
    episodes = {int(r): int(np.unique(data.episode[data.regime == r]).size) for r in np.unique(data.regime)}
    return AssumptionReport(
        gate=confounding_gate(env, data, cols), markov=markov_test(env, data, cols),
        stationarity=stationarity_test(env, data, cols), episodes=sum(episodes.values()),
        episodes_per_logger=episodes, joint_coverage=float((row_counts(env, data) > 0).mean()), alpha=alpha,
    )


def episode_split(data: Transitions, frac: float = 0.5, seed: int = 0) -> tuple[Transitions, Transitions]:
    """Split by whole episodes (within each logger): (discovery part, estimation part)."""
    rng = np.random.default_rng(seed)
    first = np.zeros(len(data), dtype=bool)
    for r in np.unique(data.regime):
        eps = np.unique(data.episode[data.regime == r])
        pick = rng.permutation(eps)[: int(round(frac * eps.size))]
        first |= (data.regime == r) & np.isin(data.episode, pick)
    return data.take(first), data.take(~first)
