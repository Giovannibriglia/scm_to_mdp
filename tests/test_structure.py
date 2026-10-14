"""S2/S3 per-target structure learning: BIC, exact posterior, ICP with loggers as environments."""
from __future__ import annotations

import numpy as np
import pytest

from scm_to_mdp.data import collect, rollout
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger
from scm_to_mdp.structure import (
    bdeu_score,
    best_bic,
    candidate_sets,
    contains_superset,
    credible_family,
    family_counts,
    icp_family,
    invariance_pvalue,
    posterior,
)

ENV = SysAdminRing(n=4)
INPUTS = ENV.state_vars + ENV.action_flags
CANDS = candidate_sets(INPUTS, 3)
TRUE = ENV.true_parents()


def _uniform(episodes=400, seed=0):
    return rollout(ENV, make_logger("uniform", ENV), episodes, 30, np.random.default_rng(seed)).columns(ENV)


def test_candidate_count_and_counts():
    assert len(CANDS) == 1 + 8 + 28 + 56 == 93
    cols = _uniform(20)
    c = family_counts(cols, "m0'", ("m0", "rep0"))
    assert c.shape == (4, 2) and c.sum() == 600


def test_bic_recovers_the_true_parents_with_rich_data():
    cols = _uniform(400)
    for target, parents in TRUE.items():
        assert set(best_bic(cols, target, CANDS)) == set(parents)


def test_bic_drops_a_weak_parent_with_little_data():
    weak = SysAdminRing(n=4, p_bad=0.90)                      # left neighbour barely matters
    cols = rollout(weak, make_logger("uniform", weak), 10, 30, np.random.default_rng(1)).columns(weak)
    missed = [t for t, p in weak.true_parents().items() if not set(p) <= set(best_bic(cols, t, CANDS))]
    assert missed                                            # some weak edge is missed at 300 rows


def test_posterior_is_a_distribution_and_concentrates():
    post = posterior(_uniform(400), "m1'", CANDS)
    assert sum(post.values()) == pytest.approx(1.0)
    fam = credible_family(post, 0.95)
    assert contains_superset(fam, TRUE["m1'"]) and len(fam) < len(CANDS)
    assert bdeu_score(np.array([[3, 1], [0, 0]])) < 0


def test_icp_accepts_the_true_set_and_rejects_with_diverse_loggers():
    loggers = [make_logger("aliasing", ENV, offset=-1, eps=0.1),
               make_logger("aliasing", ENV, offset=1, eps=0.1)]
    data = collect(ENV, loggers, [300, 300], 30, seed=2)
    cols = data.columns(ENV)
    fam = icp_family(cols, "m1'", CANDS, data.regime, alpha=0.05)
    assert set(TRUE["m1'"]) in [set(c) for c in fam]
    assert len(fam) < len(CANDS)                              # diverse loggers reject wrong sets


def test_icp_with_one_logger_rejects_nothing():
    data = collect(ENV, [make_logger("aliasing", ENV, offset=-1, eps=0.1)], [100], 30, seed=3)
    assert invariance_pvalue(data.columns(ENV), "m1'", ("m1",), data.regime) == 1.0
    assert len(icp_family(data.columns(ENV), "m1'", CANDS, data.regime, 0.05)) == len(CANDS)


def test_icp_level_holds_for_the_true_parents_across_seeds():
    """P(true set rejected) <= alpha: the guarantee the hypothesis set relies on."""
    rejected = 0
    for seed in range(40):
        loggers = [make_logger("specialist", ENV, region=(0, 1), eps=0.1),
                   make_logger("specialist", ENV, region=(2, 3), eps=0.1)]
        data = collect(ENV, loggers, [60, 60], 30, seed=100 + seed)
        true_set = tuple(sorted(TRUE["m0'"], key=INPUTS.index))
        rejected += invariance_pvalue(data.columns(ENV), "m0'", true_set, data.regime) <= 0.1
    assert rejected / 40 <= 0.1 + 3 * np.sqrt(0.1 * 0.9 / 40)


def test_jpcmci_recovers_the_ring_and_flags_a_confounded_logger():
    pytest.importorskip("tigramite")
    from scm_to_mdp.discovery import jpcmci_parents

    loggers = [make_logger("aliasing", ENV, offset=-1, eps=0.1),
               make_logger("aliasing", ENV, offset=1, eps=0.1)]
    res = jpcmci_parents(ENV, collect(ENV, loggers, [80, 80], 30, seed=0), alpha=0.01)
    for target, parents in TRUE.items():
        assert set(res.parents[target]) == set(parents)
    assert all(mark == "" for mark in res.context_into_state.values())     # no confounding
    conf_env = SysAdminRing(n=4, load_effect=0.4)
    loggers = [make_logger("uniform", conf_env), make_logger("confounded", conf_env)]
    res = jpcmci_parents(conf_env, collect(conf_env, loggers, [200, 200], 30, seed=1), alpha=0.01)
    assert any(mark == "-->" for mark in res.context_into_state.values())  # the logger moves the dynamics


def test_permutation_reference_holds_its_level_where_chi_square_does_not():
    """Sparse cells (64 configurations): chi-square over-rejects under the null; permutation does not."""
    rng = np.random.default_rng(1)
    rej = {"permutation": 0, "chi2": 0}
    for rep in range(150):
        cols = {f"x{i}": rng.integers(0, 2, 1500) for i in range(6)}
        regime = rng.integers(0, 2, 1500)
        cols["y"] = (rng.random(1500) < 0.2 + 0.5 * cols["x0"]).astype(int)
        for ref in rej:
            preds = tuple(f"x{i}" for i in range(6))
            p = invariance_pvalue(cols, "y", preds, regime, reference=ref, seed=rep)
            rej[ref] += p <= 0.05
    assert rej["permutation"] / 150 <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 150)
    assert rej["chi2"] / 150 > 0.1                      # the reason for the default
