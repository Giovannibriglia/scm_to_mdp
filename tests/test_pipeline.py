"""S4/S7 bridge to NBN and one end-to-end run per arm (S2 to S9)."""
from __future__ import annotations

import numpy as np
import pytest

from scm_to_mdp.assumptions import episode_split
from scm_to_mdp.data import collect
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger
from scm_to_mdp.pipeline import ARMS, PLUG_IN, Truth, disagreement, run_arm
from scm_to_mdp.sets import interval_model, node_sets, planning_row, plug_in_kernel


@pytest.fixture(scope="module")
def ring():
    env = SysAdminRing(n=4, p_bad=0.85)
    loggers = [make_logger("specialist", env, region=(0, 1), eps=0.2),
               make_logger("specialist", env, region=(2, 3), eps=0.2)]
    data = collect(env, loggers, [150, 150], horizon=30, seed=0)
    disc, est = episode_split(data, 0.5, seed=0)
    return env, disc, est


def test_planning_rows_follow_nbn_c_order():
    env = SysAdminRing(n=2)
    # rows over (m0, m1, rep0, rep1): state 3 = (1, 1); action 1 = repair m0 -> flags (1, 0)
    assert planning_row(env, np.array(3), np.array(1)) == int("1110", 2)
    assert planning_row(env, np.array(0), np.array(0)) == 0


def test_oracle_boxes_contain_the_true_kernel_and_flat_leaves_rows_open(ring):
    env, _, est = ring
    cols = est.columns(env)
    P = env.kernel()
    oracle = interval_model(env, node_sets(env, cols, {k: [v] for k, v in env.true_parents().items()},
                                           refute=False, delta=0.1))
    flat = interval_model(env, node_sets(env, cols, {k: [tuple(env.state_vars + env.action_flags)]
                                                     for k in env.true_parents()}, refute=False, delta=0.1))
    for m in (oracle, flat):
        assert np.all(m.L <= P + 1e-12) and np.all(P <= m.U + 1e-12)
        assert np.all(m.L.sum(-1) <= 1 + 1e-9) and np.all(m.U.sum(-1) >= 1 - 1e-9)
    assert oracle.rows_with_set == env.n_states * env.n_actions
    assert flat.rows_with_set < oracle.rows_with_set                      # structure buys coverage
    assert oracle.manifest["schema"] == "nbn.factored_sets/1"


def test_an_empty_family_falls_back_to_flat(ring):
    env, _, est = ring
    fams = {k: [v] for k, v in env.true_parents().items()}
    fams["m0'"] = []
    sets = node_sets(env, est.columns(env), fams, refute=False, delta=0.1)
    assert sets["m0'"].empty_family and sets["m0'"].family == [tuple(env.state_vars + env.action_flags)]
    assert not sets["m1'"].empty_family


def test_plug_in_kernel_with_the_true_parents_converges():
    env = SysAdminRing(n=3)
    loggers = [make_logger("uniform", env)]
    data = collect(env, loggers, [600], horizon=30, seed=1)
    P_hat = plug_in_kernel(env, data.columns(env), env.true_parents())
    assert np.abs(P_hat - env.kernel()).max() < 0.05
    np.testing.assert_allclose(P_hat.sum(-1), 1.0)


def test_disagreement_is_zero_when_every_hypothesis_is_the_same(ring):
    env, _, est = ring
    truth = Truth.of(env)
    fams = {k: [v, v] for k, v in env.true_parents().items()}
    out = disagreement(env, est.columns(env), fams, truth, n_hyp=8, rng=np.random.default_rng(0))
    assert out["n_hypotheses"] == 16 and out["s5_sampled"] == 8
    assert out["s5_disagree_share"] == 0.0


def test_every_arm_runs_and_set_arms_keep_their_promise(ring):
    env, disc, est = ring
    truth = Truth.of(env)
    rng = np.random.default_rng(0)
    for arm in ARMS:
        row = run_arm(arm, env, disc.columns(env), disc.regime, est.columns(env), truth,
                      alpha=0.1, delta=0.1, d=3, n_hyp=8, rng=rng)
        assert row["arm"] == arm and 0 <= row["regret"] <= truth.v_star
        if arm not in PLUG_IN:
            assert row["truth_in_boxes"] == 1
            assert row["sound"] == 1                       # truth in every box -> promise <= true value
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm("nope", env, disc.columns(env), disc.regime, est.columns(env), truth, 0.1, 0.1, 3, 8, rng)
