"""S4/S7 bridge to NBN and one end-to-end run per arm (S2 to S9), with S5 and the ablation plumbing."""
from __future__ import annotations

import numpy as np
import pytest

from scm_to_mdp.assumptions import confounding_gate, episode_split
from scm_to_mdp.data import collect
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger
from scm_to_mdp.mdp import value_iteration
from scm_to_mdp.pipeline import (
    ARMS,
    PLUG_IN,
    Context,
    Truth,
    collapse,
    disagreement,
    joint_visited,
    logger_value,
    reached_states,
    run_arm,
    shd,
)
from scm_to_mdp.sets import (
    confounded_sets,
    interval_model,
    node_sets,
    planning_row,
    plug_in_kernel,
)


@pytest.fixture(scope="module")
def ring():
    env = SysAdminRing(n=4, p_bad=0.85)
    loggers = [make_logger("specialist", env, region=(0, 1), eps=0.2),
               make_logger("specialist", env, region=(2, 3), eps=0.2)]
    data = collect(env, loggers, [150, 150], horizon=30, seed=0)
    disc, est = episode_split(data, 0.5, seed=0)
    return env, disc, est


def _ctx(env, disc, est, **kw):
    return Context(env, disc.columns(env, kw.get("extra", ())), disc.regime,
                   est.columns(env, kw.get("extra", ())), n_hyp=8, **kw)


def _inside(m, P):
    return bool(np.all(m.L <= P + 1e-12) and np.all(P <= m.U + 1e-12))


def test_planning_rows_follow_nbn_c_order():
    env = SysAdminRing(n=2)
    # rows over (m0, m1, rep0, rep1): state 3 = (1, 1); action 1 = repair m0 -> flags (1, 0)
    assert planning_row(env, np.array(3), np.array(1)) == int("1110", 2)
    assert planning_row(env, np.array(0), np.array(0)) == 0
    assert planning_row(env, np.array(3), np.array(1), (1,)) == int("11" "1" "10", 2)   # (m, x, rep)


def test_oracle_boxes_contain_the_true_kernel_and_flat_leaves_rows_open(ring):
    env, _, est = ring
    cols = est.columns(env)
    P = env.kernel()
    oracle = interval_model(env, node_sets(env, cols, {k: [v] for k, v in env.true_parents().items()},
                                           refute=False, delta=0.1))
    flat = interval_model(env, node_sets(env, cols, {k: [tuple(env.state_vars + env.action_flags)]
                                                     for k in env.true_parents()}, refute=False, delta=0.1))
    for m in (oracle, flat):
        assert _inside(m, P)
        assert np.all(m.L.sum(-1) <= 1 + 1e-9) and np.all(m.U.sum(-1) >= 1 - 1e-9)
    assert oracle.rows_with_set == env.n_states * env.n_actions
    assert flat.rows_with_set < oracle.rows_with_set                      # structure buys coverage
    assert oracle.manifest["schema"] == "nbn.factored_sets/1"
    assert oracle.factor_rows == 4 * 8 and oracle.coupling == pytest.approx(80 / 8)


def test_frechet_boxes_contain_the_product_boxes_and_stay_valid(ring):
    env, _, est = ring
    sets = node_sets(env, est.columns(env), {k: [v] for k, v in env.true_parents().items()},
                     refute=False, delta=0.1)
    prod, fre = interval_model(env, sets, "product"), interval_model(env, sets, "frechet")
    assert np.all(fre.L <= prod.L + 1e-12) and np.all(fre.U >= prod.U - 1e-12)
    assert np.all(fre.L.sum(-1) <= 1 + 1e-9) and np.all(fre.U.sum(-1) >= 1 - 1e-9)
    with pytest.raises(ValueError, match="joint"):
        interval_model(env, sets, "nope")


def test_the_gate_rejects_a_confounded_logger_and_confounded_rows_contain_the_truth():
    env = SysAdminRing(n=4, p_bad=0.85, load_effect=0.4)
    loggers = [make_logger("specialist", env, region=(0, 1), eps=0.2),
               make_logger("confounded", env, eps=0.2)]
    data = collect(env, loggers, [200, 200], horizon=30, seed=0)
    disc, est = episode_split(data, 0.5, seed=0)
    assert min(confounding_gate(env, disc).values()) <= 0.1 / env.n
    P = env.kernel()                                                      # interventional (load averaged)
    cs = confounded_sets(env, est.columns(env), env.true_parents(), 0.1, declared_unconfounded=[0])
    assert _inside(interval_model(env, cs, "frechet"), P)
    oracle = {k: [v] for k, v in env.true_parents().items()}
    plain = interval_model(env, node_sets(env, est.columns(env), oracle, refute=False, delta=0.1))
    assert not _inside(plain, P)                                          # ignoring the gate is unsound here
    ctx = Context(env, disc.columns(env), disc.regime, est.columns(env), gate_rejected=True,
                  declared_unconfounded=(0,))
    row = run_arm("oracle_gated", ctx, Truth.of(env), np.random.default_rng(0))
    assert row["confounded_route"] == 1 and row["truth_in_boxes"] == 1 and row["sound"] == 1


def test_an_empty_family_falls_back_to_flat(ring):
    env, _, est = ring
    fams = {k: [v] for k, v in env.true_parents().items()}
    fams["m0'"] = []
    sets = node_sets(env, est.columns(env), fams, refute=False, delta=0.1)
    assert sets["m0'"].empty_family and sets["m0'"].family == [tuple(env.state_vars + env.action_flags)]
    assert not sets["m1'"].empty_family


def test_plug_in_kernel_with_the_true_parents_converges():
    env = SysAdminRing(n=3)
    data = collect(env, [make_logger("uniform", env)], [600], horizon=30, seed=1)
    P_hat = plug_in_kernel(env, data.columns(env), env.true_parents())
    assert np.abs(P_hat - env.kernel()).max() < 0.05
    np.testing.assert_allclose(P_hat.sum(-1), 1.0)


def test_distractor_parents_are_averaged_in_plug_in_and_hulled_in_boxes():
    env = SysAdminRing(n=3)
    lg = [make_logger("distractor", env, region=(0, 1, 2), eps=0.3), make_logger("uniform", env)]
    data = collect(env, lg, [300, 300], horizon=30, seed=2, distractor_prob=0.5)
    cols = data.columns(env, ("x",))
    assert set(np.unique(cols["x"])) == {0, 1}
    with_x = {k: (*v, "x") for k, v in env.true_parents().items()}
    P_hat = plug_in_kernel(env, cols, with_x, extra=("x",))
    np.testing.assert_allclose(P_hat.sum(-1), 1.0)
    assert np.abs(P_hat - env.kernel()).max() < 0.08
    fams = {k: [v] for k, v in with_x.items()}
    m = interval_model(env, node_sets(env, cols, fams, False, 0.1, extra=("x",)), extra=("x",))
    assert _inside(m, env.kernel())


def test_collapse_merges_identical_candidates_and_disagreement_is_then_zero(ring):
    env, _, est = ring
    cols = est.columns(env)
    truth = Truth.of(env)
    fams = {k: [v, v] for k, v in env.true_parents().items()}
    reached = reached_states(env, cols)
    assert collapse(env, cols, fams, 0.025, 0.05, reached) == dict.fromkeys(fams, 1)
    out = disagreement(env, cols, fams, truth, n_hyp=8, rng=np.random.default_rng(0), reached=reached)
    assert out["n_hypotheses"] == 16 and out["s5_sampled"] == 8 and out["s5_disagree_share"] == 0.0
    wide = {k: [v, ()] for k, v in env.true_parents().items()}          # the empty set disagrees
    assert all(k == 2 for k in collapse(env, cols, wide, 0.025, 0.05, reached).values())


def test_shd_joint_visits_and_logger_value(ring):
    env, _, est = ring
    true = env.true_parents()
    assert shd(true, true) == 0
    assert shd({**true, "m0'": ("m0", "rep0", "m1")}, true) == 2
    jv = joint_visited(env, est.columns(env))
    assert jv.dtype == bool and 0 < jv.sum() <= env.n_states * env.n_actions
    v_star = value_iteration(env.kernel(), env.reward, env.gamma)[0][env.index(env.initial_state)]
    v = logger_value(env, make_logger("uniform", env), n_episodes=100, horizon=150)
    assert 0 < v < v_star


def test_every_arm_runs_and_set_arms_keep_their_promise(ring):
    env, disc, est = ring
    truth = Truth.of(env)
    rng = np.random.default_rng(0)
    ctx = _ctx(env, disc, est)
    for arm in ARMS:
        row = run_arm(arm, ctx, truth, rng)
        assert row["arm"] == arm and 0 <= row["regret"] <= truth.v_star
        if arm not in PLUG_IN:
            assert row["truth_in_boxes"] == 1
            assert row["sound"] == 1                       # truth in every box -> promise <= true value
            assert row["rows_set_joint"] + row["rows_set_factor_only"] == row["rows_with_set"]
        if row.get("s5_k_prime") is not None:
            assert 1 <= row["s5_k_prime"] <= row["n_hypotheses"]
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm("nope", ctx, truth, rng)
