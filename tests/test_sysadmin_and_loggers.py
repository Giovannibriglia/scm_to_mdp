"""SysAdmin ring (S0), loggers (S1), datasets and diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from scm_to_mdp.data import collect, rollout
from scm_to_mdp.diagnostics import aliasing_index, observational_gap, overlap, profile, row_counts
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import FAMILIES, make_logger
from scm_to_mdp.mdp import greedy, policy_evaluation, value_iteration


@pytest.fixture
def env():
    return SysAdminRing(n=4)


# ------------------------------------------------------------------ environment
def test_kernel_rows_are_distributions_and_factorise(env):
    P = env.kernel()
    assert P.shape == (16, 5, 16)
    np.testing.assert_allclose(P.sum(-1), 1.0, atol=1e-12)
    p = env.factor_probs()
    s2 = env.states
    manual = np.prod(np.where(s2[None, None] == 1, p[:, :, None, :], 1 - p[:, :, None, :]), axis=-1)
    np.testing.assert_allclose(P, manual)


def test_true_parents_and_dynamics(env):
    assert env.true_parents()["m0'"] == ("m0", "m3", "rep0")
    assert env.p_up(1, 1, 0) == pytest.approx(0.95) and env.p_up(1, 0, 0) == pytest.approx(0.60)
    assert env.p_up(0, 1, 1) == pytest.approx(0.95) and env.p_up(0, 1, 0) == pytest.approx(0.05)


def test_step_matches_the_kernel(env):
    rng = np.random.default_rng(0)
    s, a = np.array([1, 0, 1, 0], dtype=np.int8), 2
    draws = np.array([env.index(env.step(s, a, rng)) for _ in range(40_000)])
    freq = np.bincount(draws, minlength=16) / draws.size
    np.testing.assert_allclose(freq, env.kernel()[env.index(s), a], atol=0.01)


def test_hidden_load_averages_out_in_the_interventional_kernel():
    env = SysAdminRing(n=3, load_effect=0.3, load_prob=0.5)
    np.testing.assert_allclose(env.kernel(), 0.5 * env.kernel(0.0) + 0.5 * env.kernel(1.0))
    assert not np.allclose(env.kernel(0.0), env.kernel(1.0))
    with pytest.raises(ValueError):
        SysAdminRing(load_effect=0.7)                    # p_bad - 0.7 < 0


def test_value_iteration_is_optimal_and_policy_evaluation_exact(env):
    P, R = env.kernel(), env.reward
    V, Q = value_iteration(P, R, env.gamma)
    pi = greedy(Q)
    np.testing.assert_allclose(policy_evaluation(P, R, env.gamma, pi), V, atol=1e-8)
    noop = np.zeros(env.n_states, dtype=int)
    assert np.all(policy_evaluation(P, R, env.gamma, noop) <= V + 1e-9)


def test_repairing_the_left_neighbour_matters_for_the_server(env):
    """The decision the infection edge controls: with m3 down, repairing m3 protects the server m0."""
    _, Q = value_iteration(env.kernel(), env.reward, env.gamma)
    s = env.index(np.array([1, 1, 1, 0]))
    assert Q[s].argmax() == 4                            # repair m3


# ------------------------------------------------------------------ loggers
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_family_plays_valid_actions(env, family):
    lg = make_logger(family, env, eps=0.1)
    data = rollout(env, lg, n_episodes=20, horizon=30, rng=np.random.default_rng(1))
    assert data.action.min() >= 0 and data.action.max() <= env.n
    assert len(data) == 600


def test_specialist_repairs_only_its_region(env):
    lg = make_logger("specialist", env, region=(0, 1))
    data = rollout(env, lg, 50, 30, np.random.default_rng(2))
    assert set(np.unique(data.action)) <= {0, 1, 2}


def test_aliasing_rule_ties_the_repair_flag_to_the_left_neighbour(env):
    data = rollout(env, make_logger("aliasing", env, offset=-1), 200, 30, np.random.default_rng(3))
    rep, left = env.flags(data.action)[:, 1], data.state[:, 0]
    assert np.all(left[rep == 1] == 0)                   # repair m1 only when m0 is down
    assert aliasing_index(env, data, 1) > aliasing_index(
        env, rollout(env, make_logger("uniform", env), 200, 30, np.random.default_rng(3)), 1)


def test_softmax_logger_concentrates_on_good_actions(env):
    lg = make_logger("softmax_q", env, temperature=0.05)
    _, Q = value_iteration(env.kernel(), env.reward, env.gamma)
    assert np.all(lg.probs.argmax(1) == Q.argmax(1))


def test_recent_failure_logger_uses_history(env):
    lg = make_logger("recent_failure", env)
    assert lg.history_dependent and not lg.reads_hidden
    lg.reset()
    lg.observe(np.array([1, 1, 1, 1]), 0, np.array([0, 1, 1, 1]))
    lg.observe(np.array([0, 1, 1, 1]), 0, np.array([0, 1, 0, 1]))
    assert lg.act(np.array([0, 1, 0, 1]), np.random.default_rng(0)) == 3     # m2 failed last


def test_confounded_logger_biases_the_observational_kernel():
    env = SysAdminRing(n=4, load_effect=0.3)
    unconf = rollout(env, make_logger("uniform", env), 400, 30, np.random.default_rng(4))
    conf = rollout(env, make_logger("confounded", env), 400, 30, np.random.default_rng(4))
    assert make_logger("confounded", env).reads_hidden
    assert observational_gap(env, unconf) == 0.0          # within 4 standard errors everywhere
    assert observational_gap(env, conf) > 0.03            # repairs happen under high load


# ------------------------------------------------------------------ data and diagnostics
def test_collect_pools_regimes_and_hides_the_load(env):
    loggers = [make_logger("specialist", env, region=(0, 1), eps=0.05),
               make_logger("specialist", env, region=(2, 3), eps=0.05)]
    data = collect(env, loggers, [10, 20], horizon=30, seed=0)
    assert np.bincount(data.regime).tolist() == [300, 600]
    cols = data.columns(env)
    assert "hidden_load" not in cols and set(cols) >= {"m0", "rep0", "m0'", "regime"}
    again = collect(env, loggers, [10, 20], horizon=30, seed=0)
    assert np.array_equal(again.action, data.action)


def test_profile_and_overlap(env):
    a = rollout(env, make_logger("specialist", env, region=(0, 1)), 100, 30, np.random.default_rng(5))
    b = rollout(env, make_logger("specialist", env, region=(2, 3)), 100, 30, np.random.default_rng(6))
    u = rollout(env, make_logger("uniform", env), 300, 30, np.random.default_rng(7))
    pa, pu = profile(env, a), profile(env, u)
    assert pu["joint_rows"] > pa["joint_rows"]
    assert 0.0 <= pa["factor_only_rows"] <= pa["factor_rows"] <= 1.0
    assert 0.0 <= overlap(env, a, b) < 1.0
    assert row_counts(env, a).sum() == len(a)


def test_empty_cells_flag_parent_configurations_a_deterministic_logger_never_shows(env):
    from scm_to_mdp.diagnostics import aliasing_empty_cells

    rng = np.random.default_rng(8)
    aliased = rollout(env, make_logger("aliasing", env, offset=-1, eps=0.0), 200, 30, rng)
    uniform = rollout(env, make_logger("uniform", env), 200, 30, rng)
    assert aliasing_empty_cells(env, aliased) >= 1          # rep_1 = 1 with m_0 up never happens
    assert aliasing_empty_cells(env, uniform) == 0
