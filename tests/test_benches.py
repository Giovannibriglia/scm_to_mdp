"""Per-step bench building blocks: S4 builders, S7 joint boxes, S8 exact factored solver, S6 criteria."""
from __future__ import annotations

import numpy as np
import pytest

from scm_to_mdp.assumptions import episode_split
from scm_to_mdp.benches import (
    BUILDERS,
    boxes_from_factors,
    criterion_vi,
    factor_bounds,
    factored_policy_value,
    factored_robust_vi,
    q_intervals,
)
from scm_to_mdp.data import collect
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger
from scm_to_mdp.mdp import policy_evaluation
from scm_to_mdp.robust import robust_policy_value, robust_value_iteration
from scm_to_mdp.sets import interval_model, node_sets


@pytest.fixture(scope="module")
def setup():
    env = SysAdminRing(n=4, p_bad=0.85)
    loggers = [make_logger("specialist", env, region=(0, 1), eps=0.2),
               make_logger("specialist", env, region=(2, 3), eps=0.2)]
    data = collect(env, loggers, [150, 150], horizon=30, seed=0)
    _, est = episode_split(data, 0.5, seed=0)
    return env, est.columns(env)


def test_every_builder_gives_valid_intervals_and_l1_matches_the_nbn_pipeline_boxes(setup):
    env, cols = setup
    parents = env.true_parents()
    for b in BUILDERS:
        fb = factor_bounds(env, cols, parents, b, 0.1)
        for a, c in fb.values():
            assert np.all((0 <= a) & (a <= c) & (c <= 1))
    qa, qb = q_intervals(env, parents, factor_bounds(env, cols, parents, "l1", 0.1))
    L, U = boxes_from_factors(env, qa, qb, "product")
    ref = interval_model(env, node_sets(env, cols, {k: [v] for k, v in parents.items()}, False, 0.1))
    np.testing.assert_allclose(L, ref.L, atol=1e-12)
    np.testing.assert_allclose(U, ref.U, atol=1e-12)
    with pytest.raises(ValueError, match="unknown builder"):
        factor_bounds(env, cols, parents, "nope", 0.1)


def test_exact_factored_promise_lies_between_the_box_promise_and_the_true_value(setup):
    env, cols = setup
    P, R = env.kernel(), env.reward
    qa, qb = q_intervals(env, env.true_parents(), factor_bounds(env, cols, env.true_parents(), "l1", 0.1))
    L, U = boxes_from_factors(env, qa, qb)
    V_box, _, pi_box = robust_value_iteration(L, U, R, env.gamma)
    V_fac, pi_fac = factored_robust_vi(env, qa, qb, R, env.gamma)
    assert np.all(V_box <= V_fac + 1e-8)                      # the box is an outer relaxation
    for pi in (pi_box, pi_fac):
        true = policy_evaluation(P, R, env.gamma, pi)
        box = robust_policy_value(L, U, R, env.gamma, pi)
        fac = factored_policy_value(env, qa, qb, R, env.gamma, pi)
        assert np.all(box <= fac + 1e-8) and np.all(fac <= true + 1e-8)


def test_point_intervals_reduce_to_the_true_mdp():
    env = SysAdminRing(n=3)
    q = env.factor_probs()
    L, U = boxes_from_factors(env, q, q)
    np.testing.assert_allclose(L, env.kernel(), atol=1e-12)
    V, _ = factored_robust_vi(env, q, q, env.reward, env.gamma)
    np.testing.assert_allclose(V, robust_value_iteration(L, U, env.reward, env.gamma)[0], atol=1e-8)
    Lf, Uf = boxes_from_factors(env, q, q, "frechet")
    assert np.all(Lf <= L + 1e-12) and np.all(Uf >= U - 1e-12)


def test_criteria_order_pessimistic_hurwicz_optimistic(setup):
    env, cols = setup
    qa, qb = q_intervals(env, env.true_parents(), factor_bounds(env, cols, env.true_parents(), "l1", 0.1))
    L, U = boxes_from_factors(env, qa, qb)
    V1, _ = criterion_vi(L, U, env.reward, env.gamma, 1.0)
    V5, _ = criterion_vi(L, U, env.reward, env.gamma, 0.5)
    V0, _ = criterion_vi(L, U, env.reward, env.gamma, 0.0)
    np.testing.assert_allclose(V1, robust_value_iteration(L, U, env.reward, env.gamma)[0], atol=1e-8)
    assert np.all(V1 <= V5 + 1e-9) and np.all(V5 <= V0 + 1e-9)
    with pytest.raises(ValueError, match="pessimism"):
        criterion_vi(L, U, env.reward, env.gamma, 1.5)
