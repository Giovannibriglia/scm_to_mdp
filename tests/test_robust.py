"""S8: interval robust value iteration."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.mdp import policy_evaluation, value_iteration
from scm_to_mdp.robust import (
    check_boxes,
    robust_policy_value,
    robust_value_iteration,
    worst_case_distributions,
)


def _random_boxes(rng, S=5, A=3, width=0.2):
    P = rng.dirichlet(np.ones(S), size=(S, A))
    L = np.clip(P - rng.uniform(0, width, P.shape), 0, 1)
    U = np.clip(P + rng.uniform(0, width, P.shape), 0, 1)
    return P, L, U


def test_greedy_inner_minimum_matches_a_linear_program():
    rng = np.random.default_rng(0)
    for _ in range(50):
        _, L, U = _random_boxes(rng, S=6, A=1)
        V = rng.normal(size=6)
        p = worst_case_distributions(L[0, 0], U[0, 0], V)
        lp = linprog(V, A_eq=np.ones((1, 6)), b_eq=[1.0], bounds=list(zip(L[0, 0], U[0, 0], strict=True)))
        assert lp.status == 0
        assert p @ V == pytest.approx(lp.fun, abs=1e-10)
        assert p.sum() == pytest.approx(1.0) and np.all(p >= L[0, 0] - 1e-12) and np.all(p <= U[0, 0] + 1e-12)


def test_point_boxes_give_ordinary_value_iteration():
    env = SysAdminRing(n=3)
    P, R = env.kernel(), env.reward
    V_rob, Q_rob, pi = robust_value_iteration(P, P, R, env.gamma)
    V, Q = value_iteration(P, R, env.gamma)
    np.testing.assert_allclose(V_rob, V, atol=1e-8)
    np.testing.assert_allclose(Q_rob, Q, atol=1e-8)


def test_promise_is_a_lower_bound_when_the_truth_is_in_every_box():
    rng = np.random.default_rng(1)
    R = rng.uniform(0, 1, 5)
    for _ in range(20):
        P, L, U = _random_boxes(rng)
        V_rob, _, pi = robust_value_iteration(L, U, R, 0.9)
        true = policy_evaluation(P, R, 0.9, pi)
        assert np.all(V_rob <= true + 1e-8)
        np.testing.assert_allclose(robust_policy_value(L, U, R, 0.9, pi), V_rob, atol=1e-8)


def test_robust_value_is_the_max_over_deterministic_policies_of_their_promise():
    rng = np.random.default_rng(2)
    P, L, U = _random_boxes(rng, S=3, A=2)
    R = rng.uniform(0, 1, 3)
    V_rob, _, _ = robust_value_iteration(L, U, R, 0.8)
    policies = itertools.product(range(2), repeat=3)
    promises = [robust_policy_value(L, U, R, 0.8, np.array(pi)) for pi in policies]
    best = max(promises, key=lambda v: v.sum())
    np.testing.assert_allclose(V_rob, best, atol=1e-8)
    assert all(np.all(v <= V_rob + 1e-8) for v in promises)     # the robust optimum dominates every policy


def test_wider_boxes_never_promise_more():
    rng = np.random.default_rng(3)
    P, L, U = _random_boxes(rng)
    R = rng.uniform(0, 1, 5)
    V_tight, _, _ = robust_value_iteration(L, U, R, 0.9)
    V_wide, _, _ = robust_value_iteration(np.maximum(L - 0.1, 0), np.minimum(U + 0.1, 1), R, 0.9)
    assert np.all(V_wide <= V_tight + 1e-9)


def test_masked_actions_and_box_validation():
    env = SysAdminRing(n=2)
    P, R = env.kernel(), env.reward
    allowed = np.zeros(P.shape[:2], dtype=bool)
    allowed[:, 0] = True
    _, _, pi = robust_value_iteration(P, P, R, 0.9, allowed=allowed)
    assert np.all(pi == 0)
    with pytest.raises(ValueError, match="sum L"):
        check_boxes(np.full((1, 1, 2), 0.6), np.ones((1, 1, 2)))
    with pytest.raises(ValueError, match="0 <= L <= U"):
        check_boxes(np.full((1, 1, 2), 0.5), np.full((1, 1, 2), 0.4))
