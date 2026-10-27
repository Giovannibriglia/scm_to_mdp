"""The robustness checks of a promise (robustness.py), robust policy evaluation (robust_vi.evaluate)
and the random rewards (reward_family.py)."""

import numpy as np
import pytest

from scm_to_mdp import reward_family, robustness
from scm_to_mdp.core import IntervalMDP, Reward
from scm_to_mdp.discovery import full_factored, oracle
from scm_to_mdp.evaluation import random_starts, true_values
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import given
from scm_to_mdp.solving import robust_vi, value_iteration
from scm_to_mdp.tool import standard_methods
from scm_to_mdp.tool.evaluate import evaluate
from scm_to_mdp.tool.methods import learn, solve

GAMMA = 0.9


def test_robust_evaluation_without_uncertainty_is_policy_evaluation(truth, dataset):
    P = truth.P[:, truth.actions_of(dataset)]
    pi = np.eye(P.shape[1])[np.arange(P.shape[0]) % P.shape[1]]  # some deterministic policy
    exact = value_iteration.policy_value(P, dataset.reward, pi, GAMMA)
    for nature in robust_vi.NATURES:
        V = robust_vi.evaluate(IntervalMDP(P, P, dataset.reward), pi, GAMMA, tol=1e-10, nature=nature)
        assert np.allclose(V, exact, atol=1e-6)


def test_the_bracket_contains_the_promise_of_the_solved_policy(env, dataset):
    graph = given.select(oracle.fit(dataset, env.true_parents()), dataset)
    mdp = robustness.interval_mdp(nbn_intervals.learn(dataset, graph, delta=0.05))
    best = robust_vi.solve(mdp, GAMMA)
    pi = robustness.policy_table(best, mdp.L.shape[1])
    lo = robust_vi.evaluate(mdp, pi, GAMMA, nature="adversarial")
    up = robust_vi.evaluate(mdp, pi, GAMMA, nature="cooperative")
    assert np.all(lo <= up + 1e-9)
    assert np.allclose(lo, best.V, atol=1e-5)  # the solver's promise is its policy's lower bound


def test_failure_states_are_the_terminals_that_pay_less():
    r = Reward(np.zeros((4, 1)), np.array([0.0, 1.0, 0.0, 0.5]), np.array([True, True, False, True]))
    assert robustness.failure_states(r).tolist() == [True, False, False, True]
    none = Reward(np.zeros((2, 1)), np.zeros(2), np.zeros(2, dtype=bool))
    assert not robustness.failure_states(none).any()


def test_reach_probability_of_a_two_state_chain():
    # from state 0 the chain fails (state 1) w.p. 0.5 per step: P(fail within h) = 1 - 0.5**h
    step = lambda f: np.array([0.5 * f[1] + 0.5 * f[0], f[1]])  # noqa: E731
    p = robustness.reach_probability(step, np.array([False, True]), np.array([False, True]), 3)
    assert p[0] == pytest.approx(1 - 0.5**3) and p[1] == 1.0


def test_every_check_holds_for_the_true_graph(env, truth, dataset):
    method = standard_methods(exact=False)["true graph -> IMDP"]
    result = solve(learn(dataset, method, None, truth.true_parents), dataset, GAMMA)
    metrics = evaluate(result, dataset, truth, GAMMA, random_starts(truth, 5, 0), horizon=50)
    assert metrics["contain_all"]
    for check in ("safe", "bracket_holds", "opt_holds"):
        assert metrics[check], check
    assert metrics["cert_residual"] >= -1e-8  # V <= T_pi V: the promise is below the policy's robust value
    assert np.isfinite(metrics["bracket_width"]) and metrics["bracket_width"] >= 0
    V_true = true_values(truth, dataset, result.solution, GAMMA)
    assert truth.start @ V_true == pytest.approx(metrics["true"])


def test_random_rewards_are_reproducible_and_keep_the_terminals():
    B = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    terminal = np.array([False, False, False, True])
    family = reward_family.RewardFamily(per_family=3)
    draw = lambda seed: reward_family.draw(family, B, [2, 2], terminal, 3, np.array([5, 1, 9, 0]), seed)  # noqa: E731
    tasks = draw(0)
    assert len(tasks) == 3 * len(reward_family.FAMILIES)
    assert all(np.all(t.terminal >= terminal) for t in tasks)  # the new terminal set contains the old one
    assert all(t.R.shape == (4, 3) and -0.2 - 1e-12 <= t.R.min() and t.R.max() <= 1 for t in tasks)
    stress = [t for t in tasks if t.family == "stress"]
    assert all(t.r_next[1] == 1.0 for t in stress)  # state 1 is the least visited live state
    again = draw(0)
    assert all(
        np.array_equal(a.R, b.R) and np.array_equal(a.r_next, b.r_next)
        for a, b in zip(tasks, again, strict=True)
    )
    assert not all(np.array_equal(a.R, b.R) for a, b in zip(tasks, draw(1), strict=True))


def test_random_reward_metrics():
    outcomes = [
        {"family": "dense", "safe": True, "value": 0.9},
        {"family": "sparse", "safe": False, "value": 0.5},
    ]
    m = reward_family.metrics(outcomes)
    assert m["rf_rewards"] == 2 and m["rf_safe_all"] is False and m["rf_safe_share"] == 0.5
    assert m["rf_value_min"] == 0.5 and m["rf_dense_safe_all"] is True
    no_promise = reward_family.metrics([{"family": "noise", "safe": float("nan"), "value": 0.2}])
    assert np.isnan(no_promise["rf_safe_all"]) and no_promise["rf_value_mean"] == 0.2
    with pytest.raises(ValueError):
        reward_family.RewardFamily(families=("dense", "unknown"))


def test_the_full_factored_graph_gives_every_input_to_every_variable(dataset):
    graph = given.select(full_factored.fit(dataset), dataset)
    assert all(set(graph.parents[v]) == set(dataset.inputs) for v in dataset.state_vars)
