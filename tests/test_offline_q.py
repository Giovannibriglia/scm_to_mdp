"""Offline DQN and offline CQL (baselines/offline_q.py): they learn from the data only, CQL is more
pessimistic than DQN where the data are thin, runs are reproducible, and the tool wires them in."""

import numpy as np
import pytest

from scm_to_mdp.baselines import offline_q
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Method, evaluate, learn, solve, standard_methods

GAMMA = 0.9
FAST = {"steps": 3_000, "hidden": 64, "target_every": 500}  # small and quick: the defaults are larger


@pytest.fixture(scope="module")
def problem():
    env = SysAdmin(2, 0)
    truth = env.ground_truth()
    uniform = truth.learner_spec.read(env.sample(200, 20, env.behaviour("uniform", GAMMA)))
    expert = truth.learner_spec.read(env.sample(200, 20, env.behaviour("expert", GAMMA, epsilon=0.02)))
    return truth, uniform, expert


@pytest.mark.parametrize("learner", ["offline_dqn", "offline_cql"])
def test_uniform_data_give_a_near_optimal_policy(problem, learner):
    truth, uniform, _ = problem
    result = solve(learn(uniform, Method(learner, "greedy_q", solver_options=FAST)), uniform, GAMMA)
    assert result.ok and result.steps["training"]["status"] == "ok"
    metrics = evaluate(result, uniform, truth, GAMMA)
    assert metrics["optimal_share"] > 0.95


def test_cql_is_more_pessimistic_where_the_data_are_thin(problem):
    _, _, expert = problem
    dqn = offline_q.learn(expert, GAMMA, **FAST)
    cql = offline_q.learn(expert, GAMMA, cql_alpha=1.0, **FAST)
    s = expert.state_index(expert.df[expert.state_vars].values)
    counts = np.zeros_like(dqn.Q)
    np.add.at(counts, (s, expert.action_index()), 1)
    rare = counts < 0.02 * counts.sum(1, keepdims=True)  # actions the expert (almost) never took
    common = counts > 0.5 * counts.sum(1, keepdims=True)
    assert (cql.Q[rare] - dqn.Q[rare]).mean() < -1.0  # pushed down
    assert abs(cql.Q[common] - dqn.Q[common]).mean() < 0.5  # left about where they are


def test_a_run_is_reproducible(problem):
    _, uniform, _ = problem
    a = offline_q.learn(uniform, GAMMA, cql_alpha=1.0, seed=3, **FAST)
    b = offline_q.learn(uniform, GAMMA, cql_alpha=1.0, seed=3, **FAST)
    assert np.array_equal(a.Q, b.Q)


def test_the_methods_are_baselines_without_guarantee():
    with pytest.raises(ValueError, match="no CQL penalty"):
        Method("offline_dqn", "greedy_q", solver_options={"cql_alpha": 1.0})
    with pytest.raises(ValueError, match="unknown options"):
        Method("offline_cql", "greedy_q", solver_options={"epochs": 3})
    methods = standard_methods(deep=True)
    for name in ("offline DQN", "offline CQL"):
        assert methods[name].role == "baseline" and not methods[name].guarantee
    assert "offline DQN" not in standard_methods()
