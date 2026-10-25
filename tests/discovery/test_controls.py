"""The control graphs of the evaluation (discovery/controls.py): the logger-Markov graph is the true graph
under a positive logger and drops a policy-redundant parent under a deterministic one; the random graph
is matched and never the truth; true graph + 1 is a superset by one."""

import pytest

from scm_to_mdp.core import NEXT_MARK
from scm_to_mdp.discovery import controls
from scm_to_mdp.envs import FrozenLake, SysAdmin


@pytest.fixture(scope="module")
def sysadmin():
    env = SysAdmin(4, 0, storm=0.1)
    return env, env.ground_truth()


def observed_true(truth):
    cards = truth.learner_spec.cards
    return {v: tuple(p for p in truth.true_parents[v] if p in cards) for v in truth.learner_spec.state_vars}


def test_logger_markov_is_the_truth_under_a_positive_logger(sysadmin):
    env, truth = sysadmin
    for table in (env.behaviour("uniform", 0.9), env.behaviour("expert", 0.9, 0.05)):
        assert controls.logger_markov(truth, table, 20) == observed_true(truth)


def test_logger_markov_drops_parents_a_deterministic_logger_makes_redundant(sysadmin):
    env, truth = sysadmin
    markov = controls.logger_markov(truth, env.behaviour("expert", 0.9, 0.0), 20)
    true = observed_true(truth)
    assert markov != true and all(len(markov[v]) <= len(true[v]) for v in true)
    lake = FrozenLake("4x4", 2, 0)
    lake_truth = lake.ground_truth()
    lake_markov = controls.logger_markov(lake_truth, lake.behaviour("expert", 0.95, 0.0), 30)
    assert lake_markov["x0"] != observed_true(lake_truth)["x0"]


def test_random_matched_and_plus_one(sysadmin):
    _, truth = sysadmin
    true = observed_true(truth)
    states, actions = truth.learner_spec.state_vars, truth.learner_spec.action_vars
    random = controls.random_matched(truth, 3)
    assert random == controls.random_matched(truth, 3)  # seeded
    for v, ps in random.items():
        assert set(ps) != set(true[v])
        assert [p for p in ps if p in actions] == [p for p in true[v] if p in actions]
        assert len([p for p in ps if p in states]) == len([p for p in true[v] if p in states])
    plus = controls.plus_one(truth, 0)
    for v, ps in plus.items():
        names = [p for p in ps if not p.endswith(NEXT_MARK)]
        assert set(true[v]) < set(names) and len(names) == len(true[v]) + 1
