"""Evaluation on the true model (evaluation.py, S9): values, regret, the promise's validity and
the structure distances (SHD, SID, action edges)."""

import numpy as np

from scm_to_mdp.core import Graph, Solution
from scm_to_mdp.evaluation import evaluate, optimal_values, random_starts, structure_distances
from scm_to_mdp.solving import value_iteration


def test_shd_and_sid(env, dataset):
    truth = env.true_parents()
    exact = structure_distances(Graph(truth), truth, dataset)
    assert [exact[k] for k in ("shd", "missing", "extra", "sid")] == [0, 0, 0, 0]
    assert exact["action_missing"] == 0 and exact["action_recall"] == 1.0  # every action edge kept
    v = dataset.state_vars[0]
    extra = {**truth, v: tuple(dataset.inputs)}  # every input as a parent: a superset
    d = structure_distances(Graph(extra), truth, dataset)
    assert d["missing"] == 0 and d["extra"] == len(dataset.inputs) - len(truth[v])
    assert d["sid"] == 0  # extra parents are still a valid adjustment set
    missing = {**truth, v: truth[v][:-1]}  # one true parent dropped
    d = structure_distances(Graph(missing), truth, dataset)
    assert (d["shd"], d["missing"], d["extra"]) == (1, 1, 0) and d["sid"] > 0


def test_random_starts_are_live_and_distinct(truth):
    starts = random_starts(truth, 5, seed=0)
    assert len(set(starts)) == 5 and not truth.reward.terminal[starts].any()
    assert np.array_equal(starts, random_starts(truth, 5, seed=0))


def test_violation_metrics(truth, dataset):
    V_star = optimal_values(truth, 0.9)
    pi = np.zeros(truth.n_states, dtype=np.int64)  # learner action 0 everywhere
    V_true = value_iteration.policy_value(truth.P, truth.reward, truth.actions_of(dataset)[pi], 0.9)
    promise = V_true - 1.0
    promise[2] = V_true[2] + 0.5  # the promise fails in one state, by 0.5
    out = evaluate(truth, dataset, Solution(promise, pi), 0.9, V_star, random_starts(truth, 4, 0))
    assert not out["safe"]
    assert np.isclose(out["violation_rate"], 1 / truth.n_states) and np.isclose(out["worst_violation"], 0.5)
    out = evaluate(truth, dataset, Solution(V_true - 1.0, pi), 0.9, V_star, random_starts(truth, 4, 0))
    assert out["safe"] and out["violation_rate"] == 0 and out["worst_violation"] == 0
