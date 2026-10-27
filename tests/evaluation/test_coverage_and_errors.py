"""The data's coverage of a task's rows (evaluation.coverage), the learned model against the true kernel
(evaluation.model_errors), and the grid report's validity and box-looseness tables."""

import importlib

import numpy as np
import pandas as pd
import pytest

from scm_to_mdp import evaluation as ev
from scm_to_mdp.core import IntervalMDP
from scm_to_mdp.envs import SysAdmin

rg = importlib.import_module("scm_to_mdp.tool.report_grid")  # the module (the package exports the function)


@pytest.fixture(scope="module")
def expert_data():
    env = SysAdmin(4, 0, storm=0.1)
    truth = env.ground_truth()
    df = env.hide(env.sample(15, 20, env.behaviour("expert", 0.9, 0.0)))
    return truth, truth.learner_spec.read(df)


def test_coverage_shares(expert_data):
    truth, data = expert_data
    cov = ev.coverage(truth, data, 0.9, truth.parents_in(data))
    for k in ev.COVERAGE_COUNTS:
        assert 0.0 <= cov[f"cov_factor_{k}"] <= cov[f"cov_joint_{k}"] <= 1.0  # a seen row has seen factors
    joint = [cov[f"cov_joint_{k}"] for k in ev.COVERAGE_COUNTS]
    assert joint == sorted(joint)  # a higher count threshold leaves more rows unseen
    assert cov["cov_alt"] > 0  # a deterministic expert never tries the other repairs everywhere
    d, _ = ev.task_occupancy(truth, data, 0.9)
    assert np.isclose(d.sum(), 1.0) and (d[data.reward.terminal] == 0).all()


def test_model_errors_of_the_truth_and_of_a_vacuous_set(expert_data):
    truth, data = expert_data
    P = truth.observed_kernel()[:, truth.actions_of(data)]
    exact = ev.model_errors(IntervalMDP(P, P, data.reward), truth, data, 0.9)
    assert np.isclose(exact["tv_uniform"], 0.0) and np.isclose(exact["kl_data"], 0.0)
    vacuous = ev.model_errors(IntervalMDP(np.zeros_like(P), np.ones_like(P), data.reward), truth, data, 0.9)
    assert vacuous["contain_all"] and np.isclose(vacuous["width_unseen"], P.shape[-1])
    narrow = ev.model_errors(IntervalMDP(P + 0.01, P + 0.02, data.reward), truth, data, 0.9)
    assert narrow["contain_uniform"] == 0.0 and not narrow["contain_all"]


def test_validity_verdicts_and_box_looseness():
    lo, hi = rg.clopper_pearson(100, 100)
    assert np.isclose(lo, 0.025 ** (1 / 100)) and hi == 1.0
    rows = []
    for method, held in (("A", 100), ("B", 85), ("C", 97)):
        for seed in range(100):
            rows.append({"knob": "k", "policy": "p", "size": "3", "n_transitions": 10, "method": method,
                         "seed": seed, "status": "ok", "guarantee": True, "role": "ours",
                         "safe": seed < held, "contain_all": True})  # fmt: skip
    verdicts = rg.validity_table(pd.DataFrame(rows), 0.05).set_index("method")["verdict"]
    assert verdicts.to_dict() == {"A": "confirmed", "B": "refuted", "C": "undecided"}
    pair = [
        {"knob": "k", "policy": "p", "size": "3", "n_transitions": 10, "seed": 0, "status": "ok",
         "method": m, "promise": promise, "true": 2.0}
        for m, promise in ((rg.BOX, 1.0), (rg.PRODUCT, 1.5))
    ]  # fmt: skip
    loose = rg.box_looseness(pd.DataFrame(pair))
    assert np.isclose(loose["promise_gain"].iloc[0], 0.5) and np.isclose(loose["box_share"].iloc[0], 0.5)


def test_a_promise_is_about_the_observed_process():
    """With a hidden storm drawn afresh at every step, the value of the observed MDP is exactly what an
    agent that sees only the machines gets on average: as a promise it holds (safe), although a storm-on
    full state lies below it (the old per-full-state check failed it)."""
    from scm_to_mdp.core import Solution
    from scm_to_mdp.solving import value_iteration

    env = SysAdmin(3, 0, storm=0.3, storm_effect=0.8)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.hide(env.sample(5, 10, "random")))
    P = truth.observed_kernel()[:, truth.actions_of(data)]
    exact = value_iteration.solve(P, data.reward, 0.9)
    _, V_target = truth.observed_optimal(0.9)
    starts = np.flatnonzero(~truth.reward.terminal)
    metrics = ev.evaluate(truth, data, Solution(exact.V, exact.pi), 0.9, V_target, starts)
    assert metrics["safe"] and metrics["worst_gap"] < 1e-6
    V_full = ev.true_values(truth, data, Solution(exact.V, exact.pi), 0.9)
    assert (exact.V[truth.observed_index()] > V_full + 1e-3).any()  # below it in some storm-on state
