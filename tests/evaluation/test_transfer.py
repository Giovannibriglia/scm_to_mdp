"""Reward transfer (methods.replan, Env.task): re-planning a learned model under another reward equals
learning on data that carry that reward; model-free methods keep their policy and lose their promise."""

import dataclasses

import numpy as np

from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Method, learn, solve
from scm_to_mdp.tool.methods import replan


def setup():
    env = SysAdmin(3, 0, storm=0.3, storm_effect=0.8)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.hide(env.sample(150, 15, "random")))
    task_truth, reward = env.task("critical_half")
    task_data = dataclasses.replace(data, reward=reward(data.B, data.actions))
    return truth, task_truth, data, task_data


def test_task_truth_shares_the_dynamics_and_changes_the_reward():
    truth, task_truth, _, _ = setup()
    assert task_truth.P is truth.P
    assert not np.allclose(task_truth.reward.R, truth.reward.R)


def test_replanning_equals_learning_with_the_new_reward():
    _, _, data, task_data = setup()
    methods = [
        Method("causal", "box_robust_vi", discovery="bic", selection="bic_best"),
        Method("causal", "vi", discovery="bic", selection="bic_best"),
        Method("flat_rmdp", "robust_vi"),
        Method("focus", "vi"),
        Method("spibb", "spibb"),
    ]
    for method in methods:
        result = solve(learn(data, method), data, 0.9)
        fresh = solve(learn(task_data, method), task_data, 0.9)
        replanned = replan(result, task_data, 0.9)
        assert np.allclose(replanned.solution.V, fresh.solution.V), method.learner
        assert np.array_equal(replanned.solution.pi, fresh.solution.pi), method.learner


def test_model_free_methods_keep_their_policy_and_have_no_promise():
    _, _, data, task_data = setup()
    method = Method("offline_dqn", "greedy_q", solver_options={"steps": 200, "hidden": 16})
    result = solve(learn(data, method), data, 0.9)
    replanned = replan(result, task_data, 0.9)
    assert np.array_equal(replanned.solution.pi, result.solution.pi) and np.isnan(replanned.solution.V).all()


def stormy_sysadmin(n, seed):  # top level: the sweep sends these to worker processes
    return SysAdmin(int(n), seed, storm=0.1, storm_effect=0.8)


def stormier(n, seed):
    return {"storm=0.5": SysAdmin(int(n), seed, storm=0.5, storm_effect=0.8)}


def random_rollouts(env, n):
    return env.hide(env.sample(n, 10, "random"))


def test_dynamics_shift_rows_evaluate_the_same_policy_on_other_dynamics(tmp_path):
    """results_shift.csv: one row per (method, shift); the promise is the original one, the true value
    is the policy's on the shifted model (lower here: storms break machines more often)."""
    import pandas as pd

    from scm_to_mdp.tool import Experiment, run_experiment

    exp = Experiment(
        name="shift", make_env=stormy_sysadmin, collect=random_rollouts, sizes=[3], n_rollouts=[60],
        seeds=[0],
        methods={"BIC -> IMDP": Method("causal", "box_robust_vi", discovery="bic", selection="bic_best"),
                 "BIC -> MDP": Method("causal", "vi", discovery="bic", selection="bic_best")},
        limits=None, workers=1, out_dir=tmp_path, save_models="none", shifts=stormier,
    )  # fmt: skip
    main = run_experiment(exp)
    shifted = pd.read_csv(tmp_path / "results_shift.csv")
    assert len(shifted) == 2 and set(shifted["task"]) == {"storm=0.5"} and (shifted["status"] == "ok").all()
    both = main.set_index("method").join(shifted.set_index("method"), rsuffix="_shift")
    assert np.allclose(both["promise"], both["promise_shift"])  # the same policy and model
    assert (both["true_shift"] < both["true"]).all()  # worse dynamics for every policy
    from scm_to_mdp.tool.report import report

    report(tmp_path, show=False)
    assert (tmp_path / "figures" / "shift" / "average" / "episodes=60" / "transfer_gap.png").exists()
    assert "# Dynamics shift" in (tmp_path / "tables.md").read_text()
