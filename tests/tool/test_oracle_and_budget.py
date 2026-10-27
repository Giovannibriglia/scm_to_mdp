"""The oracle methods (the true observed MDP given to the solver) and the solve limit relative to them."""

import numpy as np
import pandas as pd
import pytest

from scm_to_mdp.baselines import true_model
from scm_to_mdp.core import Reward
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.plotting import figures
from scm_to_mdp.solving.stopping import iterations_bound
from scm_to_mdp.tool import Experiment, Limits, Method, oracle_methods
from scm_to_mdp.tool.sweep import run_point, step_limits

ORACLES = oracle_methods()


def make_env(size, seed):
    return SysAdmin(int(size), seed, storm=0.1, storm_effect=0.8)  # a hidden storm: V*_obs < V*


def collect(env, n_rollouts):
    return env.hide(env.sample(n_rollouts, 16, "random"))


def experiment(methods, factor=None, **kw) -> Experiment:
    return Experiment(
        name="oracle", make_env=make_env, collect=collect, sizes=[3], n_rollouts=[2, 4], seeds=[0],
        methods=methods, limits=Limits(mem_gb=2, timeout_s={"solve": 999.0}), save_models="none",
        solve_factor=factor, **kw,
    )  # fmt: skip


def test_oracle_roles_and_guarantees():
    assert ORACLES["oracle -> IMDP"].role == ORACLES["oracle -> MDP"].role == "oracle"
    assert ORACLES["oracle -> IMDP"].guarantee and not ORACLES["oracle -> MDP"].guarantee
    with pytest.raises(ValueError, match="tol > 0"):
        Method("oracle", "vi", learning_options={"tol": 1e-3})
    with pytest.raises(ValueError, match="tol = 0"):
        Method("oracle", "robust_vi")


def test_the_oracle_reaches_v_star_obs_and_its_promise_holds():
    rows = pd.DataFrame(run_point(experiment(ORACLES), 3, 0, {}))
    assert (rows["status"] == "ok").all() and set(rows["role"]) == {"oracle"}
    exact = rows[rows["method"] == "oracle -> MDP"]
    assert np.allclose(exact["true"], exact["V_star_obs"]) and np.allclose(exact["promise"], exact["true"])
    robust = rows[rows["method"] == "oracle -> IMDP"]
    assert robust["safe"].all() and (robust["promise"] <= robust["true"] + 1e-9).all()
    assert (robust["true"] - robust["promise"] < 1e-3).all()  # within about tol / (1 - gamma)
    assert (rows["V_star_obs"] < rows["V_star"]).all()  # the storm is hidden


def test_the_model_is_the_true_kernel_within_tol():
    env = make_env(3, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(collect(env, 20))
    kernel = truth.observed_kernel()[:, truth.actions_of(data), :]
    mdp = true_model.model(kernel, data, tol=0.01)
    assert np.all(mdp.L <= kernel) and np.all(kernel <= mdp.U) and np.all(mdp.U - mdp.L <= 0.02 + 1e-12)
    assert np.array_equal(true_model.model(kernel, data).L, kernel)


def test_relative_solve_limit():
    flat = {"flat IMDP": Method("flat_rmdp", "robust_vi")}
    exp = experiment(ORACLES | flat, factor=3.0, solve_floor_s=0.5)
    oracle_limits = step_limits(exp, ORACLES["oracle -> IMDP"], [])
    assert "solve" not in oracle_limits.timeout_s and oracle_limits.mem_gb == 2  # memory only
    reward = Reward.immediate(np.ones((4, 2)))  # scale 1
    sweeps = iterations_bound(1.0, 1e-7, 0.9)  # the stopping rule's sweeps at the rate gamma = 0.9
    assert 150 < sweeps < 250
    assert step_limits(exp, flat["flat IMDP"], [], reward).timeout_s["solve"] == 999.0  # no oracle
    limit = step_limits(exp, flat["flat IMDP"], [0.002, 0.004], reward).timeout_s["solve"]
    assert limit == pytest.approx(3.0 * 0.004 * sweeps)  # 3 x the slowest sweep x the sweeps at rate gamma
    assert step_limits(exp, flat["flat IMDP"], [1e-6], reward).timeout_s["solve"] == 0.5  # the floor
    assert step_limits(experiment(flat), flat["flat IMDP"], [2.0], reward).timeout_s["solve"] == 999.0  # off
    rows = pd.DataFrame(run_point(exp, 3, 0, {}))
    assert list(rows["method"][:2]) == ["oracle -> IMDP", "oracle -> MDP"]  # the oracles run first
    limit = rows.loc[rows["method"] == "flat IMDP", "solve_limit_s"]
    assert (limit >= 0.5).all() and rows.loc[rows["role"] == "oracle", "solve_limit_s"].isna().all()
    assert (rows["solve_iterations"] > 0).all() and (rows["solve_iteration_s"] > 0).all()


def test_axis_by_observed_states():
    res = pd.DataFrame({"size": ["3", "2", "4"], "seed": 0, "n_observed_states": [8, 4, 16]})
    axis = figures.Axis(res, ["2", "3", "4"], ["2", "3", "4"], "N")
    assert (
        axis.order == ["2", "3", "4"] and np.allclose(axis.x, [2, 3, 4]) and axis.labels == ["4", "8", "16"]
    )
    same = pd.DataFrame({"size": ["a", "b"], "seed": 0, "n_observed_states": [500, 500]})
    axis = figures.Axis(same, ["a", "b"], ["a", "b"], "variant")
    assert np.allclose(axis.x, [0, 1]) and axis.labels == ["500\na", "500\nb"]


def test_transitions_needed():
    rows = []
    for n, t, true in [(1, 256, 0.5), (2, 512, 0.99), (4, 1000, 0.99)]:
        rows.append(
            {"size": "3", "seed": 0, "method": "m", "n_rollouts": n, "transitions": t, "true": true,
             "status": "ok", "V_star_obs": 1.0, "V_star": 1.0, "n_states": 16, "n_observed_states": 8}
        )  # fmt: skip
    needed = figures.rollouts_needed(pd.DataFrame(rows))
    assert needed["n_star"].item() == 2 and needed["t_star"].item() == 512
    summary = figures.summarize_rollouts(needed)
    assert summary["median_t"].item() == 512 and summary["n_observed_states"].item() == 8
    assert figures.samples_label(12_000) == "12k" and figures.samples_label(1_500_000) == "1.5M"
