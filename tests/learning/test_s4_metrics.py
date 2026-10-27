"""The learned sets against the truth (evaluation.interval_metrics), and every metric reaching
results.csv."""

import numpy as np

from scm_to_mdp import evaluation
from scm_to_mdp.baselines import flat_rmdp
from scm_to_mdp.core import Graph
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.scm_learning import nbn_intervals


def test_interval_metrics_of_true_graph_and_flat_sets():
    env = SysAdmin(5, 0)  # with N = 3 the factored rows (3 x 2 x 2 x 4 = 48) outnumber the (s, a) grid (32)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(400, 20, "random"))
    true_graph = Graph({v: tuple(truth.true_parents[v]) for v in data.state_vars})
    causal = evaluation.interval_metrics(nbn_intervals.learn(data, true_graph), data, truth, 0.9)
    flat = evaluation.interval_metrics(flat_rmdp.learn(data), data, truth, 0.9)
    for m in (causal, flat):
        assert m["s4_coverage"] >= 0.95 and 0 < m["s4_width"] < 1 and 0 <= m["s4_unseen"] < 1
    assert causal["s4_rows"] < flat["s4_rows"]  # a factored model has far fewer rows than the (s, a) grid
    assert causal["s4_width"] < flat["s4_width"]  # and so tighter intervals from the same data
    P = evaluation.true_factors(truth, data)
    assert np.allclose(P.sum(-1), 1.0)  # every factor's values form a distribution (binary: no padding)


def small_sysadmin(n, seed):  # top level: the sweep sends it to a worker process
    return SysAdmin(int(n), seed)


def random_rollouts(env, n):
    return env.sample(n, 10, "random")


def test_every_metric_reaches_results_csv(tmp_path):
    """The sweep writes a fixed list of columns: the S4 metrics and the per-start references must be in it."""
    from scm_to_mdp.tool import Experiment, Method, run_experiment

    exp = Experiment(
        name="cols", make_env=small_sysadmin, collect=random_rollouts, sizes=[2],
        n_rollouts=[30], seeds=[0], methods={"BIC -> IMDP": Method("causal", "box_robust_vi", discovery="bic",
        selection="bic_best", learning_options={"widen": "hull", "k": 1})}, limits=None, workers=1,
        out_dir=tmp_path, save_models="none",
    )  # fmt: skip
    res = run_experiment(exp)
    for column in ("V_random_K", "V_oracle_K", "s4_width", "s4_coverage", "s4_unseen", "s4_rows", "widen"):
        assert column in res.columns and res[column].notna().all(), column
