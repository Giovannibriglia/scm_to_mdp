"""The pictures and result figures (plotting/): the MDP pictures, the figures of a sweep, the
rollouts needed, the parameter-learning and normalised-value figures, the validity gate."""

import numpy as np
import pandas as pd
import pytest

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import IntervalMDP
from scm_to_mdp.discovery import oracle
from scm_to_mdp.plotting import figures, mdp_graph
from scm_to_mdp.plotting.style import method_styles
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import given


def test_mdp_pictures(env, truth, dataset, tmp_path):
    P = truth.P[:, truth.actions_of(dataset)]
    bounds = nbn_intervals.learn(
        dataset, given.select(oracle.fit(dataset, env.true_parents()), dataset), 0.05
    )
    labels = [truth.state_name(s) for s in range(truth.n_states)]
    actions = [f"k{k}" for k in range(P.shape[1])]
    mdp_graph.draw(P, IntervalMDP(P, P, dataset.reward), labels, actions, "truth", tmp_path / "truth")
    mdp_graph.draw(P, product_bounds.compile(bounds), labels, actions, "learned", tmp_path / "learned")
    assert (tmp_path / "truth.pdf").exists() and (tmp_path / "learned.png").exists()


def fake_results(rng) -> pd.DataFrame:
    """A results table in the layout of results.csv: 3 sizes x 3 seeds x 2 rollouts x 2 methods."""
    rows = []
    for size, states in [("3", 8), ("4", 16), ("5", 32)]:
        for seed in range(3):
            for n in [50, 100]:
                for method, role in [("BIC -> IMDP", "ours"), ("flat MDP", "baseline")]:
                    stopped = method == "BIC -> IMDP" and size == "5" and n == 100
                    row = {
                        "size": size,
                        "n_states": states,
                        "seed": seed,
                        "n_transitions": n,
                        "method": method,
                    }
                    row |= {"role": role, "guarantee": role == "ours", "V_star": 10.0, "V_random": 5.0}
                    row |= {
                        "status": "timeout" if stopped else "ok",
                        "true": 9.0 + rng.random() if n == 100 else 8.0,
                    }
                    row |= {
                        m: rng.random()
                        for m in ["regret", "gap", "regret_K", "gap_K", "violation_rate", "worst_violation"]
                    }
                    row |= {"safe": method == "BIC -> IMDP", "shd": int(size) if role == "ours" else np.nan}
                    row |= {"sid": 2 * int(size) if role == "ours" else np.nan}
                    for step in ["discovery", "learning", "solve"]:
                        row |= {f"{step}_s": rng.random() * 10, f"{step}_gb": rng.random()}
                        row[f"{step}_status"] = "timeout" if stopped and step == "discovery" else "ok"
                    rows.append(row)
    return pd.DataFrame(rows)


def test_result_figures(tmp_path):
    res = fake_results(np.random.default_rng(0))
    methods = ["BIC -> IMDP", "flat MDP"]
    styles = method_styles({"BIC -> IMDP": ("ours", True), "flat MDP": ("baseline", False)})
    assert styles["flat MDP"]["ls"] == "--" and styles["BIC -> IMDP"]["ls"] == "-"
    at = res[res["n_transitions"] == 100]
    axis = figures.Axis(at, ["3", "4", "5"], ["N=3", "N=4", "N=5"], "N")
    figures.performance(at, axis, methods, styles, 20, "t", tmp_path / "performance")
    figures.structure(at, axis, methods, styles, "t", tmp_path / "structure")
    figures.validity(at, axis, methods, styles, 0.05, "t", tmp_path / "validity")
    figures.resources(at, axis, methods, styles, "s", {"discovery": 60, "solve": 60}, "t", tmp_path / "time")
    figures.resources(
        at, axis, methods, styles, "gb", dict.fromkeys(figures.STEP_TITLES, 8.0), "t", tmp_path / "memory"
    )
    summary = figures.sample_efficiency(
        res, ["3", "4", "5"], ["3", "4", "5"], "N", methods, styles, 0.85, "t", tmp_path / "sample_efficiency"
    )
    for name in ["performance", "validity", "structure", "time", "memory", "sample_efficiency"]:
        assert (tmp_path / f"{name}.pdf").exists()
    flat = summary[summary["method"] == "flat MDP"]
    assert (flat["median"] == 100).all()  # true = 8 < 8.5 at 50 rollouts, >= 9 at 100


def test_transitions_needed_counts_only_lasting_success():
    """n* is the first n from which the policy stays 95% optimal; a stopped run is not optimal."""
    rows = []
    for n, true, status in [(10, 9.6, "ok"), (20, 9.0, "ok"), (40, 9.7, "ok"), (80, 9.8, "ok")]:
        rows.append(
            {
                "size": "a",
                "n_states": 4,
                "seed": 0,
                "method": "m",
                "n_transitions": n,
                "V_star": 10.0,
                "true": true,
                "status": status,
            }
        )
    for n, status in [(10, "ok"), (20, "ok"), (40, "timeout"), (80, "skipped")]:
        rows.append(
            {
                "size": "a",
                "n_states": 4,
                "seed": 1,
                "method": "m",
                "n_transitions": n,
                "V_star": 10.0,
                "true": 9.9,
                "status": status,
            }
        )
    needed = figures.transitions_needed(pd.DataFrame(rows), 0.95).set_index("seed")["n_star"]
    assert needed[0] == 40  # 10 was good, but 20 was not: success from 40 on
    assert np.isinf(needed[1])  # never good up to the largest n


def test_parameter_learning_figure(tmp_path):
    rng = np.random.default_rng(0)
    rows = [
        {"method": m, "role": "ours", "guarantee": True, "size": size, "seed": seed,
         "s4_width": rng.uniform(0.1, 0.5), "s4_coverage": rng.uniform(0.95, 1),
         "s4_unseen": rng.uniform(0, 0.3), "s4_rows": 10 * size}
        for m in ("A", "B") for size in (2, 3, 4) for seed in (0, 1)
    ]  # fmt: skip
    res = pd.DataFrame(rows)
    axis = figures.Axis(res, [2, 3, 4], ["2", "3", "4"], "N")
    styles = method_styles({"A": ("ours", True), "B": ("ours", True)})
    figures.parameter_learning(res, axis, ["A", "B"], styles, 0.05, "t", tmp_path / "parameter_learning")
    assert (tmp_path / "parameter_learning.png").exists() and (tmp_path / "parameter_learning.pdf").exists()


def test_normalised_value_and_its_figure(tmp_path):
    res = pd.DataFrame(
        {"method": ["A"] * 4, "role": "ours", "guarantee": True, "size": [2, 2, 3, 3], "seed": [0, 1, 0, 1],
         "true": [5.0, 3.0, 2.0, 2.0], "promise": [4.0, 3.5, 1.0, 1.0], "V_star_obs": [5.0, 5.0, 2.0, 2.0],
         "V_random": [1.0, 1.0, 2.0, 2.0], "safe": [True, False, True, True]}
    )  # fmt: skip
    res = res.assign(
        V_star_K=[6.0, 6.0, 2.0, 2.0], regret_K=[1.0, 3.0, 0.0, 0.0], V_random_K=[2.0, 2.0, 2.0, 2.0]
    )
    out = figures.normalised(res)
    assert out["norm_value"].tolist()[:2] == [1.0, 0.5] and out["norm_value_K"].tolist()[:2] == [0.75, 0.25]
    assert out["norm_value"].iloc[2:].isna().all()  # random = optimal: nothing to normalise by
    res = res.assign(V_star=[10.0, 10.0, 4.0, 4.0], V_oracle_K=[8.0, 8.0, 4.0, 4.0])
    out = figures.normalised(res)
    assert out["oracle_share"].tolist() == [0.5, 0.3, 0.5, 0.5]
    assert out["oracle_share_K"].tolist()[:2] == [0.625, 0.375]
    axis = figures.Axis(res, [2, 3], ["2", "3"], "N")
    styles = method_styles({"A": ("ours", True)})
    figures.performance_normalized(
        res, axis, ["A"], styles, 0.95, 20, "t", tmp_path / "performance_normalized"
    )
    assert (tmp_path / "performance_normalized.png").exists()


def test_validity_gate_and_figure(tmp_path):
    rows = []
    for method, safe in (("A", [True, True, True]), ("B", [True, False, True])):
        for seed, ok in enumerate(safe):
            rows.append({"method": method, "role": "ours", "guarantee": True, "size": "2", "seed": seed,
                         "safe": ok, "true": 3.0, "promise": 2.0, "V_star_obs": 4.0, "V_random": 0.0,
                         "cert_gap": 1.5})  # fmt: skip
    res = pd.DataFrame(rows)
    gate = figures.validity_gate(res, 0.05)
    assert gate[res["method"] == "A"].all() and not gate[res["method"] == "B"].any()
    out = figures.normalised(res)
    assert np.allclose(out["norm_promise"], 0.5) and np.allclose(out["norm_cert_gap"], 0.375)
    axis = figures.Axis(res, ["2"], ["2"], "N")
    styles = method_styles({"A": ("ours", True), "B": ("ours", True)})
    figures.gated_comparison(res, axis, ["A", "B"], styles, 0.05, "t", tmp_path / "gated")
    assert (tmp_path / "gated.png").exists()


def test_chain_picture_edges_and_grid_layout(tmp_path):
    """Solid edges for the likeliest next states (ties included), dashed for the less likely ones; a
    one-agent lake is laid out on its grid."""
    from scm_to_mdp.envs import FrozenLake
    from scm_to_mdp.tool import store

    row = np.array([0.1, 0.4, 0.4, 0.06, 0.04])
    assert mdp_graph.outcomes(row, 0.05, 3) == [(1, True), (2, True), (0, False)]
    env = FrozenLake("4x4", 1, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(5, 10, "random"))
    pos = store.positions(data)
    assert pos.shape == (16, 2) and set(map(tuple, pos)) == {(x, -y) for x in range(4) for y in range(4)}
    store.draw_ground_truth(truth, data, "lake", tmp_path / "lake")
    assert (tmp_path / "lake.png").exists()


def test_legend_without_handles():
    """A figure where no method drew a line (e.g. every method failed) gets no legend, not an error."""
    import matplotlib.pyplot as plt

    from scm_to_mdp.plotting import grid_figures

    fig = plt.figure()
    figures._legend(fig, [], {})
    grid_figures._legend(fig, [])
    assert not fig.legends
    plt.close(fig)


def test_iqm_and_its_bootstrap_interval():
    from scm_to_mdp.plotting.stats import iqm, iqm_ci

    x = np.arange(1.0, 11.0)  # 1..10: the IQM is the mean of the middle 50% (scipy trim_mean, 25%)
    assert iqm(x) == pytest.approx(5.5)
    assert iqm([1, 2, 3, 100]) == pytest.approx(2.5)  # the outlier does not move it
    centre, lo, hi = iqm_ci(x)
    assert lo <= centre <= hi and hi - lo > 0
    assert iqm_ci([3.0]) == (3.0, 3.0, 3.0)
    assert np.isnan(iqm_ci([np.nan])[0])
    _, lo_s, hi_s = iqm_ci(np.r_[x, x + 100], strata=[0] * 10 + [1] * 10)  # stratified by task
    assert lo_s < hi_s
