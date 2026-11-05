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


def test_coverage_efficiency_needs_lasting_coverage_and_small_sets(tmp_path):
    """Left: the first n from which coverage stays >= 95%; right: also mean width <= 0.1. The oracle and
    the methods without intervals (point models) are left out."""
    rows = []
    # (n, coverage, width) per seed for "PCMCI+ -> IMDP"; the true graph is always covered, and small at 80
    runs = {
        0: [(10, 0.99, 1.0), (20, 0.80, 0.9), (40, 0.97, 0.3), (80, 1.0, 0.08)],
        1: [(10, 0.90, 1.0), (20, 0.96, 0.5), (40, 0.98, 0.2), (80, 0.99, 0.2)],
        2: [(10, 1.0, 1.0), (20, 1.0, 0.5), (40, 1.0, 0.09), (80, 1.0, 0.05)],
    }
    for seed, cells in runs.items():
        for n, cov, width in cells:
            for method, role, c, w in [
                ("PCMCI+ -> IMDP", "ours", cov, width),
                ("true graph -> IMDP", "reference", 1.0, 1.0 if n < 80 else 0.05),
                ("oracle -> IMDP", "oracle", 1.0, 0.0),
                ("flat MDP", "baseline", np.nan, np.nan),
            ]:
                rows.append({"size": "3", "n_states": 8, "seed": seed, "n_transitions": n, "method": method,
                             "role": role, "status": "ok", "s4_coverage_seen": c, "s4_width": w})  # fmt: skip
    res = pd.DataFrame(rows)
    methods = list(res["method"].unique())
    styles = method_styles(
        {m: (r, "IMDP" in m) for m, r in res.drop_duplicates("method")[["method", "role"]].values}
    )
    out = figures.coverage_efficiency(
        res,
        ["3"],
        ["3"],
        "N",
        methods,
        styles,
        "s4_coverage_seen",
        "t",
        tmp_path / "coverage_seen",
        target=0.95,
        eps=0.1,
    )
    assert (tmp_path / "coverage_seen.pdf").exists()
    valid, tight = (out[k].set_index("method") for k in ("valid", "tight"))
    assert set(valid.index) == {"PCMCI+ -> IMDP", "true graph -> IMDP"}  # no oracle, no point model
    needed = figures.transitions_needed(
        res[res["method"] == "PCMCI+ -> IMDP"],
        condition=pd.to_numeric(res["s4_coverage_seen"]) >= 0.95,
    ).set_index("seed")["n_star"]
    assert needed.to_dict() == {0: 40, 1: 20, 2: 10}  # seed 0: covered at 10, not at 20
    assert valid.at["true graph -> IMDP", "median"] == 10
    assert tight.at["true graph -> IMDP", "median"] == 80
    assert tight.at["PCMCI+ -> IMDP", "median"] == 80  # seeds: 80, never, 40
    empty = figures.coverage_efficiency(
        res, ["3"], ["3"], "N", methods, styles, "s4_coverage", "t", tmp_path / "coverage"
    )
    assert empty["valid"].empty and not (tmp_path / "coverage.pdf").exists()


def test_guarantee_efficiency_needs_a_lasting_promise_and_a_proof_better_than_random(tmp_path):
    """Left: the first n from which the promise holds (safe); right: also (V_rob - V_rand) / (V* - V_rand)
    >= 0. The oracle and the point models (no contain_all) are left out. The per-budget certificate:
    proved (V_opt - V_rob) and actual (V* - V^pi) regret, normalised."""
    rows = []
    # per seed of "PCMCI+ -> IMDP": (n, safe, promise, cert_gap); V* = 10, V_rand = 0, V^pi = 9
    runs = {
        0: [(10, True, -5.0, 9.0), (20, False, 2.0, 5.0), (40, True, 1.0, 2.0), (80, True, 4.0, 0.5)],
        1: [(10, False, -5.0, 9.0), (20, True, -1.0, 3.0), (40, True, 2.0, 0.8), (80, True, 5.0, 0.4)],
        2: [(10, True, -5.0, 9.0), (20, True, -2.0, 6.0), (40, True, -1.0, 3.0), (80, False, 6.0, 0.2)],
    }
    for seed, cells in runs.items():
        for n, safe, promise, gap in cells:
            for method, role, sf, pr, g, c in [
                ("PCMCI+ -> IMDP", "ours", safe, promise, gap, True),
                ("oracle -> IMDP", "oracle", True, 9.0, 0.0, True),
                ("flat MDP", "baseline", False, 9.5, np.nan, np.nan),
            ]:
                rows.append({"size": "3", "n_states": 8, "seed": seed, "n_transitions": n, "method": method,
                             "role": role, "status": "ok", "safe": sf, "promise": pr, "cert_gap": g,
                             "contain_all": c, "V_star_obs": 10.0, "V_random": 0.0, "true": 9.0})  # fmt: skip
    res = pd.DataFrame(rows)
    methods = list(res["method"].unique())
    styles = method_styles(
        {m: (r, "IMDP" in m) for m, r in res.drop_duplicates("method")[["method", "role"]].values}
    )
    out = figures.guarantee_efficiency(res, ["3"], ["3"], "N", methods, styles, "t", tmp_path / "guarantee")
    assert (tmp_path / "guarantee.pdf").exists()
    valid, tight = (out[k].set_index("method") for k in ("valid", "tight"))
    assert list(valid.index) == ["PCMCI+ -> IMDP"]  # no oracle, no point model
    assert valid.at["PCMCI+ -> IMDP", "median"] == 40  # seeds: 40, 20, never (breaks at 80)
    assert tight.at["PCMCI+ -> IMDP", "median"] == 40  # seeds: 40, 40, never
    assert out["contain"].set_index("method").at["PCMCI+ -> IMDP", "median"] == 10
    at = res[res["n_transitions"] == 40]  # certificates 2.0, 0.8, 3.0 -> 0.2; regret 1 - 0.9 = 0.1
    axis = figures.Axis(at, ["3"], ["3"], "N")
    table = figures.certificate(at, axis, methods, styles, "t", tmp_path / "certificate").set_index("method")
    assert (tmp_path / "certificate.pdf").exists() and list(table.index) == ["PCMCI+ -> IMDP"]
    assert table.at["PCMCI+ -> IMDP", "median"] == pytest.approx(0.2)
    assert table.at["PCMCI+ -> IMDP", "regret"] == pytest.approx(0.1)


def test_regret_map_marks_the_frontier_where_the_regret_stays_small(tmp_path):
    """Cells hold the median normalised regret over seeds; reached = the first n from which the regret
    stays <= 0.05 (median over seeds); the oracle is left out."""
    rows = []
    # V* = 10, V_rand = 0: regret = 1 - true / 10; per seed of "PCMCI+ -> IMDP": true at n = 10, 20, 40, 80
    runs = {0: [5.0, 9.6, 9.0, 9.8], 1: [12.0, 9.7, 9.9, 9.9], 2: [5.0, 6.0, 9.6, 9.9]}
    for seed, trues in runs.items():
        for n, true in zip((10, 20, 40, 80), trues, strict=True):
            for method, role, v in [("PCMCI+ -> IMDP", "ours", true), ("oracle -> IMDP", "oracle", 10.0)]:
                row = {"size": "3", "n_states": 8, "seed": seed, "n_transitions": n, "method": method}
                rows.append(
                    row | {"role": role, "status": "ok", "true": v, "V_star_obs": 10.0, "V_random": 0.0}
                )
    res = pd.DataFrame(rows)
    methods = list(res["method"].unique())
    styles = method_styles({"PCMCI+ -> IMDP": ("ours", True), "oracle -> IMDP": ("oracle", True)})
    cells = figures.regret_map(res, ["3"], ["3"], "N", methods, styles, "t", tmp_path / "regret_map")
    assert (tmp_path / "regret_map.pdf").exists()
    assert set(cells["method"]) == {"PCMCI+ -> IMDP"}
    at = cells.set_index("n_transitions")
    assert at.at[10, "regret"] == pytest.approx(0.5)  # regrets 0.5, -0.2, 0.5 -> median 0.5
    # regret <= 0.05 at n and every larger n: seed 0 from 80 (0.1 at 40), seed 1 from 10 (-0.2, 0.03, ...),
    # seed 2 from 40 (0.04, 0.01) -> median 40
    assert at.at[10, "n_reached"] == 40


def test_regret_gain_pairs_seeds_and_scales_by_decades(tmp_path):
    """The gain is paired per seed and divided by log10 of the step; the learning rate fits r ~ n^-b."""
    rows = []
    # V* = 10, V_rand = 0: regret r = 1 - true / 10 = 0.8 * (n / 10)^-0.5 for every seed (b = 0.5)
    for seed in range(3):
        for n in (10, 100, 1000, 10000):
            true = 10 * (1 - 0.8 * (n / 10) ** -0.5)
            for method, role, v in [("PCMCI+ -> IMDP", "ours", true), ("oracle -> IMDP", "oracle", 10.0)]:
                row = {"size": "3", "n_states": 8, "seed": seed, "n_transitions": n, "method": method}
                rows.append(
                    row | {"role": role, "status": "ok", "true": v, "V_star_obs": 10.0, "V_random": 0.0}
                )
    res = pd.DataFrame(rows)
    methods = list(res["method"].unique())
    styles = method_styles({"PCMCI+ -> IMDP": ("ours", True), "oracle -> IMDP": ("oracle", True)})
    cells, rates = figures.regret_gain(res, ["3"], ["3"], "N", methods, styles, "t", tmp_path / "regret_gain")
    assert (tmp_path / "regret_gain.pdf").exists() and set(cells["method"]) == {"PCMCI+ -> IMDP"}
    first = cells.set_index("n_from").loc[10]
    assert first["gain"] == pytest.approx(0.8 - 0.8 * 10**-0.5)  # one decade: 0.8 -> 0.253
    rate = rates.set_index("method").loc["PCMCI+ -> IMDP"]
    assert rate["rate"] == pytest.approx(0.5) and rate["seeds"] == 3


def test_zoom_by_size_draws_one_figure_per_size_and_drops_empty_panels(tmp_path):
    """Per size and budget: the median share of V*_obs (every seed finished), the seeds whose sets contain
    the truth, the median of the missed edges; the oracle left out; two sizes with the same number of
    states get distinct files; a run without sets (an MDP) has no middle panel."""
    rows = []
    # V*_obs = 10; per seed of "PCMCI+ -> IMDP": (true, contain_all, missing) at n = 10, 100, 1000
    runs = {0: [(5.0, 0, 3), (9.0, 1, 1), (9.9, 1, 0)], 1: [(6.0, 0, 4), (9.6, 1, 1), (9.8, 1, 0)],
            2: [(4.0, 1, 3), (9.0, 0, 2), (9.7, 1, 0)]}  # fmt: skip
    for size in ("a", "b"):  # both 8 observed states
        for seed, cells in runs.items():
            for n, (true, contain, missing) in zip((10, 100, 1000), cells, strict=True):
                for method, role, v, c, m in [("PCMCI+ -> IMDP", "ours", true, contain, missing),
                                              ("oracle -> IMDP", "oracle", 10.0, 1, np.nan),
                                              ("flat MDP", "baseline", 8.0, np.nan, np.nan)]:  # fmt: skip
                    stopped = method == "flat MDP" and n == 1000 and seed == 2
                    rows.append({"size": size, "n_states": 8, "n_observed_states": 8, "seed": seed,
                                 "n_transitions": n, "method": method, "role": role, "V_star": 10.0,
                                 "status": "timeout" if stopped else "ok", "true": v, "V_star_obs": 10.0,
                                 "V_random": 0.0, "contain_all": c, "missing": m})  # fmt: skip
    res = pd.DataFrame(rows)
    methods = list(res["method"].unique())
    styles = method_styles(
        {m: (r, "IMDP" in m) for m, r in res.drop_duplicates("method")[["method", "role"]].values}
    )
    points = figures.zoom_by_size(res, ["a", "b"], ["a", "b"], methods, styles, 0.95, "t", tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.pdf")) == [
        "observed_states=8_a.pdf",
        "observed_states=8_b.pdf",
    ]
    assert set(points["method"]) == {"PCMCI+ -> IMDP", "flat MDP"}  # no oracle
    at = points[(points["size"] == "a") & (points["method"] == "PCMCI+ -> IMDP")].set_index("n_transitions")
    assert at.at[100, "share"] == pytest.approx(0.9)  # 0.9, 0.96, 0.9 -> median 0.9
    assert at.at[10, "contain"] == 1 and at.at[1000, "contain"] == 3
    assert at.at[10, "missing"] == 3  # 3, 4, 3
    flat = points[(points["size"] == "a") & (points["method"] == "flat MDP")]
    assert list(flat["n_transitions"]) == [10, 100]  # a seed stopped at 1000: no point
    assert flat["contain"].isna().all() and flat["missing"].isna().all()
    # only point models: the figure has no "contain" panel (and no "missing" panel without a graph)
    mdp = res[res["method"] == "flat MDP"].assign(contain_all=np.nan)
    out = figures.zoom_by_size(mdp, ["a"], ["a"], ["flat MDP"], styles, 0.95, "t", tmp_path / "mdp")
    assert (tmp_path / "mdp" / "observed_states=8.png").exists() and out["contain"].isna().all()


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


def test_median_and_its_order_statistic_interval():
    """The transitions needed (censored at inf): the median with the [k-th smallest, k-th largest]
    interval of coverage >= 95% (10 runs: 2nd and 9th; 100 runs: 40th and 61st)."""
    from scm_to_mdp.plotting.stats import median_ci

    assert median_ci(np.arange(1.0, 11.0)) == (5.0, 2.0, 9.0)
    assert median_ci(np.arange(1.0, 101.0)) == (50.0, 40.0, 61.0)
    assert median_ci([1, 1, 1, np.inf, np.inf]) == (1.0, 1.0, np.inf)  # 5 runs: [min, max] (below 95%)
    assert median_ci([1.0] * 3 + [np.inf] * 7) == (np.inf, 1.0, np.inf)  # most seeds never get there
    assert median_ci([4.0, np.nan]) == (4.0, 4.0, 4.0)
    assert all(np.isnan(median_ci([])))
