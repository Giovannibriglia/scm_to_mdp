import numpy as np
import pandas as pd

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import Graph, IntervalMDP, Solution
from scm_to_mdp.discovery import oracle
from scm_to_mdp.evaluation import evaluate, optimal_values, random_starts, structure_distances
from scm_to_mdp.plotting import figures, mdp_graph
from scm_to_mdp.plotting.style import method_styles
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import given
from scm_to_mdp.solving import value_iteration


def test_shd_and_sid(env, dataset):
    truth = env.true_parents()
    assert structure_distances(Graph(truth), truth, dataset) == {"shd": 0, "missing": 0, "extra": 0, "sid": 0}
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
                for method, role in [("BIC -> RMDP", "ours"), ("flat MDP", "baseline")]:
                    stopped = method == "BIC -> RMDP" and size == "5" and n == 100
                    row = {"size": size, "n_states": states, "seed": seed, "n_rollouts": n, "method": method}
                    row |= {"role": role, "guarantee": role == "ours", "V_star": 10.0, "V_random": 5.0}
                    row |= {
                        "status": "timeout" if stopped else "ok",
                        "true": 9.0 + rng.random() if n == 100 else 8.0,
                    }
                    row |= {
                        m: rng.random()
                        for m in ["regret", "gap", "regret_K", "gap_K", "violation_rate", "worst_violation"]
                    }
                    row |= {"safe": method == "BIC -> RMDP", "shd": int(size) if role == "ours" else np.nan}
                    row |= {"sid": 2 * int(size) if role == "ours" else np.nan}
                    for step in ["discovery", "learning", "solve"]:
                        row |= {f"{step}_s": rng.random() * 10, f"{step}_gb": rng.random()}
                        row[f"{step}_status"] = "timeout" if stopped and step == "discovery" else "ok"
                    rows.append(row)
    return pd.DataFrame(rows)


def test_result_figures(tmp_path):
    res = fake_results(np.random.default_rng(0))
    methods = ["BIC -> RMDP", "flat MDP"]
    styles = method_styles({"BIC -> RMDP": ("ours", True), "flat MDP": ("baseline", False)})
    assert styles["flat MDP"]["ls"] == "--" and styles["BIC -> RMDP"]["ls"] == "-"
    at = res[res["n_rollouts"] == 100]
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


def test_rollouts_needed_counts_only_lasting_success():
    """n* is the first n from which the policy stays 95% optimal; a stopped run is not optimal."""
    rows = []
    for n, true, status in [(10, 9.6, "ok"), (20, 9.0, "ok"), (40, 9.7, "ok"), (80, 9.8, "ok")]:
        rows.append(
            {
                "size": "a",
                "n_states": 4,
                "seed": 0,
                "method": "m",
                "n_rollouts": n,
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
                "n_rollouts": n,
                "V_star": 10.0,
                "true": 9.9,
                "status": status,
            }
        )
    needed = figures.rollouts_needed(pd.DataFrame(rows), 0.95).set_index("seed")["n_star"]
    assert needed[0] == 40  # 10 was good, but 20 was not: success from 40 on
    assert np.isinf(needed[1])  # never good up to the largest n


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
