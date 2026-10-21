import numpy as np
import pandas as pd

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import Graph, IntervalMDP
from scm_to_mdp.discovery import oracle
from scm_to_mdp.evaluation import random_starts, structure_distances
from scm_to_mdp.plotting import figures, mdp_graph
from scm_to_mdp.scm_learning import nbn_intervals


def test_shd_and_sid(env, dataset):
    truth = env.true_parents(dataset)
    assert structure_distances(Graph(truth), truth, dataset) == {"shd": 0, "missing": 0, "extra": 0, "sid": 0}
    v = dataset.state_vars[0]
    extra = {**truth, v: tuple(dataset.inputs)}  # every input as a parent: a superset
    d = structure_distances(Graph(extra), truth, dataset)
    assert d["missing"] == 0 and d["extra"] == len(dataset.inputs) - len(truth[v])
    assert d["sid"] == 0  # extra parents are still a valid adjustment set
    missing = {**truth, v: truth[v][:-1]}  # one true parent dropped
    d = structure_distances(Graph(missing), truth, dataset)
    assert (d["shd"], d["missing"], d["extra"]) == (1, 1, 0) and d["sid"] > 0


def test_random_starts_are_live_and_distinct(env):
    starts = random_starts(env, 5, seed=0)
    assert len(set(starts)) == 5 and not env.true_reward.terminal[starts].any()
    assert np.array_equal(starts, random_starts(env, 5, seed=0))


def test_mdp_pictures(env, dataset, tmp_path):
    P = env.P[:, env.env_actions(dataset)]
    bounds = nbn_intervals.learn(dataset, oracle.fit(dataset, env.true_parents(dataset)), 0.05)
    labels = [env.state_label(b) for b in env.B]
    actions = [f"k{k}" for k in range(P.shape[1])]
    mdp_graph.draw(P, IntervalMDP(P, P, dataset.reward), labels, actions, "truth", tmp_path / "truth")
    mdp_graph.draw(P, product_bounds.compile(bounds), labels, actions, "learned", tmp_path / "learned")
    assert (tmp_path / "truth.pdf").exists() and (tmp_path / "learned.png").exists()


def test_result_figures(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for n in [3, 4, 5]:
        for seed in range(3):
            row = {"N": n, "seed": seed}
            for arm in ["BIC -> RMDP", "flat MDP"]:
                row |= {f"{arm}|{m}": rng.random() for m in ["regret", "gap", "regret_K", "gap_K"]}
                row |= {f"{arm}|safe": arm == "BIC -> RMDP", f"{arm}|shd": n, f"{arm}|sid": 2 * n}
                for step in ["discovery", "learning", "solve"]:
                    stopped = arm == "BIC -> RMDP" and step == "discovery" and n == 5
                    row |= {
                        f"{arm}|{step}_s": rng.random() * 10,
                        f"{arm}|{step}_gb": rng.random(),
                        f"{arm}|{step}_status": "timeout" if stopped else "ok",
                    }
            rows.append(row)
    res = pd.DataFrame(rows)
    axis = figures.Axis(res, "N", [3, 4, 5], None, "N")
    arms = ["BIC -> RMDP", "flat MDP"]
    figures.performance(res, axis, arms, 20, tmp_path / "performance")
    figures.structure(res, axis, ["BIC -> RMDP"], tmp_path / "structure")
    for arm in arms:
        res[f"{arm}|violation_rate"], res[f"{arm}|worst_violation"] = (
            rng.random(len(res)),
            rng.random(len(res)),
        )
    figures.validity(res, axis, arms, 0.05, tmp_path / "validity")
    steps = {"discovery": ["BIC -> RMDP"], "learning": arms, "solve": arms}
    figures.resources(
        res, axis, steps, "s", {"discovery": 60, "learning": None, "solve": 60}, tmp_path / "time"
    )
    figures.resources(res, axis, steps, "gb", dict.fromkeys(steps, 8.0), tmp_path / "memory")
    for name in ["performance", "validity", "structure", "time", "memory"]:
        assert (tmp_path / f"{name}.pdf").exists()


def test_violation_metrics(env, dataset):
    from scm_to_mdp.core import Solution
    from scm_to_mdp.evaluation import evaluate, optimal_values
    from scm_to_mdp.solving import value_iteration

    V_star = optimal_values(env, 0.9)
    pi = np.zeros(len(env.B), dtype=np.int64)  # learner action 0 everywhere
    V_true = value_iteration.policy_value(env.P, env.true_reward, env.env_actions(dataset)[pi], 0.9)
    promise = V_true - 1.0
    promise[2] = V_true[2] + 0.5  # the promise fails in one state, by 0.5
    out = evaluate(env, dataset, Solution(promise, pi), 0.9, V_star, random_starts(env, 4, 0))
    assert not out["safe"]
    assert np.isclose(out["violation_rate"], 1 / len(env.B)) and np.isclose(out["worst_violation"], 0.5)
    out = evaluate(env, dataset, Solution(V_true - 1.0, pi), 0.9, V_star, random_starts(env, 4, 0))
    assert out["safe"] and out["violation_rate"] == 0 and out["worst_violation"] == 0
