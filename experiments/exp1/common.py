"""What the exp1 scripts share: running the arms on one dataset, running the grid in waves, the MDP
pictures, the tables and the figures.

An arm is a list of (step kind, function). The first function gets the dataset, every next one gets
the output of the previous step, and the last one returns a Solution. Step kinds: "discovery",
"learning" (probabilities), "compile" (building the RMDP; recorded, never shown in a figure) and
"solve" (deployment). Every step runs in its own process with a memory limit and, for some kinds, a
time limit. A step that hits a limit stops its arm in that run, and the arm is not run on larger
instances (the grid runs in waves, from the smallest instance to the largest).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import FactorBounds, Graph, IntervalMDP
from scm_to_mdp.evaluation import evaluate, optimal_values, structure_distances
from scm_to_mdp.plotting import figures, mdp_graph
from scm_to_mdp.utils.limits import run_limited
from scm_to_mdp.utils.parallel import run_jobs

STOPPED = ("timeout", "memout")
DISCOVERY_ARMS = ["BIC -> RMDP", "PCMCI+ -> RMDP", "GES -> RMDP"]


# ==========================================================================================
# One dataset: every arm, step by step
# ==========================================================================================
def run_arms(
    env,
    data,
    arms: dict,
    gamma: float,
    mem_gb: float,
    timeout_s: dict,
    starts: np.ndarray,
    skip=(),
    keep_models: bool = False,
    label: str = "",
):
    """One row of results (status, time, memory and values per arm), and the models of the arms
    that build one (for the pictures) when keep_models."""
    true_parents = env.true_parents(data)
    V_star = optimal_values(env, gamma)
    row = {"V_star": float(env.start @ V_star), "V_star_K": float(V_star[starts].mean())}
    models = {}
    for arm, steps in arms.items():
        if arm in skip:
            row[f"{arm}|status"] = "skipped"
            continue
        x = data
        for kind, step in steps:
            out = run_limited(step, x, mem_gb=mem_gb, timeout_s=timeout_s.get(kind))
            row[f"{arm}|{kind}_status"] = out.status
            row[f"{arm}|{kind}_gb"] = out.peak_gb
            row[f"{arm}|{kind}_s"] = out.seconds
            if out.status != "ok":
                row[f"{arm}|status"] = out.status
                detail = f" ({out.value})" if out.status == "error" else ""
                print(f"   {label} {arm}: {kind} -> {out.status}{detail}", flush=True)
                break
            x = out.value
            if isinstance(x, Graph):
                row.update(
                    {
                        f"{arm}|{key}": value
                        for key, value in structure_distances(x, true_parents, data).items()
                    }
                )
                row.update({f"{arm}|{key}": value for key, value in x.report.items()})
            if arm == "DiFaRL" and kind == "discovery":
                row["DiFaRL|groups"] = len(x)
            if keep_models and isinstance(x, IntervalMDP):
                models[arm] = x
            if keep_models and isinstance(x, FactorBounds) and "compile" not in dict(steps):
                models[arm] = product_bounds.compile(x)  # the box of the factor sets, for the picture
        else:  # every step finished: x is a Solution
            row[f"{arm}|status"] = "ok"
            row.update(
                {
                    f"{arm}|{key}": value
                    for key, value in evaluate(env, data, x, gamma, V_star, starts).items()
                }
            )
    return row, models


# ==========================================================================================
# The grid, in waves from the smallest instance to the largest
# ==========================================================================================
def run_waves(run, sizes: list, seeds: list, workers: int, desc: str) -> list[dict]:
    """run(size, seed, skip) for every size (in order) and seed. An arm stopped by a limit in any
    seed of a size is skipped in every larger size."""
    rows, dead = [], set()
    bar = tqdm(
        total=len(sizes) * len(seeds), desc=f"{desc} ({workers} workers)", unit="run", dynamic_ncols=True
    )
    for size in sizes:
        jobs = [(size, seed, tuple(sorted(dead))) for seed in seeds]
        wave = [row for _, row in run_jobs(run, jobs, workers, bar=bar)]
        rows += wave
        stopped = {
            k.split("|")[0] for r in wave for k, v in r.items() if k.endswith("|status") and v in STOPPED
        }
        if stopped - dead:
            bar.write(f"stopped by a limit at {size}: {sorted(stopped - dead)} (skipped from now on)")
        dead |= stopped
    bar.close()
    return rows


# ==========================================================================================
# Pictures of the true MDP and of every learned model (seed 0, small problems)
# ==========================================================================================
def action_names(data) -> list[str]:
    return [
        "+".join(a for a, on in zip(data.action_vars, vec, strict=True) if on) or "none"
        for vec in data.actions
    ]


def draw_models(env, data, models: dict, folder, problem: str):
    P_true = env.P[:, env.env_actions(data)]  # [S, K, S'] over the learner's actions
    states = [env.state_label(b) for b in env.B]
    actions = action_names(data)
    truth = IntervalMDP(P_true, P_true, data.reward)
    mdp_graph.draw(P_true, truth, states, actions, f"{problem}: true MDP", folder / "ground_truth")
    for arm, model in models.items():
        note = ""
        if arm == "Schnitzer exact":
            note = "\n(box of the factor sets; the vertex set inside it is tighter)"
        elif np.array_equal(model.L, model.U):
            note = " (point estimates)"
        name = arm.replace(" -> ", "_").replace(" ", "_").replace("+", "plus")
        mdp_graph.draw(P_true, model, states, actions, f"{problem}: {arm}{note}", folder / name)


# ==========================================================================================
# Tables and figures (from the table of runs only)
# ==========================================================================================
def print_tables(res: pd.DataFrame, axis: figures.Axis, arms: list[str]):
    by, order = axis.by, axis.order

    def table(columns: dict, fmt):
        present = {name: res.groupby(by)[c].mean() for name, c in columns.items() if c in res}
        if present:
            print(pd.DataFrame(present).reindex(order).to_string(float_format=fmt))

    three = lambda x: f"{x:.3f}"  # noqa: E731
    print("\nV* at the nominal start and over the K starts (mean over seeds):")
    table({"nominal": "V_star", "K starts": "V_star_K"}, three)
    for metric, name in [
        ("regret", "Regret, nominal start"),
        ("regret_K", "Regret, K random starts"),
        ("gap", "Promise - true value, nominal start"),
        ("gap_K", "Promise - true value, K starts"),
        ("worst_gap", "Worst state: max(promise - true value)"),
    ]:
        print(f"\n{name} (mean over seeds):")
        table({a: f"{a}|{metric}" for a in arms}, three)
    print("\nValidity, coverage: share of runs with promise <= true value in every state:")
    table({a: f"{a}|safe" for a in arms}, lambda x: f"{x:.0%}")
    print("\nValidity, violation rate: share of states with promise > true value (mean over seeds):")
    table({a: f"{a}|violation_rate" for a in arms}, lambda x: f"{x:.1%}")
    print("\nValidity, worst violation: max over states of (promise - true value)+ (mean over seeds):")
    table({a: f"{a}|worst_violation" for a in arms}, three)
    for metric in ["shd", "missing", "extra", "sid"]:
        print(f"\nStructure, {metric} (mean over seeds):")
        table({a: f"{a}|{metric}" for a in DISCOVERY_ARMS}, lambda x: f"{x:.1f}")
    print("\nDiFaRL action groups (mean over seeds):")
    table({"groups": "DiFaRL|groups"}, lambda x: f"{x:.1f}")
    status = [c for c in res.columns if c.endswith("_status")]
    stopped = res.melt(id_vars=by, value_vars=status).query("value in @STOPPED")
    print("\nSteps stopped by a limit (count over seeds); the arm is skipped on larger instances:")
    print(
        "none" if stopped.empty else stopped.groupby([by, "variable", "value"], sort=False).size().to_string()
    )
    for unit, name in [("s", "Time in seconds"), ("gb", "Peak memory in GB")]:
        for step in ["discovery", "learning", "solve"]:
            print(f"\n{name}, {step} (mean over seeds):")
            table({a: f"{a}|{step}_{unit}" for a in arms}, lambda x: f"{x:.2f}")


def make_figures(
    res: pd.DataFrame, axis: figures.Axis, arms: list[str], k: int, limits: dict, delta: float, folder
):
    """performance, validity, structure, time and memory figures, as PDF and PNG in folder."""
    figures.performance(res, axis, arms, k, folder / "performance")
    figures.validity(res, axis, arms, delta, folder / "validity")
    figures.structure(res, axis, [a for a in DISCOVERY_ARMS if a in arms], folder / "structure")
    per_step = {
        "discovery": [a for a in arms if a in DISCOVERY_ARMS + ["DiFaRL"]],
        "learning": [
            a for a in arms if a not in ("DiFaRL", "Schnitzer exact")
        ],  # exact learns what IA learns
        "solve": arms,
    }
    time_limits = {
        "discovery": limits["time"].get("discovery"),
        "learning": None,
        "solve": limits["time"].get("solve"),
    }
    figures.resources(res, axis, per_step, "s", time_limits, folder / "time")
    figures.resources(res, axis, per_step, "gb", dict.fromkeys(per_step, limits["memory"]), folder / "memory")
    print(f"\nfigures: {folder}/{{performance,validity,structure,time,memory}}.{{pdf,png}}")
