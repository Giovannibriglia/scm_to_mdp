"""Tied CPD rows (a parametric MDP) against the factored interval MDP, on SysAdmin and Stock trading
(S2-S9; experiments/pmdp/tied.py, templates.py).

Settings (fixed before any run):
    sysadmin        SysAdmin, ring, N = 2..6 machines, no storm: the declared tying holds
    sysadmin_storm  the same with the hidden storm on (p = 0.1, effect 0.8, machine 0 only): the
                    observed P(m_0' = 1) differs from the other machines', so the declared tying is
                    WRONG for machine 0 (the control: does the promise break, does the tying test see it?)
    stock           Stock trading, scale 0.8, no news; layouts 1x2, 1x3, 1x4, 2x2, 2x3 (sectors x stocks)
Every setting: gamma 0.9, episodes reset every 64 steps; the uniform logger; the first 100, 500, 1000,
2000 transitions of one table (nested); 5 seeds; delta 0.05; evaluation as experiments 1-2 (the
nominal start and 20 random starts).

Methods:
    oracle -> MDP                    the true observed MDP, value iteration (V*, the target)
    true graph -> IMDP               factor intervals on the true graph, box relaxation, robust VI
    DYNOTEARS -> IMDP                the same on DYNOTEARS's graph (ours, as in experiment 1)
    true graph -> pMDP (a) / (b)     the declared tying on the true graph (reference)
    DYNOTEARS -> pMDP (a) / (b)      the declared tying on DYNOTEARS's graph
  (a): theta's box, every row its class's interval, box relaxation, robust VI (policy and promise).
  (b): the policy of (a) with a sound lower bound on its static worst case over theta's box (theta fixed
       for the run), at the nominal start only (the K-start columns of (b) are empty): the larger of
       Storm's parameter lifting (its value - precision, if it finishes within STORM_TIMEOUT_S) and a
       Python region refinement (b_source says which). Cross-checks in its row: promise_b_py (Python region
       refinement, a looser sound bound), sampled_upper (exact values at sampled theta: an upper
       estimate of the static minimum), storm_value_check (the exact value at Storm's theta).

Output, per setting: results/pmdp/<date>_<time>/<setting>/{results.csv, meta.json} in the format of
tool/sweep.py, then tool/report.py's tables and figures (as experiments 1 and 2). Usage:

    .venv/bin/python experiments/pmdp/run.py sysadmin sysadmin_storm stock
    .venv/bin/python experiments/pmdp/run.py stock --smoke         # 2 sizes, 100 / 500, 2 seeds
    .venv/bin/python experiments/pmdp/run.py --report FOLDER       # tables and figures again
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import draw  # noqa: E402
import templates  # noqa: E402
import tied  # noqa: E402

from scm_to_mdp.data import first_transitions  # noqa: E402
from scm_to_mdp.envs import StockTrading, SysAdmin  # noqa: E402
from scm_to_mdp.tool.evaluate import evaluate, reference_values  # noqa: E402
from scm_to_mdp.tool.methods import Result, learn, oracle_methods, solve, standard_methods  # noqa: E402
from scm_to_mdp.tool.report import report  # noqa: E402
from scm_to_mdp.tool.sweep import random_starts  # noqa: E402

N_TRANSITIONS = [100, 500, 1000, 2000]
SEEDS = list(range(5))
GAMMA, HORIZON, DELTA, N_STARTS = 0.9, 64, 0.05, 20
STORM_PRECISION, STORM_TIMEOUT_S = 0.01, 60.0  # Storm's absolute precision; over the limit: Python's bound
PY_SPLITS, PY_MAX_SPLIT = 3, 4  # Python region refinement: 3 parts on the 4 widest classes (81 sub-boxes)
HIDDEN, STORM_EFFECT = 0.1, 0.8  # as experiments/domains.py
# the promise of (b) is a start-state number: these columns of (a) do not apply to it
NOT_FOR_B = ["gap_K", "safe", "worst_gap", "violation_rate", "worst_violation", "cert_gap", "cert_gap_K",
             "cert_holds", "bracket_holds", "bracket_width", "opt_holds", "cert_residual", "V_opt"]  # fmt: skip


@dataclass(frozen=True)
class Setting:
    title: str
    make_env: Callable[[Any, int], Any]
    sizes: list
    size_name: str
    template: Any  # templates.SysAdminTemplate | templates.StockTemplate


def _stock(size: str, seed: int):
    sectors, stocks = (int(v) for v in size.split("x"))
    return StockTrading(sectors, stocks, seed, scale=0.8)


SETTINGS = {
    "sysadmin": Setting("SysAdmin (pMDP)", lambda n, s: SysAdmin(int(n), s, topology="ring"),
                        ["2", "3", "4", "5", "6"], "N", templates.SysAdminTemplate),
    "sysadmin_storm": Setting("SysAdmin with storm (pMDP control)",
                              lambda n, s: SysAdmin(int(n), s, topology="ring", storm=HIDDEN,
                                                    storm_effect=STORM_EFFECT),
                              ["2", "3", "4", "5", "6"], "N", templates.SysAdminTemplate),
    "stock": Setting("Stock trading (pMDP)", _stock, ["1x2", "1x3", "1x4", "2x2", "2x3"], "sectors x stocks",
                     templates.StockTemplate),
}  # fmt: skip


def base_methods() -> dict:
    std = standard_methods(delta=DELTA)
    return {
        "oracle -> MDP": oracle_methods()["oracle -> MDP"],
        "true graph -> IMDP": std["true graph -> IMDP"],
        "DYNOTEARS -> IMDP": std["DYNOTEARS -> IMDP"],
    }


def pmdp_rows(setting, truth, data, graph, name, imdp_method, starts, seed, figures=None) -> list[dict]:
    """Rows of `name -> pMDP (a)` and `(b)`: the declared tying on `graph`. figures: a folder for the
    pictures of the learned pMDP (draw.py), or None."""
    t0 = time.perf_counter()
    template = setting.template.learn(data, graph.parents)
    counts = tied.pooled_counts(data, template)
    lo, hi = tied.theta_box(counts, DELTA)
    result = Result(imdp_method, graph=graph, model=tied.factor_bounds(data, template, lo, hi))
    result = solve(result, data, GAMMA, seed=seed)
    metrics = evaluate(result, data, truth, GAMMA, starts)
    diag = {"pmdp_classes": template.n_classes, **tied.tying_test(data, template),
            **{f"theta_lo_{c}": lo[i] for i, c in enumerate(template.class_names)},
            **{f"theta_hi_{c}": hi[i] for i, c in enumerate(template.class_names)},
            **{f"theta_n_{c}": int(counts[i].sum()) for i, c in enumerate(template.class_names)}}  # fmt: skip
    a = {"method": f"{name} -> pMDP (a)", **result.row(), "seconds": time.perf_counter() - t0, **metrics, **diag}
    # (b): the same policy, the static worst case over theta's box
    pi = result.solution.pi
    if figures is not None:
        P = truth.observed_kernel()[:, truth.actions_of(data), :]
        tag = name.replace(" ", "_")
        title = f"{setting.title}: {name} -> pMDP, {len(data.df)} transitions"
        draw.draw_pmdp(data, template, lo, hi, P, f"{title}, the policy's parametric DTMC",
                       figures / f"{tag}_policy", policy=pi)  # fmt: skip
        if len(data.B) <= 4:
            draw.draw_pmdp(data, template, lo, hi, P, f"{title}, every action", figures / f"{tag}_all")
    obs = truth.observed_index()
    start = np.bincount(obs, weights=truth.start, minlength=len(data.B))  # the observed start
    t0 = time.perf_counter()
    storm = tied.storm_lower_bound(data, template, lo, hi, pi, GAMMA, int(np.argmax(start)), STORM_PRECISION,
                                   STORM_TIMEOUT_S)  # fmt: skip
    seconds_storm = time.perf_counter() - t0
    if np.count_nonzero(start) != 1:  # Storm's chain starts in one state
        storm = {**storm, "storm_status": "error", "storm_lower": np.nan}
    b_py = tied.region_refinement(data, template, lo, hi, pi, GAMMA, start, PY_SPLITS, PY_MAX_SPLIT)
    upper = tied.sampled_upper(data, template, lo, hi, pi, GAMMA, start)
    storm_ok = storm["storm_status"] == "ok"
    # both are sound lower bounds on the static worst case, so is their maximum; Storm's is the tighter
    # one when it finishes (it refines to the precision), Python's when Storm runs out of time
    promise = max(b_py, storm["storm_lower"]) if storm_ok else b_py
    source = "storm" if storm_ok and storm["storm_lower"] >= b_py else "python"
    b = {**a, "method": f"{name} -> pMDP (b)", **{k: np.nan for k in NOT_FOR_B}, **storm,
         "promise": promise, "gap": promise - metrics["true"], "promise_a": metrics["promise"],
         "promise_b_py": b_py, "sampled_upper": upper, "seconds": seconds_storm,
         "b_source": source, "status": "ok"}  # fmt: skip
    b["safe"] = bool(promise <= metrics["true"] + 1e-6) if np.isfinite(promise) else np.nan  # at the start only
    return [a, b]


def run_point(key: str, setting: Setting, size, seed: int, out: Path | None = None) -> list[dict]:
    env = setting.make_env(size, seed)
    truth = env.ground_truth()
    starts = random_starts(truth, N_STARTS, seed)
    refs = reference_values(truth, GAMMA, starts)
    table = env.sample(max(N_TRANSITIONS) // HORIZON + 2, HORIZON, "random")
    base = {"experiment": key, "knob": "", "policy": "uniform", "size": str(size),
            "size_label": f"{setting.size_name}={size}", "seed": seed, "n_states": truth.n_states,
            "n_observed_states": truth.learner_spec.n_states, **refs}  # fmt: skip
    rows = []
    for n in N_TRANSITIONS:
        data = truth.learner_spec.read(first_transitions(table, n))
        point = {**base, "n_transitions": n, "transitions": len(data.df)}
        graphs, kernel = {}, truth.observed_kernel()[:, truth.actions_of(data), :]
        for name, method in base_methods().items():
            t0 = time.perf_counter()
            parents = truth.true_parents if method.discovery == "oracle" else None
            result = solve(learn(data, method, None, parents, seed=seed, kernel=kernel), data, GAMMA, seed=seed)
            metrics = evaluate(result, data, truth, GAMMA, starts)
            if result.graph is not None:
                graphs[name] = (result.graph, method)
            graph = {"parents": json.dumps(result.graph.parents)} if result.graph is not None else {}
            rows.append({**point, "method": name, **result.row(), "seconds": time.perf_counter() - t0,
                         **metrics, **graph})  # fmt: skip
        for name in ("true graph", "DYNOTEARS"):
            graph, method = graphs[f"{name} -> IMDP"]
            draw_here = out is not None and seed == 0 and str(size) == setting.sizes[0] and n == max(N_TRANSITIONS)
            figures = out / "figures" / "pmdp_graph" if draw_here else None
            for row in pmdp_rows(setting, truth, data, graph, name, method, starts, seed, figures):
                rows.append({**point, **row})
        print(f"{key} size={size} seed={seed} n={n} done", flush=True)
    return rows


def run(keys: list[str], smoke: bool, out_root: Path) -> None:
    global N_TRANSITIONS, SEEDS
    if smoke:
        N_TRANSITIONS, SEEDS = [100, 500], [0, 1]
    for key in keys:
        setting = SETTINGS[key]
        sizes = setting.sizes[:2] if smoke else setting.sizes
        out = out_root / key
        out.mkdir(parents=True, exist_ok=True)
        meta = {"sizes": sizes, "size_labels": [f"{setting.size_name}={s}" for s in sizes],
                "size_name": setting.size_name, "seeds": SEEDS, "n_transitions": N_TRANSITIONS,
                "config": {"name": setting.title, "limits": None, "n_starts": N_STARTS, "gamma": GAMMA,
                           "delta": DELTA, "storm_precision": STORM_PRECISION}}  # fmt: skip
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        rows = []
        for size in sizes:
            for seed in SEEDS:
                rows += run_point(key, setting, size, seed, out)
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        make_report(out)


def make_report(out: Path) -> None:
    report(out, target=0.95, delta=DELTA, show=False)
    summary(out)


def summary(out: Path) -> None:
    """summary.md: promise and true value (normalised: 0 random, 1 V*_obs), how often the promise held,
    and the pMDP's own checks (the order a <= b_py <= b <= sampled_upper, Storm's check, the tying test)."""
    df = pd.read_csv(out / "results.csv")
    span = df["V_star_obs"] - df["V_random"]
    df["promise_norm"] = (df["promise"] - df["V_random"]) / span
    df["true_norm"] = (df["true"] - df["V_random"]) / span
    df["held"] = df["promise"] <= df["true"] + 1e-6
    lines = [f"# {out.name}: means over seeds", "", "Normalised values: 0 = random policy, 1 = V*_obs. held: share "
             "of runs with promise <= true at the start.", ""]  # fmt: skip
    for col in ("promise_norm", "true_norm", "held"):
        t = df.pivot_table(index=["size", "n_transitions"], columns="method", values=col, aggfunc="mean")
        lines += [f"## {col}", "", "```", t.round(3).to_string(), "```", ""]
    b = df[df["method"].str.endswith("pMDP (b)")].copy()
    if len(b):
        b["order_ok"] = ((b["promise_a"] <= b["promise_b_py"] + 1e-6) & (~(b["storm_status"] == "ok") | (b["promise_b_py"] <= b["storm_value"] + 1e-6))
                         & (b["storm_lower"] <= b["sampled_upper"] + 1e-6))  # fmt: skip
        b["check_err"] = (b["storm_value"] - b["storm_value_check"]).abs()
        b["tying_rejected"] = b["tying_p"] < 0.05
        t = b.groupby(["method", "size"]).agg(order_ok=("order_ok", "mean"), check_err=("check_err", "max"),
                                              storm_ok=("storm_status", lambda s: (s == "ok").mean()),
                                              from_storm=("b_source", lambda s: (s == "storm").mean()),
                                              storm_s=("seconds", "mean"), tying_rejected=("tying_rejected", "mean"),
                                              classes=("pmdp_classes", "mean"))  # fmt: skip
        lines += ["## pMDP (b): checks", "", "```", t.round(4).to_string(), "```", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("settings", nargs="*", default=list(SETTINGS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None, help="only the tables and figures of a folder")
    args = parser.parse_args()
    if args.report is not None:
        make_report(args.report)
        return
    stamp = ("smoke_" if args.smoke else "") + datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run(args.settings, args.smoke, args.out or ROOT / "results" / "pmdp" / stamp)


if __name__ == "__main__":
    main()
