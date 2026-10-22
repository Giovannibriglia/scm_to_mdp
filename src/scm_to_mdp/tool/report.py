"""Part 3 of the tool: tables and figures of a sweep, from its results.csv and meta.json only.

    report(out_dir)

writes into out_dir:
    tables.md                                    every table, for every number of rollouts
    figures/rollouts=<n>/{performance,validity,structure,time,memory}.{pdf,png}
                                                 the figures over the problem size, one folder per n
    figures/sample_efficiency.{pdf,png}          rollouts needed for a 95%-optimal policy
and prints the main tables. Baselines, oracle methods and ours are told apart by the role column
(set by the Method: flat_rmdp, flat_mdp and difarl are the baselines).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scm_to_mdp.plotting import figures
from scm_to_mdp.plotting.style import method_styles
from scm_to_mdp.tool.sweep import read_results

ROLE_ORDER = {"ours": 0, "oracle": 1, "baseline": 2}


def _methods(res: pd.DataFrame) -> list[str]:
    """Methods in the order: ours, oracle, baselines (each in order of appearance)."""
    first = res.drop_duplicates("method")
    return sorted(first["method"], key=lambda m: ROLE_ORDER.get(first.set_index("method").at[m, "role"], 3))


def _table(res: pd.DataFrame, sizes, labels, size_name, methods, column: str, fmt) -> str:
    sub = res[res["method"].isin(methods)]
    values = (
        pd.to_numeric(sub[column], errors="coerce") if column != "safe" else figures._as_float(sub["safe"])
    )
    table = values.groupby([sub["size"], sub["method"]]).mean().unstack("method")
    table = table.reindex(
        index=[s for s in sizes if s in set(sub["size"])], columns=[m for m in methods if m in table]
    )
    if table.empty or table.isna().all().all():
        return ""
    table.index = pd.Index([labels[s] for s in table.index], name=size_name)
    return table.to_string(float_format=fmt, na_rep="—")


def report(out_dir, target: float = 0.95, delta: float | None = 0.05, show: bool = True) -> dict:
    """Tables and figures of the sweep in out_dir. target: the share of V*_obs (the best policy over
    the observed state) that counts as "optimal"; delta: the failure probability of the promises
    (the 1 - delta line of the validity panel).
    Returns {"rollouts_needed": table of n* (median and quartiles over seeds)}."""
    out = Path(out_dir)
    res = read_results(out, current=True)
    if res.empty:
        raise ValueError(f"no results in {out}")
    meta = json.loads((out / "meta.json").read_text())
    sizes = [s for s in meta["sizes"] if s in set(res["size"])]
    labels = {
        s: s for s in meta["sizes"]
    }  # ticks and table rows: the sizes themselves (2, 3, ... or small, ...)
    first_label = meta["size_labels"][0]  # an older meta.json has no size_name: read it from "N=2"
    size_name = meta.get("size_name") or (first_label.split("=")[0] if "=" in first_label else "size")
    methods = _methods(res)
    first = res.drop_duplicates("method").set_index("method")
    styles = method_styles({m: (first.at[m, "role"], bool(first.at[m, "guarantee"])) for m in methods})
    roles = {m: first.at[m, "role"] for m in methods}
    limits = meta["config"]["limits"] or {"mem_gb": None, "timeout_s": {}}
    k = meta["config"]["n_starts"]
    name = meta["config"]["name"]
    fig_dir = out / "figures"
    lines = [f"# {name}: results", "", "methods: " + ", ".join(f"{m} ({roles[m]})" for m in methods), ""]
    shown = []  # what is printed: the main tables at the largest number of rollouts, and n*

    # ---- figures and tables over the size, one set per number of rollouts --------------------
    for n in meta["n_rollouts"]:
        at_n = res[res["n_rollouts"] == n]
        if at_n.empty:
            continue
        axis = figures.Axis(at_n, sizes, [labels[s] for s in sizes], size_name)
        title = f"{name}, {n} rollouts"
        folder = fig_dir / f"rollouts={n}"
        figures.performance(at_n, axis, methods, styles, k, title, folder / "performance")
        figures.validity(at_n, axis, methods, styles, delta, title, folder / "validity")
        figures.structure(at_n, axis, methods, styles, title, folder / "structure")
        timeouts = limits["timeout_s"] or {}
        time_limits = {
            "discovery": timeouts.get("discovery"),
            "learning": timeouts.get("learning"),
            "solve": timeouts.get("solve"),
        }
        figures.resources(at_n, axis, methods, styles, "s", time_limits, title, folder / "time")
        mem = dict.fromkeys(figures.STEP_TITLES, limits["mem_gb"])
        figures.resources(at_n, axis, methods, styles, "gb", mem, title, folder / "memory")
        lines += [f"## {n} rollouts", ""]
        for column, caption, fmt in [
            (
                "optimal_share",
                "True value / V* over the observed state (nominal start)",
                lambda x: f"{x:.1%}",
            ),
            ("regret", "Regret V*_obs - true value, nominal start", lambda x: f"{x:.3f}"),
            ("norm_regret", "Normalized regret (1 - gamma)(V*_obs - true value)", lambda x: f"{x:.4f}"),
            ("gap", "Promise - true value, nominal start (<= 0: the promise holds)", lambda x: f"{x:.3f}"),
            ("regret_K", f"Regret, {k} random starts", lambda x: f"{x:.3f}"),
            (
                "safe",
                "Coverage: share of seeds with promise <= true value in every state",
                lambda x: f"{x:.0%}",
            ),
            ("violation_rate", "Share of states with promise > true value", lambda x: f"{x:.1%}"),
            ("shd", "SHD of the selected graph", lambda x: f"{x:.1f}"),
            ("sid", "SID of the selected graph", lambda x: f"{x:.1f}"),
            ("discovery_s", "Seconds, discovery", lambda x: f"{x:.2f}"),
            ("learning_s", "Seconds, parameter learning", lambda x: f"{x:.2f}"),
            ("solve_s", "Seconds, solving", lambda x: f"{x:.2f}"),
        ]:
            text = _table(at_n, sizes, labels, size_name, methods, column, fmt)
            if text:
                lines += [f"{caption} (mean over seeds):", "```", text, "```", ""]
                if n == max(meta["n_rollouts"]) and column in ("optimal_share", "safe"):
                    shown += [f"{caption}, {n} rollouts (mean over seeds):", text, ""]
        stopped = at_n[~at_n["status"].isin(["ok"])]
        if len(stopped):
            counts = stopped.groupby(["size_label", "method", "failed_step", "status"], dropna=False).size()
            lines += ["Runs that did not finish (count over seeds):", "```", counts.to_string(), "```", ""]

    # ---- rollouts needed for a policy at least `target` optimal ------------------------------
    summary = figures.sample_efficiency(
        res,
        sizes,
        [labels[s] for s in sizes],
        size_name,
        methods,
        styles,
        target,
        f"{name}: rollouts for a {target:.0%}-optimal policy",
        fig_dir / "sample_efficiency",
    )
    if len(summary):

        def cell(r):
            if not np.isfinite(r["median"]):
                return f"> {max(meta['n_rollouts'])}"
            hi = "inf" if not np.isfinite(r["q75"]) else f"{r['q75']:.0f}"
            return f"{r['median']:.0f} [{r['q25']:.0f}, {hi}]"

        table = summary.assign(cell=summary.apply(cell, axis=1)).pivot(
            index="size", columns="method", values="cell"
        )
        table = table.reindex(
            index=[s for s in sizes if s in table.index], columns=[m for m in methods if m in table]
        )
        table.index = pd.Index([labels[s] for s in table.index], name=size_name)
        caption = (
            f"Rollouts for a {target:.0%}-optimal policy: median over seeds [quartiles]; '>' = not reached"
        )
        lines += [f"## {caption}", "", "```", table.to_string(), "```", ""]
        shown += [caption + ":", table.to_string(), ""]
    hard = res.drop_duplicates(["size"]).set_index("size")
    share = (hard["V_random"] / figures.target_value(hard)).reindex(sizes)
    hard_text = "\n".join(f"{size_name} = {labels[s]:<8} {v:.1%}" for s, v in share.items())
    caption = "How hard is it? Value of the random policy / V* over the observed state (nominal start)"
    lines += [caption + ":", "```", hard_text, "```", ""]
    shown += [caption + ":", hard_text, ""]
    (out / "tables.md").write_text("\n".join(lines))
    if show:
        print("\n".join(shown))
        print(f"\ntables: {out / 'tables.md'}\nfigures: {fig_dir}/")
    return {"rollouts_needed": summary}
