"""Part 3 of the tool: tables and figures of a sweep, from its results.csv and meta.json only.

    report(out_dir)

writes into out_dir:
    tables.md                                    every table, for every number of transitions
    figures/transitions=<n>/{performance,validity,structure,time,memory}.{pdf,png}
                                                 the figures over the problem size (x: observed
                                                 states), one folder per transition budget n
    figures/sample_efficiency.{pdf,png}          transitions needed for a 95%-optimal policy
n_transitions in results.csv is the dataset's budget: its first n logged transitions.
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

COVERAGE_TARGET = 0.99  # the coverage figures: share of the true probabilities inside the intervals
COVERAGE_EPS = 0.2  # and the "small sets" test: mean interval width <= 0.2
ROLE_ORDER = {"ours": 0, "oracle": 1, "reference": 1, "control": 2, "baseline": 3}


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
    Returns {"transitions_needed": table of n* (median over seeds and its 95% CI)}."""
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
    shown = []  # what is printed: the main tables at the largest number of transitions, and n*

    # ---- figures and tables over the size, one set per number of transitions --------------------
    for n in meta["n_transitions"]:
        at_n = res[res["n_transitions"] == n]
        if at_n.empty:
            continue
        axis = figures.Axis(at_n, sizes, [labels[s] for s in sizes], size_name)
        budget = f"{figures.samples_label(n)} transitions"  # e.g. "1k transitions"
        title = f"{name}, {budget}"
        folder = fig_dir / f"transitions={n}"
        figures.performance(at_n, axis, methods, styles, k, title, folder / "performance")
        normalized = folder / "performance_normalized"
        figures.performance_normalized(at_n, axis, methods, styles, target, k, title, normalized)
        figures.validity(at_n, axis, methods, styles, delta, title, folder / "validity")
        figures.structure(at_n, axis, methods, styles, title, folder / "structure")
        if delta is not None:  # the headline of exp3: value among the methods whose promise holds
            figures.gated_comparison(at_n, axis, methods, styles, delta, title, folder / "gated_comparison")
        figures.decision_edges(at_n, axis, methods, styles, title, folder / "decision_edges")
        figures.certificate(at_n, axis, methods, styles, title, folder / "certificate")
        if "s4_width" in at_n:
            stem = folder / "parameter_learning"
            figures.parameter_learning(at_n, axis, methods, styles, delta, title, stem)
        timeouts = limits["timeout_s"] or {}
        time_limits = {
            "discovery": timeouts.get("discovery"),
            "learning": timeouts.get("learning"),
            "solve": timeouts.get("solve"),
        }
        if "solve_limit_s" in at_n and pd.to_numeric(at_n["solve_limit_s"], errors="coerce").notna().any():
            per_size = pd.to_numeric(at_n["solve_limit_s"], errors="coerce").groupby(at_n["size"]).mean()
            per_size = per_size.reindex(axis.order).fillna(timeouts.get("solve") or np.nan)
            time_limits["solve"] = per_size.to_numpy(float)  # relative: one limit per size
        figures.resources(at_n, axis, methods, styles, "s", time_limits, title, folder / "time", history=res)
        mem = dict.fromkeys(figures.STEP_TITLES, limits["mem_gb"])
        figures.resources(at_n, axis, methods, styles, "gb", mem, title, folder / "memory", history=res)
        lines += [f"## {budget}", ""]
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
                if n == max(meta["n_transitions"]) and column in ("optimal_share", "safe"):
                    shown += [f"{caption}, {budget} (mean over seeds):", text, ""]
        if delta is not None and "safe" in at_n:
            gate = figures.normalised(at_n)
            valid = figures.validity_gate(gate, delta)
            for column, caption, fmt, rows in [
                ("norm_value", f"Validity-gated true value (normalised; '—' = promise held in < "
                 f"{1 - delta:.0%} of seeds)", lambda x: f"{x:.2f}",
                 gate.assign(norm_value=gate["norm_value"].where(valid))),
                ("norm_promise", "Validity-gated guaranteed value (normalised promise)", lambda x: f"{x:.2f}",
                 gate.assign(norm_promise=gate["norm_promise"].where(valid))),
                ("norm_cert_gap", "Certificate (V_opt - V_rob) / (V*_obs - V_rand), robust methods",
                 lambda x: f"{x:.2f}", gate),
                ("cert_holds", "Certificate holds: V*_obs - V^pi <= V_opt - V_rob in every state",
                 lambda x: f"{x:.0%}", gate),
                ("action_recall", "Share of the true action -> state edges kept", lambda x: f"{x:.0%}", gate),
            ]:  # fmt: skip
                if column not in rows:
                    continue
                if column == "cert_holds":
                    rows = rows.assign(cert_holds=figures._as_float(rows["cert_holds"]))
                text = _table(rows, sizes, labels, size_name, methods, column, fmt)
                if text:
                    lines += [f"{caption} (mean over seeds):", "```", text, "```", ""]
        stopped = at_n[~at_n["status"].isin(["ok"])]
        if len(stopped):
            counts = stopped.groupby(["size_label", "method", "failed_step", "status"], dropna=False).size()
            lines += ["Runs that did not finish (count over seeds):", "```", counts.to_string(), "```", ""]

    # ---- transitions needed for a policy at least `target` optimal ------------------------------
    summary = figures.sample_efficiency(
        res,
        sizes,
        [labels[s] for s in sizes],
        size_name,
        methods,
        styles,
        target,
        f"{name}: transitions for a {target:.0%}-optimal policy",
        fig_dir / "sample_efficiency",
    )
    largest = figures.samples_label(figures.sample_grid(res)[-1])

    def cell(r):
        if not np.isfinite(r["median_t"]):
            return f"> {largest}"
        hi = "inf" if not np.isfinite(r["ci_hi_t"]) else figures.samples_label(r["ci_hi_t"])
        return f"{figures.samples_label(r['median_t'])} [{figures.samples_label(r['ci_lo_t'])}, {hi}]"

    if len(summary):
        table = summary.assign(cell=summary.apply(cell, axis=1)).pivot(
            index="size", columns="method", values="cell"
        )
        table = table.reindex(
            index=[s for s in sizes if s in table.index], columns=[m for m in methods if m in table]
        )
        table.index = pd.Index([labels[s] for s in table.index], name=size_name)
        caption = (
            f"Transitions for a {target:.0%}-optimal policy: median over seeds [95% CI]; '>' = not reached"
        )
        lines += [f"## {caption}", "", "```", table.to_string(), "```", ""]
        shown += [caption + ":", table.to_string(), ""]

    # ---- transitions needed for valid (and small) probability sets (S4), and for a valid set that
    # certifies a near-optimal policy (S4 -> S8) -----------------------------------------------------
    def needed_table(summary):
        table = summary.assign(cell=summary.apply(cell, axis=1)).pivot(
            index="size", columns="method", values="cell"
        )
        table = table.reindex(
            index=[s for s in sizes if s in table.index], columns=[m for m in methods if m in table]
        )
        table.index = pd.Index([labels[s] for s in table.index], name=size_name)
        return table.to_string()

    panels = []
    for column, stem in (("s4_coverage_seen", "coverage_seen"), ("s4_coverage", "coverage")):
        title = f"{name}: transitions for valid probability sets"
        sets = figures.coverage_efficiency(
            res, sizes, [labels[s] for s in sizes], size_name, methods, styles, column, title,
            fig_dir / stem, target=COVERAGE_TARGET, eps=COVERAGE_EPS,
        )  # fmt: skip
        valid = f"{column} >= {COVERAGE_TARGET:.0%}"
        panels += [(sets, (("valid", valid), ("tight", f"{valid} and s4_width <= {COVERAGE_EPS:g}")))]
    args = (res, sizes, [labels[s] for s in sizes], size_name, methods, styles)
    sets = figures.guarantee_efficiency(*args, f"{name}: transitions for a guarantee", fig_dir / "guarantee")
    half = figures.guarantee_efficiency(*args, "", None, useful=0.5)
    cells = figures.regret_map(*args, f"{name}: normalised regret over size and data", fig_dir / "regret_map")
    if len(cells):
        frontier = cells.drop_duplicates(["size", "method"])[["size", "method", "n_reached"]]
        frontier = frontier.assign(cell=frontier["n_reached"].map(
            lambda v: figures.samples_label(v) if np.isfinite(v) else f"> {largest}"))  # fmt: skip
        table = frontier.pivot(index="size", columns="method", values="cell")
        table = table.reindex(
            index=[s for s in sizes if s in table.index], columns=[m for m in methods if m in table]
        )
        table.index = pd.Index([labels[s] for s in table.index], name=size_name)
        caption = (
            "Transitions until the normalised regret stays <= 0.05 (median over seeds); '>' = not reached"
        )
        lines += [f"## {caption}", "", "```", table.to_string(), "```", ""]
    title = f"{name}: what more data buys (normalised regret)"
    _, rates = figures.regret_gain(*args, title, fig_dir / "regret_gain")
    # one figure per size over the transitions: quality, valid sets, missed edges (figures/zoom_all_methods/)
    figures.zoom_by_size(
        res, sizes, [labels[s] for s in sizes], methods, styles, target, name, fig_dir / figures.ZOOM_FOLDER
    )
    if len(rates):
        rate_cells = [f"{r:.2f} ({n})" for r, n in rates[["rate", "seeds"]].values]
        table = rates.assign(cell=rate_cells).pivot(index="size", columns="method", values="cell")
        table = table.reindex(
            index=[s for s in sizes if s in table.index], columns=[m for m in methods if m in table]
        )
        table.index = pd.Index([labels[s] for s in table.index], name=size_name)
        caption = "Learning rate b of the normalised regret ~ n^-b, before it is <= 0.05"
        caption += ": median over seeds (seeds fitted)"
        lines += [f"## {caption}", "", "```", table.to_string(), "```", ""]
    panels += [
        (sets, (("valid", "the promise holds in every state (safe)"),
                ("tight", "it holds and V_rob >= V_random (normalised promise >= 0)"),
                ("contain", "the set contains the true MDP (contain_all)"))),
        (half, (("tight", "it holds and the normalised promise >= 0.5"),)),
    ]  # fmt: skip
    for sets_, parts in panels:
        for key, what in parts:
            if not sets_[key].empty:
                caption = f"Transitions until {what}: median over seeds [95% CI]; '>' = not reached"
                lines += [f"## {caption}", "", "```", needed_table(sets_[key]), "```", ""]
    hard = res.drop_duplicates(["size"]).set_index("size")
    share = (hard["V_random"] / figures.target_value(hard)).reindex(sizes)
    hard_text = "\n".join(f"{size_name} = {labels[s]:<8} {v:.1%}" for s, v in share.items())
    caption = "How hard is it? Value of the random policy / V* over the observed state (nominal start)"
    lines += [caption + ":", "```", hard_text, "```", ""]
    shown += [caption + ":", hard_text, ""]
    # ---- reward transfer: the same models under the environment's other rewards ---------------
    transfer = read_results(out, current=True, name="results_transfer.csv")
    if len(transfer):
        from scm_to_mdp.tool.transfer import report_transfer

        def table(rows, column, fmt):
            return _table(rows, sizes, labels, size_name, methods, column, fmt)

        lines += report_transfer(
            out, res, transfer, meta, sizes, labels, size_name, methods, styles, k, target, delta, table
        )
        shown += [f"reward transfer: {fig_dir / 'transfer'}/", ""]
    # ---- dynamics shift: the same policies where the hidden disturbance is more frequent -----------
    shifted = read_results(out, current=True, name="results_shift.csv")
    if len(shifted):
        from scm_to_mdp.tool.transfer import report_transfer

        def table(rows, column, fmt):
            return _table(rows, sizes, labels, size_name, methods, column, fmt)

        lines += report_transfer(
            out, res, shifted, meta, sizes, labels, size_name, methods, styles, k, target, delta, table,
            kind="shift",
        )  # fmt: skip
        shown += [f"dynamics shift: {fig_dir / 'shift'}/", ""]
    (out / "tables.md").write_text("\n".join(lines))
    if show:
        print("\n".join(shown))
        print(f"\ntables: {out / 'tables.md'}\nfigures: {fig_dir}/")
    return {"transitions_needed": summary}
