"""Reward transfer: the report of results_transfer.csv (tool/sweep.py, reward_transfer=True).

Every method learned once from data logged under the original reward is re-planned under each of the
environment's alternative rewards (Env.alternative_rewards; methods.replan: model-based methods
re-solve their learned model, model-free ones keep their policy and have no promise) and evaluated
exactly on the same true transitions with that reward. For one setting this writes

    figures/transfer/<task>/transitions=<n>/   performance, performance_normalized, validity
    figures/transfer/average/transitions=<n>/  performance_normalized and validity averaged over the
                                            tasks (normalised per task first), and transfer_gap: the
                                            average over the tasks minus the original reward
and a "Reward transfer" section of tables.md (per task and averaged, at every number of transitions).
"""

from pathlib import Path

import pandas as pd

from scm_to_mdp.plotting import figures

KEYS = ["size", "seed", "n_transitions", "method"]
AVERAGED = ["norm_value", "norm_value_K", "oracle_share", "oracle_share_K", "safe", "violation_rate"]
AVERAGED += ["worst_violation"]


def average(transfer: pd.DataFrame) -> pd.DataFrame:
    """One row per (size, seed, n, method): every task normalised first, then the mean over the tasks
    (a task where the method did not finish is left out of its mean)."""
    rows = figures.normalised(
        transfer[transfer["status"] == "ok"].drop(columns=["norm_value"], errors="ignore")
    )
    rows = rows.assign(safe=figures._as_float(rows["safe"]))
    for c in ("violation_rate", "worst_violation"):
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    first = transfer.drop_duplicates("method").set_index("method")[["role", "guarantee"]]
    mean = rows.groupby(KEYS)[[c for c in AVERAGED if c in rows]].mean().reset_index()
    return mean.join(first, on="method")


def gaps(main: pd.DataFrame, averaged: pd.DataFrame) -> pd.DataFrame:
    """The averaged normalised value minus the original reward's, per (size, seed, n, method)."""
    original = figures.normalised(main[main["status"] == "ok"])
    cols = [c for c in ("norm_value", "norm_value_K") if c in original]
    both = averaged.merge(original[KEYS + cols], on=KEYS, suffixes=("", "_original"))
    out = both[KEYS + ["role", "guarantee"]].copy()
    for c in cols:
        out[f"gap_{c.removeprefix('norm_')}"] = both[c] - both[f"{c}_original"]
    return out


def report_transfer(
    out: Path, main: pd.DataFrame, transfer: pd.DataFrame, meta: dict, sizes, labels, size_name, methods,
    styles, k: int, target: float, delta, table, kind: str = "reward",
) -> list[str]:  # fmt: skip
    """Figures of every task and of the average; returns the tables.md lines. kind "reward": the
    alternative rewards (results_transfer.csv, figures/transfer/); "shift": the shifted dynamics
    (results_shift.csv, figures/shift/; the policies as they are, no re-planning)."""
    if transfer.empty:
        return []
    name = meta["config"]["name"]
    tasks = list(dict.fromkeys(transfer["task"]))
    shift = kind == "shift"
    fig_dir = out / "figures" / ("shift" if shift else "transfer")
    what, word = ("dynamics", "shift") if shift else ("rewards", "reward")
    heading, note = (
        ("Dynamics shift", "policies deployed as they are, no re-learning or re-planning")
        if shift else ("Reward transfer", "models re-planned, not re-learned")
    )  # fmt: skip
    lines = [f"# {heading}", "", f"tasks: {', '.join(tasks)} ({note})", ""]
    for n in meta["n_transitions"]:
        at_n, main_n = transfer[transfer["n_transitions"] == n], main[main["n_transitions"] == n]
        if at_n.empty:
            continue
        axis = figures.Axis(at_n, sizes, [labels[s] for s in sizes], size_name)
        budget = f"{figures.samples_label(n)} transitions"  # e.g. "1k transitions"
        for task in tasks:
            rows = at_n[at_n["task"] == task]
            folder, title = fig_dir / task / f"transitions={n}", f"{name}, {word} {task}, {budget}"
            figures.performance(rows, axis, methods, styles, k, title, folder / "performance")
            figures.performance_normalized(
                rows, axis, methods, styles, target, k, title, folder / "performance_normalized"
            )
            figures.validity(rows, axis, methods, styles, delta, title, folder / "validity")
            text = table(figures.normalised(rows), "norm_value", lambda x: f"{x:.2f}")
            if text:
                lines += [
                    f"## {task}, {budget}: normalised value (0 random, 1 optimal), nominal start",
                    "```",
                    text,
                    "```",
                    "",
                ]
        averaged = average(at_n)
        folder, title = (
            fig_dir / "average" / f"transitions={n}",
            f"{name}, mean over {len(tasks)} {what}, {budget}",
        )
        figures.performance_normalized(
            averaged, axis, methods, styles, target, k, title, folder / "performance_normalized"
        )
        figures.validity(averaged, axis, methods, styles, delta, title, folder / "validity")
        gap = gaps(main_n, averaged)
        ylabel = f"other {what} − original"
        figures.transfer_gap(gap, axis, methods, styles, title, folder / "transfer_gap", ylabel)
        for column, caption, fmt in [
            ("norm_value", f"normalised value, mean over the {what}", lambda x: f"{x:.2f}"),
            ("safe", f"promise held, mean over the {what}", lambda x: f"{x:.0%}"),
        ]:
            text = table(averaged, column, fmt)
            if text:
                lines += [f"## average, {budget}: {caption}", "```", text, "```", ""]
        text = table(gap, "gap_value", lambda x: f"{x:+.2f}")
        if text:
            lines += [
                f"## average, {budget}: transfer gap (mean over the {what} - original)",
                "```",
                text,
                "```",
                "",
            ]
    return lines
