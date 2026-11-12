"""Figures of the ablation "the methods along a ladder of rewards" (run.py).

Run from the repository root, after (or during) run.py:
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py               # the latest run
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --machines 3
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --run <run folder>

Input:  results/clear27/ablations/ablation_rewards/sysadmin_<date>_<time>/ (run.py)
Output: outputs/clear27/ablations/ablation_rewards/N=<N>/<figure>.{pdf,png}      one set per size
        outputs/clear27/ablations/ablation_rewards/summary/<figure>.{pdf,png}   across the sizes
        outputs/clear27/ablations/ablation_rewards/data/<where>_<figure>.csv    the numbers drawn

Every figure has two rows of panels, with PCMCI+ and DYNOTEARS in both:
    structure learners   DYNOTEARS, PCMCI+, Strehl, Met-Rmax
    assumed structure    DYNOTEARS, PCMCI+, the full graph, the flat model

Figures, per size (x: the level of the ladder, 1 = the original reward ... 6 = traps; one column per
number of transitions):
  value_by_level     the normalised value of the policy
  promise_by_level   the normalised promise (the certified lower bound)
  held_by_level      the share of runs whose promise held (promise <= true value)
  solved_by_level    the share of runs with a normalised value of at least SOLVED
Across the sizes (summary/):
  value_by_size_n=<n>   the value against N, one column per level, at the two largest data sizes
  solved_by_level       as above, the sizes pooled
  time                  seconds to learn one model, and to re-plan it under one reward, against N

A point is the mean over the seeds of the mean over the rewards of that level; the band is one standard
error over the seeds. A run that stopped (time or memory) counts as not solved and has no value.
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
NAME = "ablation_rewards"
RESULTS = ROOT / "results" / "clear27" / "ablations" / NAME
OUTPUTS = ROOT / "outputs" / "clear27" / "ablations" / NAME
STAMP = "%Y-%m-%d_%H-%M-%S"
SOLVED = 0.95  # "solved": at least this normalised value


def _paper():
    """clear27's plot.py: the style and colours of the paper figures."""
    spec = importlib.util.spec_from_file_location("clear27_plot", HERE.parents[2] / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _paper()
PANELS = {
    "structure learners": ["DYNOTEARS", "PCMCI+", "Strehl", "Met-Rmax"],
    "assumed structure": ["DYNOTEARS", "PCMCI+", "full graph", "flat"],
}
STYLE = {  # method -> (colour, marker), the paper's
    "DYNOTEARS": (paper.BLUE, "D"),
    "PCMCI+": (paper.RED, "s"),
    "Strehl": (paper.GREEN, "^"),
    "Met-Rmax": (paper.PURPLE, "v"),
    "full graph": (paper.TEAL, "^"),
    "flat": (paper.PURPLE, "X"),
    "true graph": (paper.GOLD, "o"),  # the reference of run.py: in the time figure, in no panel
}
LEVELS = {1: "additive", 2: "subset", 3: "cost", 4: "threshold", 5: "goal", 6: "trap"}
# quantity -> (column, y label, limits, in percent)
QUANTITIES = {
    "value": ("value", "normalised value", (-1.3, 1.1), False),
    "promise": ("promise", "normalised promise", (-3.2, 1.1), False),
    "held": ("held", "runs with\npromise held (%)", (-4, 104), True),
    "solved": ("solved", f"runs with value\n≥ {SOLVED} (%)", (-4, 104), True),
}


# ---- loading --------------------------------------------------------------------------------------
def latest_run(results: Path) -> Path | None:
    found = []
    for path in results.glob("sysadmin_*") if results.is_dir() else []:
        try:
            datetime.strptime(path.name[len("sysadmin_") :], STAMP)
        except ValueError:
            continue
        found.append(path)
    return max(found) if found else None


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, timing): every finished (size, seed) of the run. runs gets `ok`, `held` as 0/1 and
    `solved` (a stopped run is not solved)."""
    files = [p.with_suffix(".csv") for p in sorted(run.glob("N=*/runs/*.json"))]
    runs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    runs["ok"] = runs["status"].eq("ok")
    runs["held"] = runs["held"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    runs["solved"] = (runs["ok"] & (runs["value"] >= SOLVED)).astype(float)
    timing = pd.DataFrame([json.loads(p.read_text()) for p in sorted(run.glob("N=*/runs/*.json"))])
    return runs, timing


def by_seed(runs: pd.DataFrame, column: str, by: list[str]) -> pd.DataFrame:
    """Mean and standard error over the seeds of the per-seed mean of `column`, per group `by`."""
    per_seed = runs.groupby([*by, "seed"])[column].mean().reset_index()
    out = per_seed.groupby(by)[column].agg(["mean", "std", "count"]).reset_index()
    out["se"] = (out["std"] / np.sqrt(out["count"])).fillna(0)
    return out


# ---- helpers --------------------------------------------------------------------------------------
def save(out: Path, where: str, fig, name: str, data: pd.DataFrame) -> None:
    folder = out / where
    folder.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    data.to_csv(out / "data" / f"{where}_{name}.csv", index=False)
    print(f"  wrote {where}/{name}")


def k(n: int) -> str:
    return f"{n:,}"


def handles_of(methods: list[str]) -> list[Line2D]:
    return [Line2D([], [], **paper.paper_line(*STYLE[m], m, ms=5)) for m in methods]


def draw_lines(ax, table: pd.DataFrame, x: str, positions: dict, methods: list[str], percent: bool) -> None:
    """One line per method: mean over the seeds, band of one standard error."""
    scale = 100.0 if percent else 1.0
    for method in methods:
        t = table[table["method"] == method].sort_values(x)
        if t.empty:
            continue
        color, marker = STYLE[method]
        at = t[x].map(positions)
        low, high = scale * (t["mean"] - t["se"]), scale * (t["mean"] + t["se"])
        ax.fill_between(at, low, high, color=color, alpha=0.15,
                        lw=0)  # fmt: skip
        ax.plot(at, scale * t["mean"], **paper.paper_line(color, marker))


def panel_legends(fig, axes) -> None:
    """Each row of panels has its own methods: a legend at the right of the row."""
    for row, (title, methods) in enumerate(PANELS.items()):
        axes[row, -1].legend(handles=handles_of(methods), loc="center left", bbox_to_anchor=(1.03, 0.5),
                             title=title, title_fontsize=8.5, fontsize=8.5, handlelength=2.0)  # fmt: skip


# ---- figures --------------------------------------------------------------------------------------
def by_level(runs: pd.DataFrame, quantity: str, where: str, out: Path) -> None:
    """Rows: the two panels; columns: the numbers of transitions; x: the level of the ladder."""
    column, ylabel, limits, percent = QUANTITIES[quantity]
    ns = sorted(runs["n_transitions"].unique())
    levels = sorted(runs["level"].unique())
    positions = {level: i for i, level in enumerate(levels)}
    table = by_seed(runs, column, ["n_transitions", "method", "level"])
    fig, axes = plt.subplots(len(PANELS), len(ns), figsize=(2.3 * max(len(ns), 2) + 1.6, 4.6), sharex=True,
                             sharey=True, squeeze=False)  # fmt: skip
    for row, methods in enumerate(PANELS.values()):
        for ax, n in zip(axes[row], ns, strict=True):
            draw_lines(ax, table[table["n_transitions"] == n], "level", positions, methods, percent)
            if not percent:
                ax.axhline(0, **paper.ZERO_LINE)
            ax.set_ylim(*limits)
            ax.set_xlim(-0.4, len(levels) - 0.6)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([LEVELS[level] for level in levels], rotation=40, ha="right", fontsize=8)
            if row == 0:
                ax.set_title(f"{k(n)} transitions", pad=4)
        axes[row, 0].set_ylabel(ylabel)
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    panel_legends(fig, axes)
    save(out, where, fig, f"{quantity}_by_level", table)


def value_by_size(runs: pd.DataFrame, n: int, out: Path) -> None:
    """Rows: the two panels; columns: the levels; x: the number of machines; at n transitions."""
    d = runs[runs["n_transitions"] == n]
    sizes, levels = sorted(d["machines"].unique()), sorted(d["level"].unique())
    positions = {m: i for i, m in enumerate(sizes)}
    table = by_seed(d, "value", ["method", "level", "machines"])
    size = (1.9 * max(len(levels), 2) + 1.6, 4.4)
    fig, axes = plt.subplots(len(PANELS), len(levels), figsize=size, sharex=True,
                             sharey=True, squeeze=False)  # fmt: skip
    for row, methods in enumerate(PANELS.values()):
        for ax, level in zip(axes[row], levels, strict=True):
            draw_lines(ax, table[table["level"] == level], "machines", positions, methods, False)
            ax.axhline(0, **paper.ZERO_LINE)
            ax.set_ylim(*QUANTITIES["value"][2])
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes)
            if row == 0:
                ax.set_title(f"{level}: {LEVELS[level]}", pad=4)
            else:
                ax.set_xlabel("machines N")
        axes[row, 0].set_ylabel("normalised value")
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    panel_legends(fig, axes)
    save(out, "summary", fig, f"value_by_size_n={n}", table.assign(n_transitions=n))


def time(runs: pd.DataFrame, timing: pd.DataFrame, out: Path) -> None:
    """Seconds to learn one model and to re-plan it under one reward, against N, per method."""
    models = runs.drop_duplicates(["machines", "seed", "n_transitions", "method"])
    learn = models.groupby(["method", "machines"])["learn_s"].mean().reset_index(name="seconds")
    replan = runs.groupby(["method", "machines"])["replan_s"].mean().reset_index(name="seconds")
    sizes = sorted(runs["machines"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    panels = (("learning one model", learn), ("re-planning under one reward", replan))
    for ax, (title, table) in zip(axes, panels, strict=True):
        for method, (color, marker) in STYLE.items():
            t = table[table["method"] == method].sort_values("machines")
            ax.plot(t["machines"], t["seconds"], **paper.paper_line(color, marker))
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xlabel("machines N")
        ax.set_title(title, pad=4)
    axes[0].set_ylabel("seconds (mean)")
    fig.tight_layout(w_pad=1.2)
    paper.legend_below(fig, handles_of(list(STYLE)), y=0.03)
    total = timing.groupby("machines")[["seconds", "learn_s", "replan_s", "setup_s"]].sum().reset_index()
    data = pd.concat(
        [learn.assign(what="learn"), replan.assign(what="replan"), total.assign(what="job total")]
    )
    save(out, "summary", fig, "time", data)


PER_SIZE = [f"{quantity}_by_level" for quantity in QUANTITIES]
SUMMARY = ["value_by_size", "solved_by_level", "time"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Figures of the ablation: the methods along a ladder of rewards."
    )
    parser.add_argument("--run", type=Path, help="the run folder (default: the latest)")
    parser.add_argument(
        "--machines", nargs="+", type=int, metavar="N", help="sizes (default: all of the run)"
    )
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=OUTPUTS, help="where the figures go")
    args = parser.parse_args(argv)
    run = args.run or latest_run(args.results)
    if run is None or not run.is_dir() or not any(run.glob("N=*/runs/*.json")):
        parser.error(f"no finished run in {args.results}: start run.py first")
    runs, timing = load(run)
    sizes = sorted(runs["machines"].unique())
    missing = [m for m in args.machines or [] if m not in sizes]
    if missing:
        parser.error(f"no finished job of N={missing} in {run}; sizes with results: {sizes}")
    chosen = args.machines or sizes
    runs, timing = runs[runs["machines"].isin(chosen)], timing[timing["machines"].isin(chosen)]
    print(f"===== {NAME}: {run} -> {args.out} =====")
    print(f"sizes {chosen}: {len(runs):,} policies, {runs['seed'].nunique()} seeds, "
          f"{100 * runs['ok'].mean():.1f}% finished")  # fmt: skip
    with mpl.rc_context(paper.PAPER_STYLE):
        for machines in chosen:
            for quantity in QUANTITIES:
                by_level(runs[runs["machines"] == machines], quantity, f"N={machines}", args.out)
        for n in sorted(runs["n_transitions"].unique())[-2:]:
            value_by_size(runs, n, args.out)
        by_level(runs, "solved", "summary", args.out)
        time(runs, timing, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
