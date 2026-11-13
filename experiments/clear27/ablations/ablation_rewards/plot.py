"""Figures of the ablation "the methods along a ladder of rewards" (run.py), one domain at a time.

Run from the repository root, after (or during) run.py:
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py                  # SysAdmin
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --domain sepsis  # or drone
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --sizes 3
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --run <run folder>

Input:  results/clear27/ablations/ablation_rewards/<domain>/<domain>_<date>_<time>/ (run.py)
Output: outputs/clear27/ablations/ablation_rewards/<domain>/
            <size>=<s>/<figure>.{pdf,png}   one set per size (<size>: N, vitals or side)
            summary/<figure>.{pdf,png}      across the sizes
            data/<where>_<figure>.csv       the numbers drawn

Two rows of panels (PANELS), the same structures in both:
    with a promise   the interval MDPs: DYNOTEARS, PCMCI+, the full graph, the flat model, the true graph
    no promise       their plug-in MDPs (the same colours), and SLF-Rmax and Met-Rmax as their papers run them
The figures of the promise (promise, held, promised) have the first row only: a point model gives no bound.

Figures, per size (x: the level of the ladder, 1 = the original reward ... 6 = traps; one column per
number of transitions):
  value_by_level     the normalised value of the policy
  solved_by_level    the share of runs with a normalised value of at least SOLVED
  promise_by_level   the normalised promise (the certified lower bound)
  held_by_level      the share of runs whose promise held (promise <= true value)
  promised_by_level  the share of runs with a normalised promise of at least PROMISED (0: the policy is
                     certified to be at least as good as the random one)
Across the sizes (summary/):
  value_by_size_n=<n>   the value against the size, one column per level; one figure per number of
                        transitions of the run
  solved_by_level       as above, the sizes pooled
  promised_by_level     as above, the sizes pooled
  time                  seconds of every step against the size: discovery, parameters, the box, solving

The output folder follows the run: a figure (or its csv) left by an earlier run that this run does not
draw is removed, e.g. the folder of a size the run does not have. With --sizes, the figures of the
other sizes of the run stay.

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
# "promised": at least this normalised promise. 0 is the value of the random policy; the promise of the
# box is far below the value (on SysAdmin it never reaches SOLVED), so SOLVED here would draw zeros
PROMISED = 0.0


def _paper():
    """clear27's plot.py: the style and colours of the paper figures."""
    spec = importlib.util.spec_from_file_location("clear27_plot", HERE.parents[2] / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _paper()
MDP = " MDP"  # run.py's suffix of a structure's plug-in MDP
PANELS = {
    "with a promise\n(interval MDP)": ["DYNOTEARS", "PCMCI+", "full graph", "flat", "true graph"],
    "no promise\n(point MDP)": ["DYNOTEARS MDP", "PCMCI+ MDP", "full graph MDP", "flat MDP", "true graph MDP",
                                "SLF-Rmax", "Met-Rmax"],
}  # fmt: skip
STYLE = {  # method -> (colour, marker), the paper's; a structure's plug-in MDP is drawn as the structure
    "DYNOTEARS": (paper.BLUE, "D"),
    "PCMCI+": (paper.RED, "s"),
    "full graph": (paper.TEAL, "^"),
    "flat": (paper.PURPLE, "X"),
    "true graph": (paper.GOLD, "o"),
    "SLF-Rmax": (paper.GREEN, "P"),
    "Met-Rmax": (paper.WINE, "v"),
}
STYLE |= {name + MDP: STYLE[name] for name in ["DYNOTEARS", "PCMCI+", "full graph", "flat", "true graph"]}
# DYNOTEARS and PCMCI+ often find the same graph, so their lines coincide: DYNOTEARS is drawn wider and
# behind, and stays visible around the line of PCMCI+
WIDE = {name: dict(lw=3.4, ms=6.0, zorder=1.5) for name in ("DYNOTEARS", "DYNOTEARS" + MDP)}
# What the figures read from the run's config.json (main): the names of the levels and of a size
RUN = {
    "levels": {1: "additive", 2: "subset", 3: "cost", 4: "threshold", 5: "goal", 6: "trap"},
    "size_name": "machines N",
    "prefix": "N",
}
# quantity -> (column, y label, limits, in percent)
QUANTITIES = {
    "value": ("value", "normalised value", (-1.3, 1.1), False),
    "promise": ("promise", "normalised promise", (-3.2, 1.1), False),
    "held": ("held", "runs with\npromise held (%)", (-4, 104), True),
    "solved": ("solved", f"runs with value\n≥ {SOLVED} (%)", (-4, 104), True),
    "promised": ("promised", f"runs with promise\n≥ {PROMISED:g} (%)", (-4, 104), True),
}
OF_THE_PROMISE = ("promise", "held", "promised")  # the first row of panels only


def panels_of(quantity: str) -> dict:
    return dict(list(PANELS.items())[:1]) if quantity in OF_THE_PROMISE else PANELS


# ---- loading --------------------------------------------------------------------------------------
def latest_run(results: Path, domain: str) -> Path | None:
    found = []
    for path in results.glob(f"{domain}_*") if results.is_dir() else []:
        try:
            datetime.strptime(path.name[len(domain) + 1 :], STAMP)
        except ValueError:
            continue
        found.append(path)
    return max(found) if found else None


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, timing): every finished (size, seed) of the run. runs gets `ok`, `held` as 0/1, `solved`
    and `promised` (the value at least SOLVED, the promise at least PROMISED; a stopped run is neither).
    A run of before the domains has the column `machines`: it is read as `size`."""
    done = sorted(run.glob("*=*/runs/*.json"))
    runs = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in done], ignore_index=True)
    timing = pd.DataFrame([json.loads(p.read_text()) for p in done])
    runs, timing = (t.rename(columns={"machines": "size"}) for t in (runs, timing))
    runs["ok"] = runs["status"].eq("ok")
    runs["held"] = runs["held"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    runs["solved"] = (runs["ok"] & (runs["value"] >= SOLVED)).astype(float)
    runs["promised"] = (runs["ok"] & (runs["promise"] >= PROMISED)).astype(float)
    return runs, timing


def read_config(run: Path) -> dict:
    """The domain of the run and the names its figures use (RUN), from config.json."""
    path = run / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    if "levels" in config:
        RUN["levels"] = {int(level): name for level, name in config["levels"].items()}
    RUN["size_name"] = config.get("size_name", RUN["size_name"])
    RUN["prefix"] = config.get("size_prefix", RUN["prefix"])
    return config


def by_seed(runs: pd.DataFrame, column: str, by: list[str]) -> pd.DataFrame:
    """Mean and standard error over the seeds of the per-seed mean of `column`, per group `by`."""
    per_seed = runs.groupby([*by, "seed"])[column].mean().reset_index()
    out = per_seed.groupby(by)[column].agg(["mean", "std", "count"]).reset_index()
    out["se"] = (out["std"] / np.sqrt(out["count"])).fillna(0)
    return out


# ---- helpers --------------------------------------------------------------------------------------
WRITTEN: set[Path] = set()  # the files of this call (save), so that remove_stale knows the others


def save(out: Path, where: str, fig, name: str, data: pd.DataFrame) -> None:
    folder = out / where
    folder.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    data.to_csv(out / "data" / f"{where}_{name}.csv", index=False)
    WRITTEN.update({folder / f"{name}.pdf", folder / f"{name}.png", out / "data" / f"{where}_{name}.csv"})
    print(f"  wrote {where}/{name}")


def remove_stale(out: Path, kept_sizes: list[int]) -> list[Path]:
    """Remove the figures and csv files in `out` that this call did not write: what an earlier run left
    (a size or a number of transitions this run does not have). The per-size figures of `kept_sizes`
    (sizes of the run that were not asked for) stay. Returns the removed files."""
    kept = {f"{RUN['prefix']}={size}" for size in kept_sizes}
    found = [
        p for where in ("*=*", "summary") for ext in ("pdf", "png") for p in out.glob(f"{where}/*.{ext}")
    ]
    found += list(out.glob("data/*.csv"))
    stale = sorted(
        p for p in found
        if p not in WRITTEN and p.parent.name not in kept and p.name.split("_")[0] not in kept
    )  # fmt: skip
    for path in stale:
        path.unlink()
    for folder in out.glob("*=*"):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    return stale


def k(n: int) -> str:
    return f"{n:,}"


def line_of(method: str, label: str | None = None, **more) -> dict:
    """The paper's line in the colour and marker of the method; wider and behind if it is in WIDE."""
    return paper.paper_line(*STYLE[method], label, **(more | WIDE.get(method, {})))


def handles_of(methods: list[str]) -> list[Line2D]:
    return [Line2D([], [], **line_of(m, m.removesuffix(MDP), ms=5)) for m in methods]


def draw_lines(ax, table: pd.DataFrame, x: str, positions: dict, methods: list[str], percent: bool) -> None:
    """One line per method: mean over the seeds, band of one standard error."""
    scale = 100.0 if percent else 1.0
    for method in methods:
        t = table[table["method"] == method].sort_values(x)
        if t.empty:
            continue
        color = STYLE[method][0]
        at = t[x].map(positions)
        low, high = scale * (t["mean"] - t["se"]), scale * (t["mean"] + t["se"])
        ax.fill_between(at, low, high, color=color, alpha=0.15,
                        lw=0)  # fmt: skip
        ax.plot(at, scale * t["mean"], **line_of(method))


def panel_legends(fig, axes, panels: dict) -> None:
    """Each row of panels has its own methods: a legend at the right of the row."""
    for row, (title, methods) in enumerate(panels.items()):
        axes[row, -1].legend(handles=handles_of(methods), loc="center left", bbox_to_anchor=(1.03, 0.5),
                             title=title, title_fontsize=8.5, fontsize=8.5, handlelength=2.0)  # fmt: skip


# ---- figures --------------------------------------------------------------------------------------
def by_level(runs: pd.DataFrame, quantity: str, where: str, out: Path) -> None:
    """Rows: the panels; columns: the numbers of transitions; x: the level of the ladder."""
    column, ylabel, limits, percent = QUANTITIES[quantity]
    panels = panels_of(quantity)
    ns = sorted(runs["n_transitions"].unique())
    levels = sorted(runs["level"].unique())
    positions = {level: i for i, level in enumerate(levels)}
    table = by_seed(runs, column, ["n_transitions", "method", "level"])
    size = (2.3 * max(len(ns), 2) + 1.6, 2.3 * len(panels))
    fig, axes = plt.subplots(len(panels), len(ns), figsize=size, sharex=True, sharey=True, squeeze=False)
    for row, methods in enumerate(panels.values()):
        for ax, n in zip(axes[row], ns, strict=True):
            draw_lines(ax, table[table["n_transitions"] == n], "level", positions, methods, percent)
            if not percent:
                ax.axhline(0, **paper.ZERO_LINE)
            ax.set_ylim(*limits)
            ax.set_xlim(-0.4, len(levels) - 0.6)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(
                [RUN["levels"][level] for level in levels], rotation=40, ha="right", fontsize=8
            )
            if row == 0:
                ax.set_title(f"{k(n)} transitions", pad=4)
        axes[row, 0].set_ylabel(ylabel)
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    panel_legends(fig, axes, panels)
    save(out, where, fig, f"{quantity}_by_level", table)


def value_by_size(runs: pd.DataFrame, n: int, out: Path) -> None:
    """Rows: the panels; columns: the levels; x: the size; at n transitions."""
    d = runs[runs["n_transitions"] == n]
    sizes, levels = sorted(d["size"].unique()), sorted(d["level"].unique())
    positions = {m: i for i, m in enumerate(sizes)}
    table = by_seed(d, "value", ["method", "level", "size"])
    size = (1.9 * max(len(levels), 2) + 1.6, 4.4)
    fig, axes = plt.subplots(len(PANELS), len(levels), figsize=size, sharex=True,
                             sharey=True, squeeze=False)  # fmt: skip
    for row, methods in enumerate(PANELS.values()):
        for ax, level in zip(axes[row], levels, strict=True):
            draw_lines(ax, table[table["level"] == level], "size", positions, methods, False)
            ax.axhline(0, **paper.ZERO_LINE)
            ax.set_ylim(*QUANTITIES["value"][2])
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes)
            if row == 0:
                ax.set_title(f"{level}: {RUN['levels'][level]}", pad=4)
            else:
                ax.set_xlabel(RUN["size_name"])
        axes[row, 0].set_ylabel("normalised value")
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    panel_legends(fig, axes, PANELS)
    save(out, "summary", fig, f"value_by_size_n={n}", table.assign(n_transitions=n))


STEPS = {  # column -> (panel title, once per model or once per reward)
    "discovery_s": ("structure: discovery", "model"),
    "learning_s": ("parameters: the interval sets", "model"),
    "relax_s": ("relaxation: the box, per reward", "reward"),
    "solve_s": ("solving, per reward", "reward"),
}


def time(runs: pd.DataFrame, timing: pd.DataFrame, out: Path) -> None:
    """Mean seconds of every step against the size, per method: discovery and parameter learning (once per
    model), relaxation and solving (once per reward). Selection is in the data, not drawn (it is tiny).
    Rows: the panels (a point model has no box to compile)."""
    models = runs.drop_duplicates(["size", "seed", "n_transitions", "method"])
    sizes = sorted(runs["size"].unique())
    fig, axes = plt.subplots(len(PANELS), len(STEPS), figsize=(2.75 * len(STEPS) + 1.6, 2.6 * len(PANELS)),
                             sharex=True, sharey=True, squeeze=False)  # fmt: skip
    rows = []
    for col, (column, (title, per)) in enumerate(STEPS.items()):
        source = models if per == "model" else runs
        table = source.groupby(["method", "size"])[column].mean().reset_index(name="seconds")
        for row, methods in enumerate(PANELS.values()):
            ax = axes[row, col]
            for method in methods:
                t = table[(table["method"] == method) & table["seconds"].notna()].sort_values("size")
                ax.plot(t["size"], t["seconds"], **line_of(method))
            ax.set_yscale("log")
            ax.set_xticks(sizes)
            if row == 0:
                ax.set_title(title, pad=4, fontsize=9)
            else:
                ax.set_xlabel(RUN["size_name"])
        rows.append(table.assign(step=column))
    for ax in axes[:, 0]:
        ax.set_ylabel("seconds (mean)")
    fig.tight_layout(w_pad=1.0)
    panel_legends(fig, axes, PANELS)
    selection = models.groupby(["method", "size"])["selection_s"].mean().reset_index(name="seconds")
    rows.append(selection.assign(step="selection_s"))
    total = timing.groupby("size").sum(numeric_only=True).drop(columns=["seed"]).reset_index()
    save(out, "summary", fig, "time", pd.concat([*rows, total.assign(step="job totals")]))


PER_SIZE = [f"{quantity}_by_level" for quantity in QUANTITIES]
SUMMARY = ["value_by_size", "solved_by_level", "promised_by_level", "time"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Figures of the ablation: the methods along a ladder of rewards."
    )
    parser.add_argument("--domain", default="sysadmin", help="sysadmin (default), sepsis or drone")
    parser.add_argument("--run", type=Path, help="the run folder (default: the latest of the domain)")
    parser.add_argument("--sizes", "--machines", dest="sizes", nargs="+", type=int, metavar="N",
                        help="sizes (default: all of the run)")  # fmt: skip
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out", type=Path, default=OUTPUTS, help="where the figures go (a folder per domain)"
    )
    args = parser.parse_args(argv)
    run = args.run or latest_run(args.results / args.domain, args.domain)
    if run is None or not run.is_dir() or not any(run.glob("*=*/runs/*.json")):
        parser.error(f"no finished run of {args.domain} in {args.results / args.domain}: start run.py first")
    domain = read_config(run).get("domain", args.domain)
    out = args.out / domain
    runs, timing = load(run)
    sizes = sorted(runs["size"].unique())
    missing = [m for m in args.sizes or [] if m not in sizes]
    if missing:
        parser.error(f"no finished job of size {missing} in {run}; sizes with results: {sizes}")
    chosen = args.sizes or sizes
    runs, timing = runs[runs["size"].isin(chosen)], timing[timing["size"].isin(chosen)]
    print(f"===== {NAME}, {domain}: {run} -> {out} =====")
    print(f"sizes {[int(m) for m in chosen]}: {len(runs):,} policies, {runs['seed'].nunique()} seeds, "
          f"{100 * runs['ok'].mean():.1f}% finished")  # fmt: skip
    with mpl.rc_context(paper.PAPER_STYLE):
        for size in chosen:
            for quantity in QUANTITIES:
                by_level(runs[runs["size"] == size], quantity, f"{RUN['prefix']}={size}", out)
        for n in sorted(runs["n_transitions"].unique()):
            value_by_size(runs, n, out)
        by_level(runs, "solved", "summary", out)
        by_level(runs, "promised", "summary", out)
        time(runs, timing, out)
    stale = remove_stale(out, [m for m in sizes if m not in chosen])
    if stale:
        print(f"  removed {len(stale)} stale file(s) of an earlier run: "
              + ", ".join(str(p.relative_to(out)) for p in stale))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
