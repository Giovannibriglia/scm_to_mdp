"""Figures of the ablation "BIC against performance, every graph" (run.py), one set per size N.

Run from the repository root, after (or during) run.py:
    .venv/bin/python experiments/clear27/ablations/ablation_bic_score/plot.py               # the latest run
    .venv/bin/python experiments/clear27/ablations/ablation_bic_score/plot.py --machines 3
    .venv/bin/python experiments/clear27/ablations/ablation_bic_score/plot.py --run <run folder>

Input:  results/clear27/ablations/ablation_bic_score/sysadmin_<date>_<time>/N=<N>/ (run.py)
Output: outputs/clear27/ablations/ablation_bic_score/N=<N>/<figure>.{pdf,png}
        outputs/clear27/ablations/ablation_bic_score/data/N=<N>_<figure>.csv   the numbers drawn

Figures
  bic_vs_value     every graph as a dot: its BIC rank in its dataset (0 = worst, 1 = best BIC) against the
                   normalised value of its policy; one panel per number of transitions, the seeds pooled.
                   Line: the mean value of the graphs in each twentieth of the BIC ranks.
  bic_vs_promise   the same with the normalised promise (the certified lower bound).
  correlation      Spearman's rank correlation between BIC and value, and BIC and promise, one per dataset.
  bic_choice       what choosing the best-BIC graph gives: value and promise of that graph, of the true
                   graph, of the graphs PCMCI+ and DYNOTEARS find, and of the best graph of all; and the
                   share of graphs whose policy is strictly better than the best-BIC graph's.
  edges            the value against the number of true edges a graph misses and of extra edges it has.

Marked graphs: true graph (gold), best BIC (black ring), PCMCI+ (red), DYNOTEARS (blue), as in the paper.
A dataset where every graph has the same value has no correlation, and is left out of `correlation`.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
NAME = "ablation_bic_score"
RESULTS = ROOT / "results" / "clear27" / "ablations" / NAME
OUTPUTS = ROOT / "outputs" / "clear27" / "ablations" / NAME
STAMP = "%Y-%m-%d_%H-%M-%S"


def _paper():
    """clear27's plot.py: the style and colours of the paper figures."""
    spec = importlib.util.spec_from_file_location("clear27_plot", HERE.parents[2] / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _paper()
BINS = 20  # the BIC ranks are cut in twentieths for the mean line
# mark -> (column of datasets.csv, label, colour, marker)
MARKS = {
    "true": ("true_graph", "true graph", paper.GOLD, "o"),
    "pcmci": ("pcmci_graph", "PCMCI+", paper.RED, "s"),
    "dynotears": ("dynotears_graph", "DYNOTEARS", paper.BLUE, "D"),
    "best_bic": ("best_bic_graph", "best BIC", "#0b0b0b", "o"),
}
QUANTITIES = {"value": "normalised value", "promise": "normalised promise"}
LIMITS = {"value": (-1.3, 1.15), "promise": (-3.2, 1.15)}


# ---- loading --------------------------------------------------------------------------------------
def latest_run(results: Path) -> Path | None:
    from datetime import datetime

    found = []
    for path in results.glob("sysadmin_*") if results.is_dir() else []:
        try:
            datetime.strptime(path.name[len("sysadmin_") :], STAMP)
        except ValueError:
            continue
        found.append(path)
    return max(found) if found else None


def sizes_of(run: Path) -> list[int]:
    return sorted(int(p.name[2:]) for p in run.glob("N=*") if any((p / "graphs").glob("*.json")))


def load(run: Path, machines: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(graphs, datasets) of one size: every finished dataset of the run. graphs gets, per dataset, the
    BIC rank of each graph in [0, 1] (1 = the best BIC)."""
    import json

    folder = run / f"N={machines}" / "graphs"
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    datasets = pd.DataFrame(records).sort_values(["n_transitions", "seed"]).reset_index(drop=True)
    files = [folder / f"seed={r['seed']}_n={r['n_transitions']}.csv" for r in records]
    graphs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    graphs["held"] = graphs["held"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    by_dataset = graphs.groupby(["seed", "n_transitions"])["bic"]
    graphs["bic_rank"] = (by_dataset.rank(method="average") - 1) / (by_dataset.transform("size") - 1)
    return graphs, datasets


def marked(graphs: pd.DataFrame, datasets: pd.DataFrame) -> pd.DataFrame:
    """The marked graphs of every dataset, one row each, with the column `mark`."""
    rows = []
    for mark, (column, _, _, _) in MARKS.items():
        keys = datasets[["seed", "n_transitions", column]].dropna().rename(columns={column: "graph"})
        keys["graph"] = keys["graph"].astype(np.int64)
        rows.append(keys.merge(graphs, on=["seed", "n_transitions", "graph"]).assign(mark=mark))
    return pd.concat(rows, ignore_index=True)


# ---- helpers --------------------------------------------------------------------------------------
def save(out: Path, machines: int, fig, name: str, data: pd.DataFrame) -> None:
    folder = out / f"N={machines}"
    folder.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    data.to_csv(out / "data" / f"N={machines}_{name}.csv", index=False)
    print(f"  wrote N={machines}/{name}")


def mark_handle(mark: str) -> Line2D:
    _, label, color, marker = MARKS[mark]
    ring = mark == "best_bic"
    face, edge = ("none", color) if ring else (color, "white")
    return Line2D([], [], ls="", marker=marker, ms=6, mfc=face, mec=edge,
                  mew=1.2 if ring else 0.5, label=label)  # fmt: skip


def draw_marks(ax, points: pd.DataFrame, x: str, y: str, limits) -> None:
    for mark, (_, _, color, marker) in MARKS.items():
        p = points[points["mark"] == mark]
        ring = mark == "best_bic"
        ax.scatter(p[x], p[y].clip(*limits), s=46 if ring else 24, marker=marker, zorder=5 if ring else 4,
                   facecolor="none" if ring else color, edgecolor=color if ring else "white",
                   linewidth=1.1 if ring else 0.5)  # fmt: skip


def k(n: int) -> str:
    return f"{n:,}"


# ---- figures --------------------------------------------------------------------------------------
def bic_against(graphs, datasets, quantity: str, machines: int, out: Path) -> None:
    ns = sorted(graphs["n_transitions"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(2.5 * max(len(ns), 2), 2.7), sharex=True, sharey=True,
                             squeeze=False)  # fmt: skip
    points, limits, rows = marked(graphs, datasets), LIMITS[quantity], []
    for ax, n in zip(axes.flat, ns, strict=True):
        g = graphs[(graphs["n_transitions"] == n) & graphs[quantity].notna()]
        ax.scatter(g["bic_rank"], g[quantity].clip(*limits), s=3, color=paper.MUTED, alpha=0.18, lw=0,
                   rasterized=True, zorder=1)  # fmt: skip
        bins = np.minimum((g["bic_rank"] * BINS).astype(int), BINS - 1)
        mean = g.groupby(bins)[quantity].mean()
        ax.plot((mean.index + 0.5) / BINS, mean.clip(*limits), color=paper.INK, lw=1.3, zorder=3)
        draw_marks(ax, points[points["n_transitions"] == n], "bic_rank", quantity, limits)
        rho = [
            stats.spearmanr(d["bic"], d[quantity]).statistic
            for _, d in g.groupby("seed")
            if d[quantity].nunique() > 1
        ]
        note = f"ρ = {np.median(rho):.2f}" if rho else "ρ: no spread"
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=8.5, color=paper.INK2)
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_title(f"{k(n)} transitions", pad=4)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(*limits)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("BIC rank (1 = best)")
        centres = (mean.index + 0.5) / BINS
        rows.append(pd.DataFrame({"n_transitions": n, "bic_rank_bin": centres, "mean": mean.values,
                                  "median_spearman": np.median(rho) if rho else np.nan}))  # fmt: skip
    axes[0, 0].set_ylabel(QUANTITIES[quantity])
    fig.tight_layout(w_pad=1.0)
    handles = [Line2D([], [], ls="", marker="o", ms=3, color=paper.MUTED, label="a graph"),
               Line2D([], [], color=paper.INK, lw=1.3, label="mean")]  # fmt: skip
    handles += [mark_handle(m) for m in MARKS]
    paper.legend_below(fig, handles, y=0.03)
    save(out, machines, fig, f"bic_vs_{quantity}", pd.concat(rows))


def correlations(graphs: pd.DataFrame) -> pd.DataFrame:
    """Spearman's correlation between BIC and each quantity, one row per dataset and quantity."""
    rows = []
    for (seed, n), d in graphs.groupby(["seed", "n_transitions"]):
        for quantity in QUANTITIES:
            x = d.dropna(subset=[quantity])
            spread = x[quantity].nunique() > 1 and x["bic"].nunique() > 1
            rho = stats.spearmanr(x["bic"], x[quantity]).statistic if spread else np.nan
            rows.append(
                {"seed": seed, "n_transitions": n, "quantity": quantity, "spearman": rho, "graphs": len(x)}
            )
    return pd.DataFrame(rows)


def correlation(graphs, datasets, machines: int, out: Path) -> None:
    table = correlations(graphs)
    ns = sorted(table["n_transitions"].unique())
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    series = (("value", paper.BLUE, "o"), ("promise", paper.RED, "s"))
    for shift, (quantity, color, marker) in zip((-0.09, 0.09), series,
                                                strict=True):  # fmt: skip
        t = table[(table["quantity"] == quantity) & table["spearman"].notna()]
        x = t["n_transitions"].map({n: i for i, n in enumerate(ns)}) + shift
        ax.scatter(x, t["spearman"], s=14, color=color, alpha=0.45, lw=0, marker=marker)
        median = t.groupby("n_transitions")["spearman"].median().reindex(ns)
        ax.plot(np.arange(len(ns)) + shift, median.values, **paper.paper_line(color, marker, ms=5))
    ax.axhline(0, **paper.ZERO_LINE)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([k(n) for n in ns])
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("transitions")
    ax.set_ylabel("rank correlation with BIC")
    handles = [Line2D([], [], **paper.paper_line(paper.BLUE, "o", "normalised value", ms=5)),
               Line2D([], [], **paper.paper_line(paper.RED, "s", "normalised promise", ms=5))]  # fmt: skip
    fig.tight_layout()
    paper.legend_below(fig, handles, y=0.03)  # median over seeds; a dot is one dataset
    save(out, machines, fig, "correlation", table)


def bic_choice(graphs, datasets, machines: int, out: Path) -> None:
    points = marked(graphs, datasets)
    ns = sorted(graphs["n_transitions"].unique())
    x = np.arange(len(ns))
    by = ["seed", "n_transitions"]
    best = graphs.groupby(by)[["value", "promise"]].max().reset_index().assign(mark="best graph")
    table = pd.concat([points[by + ["mark", "value", "promise", "held"]], best], ignore_index=True)
    chosen = points[points["mark"] == "best_bic"][by + ["value"]].rename(columns={"value": "chosen"})
    better = graphs.merge(chosen, on=by)
    better = (
        better.assign(better=better["value"] > better["chosen"] + 1e-9)
        .groupby(by)["better"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    styles = {m: (color, marker) for m, (_, _, color, marker) in MARKS.items()} | {
        "best graph": (paper.MUTED, "^")
    }
    labels = {m: label for m, (_, label, _, _) in MARKS.items()} | {"best graph": "best graph of all"}
    dashed_marks = ("best graph", "best_bic")
    for ax, quantity in zip(axes[:2], QUANTITIES, strict=True):
        for mark, (color, marker) in styles.items():
            mean = table[table["mark"] == mark].groupby("n_transitions")[quantity].mean().reindex(ns)
            dashed = mark in dashed_marks
            ax.plot(x, mean.values, **paper.paper_line(color, marker, ls=(0, (4, 2)) if dashed else "-",
                                                       zorder=4 if mark == "best_bic" else 3))  # fmt: skip
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_ylabel(QUANTITIES[quantity])
    share = better.groupby("n_transitions")["better"]
    axes[2].bar(x, 100 * share.mean().reindex(ns).values, color=paper.MUTED, width=0.6)
    axes[2].set_ylabel("graphs better than the\nbest-BIC graph (%)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([k(n) for n in ns])
        ax.set_xlabel("transitions")
    fig.tight_layout(w_pad=1.6)
    handles = [Line2D([], [], **paper.paper_line(c, m, labels[mark], ms=5,
                                                 ls=(0, (4, 2)) if mark in dashed_marks else "-"))
               for mark, (c, m) in styles.items()]  # fmt: skip
    paper.legend_below(fig, handles, y=0.03)
    save(out, machines, fig, "bic_choice", table.merge(better, on=by, how="left"))


def edges(graphs, datasets, machines: int, out: Path) -> None:
    ns = sorted(graphs["n_transitions"].unique())
    colors = [paper.TEAL, paper.BLUE, paper.PURPLE, paper.RED, paper.GOLD, paper.PINK]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    rows = []
    complete = graphs[graphs["missing"] == 0]  # extra edges alone: the graphs that contain the truth
    panels = ((axes[0], "missing", graphs, "true edges missing"),
              (axes[1], "extra", complete, "extra edges (no true edge missing)"))  # fmt: skip
    for ax, column, source, label in panels:
        for i, n in enumerate(ns):
            mean = source[source["n_transitions"] == n].groupby(column)["value"].mean()
            ax.plot(mean.index, mean.values, **paper.paper_line(colors[i % len(colors)], "o", k(n)))
            rows.append(mean.reset_index().assign(n_transitions=n, edges=column))
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_xlabel(label)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("normalised value (mean)")
    axes[1].legend(title="transitions", fontsize=8, title_fontsize=8)
    fig.tight_layout(w_pad=1.4)
    save(out, machines, fig, "edges", pd.concat(rows))


FIGURES = {
    "bic_vs_value": lambda g, d, m, o: bic_against(g, d, "value", m, o),
    "bic_vs_promise": lambda g, d, m, o: bic_against(g, d, "promise", m, o),
    "correlation": correlation,
    "bic_choice": bic_choice,
    "edges": edges,
}


def plot_size(run: Path, machines: int, out: Path, only: list | None = None) -> list[str]:
    graphs, datasets = load(run, machines)
    print(f"N={machines}: {len(datasets)} datasets, {len(graphs):,} graphs")
    written = []
    with mpl.rc_context(paper.PAPER_STYLE):
        for name, draw in FIGURES.items():
            if only and name not in only:
                continue
            draw(graphs, datasets, machines, out)
            written.append(name)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Figures of the ablation BIC against performance.")
    parser.add_argument("--run", type=Path, help="the run folder (default: the latest)")
    parser.add_argument(
        "--machines", nargs="+", type=int, metavar="N", help="sizes (default: every size of the run)"
    )
    parser.add_argument("--only", nargs="+", metavar="FIGURE", help=f"among: {', '.join(FIGURES)}")
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=OUTPUTS, help="where the figures go")
    args = parser.parse_args(argv)
    wrong = [f for f in args.only or [] if f not in FIGURES]
    if wrong:
        parser.error(f"no figure {wrong}; the figures are: {', '.join(FIGURES)}")
    run = args.run or latest_run(args.results)
    if run is None or not run.is_dir():
        parser.error(f"no run in {args.results}: start run.py first")
    sizes = sizes_of(run)
    missing = [m for m in args.machines or [] if m not in sizes]
    if missing or not sizes:
        parser.error(f"no finished dataset of N={missing or '?'} in {run}; sizes with results: {sizes}")
    print(f"===== {NAME}: {run} -> {args.out} =====")
    for machines in args.machines or sizes:
        plot_size(run, machines, args.out, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
