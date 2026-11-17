"""Figures of the ablation "a score against performance, every graph" (run.py), one set per size N.

Run from the repository root, after (or during) run.py:
    .venv/bin/python experiments/clear27/ablations/ablation_graph_score/plot.py               # the latest run
    .venv/bin/python experiments/clear27/ablations/ablation_graph_score/plot.py --machines 3
    .venv/bin/python experiments/clear27/ablations/ablation_graph_score/plot.py --run <run folder>

Input:  results/clear27/ablations/ablation_graph_score/sysadmin_<date>_<time>/N=<N>/ (run.py)
Output: outputs/clear27/ablations/ablation_graph_score/N=<N>/<figure>.{pdf,png}
        outputs/clear27/ablations/ablation_graph_score/data/N=<N>_<figure>.csv   the numbers drawn

The scores: BIC, AIC and BDeu (run.py); for each, higher is better.

Figures
  <score>_vs_value     every graph as a dot: its rank by the score in its dataset (0 = worst, 1 = best)
                       against the normalised value of its policy; one panel per number of transitions,
                       the seeds pooled, so a marked graph appears once per seed (ten rings "best BIC"
                       are the ten datasets' best graphs, not ten graphs tied in one dataset). Line: the
                       mean value of the graphs in each twentieth of the ranks.
  <score>_vs_promise   the same with the normalised promise (the certified lower bound).
  <score>_per_graph    one dot per GRAPH, averaged over the seeds: its mean rank by the score against its
                       mean value (top row) and its mean promise (bottom row), coloured by the number of
                       true edges it misses; with few graphs (at most ERROR_BARS), a bar of one standard
                       deviation over the seeds. Marked: the true graph, the empty graph, and the best
                       single graph (below). Only graphs that finished in every seed.
  correlation          Spearman's rank correlation between each score and the value, and the promise,
                       one per dataset.
  correlation_per_graph  the same correlation computed on the graphs averaged over the seeds (the dots of
                       <score>_per_graph): one number per score and number of transitions.
  choice               what choosing by a score gives: value and promise of the best graph of each score,
                       of the true graph, of the graphs PCMCI+ and DYNOTEARS find, and of the best single
                       graph: the ONE graph with the highest mean value over the seeds at that number of
                       transitions (the same graph in every dataset, so it is a choice one could make,
                       unlike the best graph of each dataset taken after the fact). Third panel: the share
                       of graphs whose policy is strictly better than the graph a score picks.
  edges                the value against the number of true edges a graph misses and of extra edges.

Marked graphs: true graph (gold), PCMCI+ (red), DYNOTEARS (blue), as in the paper; the best graph of a
score is a ring. A dataset where every graph has the same value has no correlation, and is left out.
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
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
NAME = "ablation_graph_score"
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
BINS = 20  # the ranks are cut in twentieths for the mean line
ERROR_BARS = 200  # per_graph: the spread over the seeds is drawn up to this number of graphs
SCORES = {"bic": ("BIC", "#0b0b0b"), "aic": ("AIC", paper.TEAL), "bdeu": ("BDeu", paper.PURPLE)}
# mark -> (column of datasets.csv, label, colour, marker); the best graph of a score is added per figure
GRAPHS = {
    "true": ("true_graph", "true graph", *paper.METHODS["true graph"][:2]),  # clear27's registry
    "pcmci": ("pcmci_graph", "PCMCI+", *paper.METHODS["PCMCI+"][:2]),
    "dynotears": ("dynotears_graph", "DYNOTEARS", *paper.METHODS["DYNOTEARS"][:2]),
}
BEST = {
    score: (f"best_{score}_graph", f"best {label}", color, "o") for score, (label, color) in SCORES.items()
}
SINGLE = "single"  # the best single graph (see the module docstring)
QUANTITIES = {"value": "normalised value", "promise": "normalised lower bound"}
LIMITS = {"value": (-1.3, 1.15), "promise": (-3.2, 1.15)}
DASH = (0, (4, 2))
BY = ["seed", "n_transitions"]


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


def sizes_of(run: Path) -> list[int]:
    return sorted(int(p.name[2:]) for p in run.glob("N=*") if any((p / "graphs").glob("*.json")))


def load(run: Path, machines: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(graphs, datasets) of one size: every finished dataset of the run. graphs gets, per dataset and
    score, the rank of each graph in [0, 1] (1 = the best score): <score>_rank."""
    folder = run / f"N={machines}" / "graphs"
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    datasets = pd.DataFrame(records).sort_values(["n_transitions", "seed"]).reset_index(drop=True)
    files = [folder / f"seed={r['seed']}_n={r['n_transitions']}.csv" for r in records]
    graphs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    graphs["held"] = graphs["held"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    for score in SCORES:
        by_dataset = graphs.groupby(BY)[score]
        graphs[f"{score}_rank"] = (by_dataset.rank(method="average") - 1) / (by_dataset.transform("size") - 1)
    return graphs, datasets


def marked(graphs: pd.DataFrame, datasets: pd.DataFrame, marks: dict) -> pd.DataFrame:
    """The marked graphs of every dataset, one row each, with the column `mark`."""
    rows = []
    for mark, (column, _, _, _) in marks.items():
        keys = datasets[[*BY, column]].dropna().rename(columns={column: "graph"})
        keys["graph"] = keys["graph"].astype(np.int64)
        rows.append(keys.merge(graphs, on=[*BY, "graph"]).assign(mark=mark))
    return pd.concat(rows, ignore_index=True)


def best_single(graphs: pd.DataFrame) -> pd.DataFrame:
    """Per number of transitions, the one graph with the highest mean value over the seeds: its rows
    (one per seed), with mark SINGLE. Only graphs that finished in every seed count; a tie goes to the
    graph with fewer edges, then to the lower number."""
    rows = []
    for n, d in graphs.groupby("n_transitions"):
        seeds = d["seed"].nunique()
        per = d.dropna(subset=["value"]).groupby("graph").agg(
            mean=("value", "mean"), seeds=("seed", "nunique"), edges=("n_edges", "first")
        )  # fmt: skip
        per = per[per["seeds"] == seeds].reset_index()
        if per.empty:
            continue
        per["mean"] = per["mean"].round(9)
        graph = per.sort_values(["mean", "edges", "graph"], ascending=[False, True, True])["graph"].iloc[0]
        rows.append(d[d["graph"] == graph].assign(mark=SINGLE, n_transitions=n))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[*graphs.columns, "mark"])


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


def mark_handle(label: str, color: str, marker: str, hollow: bool) -> Line2D:
    face, edge = ("none", color) if hollow else (color, "white")
    return Line2D([], [], ls="", marker=marker, ms=6, mfc=face, mec=edge, mew=1.2 if hollow else 0.5,
                  label=label)  # fmt: skip


def k(n: int) -> str:
    return f"{n:,}"


def spearman(x: pd.Series, y: pd.Series) -> float:
    return stats.spearmanr(x, y).statistic if x.nunique() > 1 and y.nunique() > 1 else np.nan


# ---- figures --------------------------------------------------------------------------------------
def score_against(graphs, datasets, score: str, quantity: str, machines: int, out: Path) -> None:
    label, rank = SCORES[score][0], f"{score}_rank"
    marks = GRAPHS | {score: BEST[score]}
    ns = sorted(graphs["n_transitions"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(2.5 * max(len(ns), 2), 2.7), sharex=True, sharey=True,
                             squeeze=False)  # fmt: skip
    points, limits, rows = marked(graphs, datasets, marks), LIMITS[quantity], []
    for ax, n in zip(axes.flat, ns, strict=True):
        g = graphs[(graphs["n_transitions"] == n) & graphs[quantity].notna()]
        ax.scatter(g[rank], g[quantity].clip(*limits), s=3, color=paper.MUTED, alpha=0.18, lw=0,
                   rasterized=True,
                   zorder=1)  # fmt: skip
        bins = np.minimum((g[rank] * BINS).astype(int), BINS - 1)
        mean = g.groupby(bins)[quantity].mean()
        centres = (mean.index + 0.5) / BINS
        ax.plot(centres, mean.clip(*limits), color=paper.INK, lw=1.3, zorder=3)
        for mark, (_, _, color, marker) in marks.items():
            p = points[(points["mark"] == mark) & (points["n_transitions"] == n)]
            hollow = mark == score
            ax.scatter(p[rank], p[quantity].clip(*limits), s=46 if hollow else 24, marker=marker,
                       zorder=5 if hollow else 4, facecolor="none" if hollow else color,
                       edgecolor=color if hollow else "white", linewidth=1.1 if hollow else 0.5)  # fmt: skip
        rho = [spearman(d[score], d[quantity]) for _, d in g.groupby("seed")]
        rho = [r for r in rho if np.isfinite(r)]
        note = f"ρ = {np.median(rho):.2f}" if rho else "ρ: no spread"
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=8.5, color=paper.INK2)
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_title(f"{k(n)} transitions", pad=4)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(*limits)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(f"{label} rank (1 = best)")
        rows.append(pd.DataFrame({"n_transitions": n, "rank_bin": centres, "mean": mean.values,
                                  "median_spearman": np.median(rho) if rho else np.nan}))  # fmt: skip
    axes[0, 0].set_ylabel(QUANTITIES[quantity])
    fig.tight_layout(w_pad=1.0)
    handles = [Line2D([], [], ls="", marker="o", ms=3, color=paper.MUTED, label="a graph"),
               Line2D([], [], color=paper.INK, lw=1.3, label="mean")]  # fmt: skip
    handles += [mark_handle(lab, c, m, mark == score) for mark, (_, lab, c, m) in marks.items()]
    paper.legend_below(fig, handles, y=0.03)
    save(out, machines, fig, f"{score}_vs_{quantity}", pd.concat(rows))


def graph_means(graphs: pd.DataFrame, score: str) -> pd.DataFrame:
    """One row per (number of transitions, graph): the mean over the seeds of its rank by the score, its
    value and its promise, their standard deviations, and the share of seeds whose promise held. Only
    the graphs with a value in every seed."""
    rank = f"{score}_rank"
    seeds = graphs.groupby("n_transitions")["seed"].nunique().rename("all_seeds")
    done = graphs.dropna(subset=["value", "promise"])
    per = done.groupby(["n_transitions", "graph"]).agg(
        seeds=("seed", "nunique"), n_edges=("n_edges", "first"), missing=("missing", "first"),
        extra=("extra", "first"), rank=(rank, "mean"), value=("value", "mean"), value_sd=("value", "std"),
        promise=("promise", "mean"), promise_sd=("promise", "std"), held=("held", "mean"),
    ).reset_index()  # fmt: skip
    per = per.merge(seeds, on="n_transitions")
    return per[per["seeds"] == per["all_seeds"]].drop(columns="all_seeds")


def per_graph(graphs, datasets, score: str, machines: int, out: Path) -> None:
    label = SCORES[score][0]
    per = graph_means(graphs, score)
    ns = sorted(per["n_transitions"].unique())
    true_graph, single = int(datasets["true_graph"].iloc[0]), best_single(graphs)
    most = max(int(per["missing"].max()), 1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "missing", [paper.BLUE, "#b9cfee", "#e6e2da", paper.RED]
    )
    norm = mpl.colors.Normalize(0, most)
    fig, axes = plt.subplots(2, len(ns), figsize=(2.5 * max(len(ns), 2), 4.9), sharex=True, sharey="row",
                             squeeze=False)  # fmt: skip
    for row, quantity in enumerate(QUANTITIES):
        limits = LIMITS[quantity]
        for ax, n in zip(axes[row], ns, strict=True):
            g = per[per["n_transitions"] == n]
            if len(g) <= ERROR_BARS:
                ax.errorbar(g["rank"], g[quantity], yerr=g[f"{quantity}_sd"].fillna(0), fmt="none", lw=0.5,
                            ecolor=paper.MUTED, alpha=0.5, zorder=1)  # fmt: skip
            few = len(g) <= ERROR_BARS
            ax.scatter(g["rank"], g[quantity].clip(*limits), c=g["missing"], cmap=cmap, norm=norm,
                       s=16 if few else 4, lw=0.3 if few else 0, edgecolor="white", rasterized=not few,
                       zorder=2)  # fmt: skip
            marks = {
                "true graph": (true_graph, paper.GOLD, "o"),
                "empty graph": (0, "#0b0b0b", "s"),
                "best single graph": (
                    int(single[single["n_transitions"] == n]["graph"].iloc[0]),
                    "#0b0b0b",
                    "^",
                ),
            }
            for graph, color, marker in marks.values():
                m = g[g["graph"] == graph]
                ax.scatter(m["rank"], m[quantity].clip(*limits), s=52, marker=marker, facecolor="none",
                           edgecolor=color, linewidth=1.2, zorder=4)  # fmt: skip
            rho = spearman(g["rank"], g[quantity])
            note = f"ρ = {rho:.2f}" if np.isfinite(rho) else "ρ: no spread"
            ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=8.5, color=paper.INK2)
            ax.axhline(0, **paper.ZERO_LINE)
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(*limits)
            ax.set_xticks([0, 0.5, 1])
            if row == 0:
                ax.set_title(f"{k(n)} transitions", pad=4)
            else:
                ax.set_xlabel(f"mean {label} rank (1 = best)")
        axes[row, 0].set_ylabel(f"{QUANTITIES[quantity]}\n(mean over seeds)")
    fig.tight_layout(w_pad=1.0, h_pad=0.9)
    handles = [mark_handle(text, color, marker, True) for text, (_, color, marker) in marks.items()]
    paper.legend_below(fig, handles, y=0.035)
    cax = fig.add_axes([0.72, -0.015, 0.2, 0.018])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    bar.set_label("true edges missing", fontsize=8.5)
    bar.set_ticks(sorted({0, most // 2, most}))
    bar.outline.set_visible(False)
    save(out, machines, fig, f"{score}_per_graph", per)


def correlations(graphs: pd.DataFrame) -> pd.DataFrame:
    """Spearman's correlation between each score and each quantity, one row per dataset."""
    rows = []
    for (seed, n), d in graphs.groupby(BY):
        for score in SCORES:
            for quantity in QUANTITIES:
                x = d.dropna(subset=[quantity])
                rows.append({"seed": seed, "n_transitions": n, "score": score, "quantity": quantity,
                             "spearman": spearman(x[score], x[quantity]), "graphs": len(x)})  # fmt: skip
    return pd.DataFrame(rows)


def correlation(graphs, datasets, machines: int, out: Path) -> None:
    table = correlations(graphs)
    ns = sorted(table["n_transitions"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    shifts = np.linspace(-0.16, 0.16, len(SCORES))
    for ax, quantity in zip(axes, QUANTITIES, strict=True):
        for shift, (score, (_, color)) in zip(shifts, SCORES.items(), strict=True):
            t = table[(table["quantity"] == quantity) & (table["score"] == score) & table["spearman"].notna()]
            x = t["n_transitions"].map({n: i for i, n in enumerate(ns)}) + shift
            ax.scatter(x, t["spearman"], s=12, color=color, alpha=0.4, lw=0)
            median = t.groupby("n_transitions")["spearman"].median().reindex(ns)
            ax.plot(np.arange(len(ns)) + shift, median.values, **paper.paper_line(color, "o", ms=5))
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([k(n) for n in ns])
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("transitions")
        ax.set_title(f"with the {QUANTITIES[quantity]}", pad=4)
    axes[0].set_ylabel("rank correlation of the score")
    fig.tight_layout(w_pad=1.2)
    handles = [
        Line2D([], [], **paper.paper_line(color, "o", label, ms=5)) for label, color in SCORES.values()
    ]
    paper.legend_below(fig, handles, y=0.03)  # a line: the median over the seeds; a dot: one dataset
    save(out, machines, fig, "correlation", table)


def correlation_per_graph(graphs, datasets, machines: int, out: Path) -> None:
    rows = []
    for score in SCORES:
        per = graph_means(graphs, score)
        for n, g in per.groupby("n_transitions"):
            for quantity in QUANTITIES:
                rows.append({"n_transitions": n, "score": score, "quantity": quantity, "graphs": len(g),
                             "spearman": spearman(g["rank"], g[quantity])})  # fmt: skip
    table = pd.DataFrame(rows)
    ns = sorted(table["n_transitions"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, quantity in zip(axes, QUANTITIES, strict=True):
        for score, (_, color) in SCORES.items():
            t = table[(table["quantity"] == quantity) & (table["score"] == score)].set_index("n_transitions")
            ax.plot(range(len(ns)), t["spearman"].reindex(ns).values, **paper.paper_line(color, "o", ms=5))
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([k(n) for n in ns])
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("transitions")
        ax.set_title(f"with the {QUANTITIES[quantity]}", pad=4)
    axes[0].set_ylabel("rank correlation of the score\n(graphs averaged over seeds)")
    fig.tight_layout(w_pad=1.2)
    handles = [
        Line2D([], [], **paper.paper_line(color, "o", label, ms=5)) for label, color in SCORES.values()
    ]
    paper.legend_below(fig, handles, y=0.03)
    save(out, machines, fig, "correlation_per_graph", table)


def choice(graphs, datasets, machines: int, out: Path) -> None:
    marks = GRAPHS | {f"best_{score}": BEST[score] for score in SCORES}
    points = pd.concat([marked(graphs, datasets, marks), best_single(graphs)], ignore_index=True)
    ns = sorted(graphs["n_transitions"].unique())
    x = np.arange(len(ns))
    styles = {
        m: (label, color, marker, m.startswith("best_")) for m, (_, label, color, marker) in marks.items()
    }
    styles[SINGLE] = ("best single graph", paper.MUTED, "^", True)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 2.9))
    for ax, quantity in zip(axes[:2], QUANTITIES, strict=True):
        for mark, (_, color, marker, dashed) in styles.items():
            mean = points[points["mark"] == mark].groupby("n_transitions")[quantity].mean().reindex(ns)
            ax.plot(x, mean.values, **paper.paper_line(color, marker, ls=DASH if dashed else "-",
                                                       zorder=3 if dashed else 4))  # fmt: skip
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_ylabel(QUANTITIES[quantity])
    width, rows = 0.8 / len(SCORES), []
    for i, (score, (label, color)) in enumerate(SCORES.items()):  # graphs better than the score's pick
        picked = points[points["mark"] == f"best_{score}"][[*BY, "value"]].rename(columns={"value": "picked"})
        d = graphs.merge(picked, on=BY)
        share = d.assign(better=d["value"] > d["picked"] + 1e-9).groupby(BY)["better"].mean().reset_index()
        mean = 100 * share.groupby("n_transitions")["better"].mean().reindex(ns)
        axes[2].bar(
            x + (i - (len(SCORES) - 1) / 2) * width, mean.values, width=width, color=color, label=label
        )
        rows.append(share.assign(score=score))
    axes[2].set_ylabel("graphs better than\nthe score's best graph (%)")
    axes[2].legend(fontsize=8, handlelength=1.0)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([k(n) for n in ns])
        ax.set_xlabel("transitions")
    fig.tight_layout(w_pad=1.6)
    handles = [Line2D([], [], **paper.paper_line(color, marker, label, ms=5, ls=DASH if dashed else "-"))
               for label, color, marker, dashed in styles.values()]  # fmt: skip
    paper.legend_below(fig, handles, ncol=4, y=0.03)
    drawn = points[[*BY, "mark", "graph", "n_edges", "missing", "value", "promise", "held"]]
    save(out, machines, fig, "choice", pd.concat([drawn, pd.concat(rows)], ignore_index=True))


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


def _against(score: str, quantity: str):
    return lambda graphs, datasets, machines, out: score_against(
        graphs, datasets, score, quantity, machines, out
    )


def _per_graph(score: str):
    return lambda graphs, datasets, machines, out: per_graph(graphs, datasets, score, machines, out)


FIGURES = {f"{score}_vs_{quantity}": _against(score, quantity) for score in SCORES for quantity in QUANTITIES}
FIGURES |= {f"{score}_per_graph": _per_graph(score) for score in SCORES}
FIGURES |= {"correlation": correlation, "correlation_per_graph": correlation_per_graph, "choice": choice}
FIGURES |= {"edges": edges}


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
    parser = argparse.ArgumentParser(description="Figures of the ablation: a score against performance.")
    parser.add_argument("--run", type=Path, help="the run folder (default: the latest)")
    parser.add_argument(
        "--machines", nargs="+", type=int, metavar="N", help="sizes (default: all of the run)"
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
