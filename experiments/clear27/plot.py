"""clear27: every figure of one domain, from the results of its runs (S9).

Run from the repository root, after (or during) the runs of a domain:

    .venv/bin/python experiments/clear27/plot.py sysadmin                 # every figure of SysAdmin
    .venv/bin/python experiments/clear27/plot.py sepsis drone             # several domains
    .venv/bin/python experiments/clear27/plot.py sysadmin --only value coverage
    .venv/bin/python experiments/clear27/plot.py sysadmin --solver "IMDP (IA)"   # the box instead of exact
    .venv/bin/python experiments/clear27/plot.py --list                   # the figure names

Input:  results/clear27/<domain>/<domain>_<source>_<date>_<time>/<knob>/uniform/results.csv, the latest
        run of every graph source (a figure whose methods have no run yet is skipped).
Output: outputs/clear27/<domain>/paper/<name>.{pdf,png}     the paper figures (section 5 of this file)
        outputs/clear27/<domain>/explore/<name>.{pdf,png}   the exploratory figures (section 6)
        outputs/clear27/<domain>/data/<name>.csv            the numbers drawn in each figure

The file, in order: 1 settings, 2 loading the runs, 3 what is read from the data of a domain (its sizes,
panels and labels), 4 style and helpers, 5 paper figures, 6 exploratory figures, 7 the command line.

Conventions
- normalised value = (V_true - V_random) / (V*_obs - V_random): 0 is the uniform random policy, 1 the best
  policy over the observed state. The promise is normalised in the same way.
- "certified" = the interval MDP of a graph source solved with the domain's solver: "IMDP (exact)" for
  SysAdmin, the box "IMDP (IA)" for Sepsis, Drone and Aircraft, where the exact solver stops beyond the
  smallest sizes (DOMAIN_SETTINGS; --solver overrides it). The flat model has one interval MDP ("flat IMDP").
- A point (method, size, transitions) is drawn only if at least MIN_OK of its seeds finished (status ok):
  the mean over the finished seeds, and a 95% t-interval as the band.
- "Promise held" = promise <= true value in every observed live state (column `safe`).
- A domain's sizes are the ones where the reference (the true graph, certified) finished at least MIN_OK
  of its runs; the larger sizes are left out of the figures that pool or compare sizes.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # files only: no window, also on a server

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# ==== 1. settings ==================================================================================
EXPERIMENT = "clear27"
MIN_OK = 0.5  # share of the seeds that must finish for a point to be drawn, and for a size to be kept
DELTA = 0.05  # the confidence of the interval sets: the 1 - delta line of the "held" figures
N_FIXED = 10_000  # the number of transitions of the figures along the problem size
THRESHOLD = 0.9  # "near-optimal": the normalised value of the data-needed figures
MAX_PANELS = 5  # panels (sizes) of a paper figure
SAFETY_BUDGETS = (1000, 10_000)  # the two panels of value_vs_safety
DEFAULT_SOLVER = "IMDP (exact)"

# The colours of the paper's figures (clear2027.tex): one per graph source and one per baseline.
GOLD, RED, BLUE, PINK, TEAL, PURPLE = "#cd9c1f", "#ce4522", "#3070c7", "#dd8cc3", "#12a7a7", "#7945ab"
GREEN, OLIVE, NAVY, WINE = "#348f4f", "#8ea43d", "#00679c", "#aa3f4e"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# The six series of the paper figures: (model, solver or None = certified, label, colour, marker).
# PCMCI+ and DYNOTEARS are the runs without same-step edges (the "_lagged" sources).
SERIES = [
    ("true graph", None, "true graph", GOLD, "o"),
    ("PCMCI+ (lagged)", None, "PCMCI+", RED, "s"),
    ("DYNOTEARS (lagged)", None, "DYNOTEARS", BLUE, "D"),
    ("sparse", None, "sparse", PINK, "v"),
    ("full-fact", None, "full graph", TEAL, "^"),
    ("flat", "IMDP", "flat", PURPLE, "X"),
]
# The figure against the offline RL baselines: (model, solver or None, label, colour, marker, dashed).
# Colours and markers as in the paper's figure against the state of the art (sota_<domain>.pdf): ours is
# its thick blue line, SPIBB on a graph its red "+", and each baseline keeps its colour and marker.
BASELINE_SERIES = [
    ("true graph", None, "ours (true graph)", BLUE, "o", False),
    ("true graph", "Factored SPIBB", "Factored SPIBB", RED, "+", True),
    ("DiFaRL", "DiFaRL", "DiFaRL", GREEN, "D", True),
    ("FOCUS", "FOCUS", "FOCUS", PURPLE, "x", True),
    ("SL-SPIBB", "SL-SPIBB", "SL-SPIBB", PINK, "v", True),
    ("flat SPIBB", "flat SPIBB", "flat SPIBB", OLIVE, "<", True),
    ("offline CQL", "offline CQL", "CQL", NAVY, ">", True),
    ("offline DQN", "offline DQN", "DQN", WINE, "*", True),
]
BASELINES = ["DiFaRL", "FOCUS", "SL-SPIBB", "flat SPIBB", "offline CQL", "offline DQN"]

# The exploratory figures draw more models: every graph source, the KWIK learners and the baselines.
GRAPH_SOURCES = ["true graph", "PCMCI+", "PCMCI+ (lagged)", "DYNOTEARS", "DYNOTEARS (lagged)", "sparse",
                 "full-fact", "flat"]  # fmt: skip
KWIK = ["true graph", "Strehl (fixed hull)", "Strehl (accepted hull)", "Strehl-KWIK",
        "Met-Rmax (fixed hull)", "Met-Rmax (accepted hull)", "Met-Rmax-KWIK"]  # fmt: skip
COLOR = {
    "true graph": GOLD, "PCMCI+ (lagged)": RED, "PCMCI+": WINE, "DYNOTEARS (lagged)": BLUE,
    "DYNOTEARS": NAVY, "sparse": PINK, "full-fact": TEAL, "flat": PURPLE,
    # a KWIK learner has one colour, and one line style per variant (as in the paper)
    "Strehl (fixed hull)": GREEN, "Strehl (accepted hull)": GREEN, "Strehl-KWIK": GREEN,
    "Met-Rmax (fixed hull)": PURPLE, "Met-Rmax (accepted hull)": PURPLE, "Met-Rmax-KWIK": PURPLE,
    "DiFaRL": GREEN, "FOCUS": PURPLE, "SL-SPIBB": PINK, "flat SPIBB": OLIVE, "offline CQL": NAVY,
    "offline DQN": WINE,
}  # fmt: skip
DASH, DOTS = (0, (4, 2)), (0, (1, 1.6))
LINE = {m: DASH for m in BASELINES} | {
    "Strehl (accepted hull)": DASH, "Met-Rmax (accepted hull)": DASH,
    "Strehl-KWIK": DOTS, "Met-Rmax-KWIK": DOTS,
}  # fmt: skip

# Choices made by hand for one domain; a domain without an entry takes them from its data (section 3).
#   solver         the certified solver of the domain (default DEFAULT_SOLVER)
#   panels         the sizes of the paper figures with one panel per size
#   detail_size    the size of the exploratory figures drawn at one size
#   graph_size     the size of the paper figure graph_errors
#   safety_labels  value_vs_safety, 10,000 transitions: where each label goes (dx, dy in points, alignment)
DOMAIN_SETTINGS = {
    "sysadmin": {
        "panels": ["2", "4", "6", "8", "9"],
        "detail_size": "7",
        "graph_size": "9",
        "safety_labels": {
            "true graph, PCMCI+, DYNOTEARS": (5, 7, "left"),
            "sparse": (6, -7, "left"),
            "true graph, plug-in": (4, 8, "right"),
            "offline CQL": (0, -11, "center"),
            "offline DQN": (0, 9, "center"),
        },
    },
    # the exact solver runs out of memory at vitals=4 (Sepsis) and from side 5 on (Drone): the box finishes
    "sepsis": {"solver": "IMDP (IA)", "panels": ["2", "3", "4"]},
    "drone": {"solver": "IMDP (IA)", "panels": ["3-1", "5-1", "7-1", "10-1", "12-1"]},
    "aircraft": {"solver": "IMDP (IA)"},  # the exact solver finishes at 4x3 only; panels from the data
}

# ==== 2. loading the runs ==========================================================================
KEEP = [
    "size", "size_label", "n_states", "n_observed_states", "seed", "n_transitions", "method", "role",
    "guarantee", "status", "failed_step", "promise", "true", "V_star_obs", "V_star", "V_random", "regret",
    "optimal_share", "gap", "safe", "violation_rate", "worst_violation",
    "discovery_s", "selection_s", "learning_s", "relax_s", "training_s", "solve_s",
    "discovery_gb", "learning_gb", "relax_gb", "training_gb", "solve_gb",
    "shd", "missing", "extra", "n_true_edges", "sid", "true_parents", "intra_next", "dags", "tied",
    "candidates", "C_star", "bias_max", "cert_gap", "cert_holds", "kwik_dont_know",
    "contain_all", "contain_all_seen", "contain_data", "s4_coverage", "s4_coverage_seen", "s4_width",
    "s4_unseen",
]  # fmt: skip
STAMP = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
TRUTH = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}


def latest_runs(folder: Path, domain: str) -> dict[str, Path]:
    """source -> its latest run folder (<domain>_<source>_<date>_<time>) in results/clear27/<domain>."""
    name = re.compile(rf"^{re.escape(domain)}_(.+)_{STAMP}$")
    found = {}
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:  # by name: the latest stamp is last
        match = name.match(path.name)
        if match and path.is_dir():
            found[match.group(1)] = path
    return found


def split_method(method: str, source: str) -> tuple[str, str]:
    """A method's name as (model, solver): "PCMCI+ -> IMDP (exact)" is ("PCMCI+", "IMDP (exact)"); a
    baseline is its own model and solver."""
    if method == "oracle -> MDP":
        return "oracle", "MDP"
    if source == "sota":
        return method, method
    if method == "flat IMDP":
        return "flat", "IMDP"
    if " -> " in method:
        model, solver = method.split(" -> ", 1)
        return model, solver
    return method, method


def load(results: Path, domain: str) -> pd.DataFrame:
    """Every run of the domain as one table (one row per size, seed, number of transitions and method),
    with: source, model, solver, ok, value and promise_value (normalised), total_s, and the graph errors
    recall, precision, missing_share, extra_per_var."""
    frames = []
    for source, run in latest_runs(results / domain, domain).items():
        for file in sorted(run.glob("*/uniform/results.csv")):
            d = pd.read_csv(file, low_memory=False)
            if d.empty:
                continue
            d = d[[c for c in KEEP if c in d.columns]].copy()
            d["source"] = source
            d["size"] = d["size"].astype(str)
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    runs = pd.concat(frames, ignore_index=True)
    parts = [split_method(m, s) for m, s in zip(runs["method"], runs["source"], strict=False)]
    runs["model"] = [p[0] for p in parts]
    runs["solver"] = [p[1] for p in parts]
    oracle = runs["model"] == "oracle"  # the oracle runs with every source: keep one copy
    one = runs[oracle].drop_duplicates(["size", "seed", "n_transitions"]).assign(source="oracle")
    runs = pd.concat([runs[~oracle], one], ignore_index=True)
    runs["ok"] = runs["status"].eq("ok")
    runs["safe"] = runs["safe"].map(TRUTH)
    span = runs["V_star_obs"] - runs["V_random"]
    runs["value"] = (runs["true"] - runs["V_random"]) / span
    runs["promise_value"] = (runs["promise"] - runs["V_random"]) / span
    runs["total_s"] = runs[["discovery_s", "selection_s", "learning_s", "solve_s"]].sum(axis=1, min_count=1)
    hits = runs["n_true_edges"] - runs["missing"]
    runs["recall"] = 1 - runs["missing"] / runs["n_true_edges"]
    runs["precision"] = (hits / (hits + runs["extra"])).where(hits + runs["extra"] > 0)
    runs["missing_share"] = runs["missing"] / runs["n_true_edges"]
    variables = runs.dropna(subset=["true_parents"]).groupby("size")["true_parents"].first()
    variables = variables.map(lambda text: len(json.loads(text)))  # next-state variables of the true graph
    runs["extra_per_var"] = runs["extra"] / runs["size"].map(variables)
    if "contain_all" in runs:
        runs["covered"] = 100 * runs["contain_all"].map(TRUTH)
    return runs


# ==== 3. what is read from the data of a domain ====================================================
@dataclass
class Domain:
    name: str
    runs: pd.DataFrame  # every run (section 2)
    solver: str  # the certified solver, e.g. "IMDP (exact)"
    sizes: list  # every size that ran, by number of observed states
    kept: list  # the sizes where the reference finished (see the module docstring)
    panels: list  # the sizes of the paper figures with one panel per size
    detail_size: str
    graph_size: str
    states: dict  # size -> observed states
    labels: dict  # size -> its label in the results, e.g. "N=7", "vitals=3"
    safety_labels: dict
    out: Path

    def pick(self, model: str, solver: str | None = None) -> pd.DataFrame:
        """The runs of one model with one solver (solver None: a baseline, whose solver is its name)."""
        x = self.runs[self.runs["model"] == model]
        return x[x["solver"] == (solver or model)]

    def certified(self, model: str) -> pd.DataFrame:
        return self.pick(model, "IMDP" if model == "flat" else self.solver)

    def series(self, model: str, solver: str | None) -> pd.DataFrame:
        """A row of SERIES / BASELINE_SERIES: solver None is the certified solver."""
        return self.certified(model) if solver is None else self.pick(model, solver)

    def title(self, size) -> str:
        """ "N=7" -> "$N = 7$" (a one-letter name is set as a symbol), "vitals=3" -> "vitals = 3"."""
        name, _, value = str(self.labels.get(size, size)).partition("=")
        if not value:
            return name
        return rf"${name} = {value}$" if len(name) == 1 else f"{name} = {value}"

    def size_range(self) -> str:
        first, last = self.labels[self.kept[0]], self.labels[self.kept[-1]]
        return first if first == last else f"{first} to {last}"


def spread(items: list, n: int) -> list:
    """At most n of the items, evenly spread, always with the first and the last."""
    if len(items) <= n:
        return list(items)
    return [items[i] for i in sorted({round(i) for i in np.linspace(0, len(items) - 1, n)})]


def describe(name: str, runs: pd.DataFrame, solver: str, out: Path) -> Domain:
    """The domain's sizes, panels and labels, from its runs and DOMAIN_SETTINGS."""
    states = runs.groupby("size")["n_observed_states"].max().dropna().sort_values()
    sizes = list(states.index)
    reference = runs[(runs["model"] == "true graph") & (runs["solver"] == solver)]
    finished = reference.groupby("size")["ok"].mean()
    kept = [s for s in sizes if finished.get(s, 0.0) >= MIN_OK]
    chosen = DOMAIN_SETTINGS.get(name, {})
    panels = [s for s in chosen.get("panels", []) if s in kept]
    if len(panels) != len(chosen.get("panels", [None])):  # no choice, or a chosen size did not finish
        panels = spread(kept, MAX_PANELS)
    largest = kept[-1] if kept else None
    return Domain(
        name=name,
        runs=runs,
        solver=solver,
        sizes=sizes,
        kept=kept,
        panels=panels,
        detail_size=chosen["detail_size"] if chosen.get("detail_size") in kept else largest,
        graph_size=chosen["graph_size"] if chosen.get("graph_size") in kept else largest,
        states={s: int(v) for s, v in states.items()},
        labels=runs.groupby("size")["size_label"].first().to_dict() if "size_label" in runs else {},
        safety_labels=chosen.get("safety_labels", {}),
        out=out / name,
    )


# ==== 4. style and helpers =========================================================================
PAPER_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.edgecolor": "#3a3a3a", "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.grid": False, "legend.frameon": False,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
}  # fmt: skip
EXPLORE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "regular", "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK,
    "lines.linewidth": 1.6, "lines.markersize": 3.5,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
}  # fmt: skip
ZERO_LINE = dict(color="#9a9a9a", lw=0.6, ls=(0, (3, 2)), zorder=0)


def cells(df: pd.DataFrame, col: str = "value", by=("size", "n_transitions")) -> pd.DataFrame:
    """Mean and 95% t-interval of `col` over the finished seeds, per cell; a cell with fewer than MIN_OK
    of its seeds finished is dropped."""
    by = list(by)
    out = df.groupby(by).agg(seeds=("seed", "size"), finished=("ok", "sum")).reset_index()
    ok = df[df["ok"]].groupby(by)[col].agg(["mean", "std", "count"]).reset_index()
    out = out.merge(ok, on=by, how="left")
    t = stats.t.ppf(0.975, np.maximum(out["count"] - 1, 1))
    half = (t * out["std"] / np.sqrt(out["count"])).fillna(0)
    out["lo"], out["hi"] = out["mean"] - half, out["mean"] + half
    return out[out["finished"] >= MIN_OK * out["seeds"]]


def save(dom: Domain, group: str, fig, name: str, data: pd.DataFrame | None = None) -> None:
    folder = dom.out / group
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    if data is not None:
        (dom.out / "data").mkdir(parents=True, exist_ok=True)
        data.to_csv(dom.out / "data" / f"{name}.csv", index=False)
    print(f"  wrote {group}/{name}")


def table(rows: list) -> pd.DataFrame:
    rows = [r for r in rows if len(r)]
    return pd.concat(rows) if rows else pd.DataFrame()


def paper_axis(ax) -> None:
    """The x axis of a paper panel along the number of transitions."""
    ax.set_xscale("log")
    ax.set_xticks([1e1, 1e3, 1e5])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def paper_line(color, marker, label=None, **more) -> dict:
    return dict(color=color, lw=1.3, marker=marker, ms=3.6, mec="white", mew=0.5, label=label) | more


def legend_below(fig, handles, ncol=None, y=0.04) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
               handlelength=2.0, columnspacing=1.6, borderaxespad=0)  # fmt: skip


def legend_columns(n_handles: int, n_panels: int) -> int:
    """One row under a wide figure; a narrow one (few panels) wraps the legend."""
    return n_handles if n_panels >= 4 else min(n_handles, 3)


def states_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def needed(dom: Domain, runs: pd.DataFrame) -> pd.DataFrame:
    """Per kept size: the median (and quartiles) over seeds of the fewest transitions at which the
    normalised value reaches THRESHOLD; infinite for a seed that never reaches it."""
    runs = runs[runs["size"].isin(dom.kept)]
    reached = runs[runs["ok"] & (runs["value"] >= THRESHOLD)].groupby(["size", "seed"])["n_transitions"].min()
    every = runs.groupby(["size", "seed"]).size().index
    by_size = reached.reindex(every).fillna(np.inf).reset_index().groupby("size")["n_transitions"]
    sizes = [s for s in dom.kept if s in by_size.groups]
    out = pd.DataFrame({"size": sizes, "states": [float(dom.states[s]) for s in sizes]})
    out["median"] = by_size.median().reindex(sizes).to_numpy()
    out["q1"] = by_size.quantile(0.25).reindex(sizes).to_numpy()
    out["q3"] = by_size.quantile(0.75).reindex(sizes).to_numpy()
    return out


# ==== 5. paper figures =============================================================================
def value(dom: Domain) -> None:
    """Normalised value of the certified policy against the transitions, one panel per size."""
    n = len(dom.panels)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * max(n, 2), 2.4), sharex=True, sharey=True, squeeze=False)
    rows, handles = [], []
    for model, solver, label, color, marker in SERIES:
        c = cells(dom.series(model, solver))
        rows.append(c.assign(method=label))
        for ax, size in zip(axes.flat, dom.panels, strict=False):
            s = c[c["size"] == size].sort_values("n_transitions")
            ax.fill_between(s["n_transitions"], s["lo"], s["hi"], color=color, alpha=0.15, lw=0)
            ax.plot(s["n_transitions"], s["mean"], **paper_line(color, marker))
        handles.append(Line2D([], [], **paper_line(color, marker, label, ms=4.5)))
    for ax, size in zip(axes.flat, dom.panels, strict=False):
        ax.set_title(dom.title(size), pad=4)
        paper_axis(ax)
        ax.set_ylim(-1.15, 1.08)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.axhline(0, **ZERO_LINE)
        ax.set_xlabel("transitions")
    axes[0, 0].set_ylabel("normalised value")
    fig.tight_layout(w_pad=1.0, h_pad=0.9)
    legend_below(fig, handles, legend_columns(len(handles), n))
    save(dom, "paper", fig, "value", table(rows))


def baselines(dom: Domain) -> None:
    """Our certified policy against the offline RL baselines: value (top) and the share of runs whose
    promise held (bottom; a baseline's promise is its own value estimate, with no guarantee)."""
    n = len(dom.panels)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * max(n, 2), 4.3), sharex=True, sharey="row", squeeze=False)
    rows, handles = [], []
    for model, solver, label, color, marker, dashed in reversed(BASELINE_SERIES):  # ours drawn on top
        d = dom.series(model, solver)
        val, held = cells(d), cells(d.assign(value=100 * d["safe"]))
        rows += [val.assign(method=label, metric="value"), held.assign(method=label, metric="promise_held")]
        style = paper_line(color, marker, lw=1.1 if dashed else 2.2, ls=(0, (4, 1.6)) if dashed else "-",
                           zorder=2 if dashed else 3)  # fmt: skip
        if marker in ("+", "x"):  # markers made of lines: drawn in the line's colour, not with a white edge
            style |= {"mec": color, "mew": 1.0}
        for j, size in enumerate(dom.panels):
            s = val[val["size"] == size].sort_values("n_transitions")
            if not dashed:
                axes[0, j].fill_between(s["n_transitions"], s["lo"], s["hi"], color=color, alpha=0.15, lw=0)
            axes[0, j].plot(s["n_transitions"], s["mean"], **style)
            h = held[held["size"] == size].sort_values("n_transitions")
            axes[1, j].plot(h["n_transitions"], h["mean"], **style)
        handles.insert(0, Line2D([], [], **(style | {"ms": 4.5, "label": label})))
    for j, size in enumerate(dom.panels):
        axes[0, j].set_title(dom.title(size), pad=4)
        for ax in axes[:, j]:
            paper_axis(ax)
        axes[0, j].set_ylim(-1.15, 1.08)
        axes[0, j].set_yticks([-1, -0.5, 0, 0.5, 1])
        axes[0, j].axhline(0, **ZERO_LINE)
        axes[1, j].set_ylim(-4, 104)
        axes[1, j].set_yticks([0, 25, 50, 75, 100])
        axes[1, j].axhline(100 * (1 - DELTA), **ZERO_LINE)
        axes[1, j].set_xlabel("transitions")
    axes[0, 0].set_ylabel("normalised value")
    axes[1, 0].set_ylabel("runs with\npromise held (%)")
    axes[1, -1].annotate(r"$1-\delta$", (5e5, 100 * (1 - DELTA)), xytext=(3, -9), textcoords="offset points",
                         fontsize=8, color="#6b6b6b", annotation_clip=False)  # fmt: skip
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), handlelength=2.4,
               ncol=len(handles) if n >= 4 else 4, columnspacing=1.2, borderaxespad=0)  # fmt: skip
    save(dom, "paper", fig, "baselines", table(rows))


def coverage(dom: Domain) -> None:
    """Share of runs in which every learned interval contains the true probability (column contain_all):
    the event the guarantee relies on, at least 1 - delta of the runs."""
    n = len(dom.panels)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * max(n, 2), 2.4), sharex=True, sharey=True, squeeze=False)
    rows, handles = [], []
    for model, solver, label, color, marker in reversed(SERIES):  # the true graph on top
        c = cells(dom.series(model, solver), "covered")
        rows.append(c.assign(method=label))
        for ax, size in zip(axes.flat, dom.panels, strict=False):
            s = c[c["size"] == size].sort_values("n_transitions")
            ax.plot(s["n_transitions"], s["mean"], **paper_line(color, marker))
        handles.insert(0, Line2D([], [], **paper_line(color, marker, label, ms=4.5)))
    for ax, size in zip(axes.flat, dom.panels, strict=False):
        ax.set_title(dom.title(size), pad=4)
        paper_axis(ax)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.axhline(100 * (1 - DELTA), **ZERO_LINE)
        ax.set_xlabel("transitions")
    axes[0, 0].set_ylabel("runs with true\nprobabilities covered (%)")
    axes[0, -1].annotate(r"$1-\delta$", (5e5, 100 * (1 - DELTA)), xytext=(3, -9), textcoords="offset points",
                         fontsize=8, color="#6b6b6b", annotation_clip=False)  # fmt: skip
    fig.tight_layout(w_pad=1.0)
    legend_below(fig, handles, legend_columns(len(handles), n))
    save(dom, "paper", fig, "coverage", table(rows))


def paper_series(dom: Domain, label: str, solver: str | None = None) -> pd.DataFrame:
    """The kept-size runs of a SERIES label (with another solver on request), or of a baseline."""
    by_label = {lab: (model, default) for model, default, lab, _, _ in SERIES}
    if label in by_label:
        model, default = by_label[label]
        d = dom.series(model, default) if solver is None else dom.pick(model, solver)
    else:
        d = dom.pick(label)
    return d[d["size"].isin(dom.kept)]


def series_handle(label: str) -> Line2D:
    color, marker = next((c, m) for _, _, lab, c, m in SERIES if lab == label)
    return Line2D([], [], **paper_line(color, marker, label, ms=4.5))


def promise_scatter(dom: Domain) -> None:
    """Promise against true value, one dot per run, coloured by the number of transitions."""
    graphs, lim = ["true graph", "PCMCI+", "DYNOTEARS"], (-3.5, 2.0)
    panels = [
        ("certified (ours)", pd.concat([paper_series(dom, g) for g in graphs])),
        ("plug-in MDP", pd.concat([paper_series(dom, g, "MDP (plug-in)") for g in graphs])),
        ("offline RL baselines", pd.concat([paper_series(dom, b) for b in BASELINES])),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "data", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    )
    norm = mpl.colors.LogNorm(10, 5e5)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharex=True, sharey=True)
    rows = []
    for ax, (title, d) in zip(axes, panels, strict=False):
        d = d[d["ok"]].sample(frac=1, random_state=0)  # random draw order: no colour hides another
        x, y = d["promise_value"].clip(*lim), d["value"].clip(-1.5, 1.35)
        ax.fill_between(lim, lim, lim[0], color="#f0efec", lw=0, zorder=0)
        ax.plot(lim, lim, color="#3a3a3a", lw=0.8, zorder=1)
        ax.scatter(x, y, c=d["n_transitions"], cmap=cmap, norm=norm, s=6, lw=0, alpha=0.7, zorder=2,
                   rasterized=True)  # fmt: skip
        if len(d):
            held = (d["promise_value"] <= d["value"] + 1e-9).mean()
            ax.text(0.04, 0.96, f"promise ≤ true value: {100 * held:.1f}%", transform=ax.transAxes, va="top",
                    fontsize=8.5, color=INK)  # fmt: skip
        ax.text(0.96, 0.04, "promise violated", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
        ax.set_title(title, pad=4)
        ax.set_xlim(lim)
        ax.set_ylim(-1.5, 1.35)
        ax.set_xlabel("normalised promise")
        columns = ["model", "solver", "size", "seed", "n_transitions", "promise_value", "value"]
        rows.append(d[columns].assign(panel=title))
    axes[0].set_ylabel("normalised value")
    fig.tight_layout(w_pad=1.2)
    cax = fig.add_axes([0.36, -0.06, 0.3, 0.03])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label("transitions", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    save(dom, "paper", fig, "promise_scatter", table(rows))


def value_vs_safety(dom: Domain) -> None:
    """Share of runs whose promise failed against the normalised value, at two numbers of transitions."""
    from adjustText import adjust_text  # only this figure needs it

    entries = [(label, None, "ours") for _, _, label, _, _ in SERIES]
    entries += [("true graph", "MDP (plug-in)", "plug-in")]
    entries += [(b, None, "baseline") for b in BASELINES]
    kinds = {"ours": (BLUE, "o"), "plug-in": (MUTED, "s"), "baseline": (RED, "D")}
    fig, axes = plt.subplots(1, len(SAFETY_BUDGETS), figsize=(8.4, 3.4), sharey=True)
    rows = []
    for ax, n in zip(axes, SAFETY_BUDGETS, strict=False):
        marks = []
        for label, solver, kind in entries:
            d = paper_series(dom, label, solver)
            d = d[d["ok"] & (d["n_transitions"] == n)]
            if d.empty:
                continue
            violated, val = 100 * (1 - d["safe"].mean()), d["value"].mean()
            color, marker = kinds[kind]
            ax.scatter(violated, val, color=color, marker=marker, s=34, edgecolor="white", lw=0.6, zorder=3)
            name = label if solver is None else "true graph, plug-in"
            marks.append((violated, val, name))
            rows.append({"n_transitions": n, "method": name, "kind": kind, "violated_pct": violated,
                         "value": val, "runs": len(d)})  # fmt: skip
        groups = []  # one label for markers that sit on top of each other
        for violated, val, name in marks:
            for g in groups:
                if abs(g[0] - violated) < 2 and abs(g[1] - val) < 0.025:
                    g[2].append(name)
                    break
            else:
                groups.append([violated, val, [name]])
        placed = dom.safety_labels if n == 10_000 else {}  # labels placed by hand (DOMAIN_SETTINGS)
        texts = []
        for violated, val, names in groups:
            name = ", ".join(names)
            if placed:
                dx, dy, ha = placed.get(name, (6, 0, "left"))
                ax.annotate(name, (violated, val), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            va="center", fontsize=7.6, color=INK2)  # fmt: skip
            else:
                dy = -0.07 if names == ["DYNOTEARS"] else 0.0  # start below its marker (the cluster at 0%)
                texts.append(ax.text(violated + 2, val + dy, name, fontsize=7.6, color=INK2))
        if texts:
            adjust_text(texts, x=[m[0] for m in marks], y=[m[1] for m in marks], ax=ax, expand=(1.3, 1.6),
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))  # fmt: skip
        ax.axhline(0, **ZERO_LINE)
        ax.set_xlim(-5, 100)
        ax.set_ylim(-0.45, 1.1)
        ax.set_title(f"{n:,} transitions", pad=4)
        ax.set_xlabel("runs with promise violated (%)")
    axes[0].set_ylabel("normalised value")
    fig.tight_layout(w_pad=1.5)
    names = {
        "certified (ours)": "ours",
        "plug-in MDP (no guarantee)": "plug-in",
        "offline RL baselines": "baseline",
    }
    handles = [Line2D([], [], color=kinds[k][0], marker=kinds[k][1], ls="", ms=6, mec="white", label=text)
               for text, k in names.items()]  # fmt: skip
    legend_below(fig, handles, y=0.02)
    save(dom, "paper", fig, "value_vs_safety", pd.DataFrame(rows))


def data_needed(dom: Domain) -> None:
    """Transitions needed for a near-optimal policy against the number of observed states, with the
    log-log slope of each method."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    rows, handles = [], []
    for _, _, label, color, marker in SERIES:
        t = needed(dom, paper_series(dom, label))
        fin = np.isfinite(t["median"].to_numpy())
        st, med = t["states"].to_numpy()[fin], t["median"].to_numpy()[fin]
        slope = np.polyfit(np.log10(st), np.log10(med), 1)[0] if fin.sum() >= 3 else np.nan
        upper = np.minimum(t["q3"].to_numpy()[fin], 5e5)
        ax.fill_between(st, t["q1"].to_numpy()[fin], upper, color=color, alpha=0.12, lw=0)
        ax.plot(st, med, **paper_line(color, marker, ms=4))
        text = f"{label}  ({slope:.2f})" if np.isfinite(slope) else label
        handles.append(Line2D([], [], **paper_line(color, marker, text, ms=4.5)))
        rows.append(t.assign(method=label, slope=slope))
    states_axis(ax)
    ax.set_yscale("log")
    ax.set_xlabel("observed states")
    ax.set_ylabel(f"transitions to reach value {THRESHOLD}")
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="method (log-log slope)", title_fontsize=8,
              alignment="left", handlelength=1.8)  # fmt: skip
    fig.tight_layout()
    save(dom, "paper", fig, "data_needed", table(rows))


def graph_errors(dom: Domain) -> None:
    """At one size: the value, the share of true edges missed and the extra edges per variable."""
    labels = ["true graph", "PCMCI+", "DYNOTEARS", "sparse", "full graph"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), sharex=True)
    rows = []
    for label in labels:
        d = paper_series(dom, label)
        d = d[d["size"] == dom.graph_size]
        color, marker = next((c, m) for _, _, lab, c, m in SERIES if lab == label)
        for ax, col in zip(axes, ["value", "missing_share", "extra_per_var"], strict=False):
            if col != "value" and label == "true graph":
                continue
            c = cells(d, col)
            ax.fill_between(c["n_transitions"], c["lo"], c["hi"], color=color, alpha=0.15, lw=0)
            ax.plot(c["n_transitions"], c["mean"], **paper_line(color, marker))
            rows.append(c.assign(method=label, metric=col))
    for ax, ylabel in zip(
        axes, ["normalised value", "true edges missed (share)", "extra edges per variable"], strict=False
    ):
        paper_axis(ax)
        ax.set_xlabel("transitions")
        ax.set_ylabel(ylabel)
    axes[0].axhline(0, **ZERO_LINE)
    axes[0].set_ylim(-0.6, 1.08)
    axes[1].set_ylim(-0.03, 1.03)
    fig.tight_layout(w_pad=1.5)
    legend_below(fig, [series_handle(lab) for lab in labels], y=0.03)
    save(dom, "paper", fig, "graph_errors", table(rows))


# ==== 6. exploratory figures =======================================================================
def name_of(model: str) -> str:
    return {"flat": "flat IMDP"}.get(model, model)


def line_of(model: str) -> dict:
    return {"color": COLOR[model], "ls": LINE.get(model, "-"), "marker": "o"}


def handle_of(model: str) -> Line2D:
    return Line2D([], [], **line_of(model))


def top_legend(fig, handles, labels, ncol=4, y=1.0) -> None:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.8,
               columnspacing=1.4)  # fmt: skip


def log_x(ax, ns) -> None:
    ax.set_xscale("log")
    ticks = [n for n in (10, 100, 1000, 10_000, 100_000) if min(ns) <= n <= max(ns)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{n:,}".replace(",000", "k") for n in ticks])
    ax.minorticks_off()


def budgets(dom: Domain) -> list:
    return sorted(dom.runs["n_transitions"].unique())


def value_grid(dom: Domain, models: list, getter, name: str, title: str) -> None:
    """Value against data, one panel per size (up to 8 sizes), one line per model."""
    sizes = spread(dom.kept, 8)
    ncols = min(4, len(sizes))
    nrows = math.ceil(len(sizes) / ncols)
    height = 2.45 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * max(ncols, 3), height), sharex=True, sharey=True,
                             squeeze=False)  # fmt: skip
    rows, handles = [], []
    for i, model in enumerate(models):
        c = cells(getter(model)).assign(method=name_of(model))
        rows.append(c)
        for ax, size in zip(axes.flat, sizes, strict=False):
            s = c[c["size"] == size].sort_values("n_transitions")
            if s.empty:
                continue
            ax.plot(s["n_transitions"], s["mean"], **line_of(model), zorder=3 + i * 0.01)
            ax.fill_between(s["n_transitions"], s["lo"], s["hi"], color=COLOR[model], alpha=0.12, lw=0)
        handles.append(handle_of(model))
    for ax, size in zip(axes.flat, sizes, strict=False):
        ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_title(f"{dom.title(size)}  ({dom.states[size]} states)", loc="left")
        ax.set_ylim(-1.15, 1.1)
        log_x(ax, budgets(dom))
    for ax in axes.flat[len(sizes) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("transitions in the dataset")
    for ax in axes[:, 0]:
        ax.set_ylabel("normalised value")
    note = "(normalised value: 0 = random policy, 1 = best policy; mean over seeds, band: 95% CI)"
    fig.tight_layout(h_pad=1.2)
    legend_rows = math.ceil(len(models) / 4)  # the legend sits above the panels, the title above the legend
    top_legend(fig, handles, [name_of(m) for m in models], ncol=4, y=0.98)
    above = 0.98 + (0.2 * legend_rows + 0.1) / height
    fig.suptitle(f"{title}\n{note}", x=0.01, ha="left", va="bottom", y=above,
                 fontsize=11 if ncols >= 4 else 9.5, color=INK, linespacing=1.5)  # fmt: skip
    save(dom, "explore", fig, name, table(rows))


def value_graph_sources(dom: Domain) -> None:
    value_grid(dom, GRAPH_SOURCES, dom.certified, "1_value_graph_sources",
               f"Value of the certified policy ({dom.solver}) by graph source")  # fmt: skip


def value_kwik(dom: Domain) -> None:
    value_grid(dom, KWIK, dom.certified, "2_value_kwik",
               f"Value of the certified policy ({dom.solver}): KWIK learners vs. the true graph")  # fmt: skip


def value_vs_baselines(dom: Domain) -> None:
    def getter(model):
        return dom.certified(model) if model in GRAPH_SOURCES else dom.pick(model)

    value_grid(dom, ["true graph", "PCMCI+ (lagged)"] + BASELINES, getter, "3_value_vs_baselines",
               f"Ours (solid: certified, {dom.solver}) vs. offline-RL baselines (dashed)")  # fmt: skip


def promise_held(dom: Domain) -> None:
    """Heatmap: share of finished runs whose promise held, per model and number of transitions."""
    relaxation = dom.solver.partition("(")[2].rstrip(")") or "exact"
    panels = {
        f"{dom.solver}\nours, certified": dom.certified,
        "MDP (plug-in)\nno guarantee": lambda m: dom.pick(m, "MDP (plug-in)"),
        f"MDP (best, {relaxation})\noptimistic": lambda m: dom.pick(
            m, "MDP (best)" if m == "flat" else f"MDP (best, {relaxation})"
        ),
        "Factored SPIBB": lambda m: dom.pick(m, "Factored SPIBB"),
    }
    models, ns = GRAPH_SOURCES + KWIK[1:], budgets(dom)
    cmap = LinearSegmentedColormap.from_list("held", [RED, "#f0efec", BLUE])
    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(
        2, len(panels), height_ratios=[len(models), len(BASELINES)], hspace=0.35, wspace=0.08
    )
    rows = []

    def held(x: pd.DataFrame, **names) -> pd.Series:
        x = x[x["ok"] & x["size"].isin(dom.kept)]
        by_n = x.groupby("n_transitions")["safe"]
        rows.append(by_n.agg(["mean", "count"]).reset_index().assign(**names))
        return by_n.mean().reindex(ns)

    def draw(ax, grid: pd.DataFrame, ylabels, title: str):
        im = ax.imshow(grid.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (r, c), v in np.ndenumerate(grid.to_numpy(dtype=float)):
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#ffffff", edgecolor=GRID,
                                           hatch="////", lw=0))  # fmt: skip
                continue
            text = "" if v >= 0.995 else f"{v * 100:.0f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=6.3, color="#ffffff" if v < 0.25 else INK)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"{n:g}" if n < 1000 else f"{n // 1000:g}k" for n in ns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=7.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=8.5)
        return im

    for j, (title, getter) in enumerate(panels.items()):
        ax = fig.add_subplot(gs[0, j])
        solver = title.split("\n")[0]
        grid = pd.DataFrame({name_of(m): held(getter(m), method=name_of(m), solver=solver) for m in models}).T
        im = draw(ax, grid, list(grid.index) if j == 0 else [""] * len(grid), title)
    ax = fig.add_subplot(gs[1, 0])
    grid = pd.DataFrame({m: held(dom.pick(m), method=m, solver=m) for m in BASELINES}).T
    draw(ax, grid, BASELINES, "baselines (their own value estimate as the promise)")
    ax.set_xlabel("transitions in the dataset", fontsize=8)
    cax = fig.add_axes([0.42, 0.13, 0.2, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("share of runs where promise ≤ true value  (no number: 100%; hatched: not run)", fontsize=8,
                 color=INK2)  # fmt: skip
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0%", "50%", "100%"])
    fig.suptitle(f"Does the promised value hold?  (sizes {dom.size_range()} pooled, finished runs)", x=0.1,
                 ha="left", y=0.97, fontsize=11)  # fmt: skip
    save(dom, "explore", fig, "4_promise_held", table(rows))


def promise_vs_true(dom: Domain) -> None:
    """Promise and true value of the certified policy at one size, one panel per graph source."""
    size, ns = dom.detail_size, budgets(dom)
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.9), sharex=True, sharey=True)
    rows = []
    quantities = [("value", BLUE, "-", "true value"), ("promise_value", RED, (0, (4, 2)), "promise")]
    for ax, model in zip(axes.flat, GRAPH_SOURCES, strict=False):
        x = dom.certified(model)
        x = x[x["size"] == size]
        for col, color, ls, quantity in quantities:
            c = cells(x, col)
            ax.plot(c["n_transitions"], c["mean"], color=color, ls=ls, marker="o")
            ax.fill_between(c["n_transitions"], c["lo"], c["hi"], color=color, alpha=0.12, lw=0)
            rows.append(c.assign(method=name_of(model), quantity=quantity))
        ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(name_of(model), loc="left")
        log_x(ax, ns)
    for ax in axes[1]:
        ax.set_xlabel("transitions in the dataset")
    for ax in axes[:, 0]:
        ax.set_ylabel("normalised value")
    handles = [Line2D([], [], color=BLUE, marker="o"), Line2D([], [], color=RED, ls=(0, (4, 2)), marker="o")]
    fig.suptitle(f"Promise vs. true value of the certified policy ({dom.solver})  ·  {dom.title(size)} "
                 f"({dom.states[size]} states)", x=0.01, ha="left", y=1.08, fontsize=11)  # fmt: skip
    top_legend(fig, handles, ["true value of the policy", "promise (certified lower bound)"], ncol=2, y=0.98)
    fig.tight_layout(h_pad=1.2)
    save(dom, "explore", fig, "5_promise_vs_true", table(rows))


def solvers(dom: Domain) -> None:
    """The three relaxations on the true graph: promise and value at one size, solve time by size."""
    size, ns = dom.detail_size, budgets(dom)
    names = {"IMDP (IA)": BLUE, "IMDP (McCormick)": PINK, "IMDP (exact)": PURPLE}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    rows = []
    for solver, color in names.items():
        x = dom.pick("true graph", solver)
        for ax, col in zip(axes[:2], ["promise_value", "value"], strict=False):
            c = cells(x[x["size"] == size], col)
            ax.plot(c["n_transitions"], c["mean"], color=color, marker="o")
            rows.append(c.assign(method=solver, panel=col))
        y = x[x["ok"] & (x["n_transitions"] == N_FIXED)].groupby("size")["solve_s"].median().reindex(dom.kept)
        axes[2].plot([dom.states[s] for s in y.index], y.to_numpy(), color=color, marker="o")
        rows.append(y.reset_index().assign(method=solver, panel="solve_s"))
    for ax, text in zip(
        axes[:2], ["promise (certified lower bound)", "true value of the policy"], strict=False
    ):
        log_x(ax, ns)
        ax.set_xlabel("transitions in the dataset")
        ax.set_title(f"{text}, {dom.title(size)}", loc="left")
        ax.axhline(1, color=MUTED, lw=0.8, ls=(0, (4, 3)))
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("normalised value")
    axes[2].set(yscale="log", xlabel="observed states", ylabel="seconds (median)")
    states_axis(axes[2])
    axes[2].set_title(f"solve time, {N_FIXED:,} transitions", loc="left")
    fig.suptitle("Robust solvers on the true graph", x=0.01, ha="left", y=1.12, fontsize=11)
    top_legend(fig, [Line2D([], [], color=c, marker="o") for c in names.values()],
               ["interval arithmetic (box)", "McCormick", "exact (vertex products)"], 3, 0.98)  # fmt: skip
    fig.tight_layout()
    save(dom, "explore", fig, "6_solvers", table(rows))


def discovery(dom: Domain) -> None:
    """The learned graph against the true one: recall, precision and structural Hamming distance."""
    models = ["PCMCI+", "PCMCI+ (lagged)", "DYNOTEARS", "DYNOTEARS (lagged)", "Strehl (fixed hull)",
              "Met-Rmax (fixed hull)", "sparse", "full-fact"]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    rows = []
    for model in models:
        x = dom.certified(model)
        x = x[x["size"].isin(dom.kept)]
        for ax, col in zip(axes, ["recall", "precision", "shd"], strict=False):
            c = cells(x, col, by=["n_transitions"])
            ax.plot(c["n_transitions"], c["mean"], **line_of(model))
            rows.append(c.assign(method=model, metric=col))
    titles = [
        "recall (true edges found)",
        "precision (found edges that are true)",
        "structural Hamming distance",
    ]
    for ax, text in zip(axes, titles, strict=False):
        log_x(ax, budgets(dom))
        ax.set_xlabel("transitions in the dataset")
        ax.set_title(text, loc="left")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    fig.suptitle(f"Learned graph vs. true graph  (mean over sizes {dom.size_range()} and seeds)", x=0.01,
                 ha="left", y=1.18, fontsize=11)  # fmt: skip
    top_legend(fig, [handle_of(m) for m in models], models, ncol=4, y=0.98)
    fig.tight_layout()
    save(dom, "explore", fig, "7_discovery", table(rows))


def scalability(dom: Domain) -> None:
    """Over every size that ran: the share of seeds that finish, and the median time, at N_FIXED."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    rows, st = [], [dom.states[s] for s in dom.sizes]
    for model in GRAPH_SOURCES:
        x = dom.certified(model)
        x = x[x["n_transitions"] == N_FIXED]
        done = x.groupby("size")["ok"].mean().reindex(dom.sizes)
        seconds = x[x["ok"]].groupby("size")["total_s"].median().reindex(dom.sizes)
        axes[0].plot(st, done.to_numpy() * 100, **line_of(model))
        axes[1].plot(st, seconds.to_numpy(), **line_of(model))
        rows.append(pd.DataFrame({"size": dom.sizes, "states": st, "finished": done.to_numpy(),
                                  "median_s": seconds.to_numpy(), "method": name_of(model)}))  # fmt: skip
    for ax in axes:
        states_axis(ax)
        ax.set_xlabel("observed states")
    axes[0].set(ylabel="% of seeds finished", ylim=(-3, 103))
    axes[0].set_title(f"runs that finish (limits: time, 6 GB), {N_FIXED:,} transitions", loc="left")
    axes[1].set(yscale="log", ylabel="seconds (median)")
    axes[1].set_title("discovery + learning + solve time", loc="left")
    fig.suptitle(
        f"Scalability of the certified pipeline ({dom.solver})", x=0.01, ha="left", y=1.17, fontsize=11
    )
    top_legend(
        fig, [handle_of(m) for m in GRAPH_SOURCES], [name_of(m) for m in GRAPH_SOURCES], ncol=4, y=0.98
    )
    fig.tight_layout()
    save(dom, "explore", fig, "8_scalability", table(rows))


def transitions_needed(dom: Domain) -> None:
    """Transitions needed for a near-optimal policy: ours (solid) against three baselines (dashed)."""
    ours = ["true graph", "PCMCI+ (lagged)", "DYNOTEARS (lagged)", "sparse", "flat"]
    entries = [(m, dom.certified) for m in ours] + [(m, dom.pick) for m in ("DiFaRL", "FOCUS", "offline DQN")]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    rows = []
    for model, getter in entries:
        t = needed(dom, getter(model))
        fin = np.isfinite(t["median"].to_numpy())
        st = t["states"].to_numpy()[fin]
        ax.plot(st, t["median"].to_numpy()[fin], **line_of(model), label=name_of(model))
        ax.fill_between(st, t["q1"].to_numpy()[fin], np.minimum(t["q3"].to_numpy()[fin], 5e5),
                        color=COLOR[model], alpha=0.08, lw=0)  # fmt: skip
        rows.append(t.assign(method=name_of(model)))
    ax.set(yscale="log", xlabel="observed states", ylabel=f"transitions to reach value ≥ {THRESHOLD}")
    states_axis(ax)
    note = "(solid: ours, certified; dashed: baselines; median over seeds, band: IQR;"
    stops = "a line stops where most seeds never reach it)"
    ax.set_title(f"Data needed for a near-optimal policy\n{note}\n{stops}", loc="left", fontsize=8.5)
    ax.legend(ncol=2, loc="upper left", fontsize=7.8)
    fig.tight_layout()
    save(dom, "explore", fig, "9_transitions_needed", table(rows))


# ==== 7. the command line ==========================================================================
# name -> (group, function, the sources whose runs it needs: without them the figure is skipped)
GRAPHS_NEEDED = ["truegraph", "pcmci_lagged", "dynotears_lagged", "sparse", "fullfact", "flat"]
FIGURES = {
    "value": ("paper", value, GRAPHS_NEEDED),
    "baselines": ("paper", baselines, ["truegraph", "sota"]),
    "coverage": ("paper", coverage, GRAPHS_NEEDED),
    "promise_scatter": ("paper", promise_scatter, ["truegraph", "pcmci_lagged", "dynotears_lagged", "sota"]),
    "value_vs_safety": ("paper", value_vs_safety, GRAPHS_NEEDED + ["sota"]),
    "data_needed": ("paper", data_needed, GRAPHS_NEEDED),
    "graph_errors": ("paper", graph_errors, GRAPHS_NEEDED[:-1]),
    "1_value_graph_sources": ("explore", value_graph_sources, ["truegraph"]),
    "2_value_kwik": ("explore", value_kwik, ["truegraph", "strehl", "met"]),
    "3_value_vs_baselines": ("explore", value_vs_baselines, ["truegraph", "sota"]),
    "4_promise_held": ("explore", promise_held, ["truegraph", "sota"]),
    "5_promise_vs_true": ("explore", promise_vs_true, ["truegraph"]),
    "6_solvers": ("explore", solvers, ["truegraph"]),
    "7_discovery": ("explore", discovery, ["pcmci", "dynotears"]),
    "8_scalability": ("explore", scalability, ["truegraph"]),
    "9_transitions_needed": ("explore", transitions_needed, ["truegraph", "sota"]),
}
STYLES = {"paper": PAPER_STYLE, "explore": EXPLORE_STYLE}


def plot_domain(
    name: str, results: Path, out: Path, solver: str | None = None, only: list | None = None
) -> list[str]:
    """Draw the figures of one domain; returns the names of the figures written. solver None: the
    domain's (DOMAIN_SETTINGS), else DEFAULT_SOLVER."""
    solver = solver or DOMAIN_SETTINGS.get(name, {}).get("solver", DEFAULT_SOLVER)
    print(f"\n===== {name}: {results / name} -> {out / name} =====")
    runs = load(results, name)
    if runs.empty:
        print(f"  no results in {results / name}: nothing to draw")
        return []
    dom = describe(name, runs, solver, out)
    sources = set(runs["source"])
    print(f"  sources: {', '.join(sorted(sources - {'oracle'}))}")
    if not dom.kept:
        print(f"  the reference (true graph -> {solver}) finished at no size: nothing to draw")
        return []
    print(f"  certified solver: {solver}")
    print(f"  sizes kept: {dom.size_range()} ({len(dom.kept)} of {len(dom.sizes)}); panels: "
          f"{', '.join(str(dom.labels[s]) for s in dom.panels)}")  # fmt: skip
    written = []
    for figure, (group, draw, required) in FIGURES.items():
        if only and figure not in only:
            continue
        missing = [s for s in required if s not in sources]
        if missing:
            print(f"  skipped {group}/{figure}: no run of {', '.join(missing)} yet")
            continue
        with mpl.rc_context(STYLES[group]):
            draw(dom)
        written.append(figure)
    return written


def domains_with_results(results: Path) -> list[str]:
    return (
        sorted(p.name for p in results.iterdir() if p.is_dir() and latest_runs(p, p.name))
        if results.is_dir()
        else []
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27: the figures of one or more domains.")
    parser.add_argument("domains", nargs="*", help="e.g. sysadmin sepsis drone")
    parser.add_argument("--list", action="store_true", help="print the figure names and exit")
    parser.add_argument("--only", nargs="+", metavar="FIGURE", help="draw only these figures")
    parser.add_argument("--solver", help='the certified solver, e.g. "IMDP (IA)" (default: the domain\'s)')
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results" / EXPERIMENT, help="the results folder"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "outputs" / EXPERIMENT, help="where the figures go"
    )
    args = parser.parse_args(argv)
    if args.list:
        for figure, (group, draw, _) in FIGURES.items():
            print(f"{group:<8} {figure:<24} {(draw.__doc__ or figure).strip().splitlines()[0]}")
        return 0
    wrong = [f for f in args.only or [] if f not in FIGURES]
    if wrong:
        parser.error(f"no figure {wrong}; the figures are: {', '.join(FIGURES)}")
    known = domains_with_results(args.results)
    if not args.domains:
        parser.error(f"give a domain; with results in {args.results}: {', '.join(known) or 'none'}")
    for name in args.domains:
        if name not in known:
            parser.error(
                f"no results of {name!r} in {args.results}; with results: {', '.join(known) or 'none'}"
            )
    for name in args.domains:
        plot_domain(name, args.results, args.out, args.solver, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
