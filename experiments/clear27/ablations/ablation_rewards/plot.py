"""Figures of the ablation "the methods along a ladder of rewards" (run.py), one domain at a time.

Run from the repository root, after (or during) run.py:
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py                  # SysAdmin
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --domain sepsis  # or drone
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --sizes 3
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --run <run folder>
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --domain all_domains

Input:  results/clear27/ablations/ablation_rewards/<domain>/<domain>_<date>_<time>/ (run.py): every run
        folder of the domain, read together (below); with --run, that run and the runs it extends
Output: outputs/clear27/ablations/ablation_rewards/<domain>/
            <size>=<s>/<figure>.{pdf,png}   one set per size (<size>: N, vitals or side)
            summary/<figure>.{pdf,png}      across the sizes
            paper/..., appendix/...         the thirty-four pooled figures (below), in their two sets
            data/<where>_<figure>.csv       the numbers drawn (a pooled figure: <set>_<figure>.csv)
        outputs/clear27/ablations/ablation_rewards/all_domains/{paper,appendix,data}/ with --domain
            all_domains: the pooled figures over the runs of SysAdmin, Drone and Sepsis together

Three rows of panels (PANELS; a row shows the methods the run has, and is left out when it has none):
    with a promise   the interval MDPs: DYNOTEARS, PCMCI+, the sparse and the full graph, the flat model,
                     the true graph
    no promise       their plug-in MDPs (the same colours), and SLF-Rmax and Met-Rmax as their papers run them
    other methods    SPIBB on a graph (on a learned graph: a baseline, with its own colour; on the true
                     graph: the colour of the graph), SL-SPIBB, flat SPIBB and FOCUS
The figures of the promise (promise, held, promised) have the first row only: the others give no bound
(a run with no method of the first row has no such figure).
The runs of a domain are parts of one experiment and are drawn together: a run with more methods on the
same problems (run.py --extends), with more sizes, or with more numbers of transitions. Their rows are
put together (a row of two runs is the later run's), and so are the times of their jobs. Where a run is
behind (a size or a seed it has not finished), its methods, sizes or data sizes are missing. A job that
did not finish (a .partial file, no .json beside it) is not read.
The sizes of LEFT_OUT are in no figure: SysAdmin N = 10 (no job finished: the workers died) and N = 11
(2 of 10 seeds, every step out of memory).

Figures, per size (x: the level of the ladder, 1 = the original reward ... 6 = traps; one column per
number of transitions):
  value_by_level     the normalised value of the policy
  solved_by_level    the share of runs with a normalised value of at least SOLVED (0.9, clear27's
                     THRESHOLD)
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
Pooled (paper/ and appendix/): the thirty-four figures of clear27's plot.py (its section 6: imdp_by_graph/,
mdp_by_graph/ and vs_baselines/ with solved_vs_transitions, solved_vs_optimality and solved_vs_time, and
tradeoff, tradeoff_pareto, tradeoff_data_needed, reported_vs_true and reported_vs_true_common: here one
dot per run of every reward, in the nine panels of the methods the ablation has), drawn by the same
code from the instances of this ablation, in its two sets: paper/ with DYNOTEARS as ours ("CD") and
without PCMCI+, appendix/ with DYNOTEARS and PCMCI+ both as ours. An instance is a (size, seed, reward),
every size of the run; it is
solved with a normalised value of at least SOLVED and safe when its promise (or, without one, its own
estimate) held; its time is learn_s + replan_s. --sizes does not change them.
The colour and marker of a method are those of clear27's registry (paper.METHODS), in every figure.

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
MAIN = ROOT / "results" / "clear27"  # the main runs: panel (a) of figure 5 (paper.tradeoff_and_rewards)
MAIN_OUTPUTS = ROOT / "outputs" / "clear27"  # figure 5 goes beside the main figures: <set>/ of all_domains
STAMP = "%Y-%m-%d_%H-%M-%S"
# "promised": at least this normalised promise. 0 is the value of the random policy; the promise of the
# box is far below the value (on SysAdmin it never reaches SOLVED), so SOLVED here would draw zeros
PROMISED = 0.0
# domain -> sizes dropped when its runs are loaded: in no figure. SysAdmin N = 10 finished no job (its
# workers were killed twice, 2026-11-14/15) and N = 11 finished 2 of 10 seeds with every step stopped by the
# memory limit: neither says anything about a method. The result files stay where they are.
LEFT_OUT = {"sysadmin": [10, 11]}


def _paper():
    """clear27's plot.py: the style and colours of the paper figures."""
    spec = importlib.util.spec_from_file_location("clear27_plot", HERE.parents[2] / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _paper()
SOLVED = paper.THRESHOLD  # "solved": at least this normalised value, as in the figures of clear27
ALL_DOMAINS = paper.ALL_DOMAINS  # --domain all_domains: the pooled figures over paper.POOLED_DOMAINS
MDP = " MDP"  # run.py's suffix of a structure's plug-in MDP
SPIBB = " SPIBB"  # and of Factored SPIBB on a structure
PANELS = {
    "with a lower bound\n(interval MDP)": [
        "DYNOTEARS", "PCMCI+", "sparse", "full graph", "flat", "true graph"],
    "no lower bound\n(point MDP)": [
        "DYNOTEARS MDP", "PCMCI+ MDP", "sparse MDP", "full graph MDP", "flat MDP", "true graph MDP",
        "SLF-Rmax", "Met-Rmax"],
    "other methods\n(no lower bound)": [
        "DYNOTEARS SPIBB", "PCMCI+ SPIBB", "true graph SPIBB", "SL-SPIBB", "flat SPIBB", "FOCUS"],
    "structure learners of\nfactored RL (other forms)": [
        "SLF-Rmax KWIK", "Met-Rmax KWIK", "SLF-Rmax (theorem)", "Met-Rmax (theorem)",
        "SLF-Rmax KWIK (theorem)", "Met-Rmax KWIK (theorem)"],
    "model-free\n(retrained per reward)": ["DiFaRL", "offline DQN", "offline CQL"],
}  # fmt: skip
# a form of a KWIK learner is drawn in the colour and marker of the learner, with a line of its own
TWIN = {name: name.split(" ")[0] for name in PANELS["structure learners of\nfactored RL (other forms)"]}
FORM_LINE = {
    "SLF-Rmax KWIK": "--", "Met-Rmax KWIK": "--", "SLF-Rmax (theorem)": ":", "Met-Rmax (theorem)": ":",
    "SLF-Rmax KWIK (theorem)": "-.", "Met-Rmax KWIK (theorem)": "-.",
}  # fmt: skip
# method -> (colour, marker): those of clear27's registry (paper.METHODS), one per method in every figure;
# a structure's plug-in MDP is drawn as the structure (that of DYNOTEARS: in its own colour,
# paper.OWN_COLOUR); SPIBB on a learned graph is a method of its own
STYLE = {
    method: paper.look_of(TWIN.get(method, method))[:2]
    for methods in PANELS.values()
    for method in methods
}
# DYNOTEARS and PCMCI+ often find the same graph, so their lines coincide: DYNOTEARS is drawn wider and
# behind, and stays visible around the line of PCMCI+
WIDE = {"DYNOTEARS" + form: dict(lw=3.4, ms=6.0, zorder=1.5) for form in ("", MDP)}
# What the figures read from the run's config.json (main): the names of the levels and of a size
RUN = {
    "levels": {1: "additive", 2: "subset", 3: "cost", 4: "threshold", 5: "goal", 6: "trap"},
    "size_name": "machines N",
    "prefix": "N",
}
# quantity -> (column, y label, limits, in percent)
QUANTITIES = {
    "value": ("value", "normalised value", (-1.3, 1.1), False),
    "promise": ("promise", "normalised lower bound", (-3.2, 1.1), False),
    "held": ("held", "runs\nreliable (%)", (-4, 104), True),
    "solved": ("solved", f"runs with value\n≥ {SOLVED} (%)", (-4, 104), True),
    "promised": ("promised", f"runs with lower bound\n≥ {PROMISED:g} (%)", (-4, 104), True),
}
OF_THE_PROMISE = ("promise", "held", "promised")  # the first row of panels only


def panels_of(quantity: str, runs: pd.DataFrame) -> dict:
    """The rows of panels of a figure, each with its methods that `runs` has (a row with none is left
    out); for a quantity of the promise, the row of the promise only (no row when no method has one)."""
    present = set(runs["method"].unique())
    rows = dict(list(PANELS.items())[:1]) if quantity in OF_THE_PROMISE else PANELS
    rows = {title: [m for m in methods if m in present] for title, methods in rows.items()}
    return {title: methods for title, methods in rows.items() if methods}


# ---- loading --------------------------------------------------------------------------------------
def runs_of(results: Path, domain: str) -> list[Path]:
    """The run folders of the domain (<domain>_<date>_<time>) with a finished job, the earliest first."""
    found = []
    for path in results.glob(f"{domain}_*") if results.is_dir() else []:
        try:
            datetime.strptime(path.name[len(domain) + 1 :], STAMP)
        except ValueError:
            continue
        if any(path.glob("*=*/runs/*.json")):
            found.append(path)
    return sorted(found)


def latest_run(results: Path, domain: str) -> Path | None:
    found = runs_of(results, domain)
    return found[-1] if found else None


def extended(run: Path) -> list[Path]:
    """The run and, before it, the runs it extends (config.json's "extends", a folder beside the run):
    the base first. A base that is not there (or has no finished job) is told and left out."""
    chain = [run]
    while True:
        config = chain[0] / "config.json"
        name = json.loads(config.read_text()).get("extends") if config.exists() else None
        if not name:
            return chain
        base = run.parent / name
        if base in chain or not any(base.glob("*=*/runs/*.json")):
            print(f"  the run extends {name}, which has no finished job in {run.parent}: drawn alone")
            return chain
        chain.insert(0, base)


def read(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, timing) of one run folder as written: every finished (size, seed), with the name of the
    run in `run`. A run of before the domains has the column `machines`: it is read as `size`."""
    done = sorted(run.glob("*=*/runs/*.json"))
    runs = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in done], ignore_index=True)
    timing = pd.DataFrame([json.loads(p.read_text()) for p in done])
    runs, timing = (t.rename(columns={"machines": "size"}).assign(run=run.name) for t in (runs, timing))
    return runs, timing


def load(folders: list[Path], domain: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(runs, timing): every finished (size, seed) of the run folders, the earliest first (runs_of: the
    parts of a domain's experiment; or extended: a run after those it extends). A row of two runs is the
    later one's; timing keeps one row per run and (size, seed). The sizes of LEFT_OUT[domain] are dropped.
    runs gets `ok`, `held` as 0/1, `solved` and `promised` (the value at least SOLVED, the promise at
    least PROMISED; a stopped run is neither)."""
    tables = [read(folder) for folder in folders]
    runs = pd.concat([r for r, _ in tables], ignore_index=True)
    timing = pd.concat([t for _, t in tables], ignore_index=True)
    left_out = LEFT_OUT.get(domain, [])
    runs = runs[~runs["size"].isin(left_out)]
    timing = timing[~timing["size"].isin(left_out)].reset_index(drop=True)
    key = ["size", "seed", "n_transitions", "method", "reward"]
    runs = runs.drop_duplicates(key, keep="last").reset_index(drop=True)
    runs["ok"] = runs["status"].eq("ok")
    runs["held"] = runs["held"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    runs["solved"] = (runs["ok"] & (runs["value"] >= SOLVED)).astype(float)
    runs["promised"] = (runs["ok"] & (runs["promise"] >= PROMISED)).astype(float)
    return runs, timing


def instances(runs: pd.DataFrame, domain: str) -> pd.DataFrame:
    """The table of instances of the pooled figures (clear27's plot.py, section 6) from the runs of
    load: an instance is a (size, seed, reward); the methods have the names of run.py, which are those of
    the pooled figures; the time is that of the model (once per dataset) plus the planning for the reward;
    reported and true_value (reported_vs_true): the normalised value the method reported, and the true one."""
    return pd.DataFrame({
        "domain": domain, "size": runs["size"], "seed": runs["seed"], "reward": runs["reward"],
        "n_transitions": runs["n_transitions"], "method": runs["method"], "ok": runs["ok"],
        "optimality": runs["value"], "safe": runs["held"].eq(1),
        "seconds": runs.reindex(columns=["learn_s", "replan_s"]).fillna(0).sum(axis=1),
        "reported": runs["promise"], "true_value": runs["value"],
    })  # fmt: skip


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
    (sizes of the run that were not asked for) stay, and so do the pooled figures (paper/, appendix/ and
    their csv files, named after the set). Returns the removed files."""
    kept = {f"{RUN['prefix']}={size}" for size in kept_sizes}
    found = [
        p for where in ("*=*", "summary") for ext in ("pdf", "png") for p in out.glob(f"{where}/*.{ext}")
    ]
    found += [p for p in out.glob("data/*.csv") if p.name.split("_")[0] not in paper.SETS]
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
    form = {"ls": FORM_LINE[method]} if method in FORM_LINE else {}
    return paper.paper_line(*STYLE[method], label, **(more | form | WIDE.get(method, {})))


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
    panels = panels_of(quantity, runs)
    if not panels:  # a figure of the promise, and no method with one
        return
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
    panels = panels_of("value", d)
    size = (1.9 * max(len(levels), 2) + 1.6, 2.2 * len(panels))
    fig, axes = plt.subplots(len(panels), len(levels), figsize=size, sharex=True,
                             sharey=True, squeeze=False)  # fmt: skip
    for row, methods in enumerate(panels.values()):
        for ax, level in zip(axes[row], levels, strict=True):
            draw_lines(ax, table[table["level"] == level], "size", positions, methods, False)
            ax.axhline(0, **paper.ZERO_LINE)
            ax.set_ylim(*QUANTITIES["value"][2])
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels(sizes)
            if row == 0:
                ax.set_title(f"{level}: {RUN['levels'][level]}", pad=4)
            if row == len(panels) - 1:
                ax.set_xlabel(RUN["size_name"])
        axes[row, 0].set_ylabel("normalised value")
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    panel_legends(fig, axes, panels)
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
    panels = panels_of("time", runs)
    fig, axes = plt.subplots(len(panels), len(STEPS), figsize=(2.75 * len(STEPS) + 1.6, 2.6 * len(panels)),
                             sharex=True, sharey=True, squeeze=False)  # fmt: skip
    rows = []
    for col, (column, (title, per)) in enumerate(STEPS.items()):
        source = models if per == "model" else runs
        table = source.groupby(["method", "size"])[column].mean().reset_index(name="seconds")
        for row, methods in enumerate(panels.values()):
            ax = axes[row, col]
            for method in methods:
                t = table[(table["method"] == method) & table["seconds"].notna()].sort_values("size")
                ax.plot(t["size"], t["seconds"], **line_of(method))
            ax.set_yscale("log")
            ax.set_xticks(sizes)
            if row == 0:
                ax.set_title(title, pad=4, fontsize=9)
            if row == len(panels) - 1:
                ax.set_xlabel(RUN["size_name"])
        rows.append(table.assign(step=column))
    for ax in axes[:, 0]:
        ax.set_ylabel("seconds (mean)")
    fig.tight_layout(w_pad=1.0)
    panel_legends(fig, axes, panels)
    selection = models.groupby(["method", "size"])["selection_s"].mean().reset_index(name="seconds")
    rows.append(selection.assign(step="selection_s"))
    total = timing.groupby("size").sum(numeric_only=True).drop(columns=["seed"]).reset_index()
    save(out, "summary", fig, "time", pd.concat([*rows, total.assign(step="job totals")]))


PER_SIZE = [f"{quantity}_by_level" for quantity in QUANTITIES]
SUMMARY = ["value_by_size", "solved_by_level", "promised_by_level", "time"]


def all_domains(results: Path, out: Path, parser, main: Path = MAIN, main_out: Path = MAIN_OUTPUTS) -> int:
    """The pooled figures (both sets) over the instances of every domain of paper.POOLED_DOMAINS (all
    its runs together), into out/all_domains; and figure 5 of the paper (paper.tradeoff_and_rewards: the
    main runs beside these), into main_out/all_domains when the main results have every domain."""
    tables = []
    for domain in paper.POOLED_DOMAINS:
        folders = runs_of(results / domain, domain)
        if not folders:
            parser.error(f"{ALL_DOMAINS} needs a finished run of {domain}, and {results / domain} has none")
        print(f"===== {NAME}, {ALL_DOMAINS}: {', '.join(f.name for f in folders)} =====")
        tables.append(instances(load(folders, domain)[0], domain))
    rewards = pd.concat(tables, ignore_index=True)
    paper.pooled_figures(rewards, out / ALL_DOMAINS)
    mains = [paper.instances_of(paper.load(main, domain), domain) for domain in paper.POOLED_DOMAINS]
    if any(t.empty for t in mains):
        print(f"  figure 5 (tradeoff_and_rewards) not drawn: {main} lacks the main runs of a domain")
        return 0
    print(f"===== figure 5: {main} + {results} -> {main_out / ALL_DOMAINS} =====")
    with mpl.rc_context(paper.PAPER_STYLE):
        for name, ours in paper.SETS.items():
            with paper.references_named():
                paper.tradeoff_and_rewards(pd.concat(mains, ignore_index=True), rewards,
                                           paper.FigureSet(name, ours, main_out / ALL_DOMAINS))  # fmt: skip
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Figures of the ablation: the methods along a ladder of rewards."
    )
    parser.add_argument("--domain", default="sysadmin", help="sysadmin (default), sepsis or drone; "
                        f"{ALL_DOMAINS}: the pooled figures over the three")  # fmt: skip
    parser.add_argument("--run", type=Path, help="one run folder, with the runs it extends (default: "
                        "every run of the domain together)")  # fmt: skip
    parser.add_argument("--sizes", "--machines", dest="sizes", nargs="+", type=int, metavar="N",
                        help="sizes (default: all of the run)")  # fmt: skip
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out", type=Path, default=OUTPUTS, help="where the figures go (a folder per domain)"
    )
    parser.add_argument("--main", type=Path, default=MAIN, help=argparse.SUPPRESS)
    parser.add_argument("--main-out", type=Path, default=MAIN_OUTPUTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.domain == ALL_DOMAINS:
        return all_domains(args.results, args.out, parser, args.main, args.main_out)
    run = args.run or latest_run(args.results / args.domain, args.domain)
    if run is None or not run.is_dir() or not any(run.glob("*=*/runs/*.json")):
        parser.error(f"no finished run of {args.domain} in {args.results / args.domain}: start run.py first")
    domain = read_config(run).get("domain", args.domain)
    out = args.out / domain
    runs, timing = load(extended(run) if args.run else runs_of(run.parent, args.domain), domain)
    if runs.empty:
        parser.error(f"every finished size of {run} is left out (LEFT_OUT): nothing to draw")
    every_size = runs
    sizes = sorted(runs["size"].unique())
    missing = [m for m in args.sizes or [] if m not in sizes]
    if missing:
        parser.error(f"no finished job of size {missing} in {run}; sizes with results: {sizes}")
    chosen = args.sizes or sizes
    runs, timing = runs[runs["size"].isin(chosen)], timing[timing["size"].isin(chosen)]
    bases = [name for name in runs["run"].unique() if name != run.name]
    print(f"===== {NAME}, {domain}: {run}{' with ' + ', '.join(bases) if bases else ''} -> {out} =====")
    unknown = sorted(set(runs["method"]) - {m for methods in PANELS.values() for m in methods})
    if unknown:
        print(f"  not drawn (in no row of PANELS): {', '.join(unknown)}")
        runs = runs[~runs["method"].isin(unknown)]
        if runs.empty:
            parser.error(f"{run} has no method that the figures draw")
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
    paper.pooled_figures(instances(every_size, domain), out)
    stale = remove_stale(out, [m for m in sizes if m not in chosen])
    if stale:
        print(f"  removed {len(stale)} stale file(s) of an earlier run: "
              + ", ".join(str(p.relative_to(out)) for p in stale))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
