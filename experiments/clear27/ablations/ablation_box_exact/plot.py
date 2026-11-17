"""Figures of the ablation "box against exact" (run.py): the interval MDP of the true graph and of DYNOTEARS
compiled into the box (the main runs) and into the exact set (this ablation) (S7, S8, S9).

Run from the repository root:
    .venv/bin/python experiments/clear27/ablations/ablation_box_exact/plot.py      # every domain,
                                                                                   # and all_domains
    .venv/bin/python experiments/clear27/ablations/ablation_box_exact/plot.py sysadmin all_domains

Input:  results/clear27/<domain>/<domain>_{truegraph,dynotears_lagged}_<date>_<time>/   the box: sources of
            the main runs, read where they are (their runs before 2026-11-17 also have the exact set)
        results/clear27/ablations/ablation_box_exact/<domain>/<domain>_<source>_exact_<date>_<time>/
            the exact set: this ablation
Output: outputs/clear27/ablations/ablation_box_exact/<domain>/ and .../all_domains/ (SysAdmin, Drone and
        Sepsis together), each with the two sets of clear27 (paper/, appendix/) and data/. Two
        figures, each as clear27's paper.save_trio writes it (each panel without legend, _a and _b,
        for \\subfigure; with its own legend, _a_full and _b_full; and the legend alone, _legend):
            box_exact        (a) % of the instances solved and (b) % reliable, against the transitions
            box_exact_cost   (a) the time to compile and solve, against the observed states, with COST_N
                             transitions, and (b) the normalised lower bound from the initial states,
                             against the transitions

An instance is a (domain, size, seed), every size of the main runs; "solved": the run finished with a
normalised value of at least THRESHOLD (0.9, clear27's); "reliable": it finished and its lower bound is at
most the true value in every state. A run that did not finish (the exact set runs out of time or memory
on the larger problems) is neither and stays in the denominator. Time: the median over the seeds that
finished, drawn where at least half of them finished; the lower bound: the mean over the runs that
finished. The colour is the graph (clear27's registry), the line the set: solid the box, dotted and
hollow the exact set.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
NAME = "ablation_box_exact"
MAIN = ROOT / "results" / "clear27"  # the box (the main results)
RESULTS = MAIN / "ablations" / NAME  # the exact set
OUTPUTS = ROOT / "outputs" / "clear27" / "ablations" / NAME


def _paper():
    """clear27's plot.py: the loading of the runs, the registry of the methods and the paper style."""
    spec = importlib.util.spec_from_file_location("clear27_plot", HERE.parents[2] / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _paper()
THRESHOLD = paper.THRESHOLD
ALL_DOMAINS = paper.ALL_DOMAINS
DOMAINS = list(paper.POOLED_DOMAINS)  # sysadmin, drone, sepsis
KEY = ["domain", "size", "seed", "n_transitions"]
COST_N = 10_000  # the number of transitions of the time panel
sys.path.insert(0, str(HERE.parents[2]))  # experiments/clear27/: protocol.py
sys.path.insert(0, str(HERE.parents[3]))  # experiments/: common.py, domains.py
import protocol  # noqa: E402

LIMIT_S = protocol.LIMITS.timeout_s["relax"] + protocol.LIMITS.timeout_s["solve"]  # compile and solve
# the four learners, in the order of the legend: (graph of the registry, its model in the runs, set)
GRAPHS = {"true graph": "true graph", "DYNOTEARS": "DYNOTEARS (lagged)"}
SETS = {"box": "IA", "exact": "exact"}
LEARNERS = [(graph, model, name) for graph, model in GRAPHS.items() for name in SETS]
LABEL = {"true graph": "True-IMDP", "DYNOTEARS": "CD-IMDP"}
EXACT_LINE = dict(lw=1.3, ls=paper.DOTS, zorder=1.5, mfc="white", mew=0.9, ms=paper.SAME_STEP_MS)


def name_of(graph: str, set_name: str) -> str:
    """A learner in the csv files and the legends: "CD-IMDP, box"."""
    return f"{LABEL[graph]}, {set_name}"


def line_of(graph: str, set_name: str) -> dict:
    """The box: a solid line in the colour and marker of the graph in clear27's registry; the exact set:
    the same colour and marker, dotted, hollow."""
    color, marker, _ = paper.look_of(graph)
    label = name_of(graph, set_name)
    if set_name == "box":
        return paper.paper_line(color, marker, label, ls="-")
    return paper.paper_line(color, marker, label, mec=color, **EXACT_LINE)


def load(main: Path, results: Path, domain: str) -> pd.DataFrame:
    """The runs of a domain as clear27 loads them (paper.load): the main results (every source: they give
    the instances of the domain, and the box) with the runs of this ablation (the exact set). Empty when
    the domain has no main results."""
    base = paper.load(main, domain)
    if base.empty:
        return base
    exact = paper.load(results, domain)
    return pd.concat([base, exact], ignore_index=True).assign(domain=domain)


def instances_of(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per learner, instance and number of transitions, over EVERY instance of the runs: ok, value
    and lower bound (normalised), solved, safe, the time to compile and solve, and the observed states. A
    learner without a row for an instance did not finish it (ok False)."""
    every = runs[[*KEY, "n_observed_states"]].drop_duplicates(KEY)
    tables = []
    for graph, model, set_name in LEARNERS:
        d = runs[(runs["model"] == model) & (runs["solver"] == f"IMDP ({SETS[set_name]})")]
        if d.empty:  # no run of this learner: it is in no figure
            continue
        d = d.assign(cost=d[["relax_s", "solve_s"]].sum(axis=1, min_count=1))
        d = every.merge(d.drop_duplicates(KEY)[[*KEY, "ok", "value", "promise_value", "safe", "cost"]],
                        on=KEY, how="left")  # fmt: skip
        ok = d["ok"].eq(True)
        value = d["value"].astype(float).where(ok, -np.inf).fillna(-np.inf)
        tables.append(pd.DataFrame({
            **{k: d[k] for k in KEY}, "states": d["n_observed_states"], "graph": graph, "set": set_name,
            "learner": name_of(graph, set_name), "ok": ok, "solved": value >= THRESHOLD,
            "safe": ok & d["safe"].eq(1), "bound": d["promise_value"].where(ok), "cost": d["cost"].where(ok),
        }))  # fmt: skip
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def shares(instances: pd.DataFrame) -> pd.DataFrame:
    """Per learner and number of transitions: the instances, those finished, the % solved and reliable (of
    all the instances) and the mean normalised lower bound (of those that finished)."""
    out = instances.groupby(["learner", "graph", "set", "n_transitions"]).agg(
        instances=("ok", "size"), finished=("ok", "sum"), solved=("solved", "mean"), safe=("safe", "mean"),
        bound=("bound", "mean"),
    ).reset_index()  # fmt: skip
    out[["solved", "safe"]] *= 100
    return out


def costs(instances: pd.DataFrame, n: int = COST_N) -> pd.DataFrame:
    """Per learner and number of observed states, with n transitions: the median time to compile and solve
    over the runs that finished, where at least half of them finished (else no point)."""
    d = instances[instances["n_transitions"] == n]
    out = d.groupby(["learner", "graph", "set", "states"]).agg(
        runs=("ok", "size"), finished=("ok", "sum"), seconds=("cost", "median"),
    ).reset_index()  # fmt: skip
    return out[out["finished"] >= out["runs"] / 2]


def drawn_lines(ax, table: pd.DataFrame, x: str, column: str) -> None:
    for graph, _, set_name in LEARNERS:
        s = table[table["learner"] == name_of(graph, set_name)].sort_values(x)
        if len(s):
            ax.plot(s[x], s[column], **line_of(graph, set_name))


def panel_share(column: str, ylabel: str, table: pd.DataFrame):
    def draw(ax) -> pd.DataFrame:
        drawn_lines(ax, table, "n_transitions", column)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_ylim(-4, 104)
        paper.percent_axis(ax.yaxis)
        ax.set_xlabel("transitions")
        ax.set_ylabel(ylabel)
        return table[["learner", "graph", "set", "n_transitions", "instances", "finished", column]]

    return draw


def panel_cost(table: pd.DataFrame):
    def draw(ax) -> pd.DataFrame:
        drawn_lines(ax, table, "states", "seconds")
        ax.axhline(LIMIT_S, color="#9a9a9a", lw=0.8, ls=paper.DASHED, zorder=0)  # the limits of the two steps
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("observed states")
        ax.set_ylabel("compile and solve (s)")  # with COST_N transitions: in the caption
        return table

    return draw


def panel_bound(table: pd.DataFrame):
    def draw(ax) -> pd.DataFrame:
        drawn_lines(ax, table, "n_transitions", "bound")
        ax.axhline(0, **paper.ZERO_LINE)
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel("transitions")
        ax.set_ylabel("normalised lower bound")
        return table[["learner", "graph", "set", "n_transitions", "finished", "bound"]]

    return draw


def draw(runs: pd.DataFrame, out: Path, sets=None) -> list[str]:
    """The two figures of a table of runs (load; one domain, or several put together), once per set of
    clear27 (`sets`: some of their names; default both). Returns the figures written, as <set>/<name>."""
    instances = instances_of(runs)
    if instances.empty:
        print("  skipped box_exact: no interval MDP of the two graphs")
        return []
    table, cost = shares(instances), costs(instances)
    drawn = [(g, s) for g, _, s in LEARNERS if name_of(g, s) in set(table["learner"])]
    handles = [Line2D([], [], **(line_of(g, s) | {"ms": 4.5})) for g, s in drawn]
    written = []
    with mpl.rc_context(paper.PAPER_STYLE):
        for name, ours in paper.SETS.items():
            if sets and name not in sets:
                continue
            to = paper.FigureSet(name, ours, out)
            panels = [panel_share("solved", paper.SPLIT_AXES["solved"], table),
                      panel_share("safe", paper.SPLIT_AXES["reliable"], table)]  # fmt: skip
            paper.save_trio(to, "", "box_exact", panels, handles, index=table, ncol=4, panel_ncol=2)
            paper.save_trio(to, "", "box_exact_cost", [panel_cost(cost), panel_bound(table)], handles,
                            index=cost, ncol=4, panel_ncol=2)  # fmt: skip
            written += [f"{name}/box_exact", f"{name}/box_exact_cost"]
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27 ablation of the set (box, exact): its figures.")
    parser.add_argument("domains", nargs="*", choices=[*DOMAINS, ALL_DOMAINS, []],
                        help=f"default: {' '.join(DOMAINS)} {ALL_DOMAINS}")  # fmt: skip
    parser.add_argument("--only", nargs="+", choices=list(paper.SETS), metavar="SET",
                        help="one set of figures: paper or appendix (default: both)")  # fmt: skip
    parser.add_argument("--main", type=Path, default=MAIN, help="the main results (the box)")
    parser.add_argument("--results", type=Path, default=RESULTS, help="the results of this ablation")
    parser.add_argument("--out", type=Path, default=OUTPUTS, help="where the figures go")
    args = parser.parse_args(argv)
    loaded = {}

    def runs_of(domain: str) -> pd.DataFrame:
        if domain not in loaded:
            loaded[domain] = load(args.main, args.results, domain)
        return loaded[domain]

    for domain in args.domains or [*DOMAINS, ALL_DOMAINS]:
        print(f"\n===== {NAME}, {domain}: {args.results} (+ {args.main}) -> {args.out / domain} =====")
        tables = [runs_of(d) for d in DOMAINS] if domain == ALL_DOMAINS else [runs_of(domain)]
        if any(t.empty for t in tables):
            print("  no main results of a domain: nothing to draw")
            continue
        draw(pd.concat(tables, ignore_index=True), args.out / domain, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
