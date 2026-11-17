"""Figures of the ablation "KWIK thresholds" (run.py): SLF-Rmax and Met-Rmax with the thresholds of their
papers' experiments (the main runs) and of their theorems (this ablation) (S2, S8, S9).

Run from the repository root:
    .venv/bin/python experiments/clear27/ablations/ablation_kwik_thresholds/plot.py      # every domain,
                                                                                         # and all_domains
    .venv/bin/python experiments/clear27/ablations/ablation_kwik_thresholds/plot.py sysadmin all_domains

Input:  results/clear27/<domain>/<domain>_{strehl,met}_<date>_<time>/   the papers' thresholds: sources of
            the main runs, read where they are (the instances of the domain come from every main source)
        results/clear27/ablations/ablation_kwik_thresholds/<domain>/<domain>_{strehl,met}_theorem_<date>_<time>/
            the theorems' thresholds: this ablation
Output: outputs/clear27/ablations/ablation_kwik_thresholds/<domain>/ and .../all_domains/ (SysAdmin, Drone and
        Sepsis together), each with the two sets of clear27 (paper/, appendix/) and data/:
            <set>/kwik_thresholds_a, _b              (a) % of the instances solved and (b) % reliable,
                                                     against the transitions, without legend (subfigures)
            <set>/kwik_thresholds_legend             their legend alone
            <set>/kwik_thresholds_a_full, _b_full    each panel with its own legend (paper.save_trio)

An instance is a (domain, size, seed); the figures count the runs (an instance and a number of
transitions) that both settings have. "Solved": the run finished with a normalised value of at least THRESHOLD (0.9); "reliable": it finished and the value the
learner reported (its own estimate, no bound) is at most the true value in every state. A run that did
not finish is neither and stays in the denominator. A learner at the theorems' thresholds has the colour
and marker of the learner, a dotted line and a hollow marker.
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
NAME = "ablation_kwik_thresholds"
MAIN = ROOT / "results" / "clear27"  # the papers' thresholds (the main results)
RESULTS = MAIN / "ablations" / NAME  # the theorems' thresholds
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
# the four learners, in the order of the legend: (method of the registry, arm in the runs, level)
LEARNERS = [(method, arm, level) for arm, method in paper.KWIK_ARMS.items() for level in ("paper", "theorem")]
LEVEL_LABEL = {"paper": "papers' thresholds", "theorem": "theorems' thresholds"}
THEOREM_LINE = dict(lw=1.3, ls=paper.DOTS, zorder=1.5, mfc="white", mew=0.9, ms=paper.SAME_STEP_MS)


def name_of(method: str, level: str) -> str:
    """A learner in the csv files and the legends: "SLF-Rmax, papers' thresholds"."""
    return f"{method}, {LEVEL_LABEL[level]}"


def line_of(method: str, level: str) -> dict:
    """The papers' thresholds: a solid line in the colour and marker of the method in clear27's registry;
    the theorems': the same colour and marker, dotted, hollow."""
    label = name_of(method, level)
    if level == "paper":
        return paper.method_line(method, []) | {"label": label, "ls": "-"}
    color, marker, _ = paper.look_of(method)
    return paper.paper_line(color, marker, label, mec=color, **THEOREM_LINE)


def load(main: Path, results: Path, domain: str) -> pd.DataFrame:
    """The runs of a domain as clear27 loads them (paper.load): the main results (every source: they give
    the instances of the domain, and the learners at the papers' thresholds) with the runs of this
    ablation (the theorems' thresholds). Empty when the domain has no main results."""
    base = paper.load(main, domain)
    if base.empty:
        return base
    theorem = paper.load(results, domain)
    return pd.concat([base, theorem], ignore_index=True).assign(domain=domain)


def instances_of(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per learner, instance and number of transitions, over the cells (domain, size, seed,
    transitions) that every learner of the runs has a row for, finished or not (the two settings are
    compared on the same runs): ok, optimality (the normalised value), solved and safe. A cell whose row
    says the learner returned nothing (a limit, or skipped after one) is neither solved nor safe."""
    present = [runs[(runs["method"] == arm) & runs["kwik_level"].eq(level)][KEY].drop_duplicates()
               for _, arm, level in LEARNERS]  # fmt: skip
    present = [p for p in present if len(p)]
    if not present:
        return pd.DataFrame()
    every = present[0]
    for p in present[1:]:
        every = every.merge(p, on=KEY)
    tables = []
    for method, arm, level in LEARNERS:
        d = runs[(runs["method"] == arm) & runs["kwik_level"].eq(level)]
        if d.empty:  # no run of this learner: it is in no figure
            continue
        d = every.merge(d.drop_duplicates(KEY)[[*KEY, "ok", "value", "safe"]], on=KEY, how="left")
        ok = d["ok"].eq(True)
        value = d["value"].astype(float).where(ok, -np.inf).fillna(-np.inf)
        tables.append(pd.DataFrame({
            **{k: d[k] for k in KEY}, "method": method, "level": level, "learner": name_of(method, level),
            "ok": ok, "optimality": value, "solved": value >= THRESHOLD, "safe": ok & d["safe"].eq(1),
        }))  # fmt: skip
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def shares(instances: pd.DataFrame) -> pd.DataFrame:
    """Per learner and number of transitions: the instances, those finished, and the % solved and safe
    (of all the instances)."""
    out = instances.groupby(["learner", "method", "level", "n_transitions"]).agg(
        instances=("ok", "size"), finished=("ok", "sum"), solved=("solved", "mean"), safe=("safe", "mean"),
    ).reset_index()  # fmt: skip
    out[["solved", "safe"]] *= 100
    return out


def panel(column: str, ylabel: str, table: pd.DataFrame):
    """A panel of kwik_thresholds: `column` (% of all the instances) against the transitions."""

    def draw(ax) -> pd.DataFrame:
        for method, _, level in LEARNERS:
            s = table[table["learner"] == name_of(method, level)].sort_values("n_transitions")
            if len(s):
                ax.plot(s["n_transitions"], s[column], **line_of(method, level))
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_ylim(-4, 104)
        paper.percent_axis(ax.yaxis)
        ax.set_xlabel("transitions")
        ax.set_ylabel(ylabel)
        return table[["learner", "method", "level", "n_transitions", "instances", "finished", column]]

    return draw


def draw(runs: pd.DataFrame, out: Path, sets=None) -> list[str]:
    """kwik_thresholds of a table of runs (load; one domain, or several put together), once per set of
    clear27 (`sets`: some of their names; default both). Returns the figures written, as <set>/<name>."""
    instances = instances_of(runs)
    if instances.empty:
        print("  skipped kwik_thresholds: no run of the KWIK learners")
        return []
    table = shares(instances)
    drawn = [(m, level) for m, _, level in LEARNERS if name_of(m, level) in set(table["learner"])]
    handles = [Line2D([], [], **(line_of(m, level) | {"ms": 4.5})) for m, level in drawn]
    written = []
    with mpl.rc_context(paper.PAPER_STYLE):
        for name, ours in paper.SETS.items():
            if sets and name not in sets:
                continue
            to = paper.FigureSet(name, ours, out)
            panels = [panel("solved", paper.SPLIT_AXES["solved"], table),
                      panel("safe", paper.SPLIT_AXES["reliable"], table)]  # fmt: skip
            paper.save_trio(to, "", "kwik_thresholds", panels, handles, index=table, ncol=2, panel_ncol=1)
            written.append(f"{name}/kwik_thresholds")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27 ablation of the KWIK thresholds: its figures.")
    parser.add_argument("domains", nargs="*", choices=[*DOMAINS, ALL_DOMAINS, []],
                        help=f"default: {' '.join(DOMAINS)} {ALL_DOMAINS}")  # fmt: skip
    parser.add_argument("--only", nargs="+", choices=list(paper.SETS), metavar="SET",
                        help="one set of figures: paper or appendix (default: both)")  # fmt: skip
    parser.add_argument("--main", type=Path, default=MAIN, help="the main results (the papers' thresholds)")
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
