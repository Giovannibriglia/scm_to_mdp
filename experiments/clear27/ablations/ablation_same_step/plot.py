"""Figures of the ablation "same-step arrows" (run.py): PCMCI+ and DYNOTEARS with and without arrows
inside a step, each as the box interval MDP and as the plug-in MDP (S2, S8, S9).

Run from the repository root:
    .venv/bin/python experiments/clear27/ablations/ablation_same_step/plot.py              # every domain,
                                                                                           # and all_domains
    .venv/bin/python experiments/clear27/ablations/ablation_same_step/plot.py sysadmin drone
    .venv/bin/python experiments/clear27/ablations/ablation_same_step/plot.py all_domains --only paper

Input:  the latest run of each of the four learners, a domain at a time:
        results/clear27/<domain>/<domain>_{dynotears,pcmci}_lagged_<date>_<time>/   without same-step
            arrows: sources of the main runs, read where they are (one copy)
        results/clear27/ablations/ablation_same_step/<domain>/<domain>_{dynotears,pcmci}_<date>_<time>/
            with same-step arrows allowed: this ablation (run.py; the runs of 2026-11-05 were moved here
            from the main results, see their provenance.json)
Output: outputs/clear27/ablations/ablation_same_step/<domain>/ and .../all_domains/ (SysAdmin, Drone and
        Sepsis together), each with the two sets of clear27 (paper/, appendix/) and data/:
            <set>/imdp/solved_safely_vs_transitions.{pdf,png}   the box interval MDP of the four graphs
            <set>/mdp/solved_safely_vs_transitions.{pdf,png}    their plug-in MDP
            <set>/graph_errors.{pdf,png}                        the four graphs against the true one
            data/<set>_imdp_solved_safely_vs_transitions.csv, data/<set>_mdp_..., data/<set>_graph_errors.csv
            <set>/solved_safely_{a,b,legend,a_full,b_full}        the two forms, and the two panels of
            <set>/graph_errors_split_{a,b,legend,a_full,b_full}   graph_errors, as clear27's paper.save_trio
                                                                  writes them: subfigures, their legend, and
                                                                  each panel with its own legend

solved_safely_vs_transitions: the number of transitions (x) against the % of ALL instances solved safely.
An instance is a (domain, size, seed), every size of the main runs; it is solved safely when the run
finished with a normalised value of at least THRESHOLD (0.9, clear27's) AND the value the method reported
is at most the true value in every state (column `safe`: the promise of the interval MDP, the own
estimate of the plug-in MDP, which is no bound). A run that did not finish (time or memory limit, or
skipped after one) is not solved safely and stays in the denominator: the csv has `instances` (all) and
`finished`, with the % solved, safe and solved_safely.
graph_errors: the share of the true arrows a graph misses and its extra arrows per next-state variable,
the mean over the instances whose discovery finished (band: 95% t-interval); the csv also has the mean
number of same-step arrows of a graph (`same_step_arrows`). The graph is the same for the two forms.

The four learners keep clear27's registry (paper.look_of): DYNOTEARS blue as an interval MDP and in
graph_errors, coral as a plug-in MDP (its two colours there); PCMCI+ red. A twin with
same-step arrows has the colour and marker of its method, a broken line and a larger hollow marker
(paper.SAME_STEP_LINES, SAME_STEP_MS), drawn under its method: where the two coincide, a ring around it.
No learner is labelled "(ours)" here: the ablation compares the two discovery methods with their twins, so
the two without same-step arrows are both drawn thick, in both sets. The sets differ in the names only:
paper/ "DYNOTEARS", appendix/ with the form ("DYNOTEARS-IMDP").
Unlike the figures of the main runs, paper/ shows PCMCI+ too: the ablation is about both methods.
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
NAME = "ablation_same_step"
MAIN = ROOT / "results" / "clear27"  # the runs without same-step arrows (the main results)
RESULTS = MAIN / "ablations" / NAME  # the runs with same-step arrows allowed
OUTPUTS = ROOT / "outputs" / "clear27" / "ablations" / NAME


def _paper():
    """clear27's plot.py: the loading of the runs, the registry of the methods and the paper style."""
    spec = importlib.util.spec_from_file_location("clear27_plot", HERE.parents[2] / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paper = _paper()
THRESHOLD = paper.THRESHOLD  # "solved": at least this normalised value
ALL_DOMAINS = paper.ALL_DOMAINS
DOMAINS = list(paper.POOLED_DOMAINS)  # sysadmin, drone, sepsis
SAME_STEP = " same-step"  # the suffix of a twin in the `method` column of the csv files
# The four learners, in the order of the legends: (discovery method of the registry, same-step arrows)
LEARNERS = [(method, twin) for method in paper.DISCOVERY for twin in (False, True)]
FORMS = {"imdp": ("IMDP", "IMDP (IA)"), "mdp": ("MDP", "MDP (plug-in)")}  # folder -> (form, solver)
HELD = {"IMDP": "lower bound held", "MDP": "own estimate held"}  # the second condition, on the y axis
KEY = ["domain", "size", "seed", "n_transitions"]


def name_of(method: str, twin: bool) -> str:
    """A learner in the csv files: "DYNOTEARS", "DYNOTEARS same-step"."""
    return method + (SAME_STEP if twin else "")


def model_of(method: str, twin: bool) -> str:
    """Its model in the runs of clear27: "DYNOTEARS (lagged)" without same-step arrows, "DYNOTEARS" with."""
    return paper.SAME_STEP[method][1] if twin else paper.MODEL[method]


def label_of(method: str, twin: bool, form: str | None, to) -> str:
    """Its name in a legend, never with "(ours)". paper/: "DYNOTEARS", "PCMCI+", "DYNOTEARS, same-step
    arrows"; appendix/: with the form, "DYNOTEARS-IMDP", "DYNOTEARS-IMDP, same-step arrows" (form None,
    the figure of the graphs: without it)."""
    base = method if form is None or len(to.ours) == 1 else f"{method}-{form}"
    return f"{base}, same-step arrows" if twin else base


def line_of(method: str, twin: bool, form: str, to, label_form: str | None = "") -> dict:
    """The line of a learner in a form: without same-step arrows thick, for both discovery methods and
    in both sets (paper.method_line, the line of ours), its twin in the same colour with a hollow marker
    and the line of
    paper.SAME_STEP_LINES. label_form: the form in the label (default: `form`; None: no form)."""
    label = label_of(method, twin, form if label_form == "" else label_form, to)
    if not twin:
        thick = list(paper.DISCOVERY)  # both methods without same-step arrows, whatever the set
        return paper.method_line(method if form == "IMDP" else f"{method} {form}", thick) | {"label": label}
    color, marker, _ = paper.look_of(method if form == "IMDP" else f"{method} {form}")  # its method's
    width, dashes, order = paper.SAME_STEP_LINES[form]
    return paper.paper_line(color, marker, label, lw=width, ls=dashes, zorder=order, mfc="white", mec=color,
                            mew=0.9, ms=paper.SAME_STEP_MS)  # fmt: skip


def load(main: Path, results: Path, domain: str) -> pd.DataFrame:
    """The runs of a domain as clear27 loads them (paper.load): the main results (every source: they
    give the instances of the domain, and the two learners without same-step arrows) with the runs of
    this ablation (the two twins). Empty when the domain has no main results."""
    base = paper.load(main, domain)
    if base.empty:
        return base
    twins = paper.load(results, domain)
    if not twins.empty:
        twins = twins[twins["source"].isin([source for source, _ in paper.SAME_STEP.values()])]
    return pd.concat([base, twins], ignore_index=True).assign(domain=domain)


def instances_of(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per learner, form, instance and number of transitions, over EVERY instance of the runs
    (a (domain, size, seed) that any source ran, at every number of transitions of the runs): ok,
    optimality (the normalised value), safe, and solved_safely. A learner without a row for an instance
    did not finish it (ok False)."""
    every = runs[KEY].drop_duplicates()
    tables = []
    for method, twin in LEARNERS:
        for form, solver in FORMS.values():
            d = runs[(runs["model"] == model_of(method, twin)) & (runs["solver"] == solver)]
            if d.empty:  # no run of this learner: it is in no figure
                continue
            d = d.drop_duplicates(KEY)[[*KEY, "ok", "value", "safe"]]
            d = every.merge(d, on=KEY, how="left")
            ok = d["ok"].eq(True)
            value = d["value"].astype(float).where(ok, -np.inf).fillna(-np.inf)
            safe = ok & d["safe"].eq(1)
            tables.append(pd.DataFrame({
                **{k: d[k] for k in KEY}, "method": name_of(method, twin), "form": form, "ok": ok,
                "optimality": value, "solved": value >= THRESHOLD, "safe": safe,
                "solved_safely": (value >= THRESHOLD) & safe,
            }))  # fmt: skip
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def shares(instances: pd.DataFrame, form: str) -> pd.DataFrame:
    """Per learner and number of transitions: the instances, those finished, and the % solved, safe and
    solved safely (of all the instances)."""
    d = instances[instances["form"] == form]
    out = d.groupby(["method", "n_transitions"]).agg(
        instances=("ok", "size"), finished=("ok", "sum"), solved=("solved", "mean"), safe=("safe", "mean"),
        solved_safely=("solved_safely", "mean"),
    ).reset_index()  # fmt: skip
    out[["solved", "safe", "solved_safely"]] *= 100
    return out


def learners_in(table: pd.DataFrame) -> list:
    return [(m, twin) for m, twin in LEARNERS if name_of(m, twin) in set(table["method"])]


def legend(fig, handles, y=0.02) -> None:
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=2, fontsize=7.8,
               handlelength=2.8, columnspacing=1.6, borderaxespad=0)  # fmt: skip


def solved_safely_vs_transitions(instances: pd.DataFrame, folder: str, to) -> bool:
    """% of all instances solved safely (normalised value of at least THRESHOLD, and the reported value
    held) against the transitions: the four learners in one form."""
    form = FORMS[folder][0]
    t = shares(instances, form) if len(instances) else pd.DataFrame()
    if t.empty:
        print(f"  skipped {to.name}/{folder}/solved_safely_vs_transitions: no run of the four learners")
        return False
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    handles, labels = [], {}
    for method, twin in learners_in(t):
        s = t[t["method"] == name_of(method, twin)].sort_values("n_transitions")
        style = line_of(method, twin, form, to)
        ax.plot(s["n_transitions"], s["solved_safely"], **style)
        handles.append(Line2D([], [], **(style | {"ms": max(style["ms"], 4.5)})))
        labels[name_of(method, twin)] = style["label"]
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(-4, 104)
    paper.percent_axis(ax.yaxis)
    ax.set_xlabel("transitions")
    ax.set_ylabel(f"instances solved reliably (%)\n(value ≥ {THRESHOLD}, {HELD[form]})")
    fig.tight_layout()
    legend(fig, handles)
    t.insert(1, "label", t["method"].map(labels))
    t.insert(2, "form", form)
    paper.save_to(to.out, f"{to.name}/{folder}", fig, "solved_safely_vs_transitions", t,
                  csv=f"{to.name}_{folder}_solved_safely_vs_transitions")  # fmt: skip
    return True


def graphs_of(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per learner, instance and number of transitions whose discovery finished: the errors of
    its graph (the same for every compilation of the graph: the first row that has them)."""
    tables = []
    for method, twin in LEARNERS:
        d = runs[runs["model"] == model_of(method, twin)]
        columns = [c for c in ("missing_share", "extra_per_var", "intra_next") if c in d]
        if d.empty or "missing_share" not in columns:
            continue
        g = d.groupby(KEY)[columns].first().dropna(subset=["missing_share"]).reset_index()
        tables.append(g.assign(method=name_of(method, twin)))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def mean_band(d: pd.DataFrame, column: str) -> pd.DataFrame:
    """The mean of a column per number of transitions, with a 95% t-interval over the instances."""
    out = d.groupby("n_transitions")[column].agg(["mean", "std", "count"]).reset_index()
    t = stats.t.ppf(0.975, np.maximum(out["count"] - 1, 1))
    half = (t * out["std"] / np.sqrt(out["count"])).fillna(0)
    return out.assign(lo=out["mean"] - half, hi=out["mean"] + half)


def graph_errors(runs: pd.DataFrame, to) -> bool:
    """The four graphs against the true one: the share of the true arrows missed and the extra arrows
    per next-state variable, against the transitions."""
    g = graphs_of(runs)
    if g.empty:
        print(f"  skipped {to.name}/graph_errors: no graph of the four learners")
        return False
    total = len(runs[["domain", "size", "seed"]].drop_duplicates())
    panels = [("missing_share", "true arrows missed (share)"), ("extra_per_var", "extra arrows per variable")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True)
    rows, handles = [], []
    for method, twin in learners_in(g):
        d = g[g["method"] == name_of(method, twin)]
        style = line_of(method, twin, "IMDP", to, label_form=None)
        for ax, (column, _) in zip(axes, panels, strict=True):
            c = mean_band(d, column)
            ax.fill_between(c["n_transitions"], c["lo"], c["hi"], color=style["color"], alpha=0.13, lw=0)
            ax.plot(c["n_transitions"], c["mean"], **(style | {"label": None}))
            row = c.rename(columns={"count": "graphs"}).assign(
                method=name_of(method, twin), label=style["label"], metric=column, instances=total
            )
            if "intra_next" in d:
                row = row.merge(d.groupby("n_transitions")["intra_next"].mean().rename("same_step_arrows"),
                                on="n_transitions", how="left")  # fmt: skip
            rows.append(row)
        handles.append(Line2D([], [], **(style | {"ms": max(style["ms"], 4.5)})))
    for ax, (_, ylabel) in zip(axes, panels, strict=True):
        paper.paper_axis(ax)
        ax.set_xlabel("transitions")
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylim(bottom=min(-0.03, axes[1].get_ylim()[0]))
    fig.tight_layout(w_pad=1.6)
    legend(fig, handles, y=0.03)
    data = pd.concat(rows, ignore_index=True)
    first = ["method", "label", "metric", "n_transitions", "instances", "graphs"]
    paper.save_to(to.out, to.name, fig, "graph_errors", data[first + [c for c in data if c not in first]],
                  csv=f"{to.name}_graph_errors")  # fmt: skip
    return True


def same_step_trios(instances: pd.DataFrame, runs: pd.DataFrame, to) -> list[str]:
    """The two figures of the appendix as clear27's paper.save_trio writes them: solved_safely, (a) the box
    interval MDP and (b) the plug-in MDP of the four learners, and graph_errors_split, (a) the true arrows
    missed and (b) the extra arrows per variable; each panel without legend and with its own, and the
    legend alone. Returns the figures written, as <set>/<name>."""
    written = []
    empty = not len(instances)
    tables = {folder: pd.DataFrame() if empty else shares(instances, FORMS[folder][0]) for folder in FORMS}
    if all(len(t) for t in tables.values()):
        learners = [x for x in LEARNERS if any(name_of(*x) in set(t["method"]) for t in tables.values())]

        def solved(folder):
            form, t = FORMS[folder][0], tables[folder]

            def on(ax) -> pd.DataFrame:
                for method, twin in learners_in(t):
                    s = t[t["method"] == name_of(method, twin)].sort_values("n_transitions")
                    ax.plot(s["n_transitions"], s["solved_safely"], **line_of(method, twin, form, to, None))
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
                ax.set_ylim(-4, 104)
                paper.percent_axis(ax.yaxis)
                ax.set_xlabel("transitions")
                ax.set_ylabel("runs solved and reliable (%)")
                return t.assign(form=form).rename(columns={"method": "learner"})  # no label of the registry

            return on

        handles = [Line2D([], [], **(line_of(m, twin, "IMDP", to, None) | {"ms": 4.5}))
                   for m, twin in learners]  # fmt: skip
        paper.save_trio(to, "", "solved_safely", [solved("imdp"), solved("mdp")], handles, ncol=2,
                        panel_ncol=1)  # fmt: skip
        written.append(f"{to.name}/solved_safely")
    g = graphs_of(runs)
    if not g.empty:
        def errors(column: str, ylabel: str):
            def on(ax) -> pd.DataFrame:
                rows = []
                for method, twin in learners_in(g):
                    style = line_of(method, twin, "IMDP", to, label_form=None)
                    c = mean_band(g[g["method"] == name_of(method, twin)], column)
                    ax.fill_between(c["n_transitions"], c["lo"], c["hi"], color=style["color"], alpha=0.13,
                                    lw=0)  # fmt: skip
                    ax.plot(c["n_transitions"], c["mean"], **(style | {"label": None}))
                    rows.append(c.assign(method=name_of(method, twin), metric=column))
                paper.paper_axis(ax)
                ax.set_xlabel("transitions")
                ax.set_ylabel(ylabel)
                if column == "missing_share":
                    ax.set_ylim(-0.03, 1.03)
                return pd.concat(rows, ignore_index=True).rename(columns={"method": "learner"})

            return on

        handles = [Line2D([], [], **(line_of(m, twin, "IMDP", to, label_form=None) | {"ms": 4.5}))
                   for m, twin in learners_in(g)]  # fmt: skip
        panels = [errors("missing_share", "true arrows missed (share)"),
                  errors("extra_per_var", "extra arrows per variable")]  # fmt: skip
        paper.save_trio(to, "", "graph_errors_split", panels, handles, ncol=2, panel_ncol=1)
        written.append(f"{to.name}/graph_errors_split")
    return written


FIGURES = ["imdp/solved_safely_vs_transitions", "mdp/solved_safely_vs_transitions", "graph_errors"]


def draw(runs: pd.DataFrame, out: Path, sets=None) -> list[str]:
    """The three figures of a table of runs (load; one domain, or several put together), once per set
    of clear27 (`sets`: some of their names; default both). Returns the figures written, as <set>/<name>."""
    instances = instances_of(runs)
    if len(instances):
        every = instances[KEY[:3]].drop_duplicates()
        by_domain = ", ".join(f"{d}: {n}" for d, n in every.groupby("domain").size().items())
        print(f"  {len(every):,} instances ({by_domain}), {instances['n_transitions'].nunique()} numbers of "
              f"transitions")  # fmt: skip
    written = []
    with mpl.rc_context(paper.PAPER_STYLE):
        for name, ours in paper.SETS.items():
            if sets and name not in sets:
                continue
            to = paper.FigureSet(name, ours, out)
            for folder in FORMS:
                if solved_safely_vs_transitions(instances, folder, to):
                    written.append(f"{name}/{folder}/solved_safely_vs_transitions")
            if graph_errors(runs, to):
                written.append(f"{name}/graph_errors")
            written += same_step_trios(instances, runs, to)
    return written


def summary(runs: pd.DataFrame, at=(10_000, 100_000, 500_000)) -> pd.DataFrame:
    """The % of instances solved safely at some numbers of transitions: learner x form (for a report)."""
    instances = instances_of(runs)
    rows = []
    for form, _ in FORMS.values():
        t = shares(instances, form)
        for row in t[t["n_transitions"].isin(at)].itertuples():
            rows.append({"form": form, "method": row.method, "n_transitions": row.n_transitions,
                         "instances": row.instances, "finished": row.finished, "solved": row.solved,
                         "safe": row.safe, "solved_safely": row.solved_safely})  # fmt: skip
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27 ablation of the same-step arrows: its figures.")
    parser.add_argument("domains", nargs="*", choices=[*DOMAINS, ALL_DOMAINS, []],
                        help=f"default: {' '.join(DOMAINS)} {ALL_DOMAINS}")  # fmt: skip
    parser.add_argument("--only", nargs="+", choices=list(paper.SETS), metavar="SET",
                        help="one set of figures: paper or appendix (default: both)")  # fmt: skip
    parser.add_argument("--main", type=Path, default=MAIN, help="the main results (the lagged learners)")
    parser.add_argument("--results", type=Path, default=RESULTS, help="the results of this ablation")
    parser.add_argument("--out", type=Path, default=OUTPUTS, help="where the figures go")
    args = parser.parse_args(argv)
    chosen = args.domains or [*DOMAINS, ALL_DOMAINS]
    loaded = {}

    def runs_of(domain: str) -> pd.DataFrame:
        if domain not in loaded:
            loaded[domain] = load(args.main, args.results, domain)
        return loaded[domain]

    for domain in chosen:
        print(f"\n===== {NAME}, {domain}: {args.results} (+ {args.main}) -> {args.out / domain} =====")
        if domain == ALL_DOMAINS:
            tables = [runs_of(d) for d in DOMAINS]
            missing = [d for d, t in zip(DOMAINS, tables, strict=True) if t.empty]
            if missing:
                print(f"  no main results of {', '.join(missing)}: nothing to draw")
                continue
            runs = pd.concat(tables, ignore_index=True)
        else:
            runs = runs_of(domain)
            if runs.empty:
                print(f"  no main results in {args.main / domain}: nothing to draw")
                continue
        draw(runs, args.out / domain, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
