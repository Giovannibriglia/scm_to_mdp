"""The tables of the appendix of the paper, from the main runs of clear27 (S9).

Run from the repository root, after the main runs:
    .venv/bin/python experiments/clear27/paper_tables.py

Output: outputs/clear27/tables/, one LaTeX body (.tex, the rows between \\midrule and \\bottomrule) and one
.csv per table:
    reliable      the share of runs (%) whose reported value (the lower bound of an interval MDP, the own
                  estimate of any other method) is at most the true value, from the initial states
                  ("initial") and in every observed state ("all"), per method and domain, over every size and
                  number of transitions (Table app-held)
    states        the share of the 10 seeds (%) solved AND reliable in every state, with STATES_N
                  transitions, per domain, size and method: CD-MDP and CD-IMDP (Table exp-states)

Conventions of Section 4: a run is a (size, seed, number of transitions) of a task; every share is over
every run of the method's grid, and a run that returns nothing (a limit, or skipped after one) counts as
neither solved nor reliable; the csv gives the runs of every row. "Solved": a normalised value of at least plot.THRESHOLD (0.9). Interval MDPs: the box.
SLF-Rmax and Met-Rmax: the papers' thresholds (plot.load, kwik_level).
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "clear27"
OUTPUTS = ROOT / "outputs" / "clear27" / "tables"
DOMAINS = ["sysadmin", "drone", "sepsis"]
KEY = ["size", "seed", "n_transitions"]
STATES_N = 500_000


def _plot():
    spec = importlib.util.spec_from_file_location("clear27_plot", HERE / "plot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = _plot()

# (group, name in the paper, method in the runs)
ROWS = [
    ("interval MDP on a graph (box)", "CD-IMDP", "DYNOTEARS (lagged) -> IMDP (IA)"),
    ("interval MDP on a graph (box)", "PCMCI+-IMDP", "PCMCI+ (lagged) -> IMDP (IA)"),
    ("interval MDP on a graph (box)", "True-IMDP", "true graph -> IMDP (IA)"),
    ("interval MDP on a graph (box)", "Full-IMDP", "full-fact -> IMDP (IA)"),
    ("interval MDP on a graph (box)", "Sparse-IMDP", "sparse -> IMDP (IA)"),
    ("interval MDP on a graph (box)", "Flat-IMDP", "flat IMDP"),
    ("MDP on a graph", "CD-MDP", "DYNOTEARS (lagged) -> MDP (plug-in)"),
    ("MDP on a graph", "CD-SPIBB", "DYNOTEARS (lagged) -> Factored SPIBB"),
    ("MDP on a graph", "True-MDP", "true graph -> MDP (plug-in)"),
    ("MDP on a graph", "Full-MDP", "full-fact -> MDP (plug-in)"),
    ("MDP on a graph", "Flat-MDP", "flat -> MDP (plug-in)"),
    ("baselines", "FOCUS", "FOCUS"),
    ("baselines", "DiFaRL", "DiFaRL"),
    ("baselines", "SL-SPIBB", "SL-SPIBB"),
    ("baselines", "flat SPIBB", "flat SPIBB"),
    ("baselines", "SLF-Rmax", "Strehl-KWIK -> MDP (plug-in)"),
    ("baselines", "Met-Rmax", "Met-Rmax-KWIK -> MDP (plug-in)"),
    ("baselines", "offline DQN", "offline DQN"),
    ("baselines", "CQL", "offline CQL"),
]


def runs_of(results: Path, domain: str) -> pd.DataFrame:
    """Every run of a domain (plot.load) without the KWIK learners at their theorems' thresholds."""
    runs = plot.load(results, domain)
    if runs.empty:
        return runs
    theorem = runs["method"].isin(plot.KWIK_ARMS) & runs["kwik_level"].eq("theorem")
    return runs[~theorem]


def judged(runs: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per run of the method's grid (a main run keeps a row for every (size, seed, transitions),
    with the status timeout, memout or skipped when it returned nothing): finished, solved, reliable from
    the initial states and in every state."""
    d = runs[runs["method"] == method].drop_duplicates(KEY)[[*KEY, "ok", "value", "promise", "true", "safe"]]
    ok = d["ok"].eq(True)
    return d.assign(
        ok=ok, solved=ok & (d["value"] >= plot.THRESHOLD),
        initial=ok & (d["promise"] <= d["true"] + 1e-9), all=ok & d["safe"].eq(1),
    )  # fmt: skip


def reliable(results: Path) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        runs = runs_of(results, domain)
        if runs.empty:
            continue
        for group, name, method in ROWS:
            if not (runs["method"] == method).any():
                continue
            j = judged(runs, method)
            rows.append({"group": group, "method": name, "domain": domain, "runs": len(j),
                         "initial": 100 * j["initial"].mean(), "all": 100 * j["all"].mean()})  # fmt: skip
    return pd.DataFrame(rows)


def states(results: Path) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        runs = runs_of(results, domain)
        if runs.empty:
            continue
        observed = runs.groupby("size")["n_observed_states"].first()
        for name, method in (("CD-MDP", "DYNOTEARS (lagged) -> MDP (plug-in)"),
                             ("CD-IMDP", "DYNOTEARS (lagged) -> IMDP (IA)")):  # fmt: skip
            j = judged(runs, method)
            j = j[j["n_transitions"] == STATES_N]
            for size, g in j.groupby("size"):
                rows.append({"domain": domain, "size": size, "states": int(observed[size]), "method": name,
                             "seeds": len(g), "solved_and_reliable": 100 * (g["solved"] & g["all"]).mean()})
    return pd.DataFrame(rows)


def latex_reliable(t: pd.DataFrame) -> str:
    lines = []
    for group in dict.fromkeys(t["group"]):
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{group}}}}} \\\\")
        for name in dict.fromkeys(t[t["group"] == group]["method"]):
            cells = []
            for domain in DOMAINS:
                r = t[(t["method"] == name) & (t["domain"] == domain)]
                cells += ["--", "--"] if r.empty else [f"${r['initial'].iloc[0]:.1f}$", f"${r['all'].iloc[0]:.1f}$"]
            lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    return "\n".join(lines[:-1]) + "\n"


def latex_states(t: pd.DataFrame) -> str:
    lines = []
    for domain in DOMAINS:
        d = t[t["domain"] == domain].sort_values("states")
        sizes = list(dict.fromkeys(d["states"]))
        lines.append(f"{plot.DOMAIN_NAMES[domain]} & states & " + " & ".join(f"${s:,}$".replace(",", "{,}")
                                                                             for s in sizes) + " \\\\")
        for name in ("CD-MDP", "CD-IMDP"):
            values = [d[(d["method"] == name) & (d["states"] == s)]["solved_and_reliable"] for s in sizes]
            lines.append(f" & {name} & " + " & ".join(f"${v.iloc[0]:.0f}$" if len(v) else "--" for v in values)
                         + " \\\\")  # fmt: skip
        lines.append("\\midrule")
    return "\n".join(lines[:-1]) + "\n"


def main(argv=None) -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for name, table, latex in (("reliable", reliable(RESULTS), latex_reliable),
                               ("states", states(RESULTS), latex_states)):  # fmt: skip
        if table.empty:
            print(f"  {name}: no main results in {RESULTS}")
            continue
        table.to_csv(OUTPUTS / f"{name}.csv", index=False)
        (OUTPUTS / f"{name}.tex").write_text(latex(table))
        print(f"  wrote {OUTPUTS / name}.tex and .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
