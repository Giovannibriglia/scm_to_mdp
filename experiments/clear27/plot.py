"""clear27: every figure of one domain, from the results of its runs (S9).

Run from the repository root, after (or during) the runs of a domain:

    .venv/bin/python experiments/clear27/plot.py sysadmin                 # every figure of SysAdmin
    .venv/bin/python experiments/clear27/plot.py sepsis drone             # several domains
    .venv/bin/python experiments/clear27/plot.py all_domains              # the pooled figures over
                                                                          # SysAdmin, Drone and Sepsis
    .venv/bin/python experiments/clear27/plot.py sysadmin --only value coverage
    .venv/bin/python experiments/clear27/plot.py sysadmin --only pooled   # the thirty-four pooled figures
    .venv/bin/python experiments/clear27/plot.py sysadmin --only paper    # one set: paper/ or appendix/
    .venv/bin/python experiments/clear27/plot.py sysadmin --solver "IMDP (exact)"   # instead of the box
    .venv/bin/python experiments/clear27/plot.py --list                   # the figure names

Input:  results/clear27/<domain>/<domain>_<source>_<date>_<time>/<knob>/uniform/results.csv, the latest
        run of every graph source (a figure whose methods have no run yet is skipped).
Output: outputs/clear27/<domain>/paper/...      the paper figures, with DYNOTEARS as ours ("CD") and
                                                without PCMCI+: the seven of section 5 and the 19 pooled
                                                ones of section 6 (in imdp_by_graph/, mdp_by_graph/,
                                                vs_baselines/, and tradeoff*)
        outputs/clear27/<domain>/appendix/...   the same seventeen with DYNOTEARS and PCMCI+ both as ours
        outputs/clear27/<domain>/explore/...    the exploratory figures (section 7)
        outputs/clear27/<domain>/data/<name>.csv   the numbers drawn in each figure; a paper figure:
                                                <set>_<name>.csv or <set>_<folder>_<name>.csv, e.g.
                                                paper_value.csv, appendix_vs_baselines_solved_vs_time.csv
        outputs/clear27/all_domains/{paper,appendix,data}/   the pooled figures over POOLED_DOMAINS
Every figure is a PDF and a PNG.

The file, in order: 1 settings (with the registry of the methods), 2 loading the runs, 3 what is read from
the data of a domain (its sizes, panels and labels), 4 style and helpers, 5 paper figures, 6 pooled
figures, 7 exploratory figures, 8 the command line.

Conventions
- normalised value = (V_true - V_random) / (V*_obs - V_random): 0 is the uniform random policy, 1 the best
  policy over the observed state. The promise is normalised in the same way.
- "certified" = the interval MDP of a graph source solved with the domain's solver: the box "IMDP (IA)"
  in every domain (DOMAIN_SETTINGS; --solver overrides it, e.g. "IMDP (exact)", which stops beyond the
  smallest sizes). The flat model has one interval MDP ("flat IMDP").
- A point (method, size, transitions) is drawn only if at least MIN_OK of its seeds finished (status ok):
  the mean over the finished seeds, and a 95% t-interval as the band.
- "Promise held" = promise <= true value in every observed live state (column `safe`).
- A domain's sizes are the ones where the reference (the true graph, certified) finished at least MIN_OK
  of its runs; the larger sizes are left out of the figures that pool or compare sizes.
- The sizes of LEFT_OUT (none now) are in no figure.
- One colour, one marker and one line per method, in every figure (the registry METHODS; the ablations
  use it too). The interval MDP and the plug-in MDP of a graph share its colour: solid with a promise,
  dash-dot without; the plug-in MDPs of DYNOTEARS, of the full graph and of the flat model have a colour
  of their own (OWN_COLOUR), and that of DYNOTEARS, as ours, the solid line. Ours has the thickest lines,
  the true graph (the reference) is dotted with a hollow
  marker. A method is never told by its colour alone: see the registry.
- Two sets of paper figures (SETS): in paper/ ours is DYNOTEARS, named "CD-IMDP (ours)" and "CD-MDP
  (ours)", and PCMCI+ is in no figure; in appendix/ DYNOTEARS and PCMCI+ are both ours, under their
  names. The other names: True-, Sparse-, Full-, Flat- with -IMDP or -MDP, and the baselines' own.
  Factored SPIBB on a learned graph is a baseline with its own colour: "CD-SPIBB" in paper/,
  "DYNOTEARS-SPIBB" and "PCMCI+-SPIBB" in appendix/.
  --only paper (or appendix) draws one set.
- PCMCI+ and DYNOTEARS are the runs without arrows inside a step (the sources pcmci_lagged and
  dynotears_lagged). Their twins with same-step arrows allowed are in no figure of this file: they are
  the ablation ablations/ablation_same_step, whose plot.py draws the four with the styles of SAME_STEP.

The pooled figures (section 6) count instances, with every size of the domain together. An instance is a
(domain, size, seed); it is "solved" when the run finished with a normalised value of at least THRESHOLD
(0.9), and "safe" when it finished and the value the method reported is at most the true value in every
state (column `safe`; a method without a promise is judged on its own estimate). An instance that did not
finish is neither, and still counts. Interval MDPs are the box ("IMDP (IA)").
SLF-Rmax and Met-Rmax are the plug-in MDPs of the two KWIK learners (sources strehl and met) with the
thresholds of their papers' experiments (protocol.KWIK_PAPER). A run of those sources with the thresholds
of the theorems (m "calibrated" in its meta.json: the runs before 2026-11-17) is left out of the pooled
figures; ablations/ablation_kwik_thresholds compares the two settings.
  imdp_by_graph/solved_vs_transitions   % of instances solved against the transitions: the interval MDP
                                        of every graph source
  vs_baselines/solved_vs_optimality_to90, _grey, solved_vs_regret, solved_vs_regret_grey: the same
                                        to 90%, with the part above 90% grey, against the regret
                                        (1 - normalised value), and against the regret with its part below
                                        10% grey; solved_vs_regret_fade: from 0 to 100%,
                                        the part above 10% grey and its curves transparent;
                                        solved_vs_regret_fade90: the same with the band from 10 to 90% only;
                                        solved_vs_regret_fade_ends: the bands at the ends, 0-10 and 90-100%;
                                        solved_vs_optimality_fade_ends: the same against the normalised value
  imdp_by_graph/solved_vs_optimality    at OPTIMALITY_N transitions: % of instances with a normalised
                                        value of at least x, x from 0 to THRESHOLD
  mdp_by_graph/<the same two>           the plug-in MDP of the graph sources, and the baselines
  vs_baselines/<the same two>           ours (interval MDP and plug-in MDP) against the baselines
  vs_baselines/solved_vs_time           at REFERENCE_N transitions: % of instances solved within a time
  tradeoff                              % solved (x) against % safe (y), one path per method as data grow
  tradeoff_pareto                       the same, with the envelope of all points and, per data size, the
                                        method that is best on each axis
  tradeoff_data_needed                  the transitions a method needs to solve 10 ... 90% of the
                                        instances (x) against the % safe with that much data (y)
  solved_and_tradeoff                   two panels with one legend, for the text width of the paper:
                                        (a) vs_baselines/solved_vs_transitions, (b) tradeoff_pareto
  vs_baselines/reliable_vs_transitions  % of the runs whose reported value is reliable (at most the true
                                        value of the policy from the start; over the runs that returned a
                                        value: the "reliability" of reported_vs_true) against the transitions
  vs_baselines/reliable_everywhere_vs_transitions   the stricter check, in every state (`safe`), as a %
                                        of all the instances (the y axis of the tradeoff planes)
  solved_and_reliable                   two panels with one legend, as solved_and_tradeoff: (a)
                                        vs_baselines/solved_vs_transitions, (b) .../reliable_vs_transitions
  solved_and_reliable_everywhere        the same with the check in every state in (b)
  tradeoff_pareto_compact               the plane of tradeoff_pareto alone, for half the text width, with
                                        no method in its legend (beside a figure whose legend names them)
  reported_vs_true                      one dot per run, a panel per method of REPORTED: the value the
                                        method reported (x: the lower bound of the interval MDP of ours,
                                        its own estimate for the others) against the true value of its
                                        policy (y), both normalised, coloured by domain (DOMAIN_COLOURS);
                                        in the grey region the method reported more than the true value
  reported_vs_true_top8                 the same for the TOP_N most reliable methods of REPORTED_TOP (the
                                        interval MDPs of the full graph and of no graph are candidates too)
  reported_vs_true_selected             the same for the eight methods of REPORTED_SELECTED, by reliability
  reported_vs_true_common               the same on the runs that every method of the figure finished
  reported_vs_true_all                  appendix/ only: the same for every method of the pooled figures
                                        (REPORTED_ALL, PCMCI+ too), by reliability
  solved_reliably_vs_states             over several domains, with 500,000 transitions: the % of the
                                        instances solved (normalised value of at least 0.9) AND reliable
                                        (a reported value at most the true one) against the observed
                                        states; a line per method, its shares per number of states
                                        smoothed (trend: SMOOTH_OCTAVES); the pair of
                                        tradeoff_pareto_compact, with no method in its legend
  solved_reliably_vs_states_everywhere  the same with the check in every state (`safe`)
  tradeoff_and_scalability              over several domains: (a) the plane of tradeoff_pareto and (b)
                                        solved_reliably_vs_states, with one legend that names the methods
  solved_and_scalability                Figure 3 of the paper, over several domains: (a)
                                        vs_baselines/solved_vs_transitions and (b)
                                        solved_reliably_vs_states_everywhere, as save_trio writes it: each
                                        panel without legend (_a, _b) and with its own (_a_full, _b_full),
                                        and the legend alone (_legend)
  tradeoff_and_optimality               over several domains: (a) the plane of tradeoff_pareto and (b)
                                        vs_baselines/solved_vs_optimality_fade_ends, with one legend
The same thirty-four are drawn for an ablation by its own plot.py, which calls pooled_figures with its
instances. Figure 5 of the paper (tradeoff_and_rewards: the plane of the main runs and that of the runs
with other rewards, as save_trio writes it) is drawn by ablations/ablation_rewards/plot.py --domain all_domains.
"""

import argparse
import json
import math
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
THRESHOLD = 0.9  # "near-optimal" / "solved": the normalised value of the data-needed and pooled figures
REFERENCE_N = 100_000  # the number of transitions of the pooled figures drawn at one data size
OPTIMALITY_N = 500_000  # the number of transitions of solved_vs_optimality (all the data)
LEFT_OUT: dict[str, list[str]] = {}  # domain -> sizes dropped when its runs are loaded: in no figure (none)
# SLF-Rmax and Met-Rmax in the pooled figures: the plug-in MDP of each KWIK learner (sources strehl and met),
# from runs with the thresholds of the papers' experiments only (load: column kwik_level).
KWIK_ARMS = {"Strehl-KWIK -> MDP (plug-in)": "SLF-Rmax", "Met-Rmax-KWIK -> MDP (plug-in)": "Met-Rmax"}
KWIK_METHODS = list(KWIK_ARMS.values())
# Runs of SLF-Rmax and Met-Rmax with the papers' thresholds copied on 2026-11-15 from the reward ablation (its
# original reward, the same problems): read as the two arms at the papers' thresholds (load).
LEGACY_KWIK = "kwik_paper"
MAX_PANELS = 5  # panels (sizes) of a paper figure
SAFETY_BUDGETS = (1000, 10_000)  # the two panels of value_vs_safety
DEFAULT_SOLVER = "IMDP (exact)"

# The colours of the paper's figures (clear2027.tex); the registry below gives one to every method.
GOLD, RED, BLUE, PINK, TEAL, PURPLE = "#cd9c1f", "#ce4522", "#3070c7", "#dd8cc3", "#12a7a7", "#7945ab"
GREEN, WINE = "#348f4f", "#aa3f4e"
LIME, INDIGO, BROWN = "#b5d334", "#2b1a8a", "#5e3c1e"
SKY, VIOLET, PLUM, ULTRAMARINE = "#84b2fd", "#b264fb", "#770255", "#3f25e4"
CORAL = "#ff8c7f"  # an orange towards coral: a purer orange is too close to the gold or to the red
ROSE, MAGENTA = "#ff3284", "#e500da"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# The registry: one colour, one marker and one line per method, in every figure of this file and of the
# ablations (no figure has colours of its own). The six graph sources keep the colours of the paper; a
# baseline that shared one of them, or was close to one, has its own (FOCUS, SL-SPIBB, SLF-Rmax, Met-Rmax,
# flat SPIBB, offline CQL). Factored SPIBB on a learned graph is a baseline, not ours: it has an entry of
# its own, and not the colour of the graph.
# The interval MDP and the plug-in MDP of a graph: DIFFERENT colours for DYNOTEARS (ours: blue and
# coral), the full graph (teal and rose) and the flat model (purple and magenta), whose two forms are drawn
# together in the pooled figures (OWN_COLOUR, below: the colour and the marker of these plug-in MDPs); for
# the others (the true graph, PCMCI+, the sparse graph) the plug-in MDP keeps the colour and the marker of
# its graph and is told by its line (dash-dot for ours, dash-dot-dot otherwise).
# With 19 colours some pairs are close, above all for a reader with a colour-vision deficiency (e.g.
# the plug-in MDP of DYNOTEARS and the true graph or the sparse graph, the plug-in MDP of the flat model
# and offline CQL, sparse and full graph, PCMCI+ and DiFaRL): no two methods share a marker, and the
# methods of a close pair have different lines, so that a method is never told by its colour alone.
DOTS, DASHDOT, DASHDOTDOT = (0, (1, 1.6)), (0, (4, 1.4, 1, 1.4)), (0, (4, 1.4, 1, 1.4, 1, 1.4))
DASH3DOT = (0, (4, 1.4, 1, 1.4, 1, 1.4, 1, 1.4))
DOT_DOT_DASH = (0, (1, 1.6, 1, 1.6, 3.2, 1.6))  # dotted with a short dash: the reference without intervals
DASHED, LONG, SHORT, LONG_SHORT = (0, (4, 1.6)), (0, (7.5, 2)), (0, (2, 1.2)), (0, (7, 1.5, 2.5, 1.5))
METHODS = {  # method -> (colour, marker, line); the line of a graph source: of its interval MDP
    "true graph": (GOLD, "o", DOTS),  # the reference: dotted, with a hollow marker
    "DYNOTEARS": (BLUE, "D", "-"),
    "PCMCI+": (RED, "s", "-"),
    "sparse": (PINK, "v", DASHED),
    "full graph": (TEAL, "H", LONG),
    "flat": (PURPLE, "X", SHORT),
    "FOCUS": (INK, "^", DASHED),
    "SL-SPIBB": (BROWN, "8", LONG),
    "flat SPIBB": (SKY, "<", LONG_SHORT),
    "DYNOTEARS SPIBB": (PLUM, "h", DASHDOT),
    "PCMCI+ SPIBB": (ULTRAMARINE, (4, 1, 0), DASHDOTDOT),
    "DiFaRL": (GREEN, "P", SHORT),
    "offline DQN": (WINE, "*", DASHED),
    "offline CQL": (VIOLET, ">", LONG_SHORT),
    "SLF-Rmax": (LIME, "p", LONG),
    "Met-Rmax": (INDIGO, "d", SHORT),
}
# The forms of a graph source with a colour of their own, in every figure: form -> (colour, marker, line
# where it is not ours; None: that of the plug-in MDP of another structure, dash-dot-dot). The plug-in MDP
# of DYNOTEARS, which is ours beside its interval MDP: with its colour it needs no line or marker of its
# own (where it is ours: the solid line of ours, the same diamond). Those of the full graph and of the
# flat model: a marker of their own too, and two different lines (their colours are close with a
# colour-vision deficiency).
OWN_COLOUR = {
    "true graph MDP": (GOLD, "s", DOT_DOT_DASH),  # the second reference: the gold of the true graph, a square
    "DYNOTEARS MDP": (CORAL, "D", None),
    "full graph MDP": (ROSE, (6, 1, 0), None),  # a six-pointed star
    "flat MDP": (MAGENTA, (5, 0, 180), DASH3DOT),  # a pentagon, point down
}
GRAPHS = ["true graph", "DYNOTEARS", "PCMCI+", "sparse", "full graph", "flat"]  # the graph sources
BASELINES = ["DiFaRL", "FOCUS", "SL-SPIBB", "flat SPIBB", "offline CQL", "offline DQN"]
REFERENCE = "true graph"
# The two models of the true graph: references, not methods. Gold, thin dotted lines and hollow markers
# (a circle for the interval MDP, a square for the plug-in MDP), under ours; in the pooled figures they
# are named "... (reference)" and left out of the envelope and of the rings of the tradeoff plane.
REFERENCES = [REFERENCE, f"{REFERENCE} MDP"]
# Factored SPIBB on the graph of a discovery method: a baseline (method -> the discovery method)
SPIBB_ON = {"DYNOTEARS SPIBB": "DYNOTEARS", "PCMCI+ SPIBB": "PCMCI+"}
# A graph source has several forms, drawn in its colour: its interval MDP (the method named as the graph),
# "<graph> MDP" (the plug-in MDP: no promise, dash-dot) and "<graph> SPIBB" (Factored SPIBB on the graph;
# on a learned graph it is a method of the registry, SPIBB_ON).
FORMS = {"IMDP": None, "MDP": "MDP (plug-in)", "SPIBB": "Factored SPIBB"}  # form -> solver (None: certified)
# graph -> its model in the runs (PCMCI+ and DYNOTEARS: the runs without same-step edges, "_lagged")
MODEL = {"true graph": "true graph", "DYNOTEARS": "DYNOTEARS (lagged)", "PCMCI+": "PCMCI+ (lagged)",
         "sparse": "sparse", "full graph": "full-fact", "flat": "flat"}  # fmt: skip
# The two sets of paper figures. A discovery method is drawn as ours (the thickest lines) in the sets
# where it is ours, and not at all in the others: paper/ has no PCMCI+. set -> the discovery methods
# that are ours; with one, it is "CD" in the legends, with two each has its name.
DISCOVERY = {"DYNOTEARS": "dynotears_lagged", "PCMCI+": "pcmci_lagged"}  # discovery method -> its source
SETS = {"paper": ["DYNOTEARS"], "appendix": ["DYNOTEARS", "PCMCI+"]}
# The twin of a discovery method with arrows inside a step allowed: in no figure of this file, only in
# the ablation of the same-step arrows (ablations/ablation_same_step/plot.py). discovery method -> (the
# source of the twin, its model in the runs). A twin has the colour and the marker of its method, a line
# of SAME_STEP_LINES (by form) and a hollow marker; it is never ours.
SAME_STEP = {"DYNOTEARS": ("dynotears", "DYNOTEARS"), "PCMCI+": ("pcmci", "PCMCI+")}
PREFIX = {"true graph": "True", "sparse": "Sparse", "full graph": "Full", "flat": "Flat"}
# line of a form -> (width, style or None: the registry's, order of drawing)
OURS_LINES = {"IMDP": (2.2, "-", 6), "MDP": (2.2, DASHDOT, 5)}
OTHER_LINES = {"IMDP": (1.1, None, 2), "MDP": (1.1, DASHDOTDOT, 2), "SPIBB": (1.1, DASHDOT, 2)}
SAME_STEP_LINES = {"IMDP": (1.3, DASHED, 1.5), "MDP": (1.3, DOTS, 1.5)}  # under its method: see SAME_STEP_MS
SAME_STEP_MS = 5.4  # a larger hollow marker: a ring around the marker of its method where the two coincide
REFERENCE_LINE = (1.3, DOTS, 3)

# The exploratory figures draw more models: every graph source, the KWIK learners and the baselines.
# A model that is a method of the registry has its colour and marker.
GRAPH_SOURCES = ["true graph", "PCMCI+ (lagged)", "DYNOTEARS (lagged)", "sparse", "full-fact", "flat"]
KWIK = ["true graph", "Strehl (fixed hull)", "Strehl (accepted hull)", "Strehl-KWIK",
        "Met-Rmax (fixed hull)", "Met-Rmax (accepted hull)", "Met-Rmax-KWIK"]  # fmt: skip
EXPLORED = (
    {MODEL[graph]: graph for graph in GRAPHS}
    | {name: name for name in BASELINES}
    # a KWIK learner has one colour, and one line style per variant (as in the paper)
    | {name: "SLF-Rmax" for name in KWIK if name.startswith("Strehl")}
    | {name: "Met-Rmax" for name in KWIK if name.startswith("Met-Rmax")}
)  # model of an exploratory figure -> its method in the registry
COLOR = {model: METHODS[method][0] for model, method in EXPLORED.items()}
MARKER = {model: METHODS[method][1] for model, method in EXPLORED.items()}
DASH = (0, (4, 2))
LINE = {m: DASH for m in BASELINES} | {
    "Strehl (accepted hull)": DASH, "Met-Rmax (accepted hull)": DASH,
    "Strehl-KWIK": DOTS, "Met-Rmax-KWIK": DOTS,
}  # fmt: skip

# Choices made by hand for one domain; a domain without an entry takes them from its data (section 3).
#   solver         the certified solver of the domain (default DEFAULT_SOLVER): the box everywhere
#   panels         the sizes of the paper figures with one panel per size
#   detail_size    the size of the exploratory figures drawn at one size
#   graph_size     the size of the paper figure graph_errors
#   baseline_panels  the sizes of the figure against the state of the art (its methods finish at every
#                    size, so it can show larger sizes than `panels`; default: spread over every size)
#   title          the title of a size's panel, from the size (default: from its label in the results)
DOMAIN_SETTINGS = {
    "sysadmin": {
        "solver": "IMDP (IA)",
        "panels": ["2", "4", "6", "8", "9"],
        "baseline_panels": ["2", "4", "6", "8", "10"],  # the paper's
        "detail_size": "7",
        "graph_size": "9",
    },
    # the exact solver runs out of memory at vitals=4 (Sepsis) and from side 5 on (Drone): the box finishes
    "sepsis": {
        "solver": "IMDP (IA)",
        "panels": ["2", "3", "4"],
        "baseline_panels": ["2", "3", "4"],  # the paper shows 2 and 3
        "title": lambda size: f"{size} vital signs",
    },
    "drone": {
        "solver": "IMDP (IA)",
        "panels": ["3-1", "5-1", "7-1", "10-1", "12-1"],
        "baseline_panels": ["3-1", "5-1", "7-1", "9-1", "12-1"],  # the paper's
        "title": lambda size: f"side {size.split('-')[0]}",
    },
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


def kwik_levels(meta: Path) -> dict:
    """method -> "theorem" or "paper", for the KWIK learners of a run (its meta.json): "theorem" when a
    threshold of the learner is "calibrated" (the theorems' level), "paper" otherwise (protocol.KWIK_PAPER)."""
    if not meta.is_file():
        return {}
    methods = json.loads(meta.read_text()).get("config", {}).get("methods", {})
    levels = {}
    for method, spec in methods.items():
        if isinstance(spec, dict) and spec.get("learner") == "kwik":
            options = spec.get("discovery_options") or {}
            calibrated = "calibrated" in (options.get("m"), options.get("m_sub"))
            levels[method] = "theorem" if calibrated else "paper"
    return levels


def load(results: Path, domain: str) -> pd.DataFrame:
    """Every run of the domain as one table (one row per size, seed, number of transitions and method),
    with: source, model, solver, ok, value and promise_value (normalised), total_s, the graph errors
    recall, precision, missing_share, extra_per_var, and kwik_level (kwik_levels; empty for the other
    methods). The sizes of LEFT_OUT are dropped."""
    frames = []
    for source, run in latest_runs(results / domain, domain).items():
        for file in sorted(run.glob("*/uniform/results.csv")):
            d = pd.read_csv(file, low_memory=False)
            if d.empty:
                continue
            d = d[[c for c in KEEP if c in d.columns]].copy()
            d["source"] = source
            d["kwik_level"] = d["method"].map(kwik_levels(file.parent / "meta.json"))
            if source == LEGACY_KWIK:  # the copies of 2026-11-15: the papers' thresholds, named as in KWIK_ARMS
                d["method"] = d["method"].map({v: k for k, v in KWIK_ARMS.items()}).fillna(d["method"])
                d["kwik_level"] = "paper"
            d["size"] = d["size"].astype(str)
            frames.append(d[~d["size"].isin(LEFT_OUT.get(domain, []))])
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
    baseline_panels: list  # the sizes of the figure against the state of the art
    size_title: object  # None, or a function size -> the title of its panel (DOMAIN_SETTINGS "title")
    out: Path
    set: str = "paper"  # the set of paper figures being drawn (SETS): their folder in `out`
    ours: tuple = tuple(SETS["paper"])  # the discovery methods that are ours in that set

    def pick(self, model: str, solver: str | None = None) -> pd.DataFrame:
        """The runs of one model with one solver (solver None: a baseline, whose solver is its name)."""
        x = self.runs[self.runs["model"] == model]
        return x[x["solver"] == (solver or model)]

    def certified(self, model: str) -> pd.DataFrame:
        return self.pick(model, "IMDP" if model == "flat" else self.solver)

    def of(self, method: str) -> pd.DataFrame:
        """The runs of a method of the registry (METHODS), or of a form of a graph source: "DYNOTEARS"
        is its certified interval MDP, "DYNOTEARS MDP" its plug-in MDP, "DYNOTEARS SPIBB" Factored SPIBB."""
        if method in SPIBB_ON:
            return self.pick(MODEL[SPIBB_ON[method]], FORMS["SPIBB"])
        graph, form = split_form(method)
        if form is None:
            return self.pick(method)
        return self.certified(MODEL[graph]) if form == "IMDP" else self.pick(MODEL[graph], FORMS[form])

    def title(self, size) -> str:
        """ "N=7" -> "$N = 7$" (a one-letter name is set as a symbol), "vitals=3" -> "vitals = 3"."""
        if self.size_title is not None:
            return self.size_title(size)
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
    wanted = chosen.get("baseline_panels", [])
    against = [s for s in wanted if s in sizes]
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
        baseline_panels=against if len(against) == len(wanted) and against else spread(sizes, MAX_PANELS),
        size_title=chosen.get("title"),
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
    "savefig.dpi": 500, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
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


def save_to(out: Path, group: str, fig, name: str, data: pd.DataFrame | None = None, csv=None) -> None:
    """The figure as out/<group>/<name>.{pdf,png} and its numbers as out/data/<csv or name>.csv."""
    folder = out / group
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.pdf")
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    if data is not None:
        (out / "data").mkdir(parents=True, exist_ok=True)
        data.to_csv(out / "data" / f"{csv or name}.csv", index=False)
    print(f"  wrote {group}/{name}")


def save(dom: Domain, group: str, fig, name: str, data: pd.DataFrame | None = None) -> None:
    """A figure of a domain: an exploratory one in explore/, a paper one in the folder of the set being
    drawn (paper/ or appendix/), with the set in the name of its csv (paper_<name>.csv)."""
    if group == "paper":
        save_to(dom.out, dom.set, fig, name, data, csv=f"{dom.set}_{name}")
    else:
        save_to(dom.out, group, fig, name, data)


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
# Every figure of this section and of the next is drawn once per set of SETS (paper/, appendix/), with
# the discovery methods of the set as ours (dom.ours); the others of DISCOVERY are left out.
def split_form(method: str) -> tuple[str, str | None]:
    """A method as (graph, form): "DYNOTEARS" is ("DYNOTEARS", "IMDP"), "flat MDP" is ("flat", "MDP");
    a baseline is (its name, None)."""
    if method in GRAPHS:
        return method, "IMDP"
    if method in METHODS:
        return method, None
    graph, _, form = method.rpartition(" ")
    if graph not in GRAPHS or form not in FORMS:
        raise KeyError(f"{method!r} is no method of the registry (METHODS) and no form of a graph source")
    return graph, form


def label_of(method: str, ours: list) -> str:
    """The name of a method in a legend: "CD-IMDP (ours)" (with two discovery methods as ours:
    "DYNOTEARS-IMDP (ours)"), "True-MDP", "Flat-IMDP"; a baseline has its plain name, and Factored SPIBB
    on a learned graph (a baseline) is "CD-SPIBB" (with two discovery methods: "DYNOTEARS-SPIBB")."""
    if method in SPIBB_ON:
        return f"{'CD' if list(ours) == [SPIBB_ON[method]] else SPIBB_ON[method]}-SPIBB"
    graph, form = split_form(method)
    if form is None:
        return method
    if graph in ours:
        return f"{'CD' if len(ours) == 1 else graph}-{form} (ours)"
    named = " (reference)" if _REFERENCES_NAMED[0] and method in REFERENCES else ""
    return f"{PREFIX.get(graph, graph)}-{form}{named}"


_REFERENCES_NAMED = [False]


@contextmanager
def references_named():
    """Inside it (the pooled figures), label_of names the models of the true graph "True-IMDP (reference)"
    and "True-MDP (reference)"."""
    before, _REFERENCES_NAMED[0] = _REFERENCES_NAMED[0], True
    try:
        yield
    finally:
        _REFERENCES_NAMED[0] = before


def look_of(method: str) -> tuple:
    """(colour, marker, line) of a method in the registry: those of its graph source, with the colour and
    the marker of OWN_COLOUR for a form that has them."""
    color, marker, dashes = METHODS[split_form(method)[0]]
    return *OWN_COLOUR.get(method, (color, marker))[:2], dashes


def method_line(method: str, ours: list, **more) -> dict:
    """The line of a method, with its label: the colour and marker of the registry (look_of); ours thick
    (solid with a promise or with a colour of its own, dash-dot without), the reference dotted with a
    hollow marker, the others in the line of the registry (the plug-in MDP of another structure:
    dash-dot-dot, or the line of OWN_COLOUR)."""
    graph, form = split_form(method)
    color, marker, dashes = look_of(method)
    if graph in ours:
        width, line, order = OURS_LINES[form]
        line = "-" if method in OWN_COLOUR else line  # its colour tells it from the interval MDP
    elif graph == REFERENCE:
        width, line, order = REFERENCE_LINE
        line = OWN_COLOUR.get(method, (None,) * 3)[2] or line  # the plug-in MDP: a dot pattern of its own
    else:
        width, line, order = OTHER_LINES[form or "IMDP"]
        line = OWN_COLOUR.get(method, (None,) * 3)[2] or line
    style = paper_line(color, marker, label_of(method, ours), lw=width, ls=line or dashes, zorder=order)
    if graph == REFERENCE and graph not in ours:
        style |= {"mfc": "white", "mec": color, "mew": 0.9}
    return style | more


def method_handle(method: str, ours: list) -> Line2D:
    return Line2D([], [], **method_line(method, ours, ms=4.5))


def drawn(methods: list, ours: list) -> list:
    """The methods of a figure in a set: a discovery method that is not ours in it is left out, with
    Factored SPIBB on its graph."""
    graphs = {m: SPIBB_ON.get(m, split_form(m)[0]) for m in methods}
    return [m for m in methods if graphs[m] not in DISCOVERY or graphs[m] in ours]


def certified(dom: Domain) -> list:
    """The graph sources of the figures of the certified policy, in the set of the domain."""
    return drawn(GRAPHS, dom.ours)


def kept(dom: Domain, method: str) -> pd.DataFrame:
    """The runs of a method at the kept sizes."""
    d = dom.of(method)
    return d[d["size"].isin(dom.kept)]


def value(dom: Domain) -> None:
    """Normalised value of the certified policy against the transitions, one panel per size."""
    n = len(dom.panels)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * max(n, 2), 2.4), sharex=True, sharey=True, squeeze=False)
    rows, handles = [], []
    for method in certified(dom):
        c = cells(dom.of(method))
        style = method_line(method, dom.ours)
        rows.append(c.assign(method=style["label"]))
        for ax, size in zip(axes.flat, dom.panels, strict=False):
            s = c[c["size"] == size].sort_values("n_transitions")
            ax.fill_between(s["n_transitions"], s["lo"], s["hi"], color=style["color"], alpha=0.15, lw=0)
            ax.plot(s["n_transitions"], s["mean"], **(style | {"label": None}))
        handles.append(method_handle(method, dom.ours))
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


def runs_of_method(dom: Domain, method: str) -> pd.DataFrame:
    """dom.of, with SLF-Rmax and Met-Rmax as their arm at the papers' thresholds (KWIK_ARMS, kwik_level)."""
    if method in KWIK_METHODS:
        arm = {v: k for k, v in KWIK_ARMS.items()}[method]
        return dom.runs[(dom.runs["method"] == arm) & dom.runs["kwik_level"].eq("paper")]
    return dom.of(method)


def value_and_held(dom: Domain, methods: list, panels: list, name: str, held_label: str) -> None:
    """A panel per size: the normalised value (top) and the share of runs whose reported value held in
    every state (bottom), against the transitions."""
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * max(n, 2), 4.3), sharex=True, sharey="row", squeeze=False)
    rows, handles = [], []
    for method in methods:
        d = runs_of_method(dom, method)
        if d.empty:
            continue
        val, held = cells(d), cells(d.assign(value=100 * d["safe"]))
        style = method_line(method, dom.ours)
        label = style.pop("label")
        rows += [val.assign(method=label, metric="value"), held.assign(method=label, metric="held")]
        for j, size in enumerate(panels):
            s = val[val["size"] == size].sort_values("n_transitions")
            axes[0, j].plot(s["n_transitions"], s["mean"], **style)
            h = held[held["size"] == size].sort_values("n_transitions")
            axes[1, j].plot(h["n_transitions"], h["mean"], **style)
        handles.append(method_handle(method, dom.ours))
    for j, size in enumerate(panels):
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
    axes[1, 0].set_ylabel(held_label)
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), handlelength=2.6,
               ncol=5 if n >= 4 else 3, columnspacing=1.2, borderaxespad=0)  # fmt: skip
    save(dom, "paper", fig, name, table(rows))


def baselines(dom: Domain) -> None:
    """The plug-in MDPs of ours and of the true graph against the state of the art (with Factored SPIBB
    on the learned graph, a baseline), as in the paper: the normalised value (top) and the share of runs
    whose own value estimate held (bottom). No method here has a promise."""
    methods = [f"{graph} MDP" for graph in dom.ours] + [f"{REFERENCE} MDP"]
    methods += drawn(list(SPIBB_ON), dom.ours)
    methods += ["SL-SPIBB", "DiFaRL", "FOCUS", "flat SPIBB", "SLF-Rmax", "Met-Rmax", "offline CQL",
                "offline DQN"]  # fmt: skip
    value_and_held(dom, methods, dom.baseline_panels, "baselines", "runs whose value\nestimate held (%)")


def imdp_value_held(dom: Domain) -> None:
    """The interval MDPs (the certified solver) of the learned and the assumed graphs: the normalised value
    (top) and the share of runs whose lower bound held in every state (bottom), one panel per size."""
    value_and_held(dom, certified(dom), dom.panels, "imdp_value_held", "runs whose lower\nbound held (%)")


def coverage(dom: Domain) -> None:
    """Share of runs in which every learned interval contains the true probability (column contain_all):
    the event the guarantee relies on, at least 1 - delta of the runs."""
    n = len(dom.panels)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * max(n, 2), 2.4), sharex=True, sharey=True, squeeze=False)
    rows, handles = [], []
    for method in reversed(certified(dom)):  # the true graph on top
        c = cells(dom.of(method), "covered")
        style = method_line(method, dom.ours, zorder=None)
        rows.append(c.assign(method=style["label"]))
        for ax, size in zip(axes.flat, dom.panels, strict=False):
            s = c[c["size"] == size].sort_values("n_transitions")
            ax.plot(s["n_transitions"], s["mean"], **(style | {"label": None}))
        handles.insert(0, method_handle(method, dom.ours))
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


def promise_scatter(dom: Domain) -> None:
    """Promise against true value, one dot per run, coloured by the number of transitions."""
    graphs, lim = [REFERENCE, *dom.ours], (-3.5, 2.0)
    panels = [
        ("certified (ours)", pd.concat([kept(dom, g) for g in graphs])),
        ("plug-in MDP", pd.concat([kept(dom, f"{g} MDP") for g in graphs])),
        ("offline RL baselines", pd.concat([kept(dom, b) for b in BASELINES])),
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
            ax.text(0.04, 0.96, f"reliability: {100 * held:.1f}%", transform=ax.transAxes, va="top",
                    fontsize=8.5, color=INK)  # fmt: skip
        ax.text(0.96, 0.04, "not reliable", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
        ax.set_title(title, pad=4)
        ax.set_xlim(lim)
        ax.set_ylim(-1.5, 1.35)
        ax.set_xlabel("normalised reported value")
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
    """Share of runs whose promise failed against the normalised value, at two numbers of transitions:
    one marker per method (the plug-in MDP of the true graph: hollow); markers that sit on top of each
    other are named beside them."""
    entries = [(method, "certified") for method in certified(dom)]
    entries += [(f"{REFERENCE} MDP", "plug-in")]
    entries += [(b, "baseline") for b in [*drawn(list(SPIBB_ON), dom.ours), *BASELINES]]
    fig, axes = plt.subplots(1, len(SAFETY_BUDGETS), figsize=(8.4, 3.4), sharey=True)
    rows, handles = [], {}
    for ax, n in zip(axes, SAFETY_BUDGETS, strict=False):
        marks = []
        for method, kind in entries:
            d = kept(dom, method)
            d = d[d["ok"] & (d["n_transitions"] == n)]
            if d.empty:
                continue
            violated, val = 100 * (1 - d["safe"].mean()), d["value"].mean()
            style = method_line(method, dom.ours)
            look = dict(marker=style["marker"], mfc=style["color"], mec="white", mew=0.6)
            if kind == "plug-in":
                look |= dict(mfc="white", mec=style["color"], mew=1.0)
            ax.plot(violated, val, ls="", ms=6.5, zorder=3, clip_on=False, **look)
            handles[method] = Line2D([], [], ls="", ms=6, label=style["label"], **look)
            marks.append((violated, val, style["label"]))
            rows.append({"n_transitions": n, "method": style["label"], "kind": kind,
                         "violated_pct": violated, "value": val, "runs": len(d)})  # fmt: skip
        groups = []  # the markers that sit on top of each other
        for violated, val, name in marks:
            for g in groups:
                if abs(g[0] - violated) < 2 and abs(g[1] - val) < 0.025:
                    g[2].append(name)
                    break
            else:
                groups.append([violated, val, [name]])
        top = -np.inf  # their names, one per line beside the markers, kept apart from bottom to top
        for violated, val, names in sorted((g for g in groups if len(g[2]) > 1), key=lambda g: g[1]):
            half = 0.03 * len(names)  # half the height of the text, in units of the y axis
            at = max(val, top + half + 0.02)
            top = at + half
            side = 1 if violated < 50 else -1
            ax.annotate(",\n".join(names), (violated, val), xytext=(violated + 7 * side, at), fontsize=6.5,
                        color=INK2, ha="left" if side > 0 else "right", va="center", linespacing=1.1,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkB=4))  # fmt: skip
        ax.axhline(0, **ZERO_LINE)
        ax.set_xlim(-5, 100)
        ax.set_ylim(-0.45, 1.1)
        ax.set_title(f"{n:,} transitions", pad=4)
        ax.set_xlabel("runs not reliable (%)")
    axes[0].set_ylabel("normalised value")
    fig.tight_layout(w_pad=1.5)
    order = [handles[method] for method, _ in entries if method in handles]
    fig.legend(handles=order, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=6, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0)  # fmt: skip
    save(dom, "paper", fig, "value_vs_safety", pd.DataFrame(rows))


def data_needed(dom: Domain) -> None:
    """Transitions needed for a near-optimal policy against the number of observed states, with the
    log-log slope of each method."""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    rows, handles = [], []
    for method in certified(dom):
        t = needed(dom, kept(dom, method))
        fin = np.isfinite(t["median"].to_numpy())
        st, med = t["states"].to_numpy()[fin], t["median"].to_numpy()[fin]
        slope = np.polyfit(np.log10(st), np.log10(med), 1)[0] if fin.sum() >= 3 else np.nan
        upper = np.minimum(t["q3"].to_numpy()[fin], 5e5)
        style = method_line(method, dom.ours, ms=4)
        label = style.pop("label")
        ax.fill_between(st, t["q1"].to_numpy()[fin], upper, color=style["color"], alpha=0.12, lw=0)
        ax.plot(st, med, **style)
        text = f"{label}  ({slope:.2f})" if np.isfinite(slope) else label
        handles.append(Line2D([], [], **(style | {"label": text, "ms": 4.5})))
        rows.append(t.assign(method=label, slope=slope))
    states_axis(ax)
    ax.set_yscale("log")
    ax.set_xlabel("observed states")
    ax.set_ylabel(f"transitions to reach value {THRESHOLD}")
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="method (log-log slope)", title_fontsize=8,
              alignment="left", handlelength=2.6)  # fmt: skip
    fig.tight_layout()
    save(dom, "paper", fig, "data_needed", table(rows))


def graph_errors(dom: Domain) -> None:
    """At one size: the value, the share of true edges missed and the extra edges per variable, for the
    learned graphs (ours) and the sparse and full graphs. The true graph has no error: it is not drawn."""
    methods = [*dom.ours, "sparse", "full graph"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), sharex=True)
    rows = []
    for method in methods:
        d = kept(dom, method)
        d = d[d["size"] == dom.graph_size]
        style = method_line(method, dom.ours)
        label = style.pop("label")
        for ax, col in zip(axes, ["value", "missing_share", "extra_per_var"], strict=False):
            c = cells(d, col)
            ax.fill_between(c["n_transitions"], c["lo"], c["hi"], color=style["color"], alpha=0.15, lw=0)
            ax.plot(c["n_transitions"], c["mean"], **style)
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
    legend_below(fig, [method_handle(m, dom.ours) for m in methods], y=0.03)
    save(dom, "paper", fig, "graph_errors", table(rows))


# ==== 6. pooled figures ============================================================================
# Twelve figures that count instances, every size together. They are drawn from a table of instances and
# know nothing of where it comes from: the main runs (instances_of, below), or the runs of an ablation,
# whose plot.py builds the same table and calls pooled_figures. One row per instance, number of
# transitions and method:
#   domain, size, seed (and reward, in an ablation)   the instance
#   n_transitions, method                             method: of the registry (METHODS), or a form of a
#                                                     graph source ("DYNOTEARS MDP")
#   ok            the run finished
#   optimality    its normalised value
#   safe          the value the method reported is at most the true value in every state
#   seconds       the time of the run
# An instance that did not finish is neither solved nor safe, and stays in the denominator.
#
# The methods of each family of figures; OURS stands for every discovery method that is ours in the set
# (FigureSet.ours), and a figure draws the methods its instances have.
OURS = "<ours>"
IMDP_BY_GRAPH = ["true graph", OURS, "sparse", "full graph", "flat"]
MDP_BY_GRAPH = ["true graph MDP", f"{OURS} MDP", "flat MDP", "full graph MDP", "FOCUS", "SL-SPIBB",
                "flat SPIBB", "SLF-Rmax", "Met-Rmax", "DiFaRL", "offline DQN", "offline CQL"]  # fmt: skip
VS_BASELINES = [*REFERENCES, OURS, f"{OURS} MDP", "FOCUS", "SL-SPIBB", "flat SPIBB", "SLF-Rmax", "Met-Rmax",
                "full graph", "full graph MDP", "flat", "flat MDP", "DiFaRL", "offline DQN",
                "offline CQL"]  # fmt: skip
# The methods a set of pooled figures leaves out: the sparse graph is in the appendix only
SET_LEFT_OUT = {"paper": ["sparse", "sparse MDP"]}
TRADEOFF = list(VS_BASELINES)  # the planes: the methods of vs_baselines, the two references included
PAIR_PANELS = {"a": VS_BASELINES, "b": TRADEOFF}  # solved_and_tradeoff: the methods of each of its panels
TARGETS = (10, 30, 50, 70, 90)  # tradeoff_data_needed: the shares of the instances to solve (%)
# tradeoff_pareto: the rings around the best method of a data size. They mark no method, so they have
# no colour of a method: neutral, a black circle and a grey square (a blue and an orange ring were the
# colours of the two methods of ours that most often carry them)
SAFEST, MOST_SOLVED = INK, INK2
# reported_vs_true: its panels, those of these methods that the instances have, in this order (OURS: the
# first discovery method that is ours); twelve are four to a row, nine (no model-free method) three
REPORTED = [OURS, f"{OURS} MDP", "flat MDP", "full graph MDP", "FOCUS", "SL-SPIBB", "flat SPIBB", "SLF-Rmax",
            "Met-Rmax", "DiFaRL", "offline DQN", "offline CQL"]  # fmt: skip
# reported_vs_true_top8: the TOP_N most reliable of these candidates (the interval MDPs of the full graph and
# of no graph report a lower bound, as ours does; the true graph is a reference and no candidate)
REPORTED_TOP = [*REPORTED, "full graph", "flat"]
TOP_N = 8
# reported_vs_true_all (appendix/ only): every method of the pooled figures, by reliability
REPORTED_ALL = [REFERENCE, "true graph MDP", "DYNOTEARS", "DYNOTEARS MDP", "PCMCI+", "PCMCI+ MDP", "sparse",
                "full graph", "full graph MDP", "flat", "flat MDP", "FOCUS", "SL-SPIBB", "flat SPIBB", "SLF-Rmax",
                "Met-Rmax", "DiFaRL", "offline DQN", "offline CQL"]  # fmt: skip
# reported_vs_true_selected: these eight, in the order of their reliability (the choice of the paper)
REPORTED_SELECTED = [REFERENCE, OURS, "FOCUS", "SL-SPIBB", "flat SPIBB", "offline DQN", "DiFaRL", "full graph"]
REPORTED_X, REPORTED_Y = (-4.0, 3.0), (-1.3, 1.0)  # the limits, normalised; a run beyond them: at the border
REPORTED_TITLE = 8.0  # the font size of the title of a panel (the name of its method and a share)
REPORTED_HEAD = 0.62  # room above REPORTED_Y for the text of a panel (no run is there: a value is at most 1)
RUN_KEY = ["domain", "size", "seed", "n_transitions", "reward"]  # a run, in the columns the instances have
# One colour per domain, wherever a domain is colour-coded (the names: those of the paper)
# (purple, the colour of Sepsis in earlier figures, is too close to the blue: 13.5 for normal vision and
# 5.1 for deuteranopia in OKLab x 100, against 25.1 and 19.6 for the wine red)
DOMAIN_COLOURS = {"sysadmin": BLUE, "drone": GOLD, "sepsis": WINE}
DOMAIN_NAMES = {"sysadmin": "SysAdmin", "drone": "Drone", "sepsis": "Sepsis"}  # the Sepsis simulator

# The method of a main run -> its name in the pooled figures: the box for every interval MDP, the lagged
# graphs of PCMCI+ and DYNOTEARS, and SLF-Rmax and Met-Rmax (KWIK_ARMS, the papers' thresholds: instances_of).
MAIN_METHODS = {
    "true graph -> IMDP (IA)": "true graph",
    "DYNOTEARS (lagged) -> IMDP (IA)": "DYNOTEARS",
    "PCMCI+ (lagged) -> IMDP (IA)": "PCMCI+",
    "sparse -> IMDP (IA)": "sparse",
    "full-fact -> IMDP (IA)": "full graph",
    "flat IMDP": "flat",
    "true graph -> MDP (plug-in)": "true graph MDP",
    "DYNOTEARS (lagged) -> MDP (plug-in)": "DYNOTEARS MDP",
    "PCMCI+ (lagged) -> MDP (plug-in)": "PCMCI+ MDP",
    "flat -> MDP (plug-in)": "flat MDP",
    "full-fact -> MDP (plug-in)": "full graph MDP",
} | KWIK_ARMS | {name: name for name in BASELINES}
STEPS_S = ["discovery_s", "selection_s", "learning_s", "relax_s", "training_s", "solve_s"]
POOLED_DOMAINS = ["sysadmin", "drone", "sepsis"]  # the domains of the pooled figures, and of all_domains
ALL_DOMAINS = "all_domains"


@dataclass
class FigureSet:
    """Where a set of figures goes and which discovery methods are ours in it (SETS)."""

    name: str  # "paper" or "appendix": the folder in `out`, and the prefix of the csv files
    ours: list
    out: Path

    def methods(self, methods: list) -> list:
        """The methods of a family, with OURS as the discovery methods that are ours."""
        out = [m.replace(OURS, graph) for m in methods for graph in (self.ours if OURS in m else [""])]
        return [m for m in out if m not in SET_LEFT_OUT.get(self.name, [])]

    def is_ours(self, method: str) -> bool:
        return split_form(method)[0] in self.ours


def instances_of(runs: pd.DataFrame, domain: str) -> pd.DataFrame:
    """The table of instances of the pooled figures from the runs of a domain (load): an instance is a
    (size, seed), every size that ran; its time is the sum of the steps of the run. SLF-Rmax and Met-Rmax
    are the rows with the thresholds of the papers' experiments (kwik_level "paper"), and of no other. reported and true_value (reported_vs_true):
    the value the method reported and the true value of its policy, from the start, normalised."""
    d = runs[runs["method"].isin(MAIN_METHODS)]
    kwik = d["method"].isin(KWIK_ARMS)
    d = d[~kwik | d.get("kwik_level", pd.Series("", index=d.index)).eq("paper")]
    d = d.drop_duplicates(["size", "seed", "n_transitions", "method"])
    return pd.DataFrame({
        "domain": domain, "size": d["size"], "seed": d["seed"], "n_transitions": d["n_transitions"],
        "method": d["method"].map(MAIN_METHODS), "ok": d["ok"], "optimality": d["value"],
        "safe": d["safe"].eq(1), "seconds": d.reindex(columns=STEPS_S).fillna(0).sum(axis=1),
        "reported": d["promise_value"], "true_value": d["value"],
    })  # fmt: skip


def judged(instances: pd.DataFrame) -> pd.DataFrame:
    """The instances with `solved` and `safe` as 0/1; one that did not finish is neither, and its
    optimality is minus infinity (below every requirement)."""
    d = instances.copy()
    ok = d["ok"].astype(bool)
    d["optimality"] = d["optimality"].astype(float).where(ok, -np.inf).fillna(-np.inf)
    d["solved"] = (d["optimality"] >= THRESHOLD).astype(float)
    d["safe"] = (ok & d["safe"].eq(1)).astype(float)
    return d


def shares(d: pd.DataFrame, methods: list) -> pd.DataFrame:
    """Per method and number of transitions: the instances, and the % of them solved and safe."""
    d = d[d["method"].isin(methods)]
    by = d.groupby(["method", "n_transitions"])
    out = by.agg(instances=("solved", "size"), solved=("solved", "mean"), safe=("safe", "mean")).reset_index()
    out[["solved", "safe"]] *= 100
    return out


def present(table: pd.DataFrame, methods: list) -> list:
    return [m for m in methods if m in set(table["method"])]


def pooled_legend(fig, methods: list, to: FigureSet, extra=(), y=0.02, ncol=3) -> None:
    handles = [method_handle(m, to.ours) for m in methods] + list(extra)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=7.8,
               handlelength=2.8, columnspacing=1.2, borderaxespad=0)  # fmt: skip


def percent_axis(axis) -> None:
    axis.set_ticks([0, 20, 40, 60, 80, 100])


def short(n: float) -> str:
    """A number of transitions as a label: 500, 2k, 100k."""
    return f"{n / 1000:g}k" if n >= 1000 else f"{n:g}"


def save_pooled(to: FigureSet, folder: str, fig, name: str, data: pd.DataFrame) -> None:
    """The figure in <set>/<folder>/ and its numbers as data/<set>_<folder>_<name>.csv, with the name of
    each method in the figure (`label`) beside its name in the registry."""
    if "method" in data:
        data.insert(1, "label", data["method"].map(lambda m: label_of(m, to.ours)))
    group = f"{to.name}/{folder}" if folder else to.name
    save_to(to.out, group, fig, name, data, csv="_".join(filter(None, [to.name, folder, name])))


def nothing(to: FigureSet, folder: str, name: str, why: str) -> bool:
    print(f"  skipped {to.name}/{folder + '/' if folder else ''}{name}: {why}")
    return False


def solved_vs_transitions(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """% of instances solved (normalised value of at least THRESHOLD) against the transitions."""
    t = shares(d, methods)
    if t.empty:
        return nothing(to, folder, "solved_vs_transitions", "no run of its methods")
    methods = present(t, methods)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    solved_on(ax, t, methods, to)
    fig.tight_layout()
    pooled_legend(fig, methods, to)
    save_pooled(to, folder, fig, "solved_vs_transitions", t)
    return True


def solved_on(ax, t: pd.DataFrame, methods: list, to: FigureSet) -> None:
    """On an axis: the % of instances solved (shares) of each method against the transitions."""
    for method in methods:
        s = t[t["method"] == method].sort_values("n_transitions")
        ax.plot(s["n_transitions"], s["solved"], **method_line(method, to.ours))
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(-4, 104)
    percent_axis(ax.yaxis)
    ax.set_xlabel("transitions")
    ax.set_ylabel(f"instances solved\n(normalised value ≥ {THRESHOLD}, %)")


def optimality_on(ax, d: pd.DataFrame, methods: list, to: FigureSet, low: float = 0.0, high: float = 1.0,
                  regret: bool = False, grey: bool = False, grey_alpha: float = 1.0, faded: tuple = ()) -> list:  # fmt: skip
    """The curves of solved_vs_optimality on an axis, from the runs at OPTIMALITY_N transitions (see it for
    the options). Returns the rows of its csv, one table per method."""
    x = np.linspace(low, high, 201)
    required = 1 - x if regret else x  # a regret of at most x: a normalised value of at least 1 - x
    if grey and not faded:
        band = (0, 100 * (1 - THRESHOLD)) if regret else (100 * THRESHOLD, 100)
        ax.axvspan(*band, color="#f0efec", alpha=grey_alpha, lw=0, zorder=0)
    for a, b in faded:
        ax.axvspan(100 * a, 100 * b, color="#f0efec", lw=0, zorder=0)
    cuts = sorted({low, high, *(c for band in faded for c in band if low < c < high)})
    rows = []
    for method in methods:
        values = np.sort(d[d["method"] == method]["optimality"].to_numpy())
        reached = 100 * (1 - np.searchsorted(values, required - 1e-12, side="left") / len(values))
        # no marker along the curve (201 points): one at its end, as in the legend
        if faded:  # one piece between consecutive cuts (they share their ends), transparent in a band
            for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
                piece = (x >= a - 1e-12) & (x <= b + 1e-12)
                look = {"alpha": FADED} if any(lo <= (a + b) / 2 <= hi for lo, hi in faded) else {}
                end = {"markevery": [-1], "clip_on": False} if k == len(cuts) - 2 else {"marker": None}
                ax.plot(100 * x[piece], reached[piece], **method_line(method, to.ours, **look, **end))
        else:
            ax.plot(100 * x, reached, **method_line(method, to.ours, markevery=[-1], clip_on=False))
        rows.append(pd.DataFrame({"method": method, "n_transitions": OPTIMALITY_N, "instances": len(values),
                                  "regret" if regret else "required_optimality": x, "reached": reached}))  # fmt: skip
    ax.set_xlim(100 * low, 100 * high)
    ax.set_ylim(-4, 104)
    percent_axis(ax.yaxis)
    ax.set_xlabel("allowed normalised regret (%)" if regret else "required normalised value (%)")
    ax.set_ylabel("instances within it (%)" if regret else "instances reaching it (%)")
    return rows


def solved_vs_optimality(d: pd.DataFrame, methods: list, folder: str, to: FigureSet,
                         name: str = "solved_vs_optimality", regret: bool = False, full: bool = True,
                         grey: bool = False, upto: float | None = None, grey_alpha: float = 1.0,
                         fade: bool = False, fade_to: float | None = None,
                         faded: tuple = ()) -> bool:  # fmt: skip
    """At OPTIMALITY_N transitions: % of instances with a normalised value of at least x, x from 0 to 100%
    (`full`) or to THRESHOLD. With `regret`, against the normalised regret (1 - normalised value): % of
    instances with a regret of at most x, x from 0 (`full`) or 1 - THRESHOLD to 100%. With `grey`, the
    part beyond the threshold of "solved" (value above THRESHOLD, regret below 1 - THRESHOLD) on a grey
    background, the curves drawn over it (`grey_alpha`: its opacity). `upto`: the right end of the x axis,
    in place of the one above. With `fade` (and `regret`), the grey band covers the UNSOLVED part instead
    (a regret above 1 - THRESHOLD), and the curves are drawn transparent (FADED) inside it; `fade_to`: the
    regret where the band ends (the axis end without it), the curves solid again beyond it. `faded`: the
    bands (from, to) of the x axis, as fractions, with a grey background and transparent curves, in place
    of the band of `fade`."""
    d = d[d["method"].isin(methods) & (d["n_transitions"] == OPTIMALITY_N)]
    if d.empty:
        return nothing(to, folder, name, f"no run at {OPTIMALITY_N:,} transitions")
    methods = present(d, methods)
    low, high = (0.0 if full or not regret else 1 - THRESHOLD), (1.0 if full or regret else THRESHOLD)
    high = high if upto is None else upto
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    if fade and not faded:
        faded = ((1 - THRESHOLD, fade_to or high),)
    rows = optimality_on(ax, d, methods, to, low, high, regret, grey, grey_alpha, faded)
    fig.tight_layout()
    pooled_legend(fig, methods, to)
    save_pooled(to, folder, fig, name, table(rows))
    return True


def solved_vs_optimality_to90(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_optimality with the required value from 0 to THRESHOLD."""
    return solved_vs_optimality(d, methods, folder, to, name="solved_vs_optimality_to90", full=False)


def solved_vs_optimality_grey(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_optimality, the values above THRESHOLD on a grey background."""
    return solved_vs_optimality(d, methods, folder, to, name="solved_vs_optimality_grey", grey=True)


def solved_vs_regret(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_optimality against the allowed regret, from 1 - THRESHOLD to 100%."""
    return solved_vs_optimality(d, methods, folder, to, name="solved_vs_regret", regret=True, full=False)


def solved_vs_regret_grey(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_regret from 0, the regrets below 1 - THRESHOLD on a grey background."""
    return solved_vs_optimality(d, methods, folder, to, name="solved_vs_regret_grey", regret=True, grey=True)


FADED = 0.2  # the opacity of the curves inside the grey bands of the faded figures (solved_vs_*_fade*)


def solved_vs_regret_fade(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_regret from 0 to 100%, the regrets above 1 - THRESHOLD (not solved) on a grey background,
    the curves transparent there."""
    return solved_vs_optimality(d, methods, folder, to, name="solved_vs_regret_fade", regret=True, fade=True)


def solved_vs_regret_fade90(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_regret_fade with the band from 1 - THRESHOLD to 90% only: the curves solid again above 90%."""
    name = "solved_vs_regret_fade90"
    return solved_vs_optimality(d, methods, folder, to, name=name, regret=True, fade=True, fade_to=0.9)


def solved_vs_regret_fade_ends(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_regret from 0 to 100%, grey and with transparent curves at its two ends: below
    1 - THRESHOLD and above 90%; solid in between."""
    name, ends = "solved_vs_regret_fade_ends", ((0.0, 1 - THRESHOLD), (0.9, 1.0))
    return solved_vs_optimality(d, methods, folder, to, name=name, regret=True, faded=ends)


def solved_vs_optimality_fade_ends(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_vs_optimality from 0 to 100%, grey and with transparent curves at its two ends: below 10% and
    above THRESHOLD; solid in between (solved_vs_regret_fade_ends against the normalised value)."""
    name = "solved_vs_optimality_fade_ends"
    return solved_vs_optimality(d, methods, folder, to, name=name, faded=((0.0, 0.1), (THRESHOLD, 1.0)))


FADED_ENDS = ((0.0, 0.1), (THRESHOLD, 1.0))  # the grey bands of solved_vs_optimality_fade_ends


def solved_vs_time(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """At REFERENCE_N transitions: % of instances solved within a time (a step at every solved instance)."""
    d = d[d["method"].isin(methods) & (d["n_transitions"] == REFERENCE_N)]
    if d.empty:
        return nothing(to, folder, "solved_vs_time", f"no run at {REFERENCE_N:,} transitions")
    methods = present(d, methods)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    rows = []
    for method in methods:
        x = d[d["method"] == method]
        seconds = np.sort(x[x["solved"] == 1]["seconds"].to_numpy())
        if not len(seconds):  # solves nothing: in the legend, with no curve
            continue
        solved = 100 * np.arange(1, len(seconds) + 1) / len(x)
        style = method_line(method, to.ours, markevery=[-1])  # a marker where the curve ends
        ax.step(np.r_[seconds[:1], seconds], np.r_[0, solved], where="post", **style)
        rows.append(pd.DataFrame({"method": method, "n_transitions": REFERENCE_N, "instances": len(x),
                                  "seconds": seconds, "solved": solved}))  # fmt: skip
    ax.set_xscale("log")
    ax.set_ylim(-4, 104)
    percent_axis(ax.yaxis)
    ax.set_xlabel("time of an instance (s)")
    ax.set_ylabel("instances solved\nwithin that time (%)")
    fig.tight_layout()
    pooled_legend(fig, methods, to)
    save_pooled(to, folder, fig, "solved_vs_time", table(rows))
    return True


def label_points(ax, points: pd.DataFrame, near: float, above, color) -> None:
    """Label the points (solved, safe, n_transitions) with their number of transitions. Successive points
    less than `near` apart share one label, e.g. "5k–500k"; above: a function of the label's rank."""
    groups = []
    for row in points.sort_values("n_transitions").itertuples():
        last = groups[-1] if groups else None
        if last and abs(last[0][-1] - row.solved) < near and abs(last[1][-1] - row.safe) < near:
            for values, value in zip(last, (row.solved, row.safe, row.n_transitions), strict=True):
                values.append(value)
        else:
            groups.append(([row.solved], [row.safe], [row.n_transitions]))
    for i, (xs, ys, ns) in enumerate(groups):
        up = above(i)
        text = short(ns[0]) if len(ns) == 1 else f"{short(ns[0])}–{short(ns[-1])}"
        ax.annotate(text, (np.mean(xs), max(ys) if up else min(ys)), xytext=(0, 5 if up else -5),
                    textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=5.5,
                    color=color, zorder=10)  # fmt: skip


def scatter_path(ax, x, y, areas, style: dict) -> None:
    """The markers of a path, of the given areas, as the line's (hollow for the reference)."""
    hollow = "mfc" in style
    ax.scatter(x, y, s=areas, marker=style["marker"], facecolor="white" if hollow else style["color"],
               lw=0.9 if hollow else 0.4, edgecolor=style["color"] if hollow else "white",
               zorder=style["zorder"] + 1)  # fmt: skip


IDEAL = "#e8000b"  # the ideal corner of the planes: a bright red, to stand out


def plane_axes(ax, small: bool = False, square: bool = True) -> None:
    """The axes of the plane % solved (x) against % safe (y), with the ideal corner marked (a star).
    small: a panel smaller than the figure of `plane`, where "ideal" goes above the star (below it, it
    would lie on the paths that end near the corner)."""
    ax.plot(100, 100, marker="*", color=IDEAL, ms=9 if small else 12, ls="", zorder=10, clip_on=False)
    if small:
        ax.annotate("ideal", (100, 100), xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.5, color=IDEAL, fontweight="bold",
                    annotation_clip=False)  # fmt: skip
    else:
        ax.annotate("ideal", (100, 100), xytext=(-7, -11), textcoords="offset points", ha="right",
                    fontsize=7.5, color=IDEAL, fontweight="bold")  # fmt: skip
    ax.set_xlim(-4, 106)
    ax.set_ylim(-4, 106)
    if square:  # a split panel (split_figures) is wider than tall, to fill half the text width
        ax.set_aspect("equal")
    percent_axis(ax.xaxis)
    percent_axis(ax.yaxis)


def plane(t: pd.DataFrame, methods: list, to: FigureSet, labels: bool):
    """The plane % solved (x) against % safe (y): one path per method as data grow, larger markers for
    more data. With `labels`, the points of ours carry their number of transitions, and the other paths
    at their end only (the starts sit on top of each other)."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    plane_on(ax, t, methods, to, labels)
    ax.set_xlabel(SOLVED_AXIS)
    ax.set_ylabel(SAFE_AXIS)
    return fig, ax


def plane_on(ax, t: pd.DataFrame, methods: list, to: FigureSet, labels: bool, size: float = 1.0,
             square: bool = True) -> None:
    """On an axis: the plane of `plane`, without the names of its axes. size: the scale of the markers
    (a panel smaller than the figure of `plane` takes smaller ones)."""
    ns = sorted(t["n_transitions"].unique())
    area = {n: size * (9 + 36 * i / max(len(ns) - 1, 1)) for i, n in enumerate(ns)}
    plane_axes(ax, small=size < 1, square=square)
    for method in methods:
        p = t[t["method"] == method].sort_values("n_transitions")
        style = method_line(method, to.ours)
        ax.plot(p["solved"], p["safe"], **(style | {"marker": None}))
        scatter_path(ax, p["solved"], p["safe"], [area[n] for n in p["n_transitions"]], style)
        if labels:
            # the two forms of a method of ours start on opposite sides, and so do two methods of ours,
            # whose paths often coincide
            graph, form = split_form(method)
            first = (form == "MDP") != (graph in to.ours[1:])
            points = p if to.is_ours(method) else p.tail(1)
            label_points(ax, points, 3, lambda i, first=first: (i % 2 == 0) == first, style["color"])


SOLVED_AXIS = f"instances solved (normalised value ≥ {THRESHOLD}, %)"
SAFE_AXIS = "instances reliable (reported value ≤ true value, %)"


def tradeoff(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """% of instances solved (x) against % safe (y): one path per method as the data grow from more than
    100 transitions, the points labelled with their number of transitions."""
    t = shares(d[d["n_transitions"] > 100], methods)
    if t.empty:
        return nothing(to, folder, "tradeoff", "no run of its methods with more than 100 transitions")
    methods = present(t, methods)
    fig, _ = plane(t, methods, to, labels=True)
    fig.tight_layout()
    pooled_legend(fig, methods, to)
    save_pooled(to, folder, fig, "tradeoff", t)
    return True


def envelope(solved: np.ndarray, safe: np.ndarray) -> tuple[float, float, float]:
    """(a, b, p) of the superellipse (x/a)^p + (y/b)^p = 1 of the smallest area with no point beyond it:
    a search over a and b in steps of 0.25 (up to 112), with the smallest p that keeps every point."""
    x, y = np.asarray(solved, dtype=float), np.asarray(safe, dtype=float)
    best = None
    for a in np.arange(x.max() + 0.25, 112.25, 0.25):
        for b in np.arange(y.max() + 0.25, 112.25, 0.25):
            u, v = x / a, y / b
            low, high = 1.0, 200.0
            if (u**high + v**high).max() > 1:
                continue
            for _ in range(40):
                mid = (low + high) / 2
                low, high = (low, mid) if (u**mid + v**mid).max() <= 1 else (mid, high)
            area = a * b * math.gamma(1 + 1 / high) ** 2 / math.gamma(1 + 2 / high)
            if best is None or area < best[0]:
                best = (area, a, b, high)
    return best[1:]


def tradeoff_pareto(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """The tradeoff plane with the envelope of all points (the smallest superellipse with none beyond it;
    the reference is left out) and, per data size, a ring around the method that is best on each axis."""
    t = shares(d[d["n_transitions"] > 100], methods)
    compared = t[~t["method"].isin(REFERENCES)]
    if compared.empty:
        return nothing(to, folder, "tradeoff_pareto", "no run of its methods with more than 100 transitions")
    methods = present(t, methods)
    fig, ax = plane(t, methods, to, labels=False)
    t = pareto_on(ax, t, compared)
    fig.tight_layout()
    extra = [Line2D([], [], color=MUTED, lw=1.6, label="envelope of all points"), *ring_handles()]
    pooled_legend(fig, methods, to, extra)
    save_pooled(to, folder, fig, "tradeoff_pareto", t)
    return True


RINGS = [  # column, the order that picks the best (a tie on the axis: the better on the other one), ...
    ("safest", ["safe", "solved"], SAFEST, "o", 110, True),
    ("most_solved", ["solved", "safe"], MOST_SOLVED, "s", 125, False),
]


def ring_handles() -> list:
    ring = dict(ls="", mfc="none", mew=1.2, ms=7)
    return [
        Line2D([], [], marker="o", mec=SAFEST, label="most reliable of a data size", **ring),
        Line2D([], [], marker="s", mec=MOST_SOLVED, label="most solved of a data size", **ring),
    ]


def pareto_on(ax, t: pd.DataFrame, compared: pd.DataFrame, labels: bool = True, size: float = 1.0,
              rings: bool = True):
    """On a plane of paths: the envelope of the points of `compared` and, per data size, a ring around the
    method that is best on each axis, with its number of transitions (`labels`; size: the scale of the
    rings). Returns `t` with the columns safest, most_solved and envelope_{a, b, p}."""
    a, b, p = envelope(compared["solved"], compared["safe"])
    angle = np.linspace(0, np.pi / 2, 800)
    cx, cy = a * np.sin(angle) ** (2 / p), b * np.cos(angle) ** (2 / p)
    ax.fill_between(cx, 0, cy, color=MUTED, alpha=0.08, lw=0, zorder=0)
    ax.plot(cx, cy, color=MUTED, lw=1.6, zorder=1.5)
    t = t.assign(safest=False, most_solved=False, envelope_a=a, envelope_b=b, envelope_p=p)
    for column, order, color, marker, area, up in RINGS if rings else []:
        best = compared.sort_values(order, ascending=False).groupby("n_transitions").head(1)
        t.loc[best.index, column] = True
        ax.scatter(best["solved"], best["safe"], s=size * area, marker=marker, facecolor="none",
                   edgecolor=color, lw=1.2 if size == 1 else 1.0, zorder=9)  # fmt: skip
        if labels:
            clear = best.assign(safe=best["safe"] + (2.2 if up else -2.2))  # the labels clear the rings
            label_points(ax, clear, 9, lambda _, up=up: up, color)
    return t


PAIR_SIZE = (7.0, 3.45)  # solved_and_tradeoff, in inches with its legend: the text width of the paper
PAIR_LEGEND = dict(ncol=6, fontsize=6.6)  # its one legend: columns, and a font no smaller than 6.5 pt
PAIR_MARKS = 0.5  # the markers and rings of its plane, against those of tradeoff_pareto


PAIR_LETTERS = dict(gap=0.045, fontsize=8.5)  # the letters under the legend: gap in inches, font size


def pair_letters(fig, axes) -> list:
    """(a), (b), ... BELOW the legend of a figure of panels, each centred under its own panel (the middle
    of the axes area, not of a half of the figure), in the ink of the text. The legend sits at the bottom
    edge of the figure, so the letters hang below it: the tight bounding box of the file includes them."""
    y = -PAIR_LETTERS["gap"] / fig.get_figheight()
    out = []
    for letter, axis in zip("abcdefgh", axes, strict=False):
        box = axis.get_position()
        out.append(fig.text((box.x0 + box.x1) / 2, y, f"({letter})", ha="center", va="top",
                            fontsize=PAIR_LETTERS["fontsize"], color=INK))  # fmt: skip
    return out


def solved_and_tradeoff(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """Two panels with ONE legend, for the text width of the paper: (a) vs_baselines/solved_vs_transitions
    and (b) tradeoff_pareto, each drawn by the code of its own figure (solved_on; plane_on and pareto_on),
    from the same tables. The methods of the two panels are PAIR_PANELS (`methods` is not used). The legend
    has every method of either panel once, then the envelope and the two rings of (b). Its csv is the
    index of that legend (method, label, panels: "a", "b" or "a+b"); the numbers are in the csv files of
    the two figures. The numbers of transitions beside the rings are left out (no room at this size)."""
    methods, plane_methods = to.methods(PAIR_PANELS["a"]), to.methods(PAIR_PANELS["b"])
    left = shares(d, methods)
    right = shares(d[d["n_transitions"] > 100], plane_methods)
    compared = right[~right["method"].isin(REFERENCES)]
    if left.empty or compared.empty:
        return nothing(to, folder, "solved_and_tradeoff", "no run of the methods of one of its panels")
    methods, plane_methods = present(left, methods), present(right, plane_methods)
    fig = plt.figure(figsize=PAIR_SIZE)
    listed = methods + [m for m in plane_methods if m not in methods]
    extra = [Line2D([], [], color=MUTED, lw=1.6, label="envelope of all points"), *ring_handles()]
    handles = [method_handle(m, to.ours) for m in listed] + extra
    rows = math.ceil(len(handles) / PAIR_LEGEND["ncol"])
    foot = (0.115 * rows + 0.06) / PAIR_SIZE[1]  # the legend: its rows, in the share of the height
    top, bottom = 0.975, foot + 0.50 / PAIR_SIZE[1]  # the letters are under the legend: room for the x names
    side = (top - bottom) * PAIR_SIZE[1] / PAIR_SIZE[0]  # the plane is square: its width, in the figure's
    plane_ax = fig.add_axes([0.985 - side, bottom, side, top - bottom])
    ax = fig.add_axes([0.085, bottom, 0.985 - side - 0.085 - 0.075, top - bottom])
    solved_on(ax, left, methods, to)
    plane_on(plane_ax, right, plane_methods, to, labels=False, size=PAIR_MARKS)
    pareto_on(plane_ax, right, compared, labels=False, size=PAIR_MARKS)
    plane_ax.set_xlabel("instances solved (%)")
    plane_ax.set_ylabel("instances reliable (%)")
    spacing = dict(handlelength=2.6, columnspacing=1.0, handletextpad=0.5, labelspacing=0.3, borderaxespad=0)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), **spacing, **PAIR_LEGEND)
    pair_letters(fig, (ax, plane_ax))
    panels = (("a", methods), ("b", plane_methods))
    index = pd.DataFrame(
        {"method": listed, "panels": ["+".join(x for x, ms in panels if m in ms) for m in listed]}
    )
    save_pooled(to, folder, fig, "solved_and_tradeoff", index)
    return True


# reliable_vs_transitions: the two checks of a reported value, name of the figure -> (column of
# reliable_shares, name of its y axis). "start": the value of the policy from the start, over the runs that
# returned a value (the "reliability" of reported_vs_true); "everywhere": in every state, over all the
# instances (the y axis of the tradeoff planes: one that did not finish is not reliable)
RELIABLE = {
    "reliable_vs_transitions": ("reliable", "runs reliable (%)"),
    "reliable_everywhere_vs_transitions": ("reliable_everywhere", "instances reliable in every state (%)"),
}
RELIABLE_PAIRS = {  # the two-panel figure of each check -> the figure of its panel (b)
    "solved_and_reliable": "reliable_vs_transitions",
    "solved_and_reliable_everywhere": "reliable_everywhere_vs_transitions",
}
COMPACT_SIZE = (3.3, 3.3)  # tradeoff_pareto_compact, in inches with its legend: half of PAIR_SIZE's width


def reliable_shares(d: pd.DataFrame, methods: list) -> pd.DataFrame:
    """Per method and number of transitions: the instances, the runs that returned a value (finished, with
    a reported and a true value), and the % reliable three ways. reliable: reported at most the true value
    from the start, over the returned runs (nan with none); reliable_of_all: the same over all the
    instances; reliable_everywhere: reported at most the true value in every state (`safe`), over all the
    instances. Without the columns reported and true_value, the first two are nan."""
    d = d[d["method"].isin(methods)].copy()
    if {"reported", "true_value"} <= set(d.columns):
        d["returned"] = d["ok"].astype(bool) & d["reported"].notna() & d["true_value"].notna()
        d["held"] = d["returned"] & (d["reported"] <= d["true_value"] + 1e-9)
    else:
        d["returned"], d["held"] = np.nan, np.nan
    by = d.groupby(["method", "n_transitions"])
    out = by.agg(instances=("safe", "size"), returned=("returned", "sum"), held=("held", "sum"),
                 reliable_everywhere=("safe", "mean")).reset_index()  # fmt: skip
    out["reliable"] = 100 * out["held"] / out["returned"].where(out["returned"] > 0)
    out["reliable_of_all"] = 100 * out["held"] / out["instances"]
    out["reliable_everywhere"] *= 100
    return out.drop(columns="held")


def reliable_on(ax, t: pd.DataFrame, methods: list, to: FigureSet, column: str, ylabel: str) -> None:
    """On an axis: the % reliable (a column of reliable_shares) of each method against the transitions,
    with the lines, markers and order of solved_on (ours on top, where lines coincide at 100%)."""
    for method in methods:
        s = t[t["method"] == method].sort_values("n_transitions").dropna(subset=[column])
        ax.plot(s["n_transitions"], s[column], **method_line(method, to.ours))
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(-4, 104)
    percent_axis(ax.yaxis)
    ax.set_xlabel("transitions")
    ax.set_ylabel(ylabel)


def reliable_table(d: pd.DataFrame, methods: list, name: str) -> pd.DataFrame:
    """reliable_shares of the methods that have a value of the check of the figure `name` (RELIABLE)."""
    t = reliable_shares(d, methods)
    return t[t["method"].isin(t.dropna(subset=[RELIABLE[name][0]])["method"])]


def reliable_vs_transitions(d: pd.DataFrame, methods: list, folder: str, to: FigureSet, name=None) -> bool:
    """% of the runs whose reported value is reliable (at most the true value of the policy, from the
    start; over the runs that returned a value: the "reliability" of reported_vs_true) against the
    transitions, with the methods and lines of solved_vs_transitions. The interval MDP of ours reports a
    lower bound, the other methods their own estimate."""
    name = name or "reliable_vs_transitions"
    column, ylabel = RELIABLE[name]
    t = reliable_table(d, methods, name)
    if t.empty:
        return nothing(to, folder, name, "no run of its methods reported a value")
    methods = present(t, methods)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    reliable_on(ax, t, methods, to, column, ylabel)
    fig.tight_layout()
    pooled_legend(fig, methods, to)
    save_pooled(to, folder, fig, name, t)
    return True


def reliable_everywhere_vs_transitions(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """reliable_vs_transitions with the stricter check: the reported value is at most the true one in
    EVERY state (`safe`), as a % of all the instances (one that did not finish is not reliable), which is
    the y axis of the tradeoff planes."""
    return reliable_vs_transitions(d, methods, folder, to, name="reliable_everywhere_vs_transitions")


def solved_and_reliable(d: pd.DataFrame, methods: list, folder: str, to: FigureSet, name=None) -> bool:
    """Two panels with ONE legend, of the size of solved_and_tradeoff: (a) vs_baselines/solved_vs_transitions
    and (b) vs_baselines/reliable_vs_transitions, each drawn by the code of its own figure (solved_on,
    reliable_on) from the same tables, with the same axis of transitions and of percentages. Its csv is
    the index of the legend (method, label, panels); the numbers are in the csv files of the two figures."""
    name = name or "solved_and_reliable"
    column, ylabel = RELIABLE[RELIABLE_PAIRS[name]]
    left, right = shares(d, methods), reliable_table(d, methods, RELIABLE_PAIRS[name])
    if left.empty or right.empty:
        return nothing(to, folder, name, "no run of the methods of one of its panels")
    left_methods, right_methods = present(left, methods), present(right, methods)
    listed = left_methods + [m for m in right_methods if m not in left_methods]
    handles = [method_handle(m, to.ours) for m in listed]
    fig = plt.figure(figsize=PAIR_SIZE)
    rows = math.ceil(len(handles) / PAIR_LEGEND["ncol"])
    foot = (0.115 * rows + 0.06) / PAIR_SIZE[1]  # the legend: its rows, in the share of the height
    top, bottom = 0.975, foot + 0.50 / PAIR_SIZE[1]  # the letters are under the legend: room for the x names
    width = (0.99 - 0.07 - 0.075) / 2  # two panels of one width, with room for the y name of the second
    ax = fig.add_axes([0.07, bottom, width, top - bottom])
    second = fig.add_axes([0.99 - width, bottom, width, top - bottom])
    solved_on(ax, left, left_methods, to)
    ax.set_ylabel("instances solved (%)")
    # the name of the y axis must fit the height of the panel: without "instances" for the longer one
    ylabel = ylabel.replace("instances reliable in", "reliable in")
    reliable_on(second, right, right_methods, to, column, ylabel)
    spacing = dict(handlelength=2.6, columnspacing=1.0, handletextpad=0.5, labelspacing=0.3, borderaxespad=0)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), **spacing, **PAIR_LEGEND)
    pair_letters(fig, (ax, second))
    panels = (("a", left_methods), ("b", right_methods))
    index = pd.DataFrame(
        {"method": listed, "panels": ["+".join(x for x, ms in panels if m in ms) for m in listed]}
    )
    save_pooled(to, folder, fig, name, index)
    return True


def solved_and_reliable_everywhere(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_and_reliable with the check in every state in (b): reliable_everywhere_vs_transitions."""
    return solved_and_reliable(d, methods, folder, to, name="solved_and_reliable_everywhere")


# solved_reliably_vs_states: how large a problem a method solves reliably, with all the data. An instance
# counts when its policy has a normalised value of at least THRESHOLD AND the value the method reported is
# at most the true value of that policy (the check of reported_vs_true); one that did not finish, or
# returned no value, is neither, and stays in the denominator. Drawn over several domains: a point per
# distinct number of observed states (the instances of the domains with that number together) in the csv,
# and in the figure the smoothed line of each method through its points (trend), not the points.
STATES_BUDGET = 500_000  # the number of transitions of the figure
SEPSIS_STATES = {2: 144, 3: 288, 4: 1440}  # vital signs -> observed states
STATES_OF = {"sysadmin": lambda n: 2**n, "drone": lambda n: n**3, "sepsis": SEPSIS_STATES.get}
STATES_AXIS = f"instances solved (≥ {THRESHOLD}) and reliable (%)"
# The line of a method (trend): a Gaussian-weighted average of its shares on the log2 of the observed states,
# weighted by the instances, with this standard deviation IN OCTAVES of problem size (a doubling of states)
SMOOTH_OCTAVES = 2.5
SMOOTH_GRID = 200  # the points of a line, evenly spaced on the log2 between its smallest and largest size
SMOOTH_THIN = 0.55  # the line width of the methods that are neither ours nor a reference, against the usual


def observed_states(domain: str, size) -> float:
    """The observed states of a size of a domain of the paper (STATES_OF: 2^N machines; side^3 of the grid
    with one drone, a size "<side>-1" of the main runs; the vital signs of Sepsis), nan when unknown. The
    main runs record the same numbers (n_observed_states); an ablation does not, hence the rule."""
    first, _, drones = str(size).partition("-")
    rule = STATES_OF.get(domain)
    if rule is None or not first.strip().isdigit() or drones not in ("", "1"):
        return float("nan")
    states = rule(int(first))
    return float("nan") if states is None else float(states)


def trend(states, shares, instances) -> tuple:
    """The line of a method: the average of its shares weighted by the instances and by a Gaussian on the log2
    of the states (SMOOTH_OCTAVES wide), on a grid between its smallest and largest size. (states, values)."""
    x, y, n = np.log2(np.asarray(states, float)), np.asarray(shares, float), np.asarray(instances, float)
    g = np.linspace(x.min(), x.max(), SMOOTH_GRID)
    w = n[None, :] * np.exp(-0.5 * ((g[:, None] - x[None, :]) / SMOOTH_OCTAVES) ** 2)
    return 2.0**g, (w * y[None, :]).sum(1) / w.sum(1)


def reliably_by_states(d: pd.DataFrame, methods: list, budget: int = STATES_BUDGET,
                       everywhere: bool = False) -> pd.DataFrame:  # fmt: skip
    """Per method and number of observed states, at `budget` transitions: the instances, how many are
    solved, reliable and both (counts `*_n`, and % of ALL the instances), and the instances of each
    domain. Reliable: the reported value from the start at most the true one; with `everywhere`, in every
    state (`safe`). Empty without the reported values."""
    if not {"reported", "true_value"} <= set(d.columns):
        return pd.DataFrame()
    d = d[d["method"].isin(methods) & (d["n_transitions"] == budget)].copy()
    d["states"] = [observed_states(a, b) for a, b in zip(d["domain"], d["size"], strict=True)]
    d = d.dropna(subset=["states"])
    if d.empty:
        return pd.DataFrame()
    returned = d["ok"].astype(bool) & d["reported"].notna() & d["true_value"].notna()
    if everywhere:
        d["reliable_n"] = (d["ok"].astype(bool) & d["safe"].eq(1)).astype(float)
    else:
        d["reliable_n"] = (returned & (d["reported"] <= d["true_value"] + 1e-9)).astype(float)
    d["solved_n"] = d["solved"].astype(float)
    d["solved_reliably_n"] = d["solved_n"] * d["reliable_n"]
    key = ["method", "states"]
    counts = ["solved_n", "reliable_n", "solved_reliably_n"]
    out = d.groupby(key).agg(instances=("solved_n", "size"), **{c: (c, "sum") for c in counts}).reset_index()
    for c in counts:
        out[c.removesuffix("_n")] = 100 * out[c] / out["instances"]
    per_domain = d.groupby([*key, "domain"]).size().unstack("domain", fill_value=0)
    per_domain.columns = [f"instances_{name}" for name in per_domain.columns]
    out = out.merge(per_domain.reset_index(), on=key).sort_values(key, ignore_index=True)
    out.insert(len(key), "n_transitions", budget)
    return out


def compact_frame(fig, references: int) -> list:
    """The axes area [left, bottom, width, height] of a compact figure (COMPACT_SIZE), a square: the one
    tradeoff_pareto_compact leaves above its legend (the envelope, the two rings, the references it has
    and the ideal corner, in two columns). solved_reliably_vs_states takes the same area, so that
    the two are a pair side by side."""
    rows = math.ceil((4 + references) / 2)
    bottom = (0.125 * rows + 0.06 + 0.42) / COMPACT_SIZE[1]  # the legend and the x name, under the plane
    side = min(0.975 - 0.155, (0.94 - bottom) * COMPACT_SIZE[1] / COMPACT_SIZE[0])  # a square plane
    left = 0.155 + (0.975 - 0.155 - side) / 2
    return [left, bottom, side, side * COMPACT_SIZE[0] / COMPACT_SIZE[1]]


def compact_canvas(fig) -> None:
    """Keep the whole canvas of a compact figure when it is saved (the paper style crops a figure to what
    is drawn): an invisible frame along its edges, so that the compact figures all have the size
    COMPACT_SIZE and their axes areas (compact_frame) line up side by side."""
    frame = mpl.patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False, lw=0, ec="none")
    fig.add_artist(frame)


COMPACT_LEGEND = dict(handlelength=2.6, columnspacing=1.0, handletextpad=0.5, labelspacing=0.3,
                      borderaxespad=0, ncol=2, fontsize=6.6)  # fmt: skip


def states_on(ax, t: pd.DataFrame, methods: list, to: FigureSet, smooth: bool = True) -> pd.DataFrame:
    """The lines of solved_reliably_vs_states on an axis, from the shares of reliably_by_states: one smoothed
    line per method (trend), ours and the references at their width, the others thinner. Without `smooth`
    (one domain: a few sizes), the shares themselves, with a marker at each size. Returns the lines
    (method, states, smoothed: the share itself without `smooth`)."""
    lines = []
    for method, s in t.groupby("method"):
        if smooth:
            states, values = trend(s["states"], s["solved_reliably"], s["instances"])
        else:
            states, values = s["states"].to_numpy(), s["solved_reliably"].to_numpy()
        lines.append(pd.DataFrame({"method": method, "states": states, "smoothed": values}))
    lines = pd.concat(lines, ignore_index=True)
    for method in methods:
        s = lines[lines["method"] == method]
        style = method_line(method, to.ours)
        if split_form(method)[0] not in to.ours and method not in REFERENCES:
            style["lw"] *= SMOOTH_THIN
        ax.plot(s["states"], s["smoothed"], **(style if not smooth else style | {"marker": None}))
    states_axis(ax)
    ax.set_xlim(t["states"].min() / 1.25, t["states"].max() * 1.25)
    ax.set_ylim(-4, 104)
    percent_axis(ax.yaxis)
    ax.set_xlabel("observed states")
    ax.set_ylabel(STATES_AXIS)
    return lines


def solved_reliably_vs_states(d: pd.DataFrame, methods: list, folder: str, to: FigureSet,
                              name: str = "solved_reliably_vs_states", everywhere: bool = False) -> bool:  # fmt: skip
    """Over several domains, with STATES_BUDGET transitions: the % of the instances that are solved
    (normalised value of at least THRESHOLD) AND reliable (a reported value at most the true one), against
    the observed states. One line per method: its shares per distinct number of states, smoothed (trend);
    the shares themselves are not drawn. On the canvas and in the axes area of
    tradeoff_pareto_compact (compact_frame), its pair side by side, with no method in the legend (the
    references only). Its csv: the share per method and number of states (row = share), the lines
    (row = line: the states of the grid and `smoothed`) and SMOOTH_OCTAVES. With `everywhere`, reliable
    in every state (`safe`)."""
    if STATES_BUDGET not in set(d["n_transitions"]):
        return nothing(to, folder, name, f"no run with {STATES_BUDGET:,} transitions")
    t = reliably_by_states(d, methods, everywhere=everywhere)
    if t.empty:
        return nothing(to, folder, name, "no reported value, or no size with known observed states")
    methods = present(t, methods)
    references = [m for m in methods if m in REFERENCES]
    fig = plt.figure(figsize=COMPACT_SIZE)
    ax = fig.add_axes(compact_frame(fig, len(references)))
    lines = states_on(ax, t, methods, to, smooth=d["domain"].nunique() > 1)  # one domain: its shares
    ax.set_ylabel(STATES_AXIS, y=0.46)  # as long as the canvas is high: a little down, to stay on it
    if references:
        fig.legend(handles=[method_handle(m, to.ours) for m in references], loc="lower center",
                   bbox_to_anchor=(0.5, 0.0), **COMPACT_LEGEND)  # fmt: skip
    compact_canvas(fig)
    table = pd.concat([t.assign(row="share"), lines.assign(row="line", n_transitions=STATES_BUDGET)])
    table.insert(0, "row", table.pop("row"))
    save_pooled(to, folder, fig, name, table.assign(octaves=SMOOTH_OCTAVES))
    return True


def solved_reliably_vs_states_everywhere(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """solved_reliably_vs_states with the check in every state (`safe`)."""
    name = "solved_reliably_vs_states_everywhere"
    return solved_reliably_vs_states(d, methods, folder, to, name=name, everywhere=True)


def tradeoff_pareto_compact(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """The plane of tradeoff_pareto as panel (b) of solved_and_tradeoff draws it (small markers and rings,
    no number of transitions beside the rings), alone and WITHOUT the methods in its legend: for half the
    text width, beside a figure whose legend names them. Its legend: the envelope, the two rings, the
    references the plane has and the ideal corner. Its csv is tradeoff_pareto's."""
    t = shares(d[d["n_transitions"] > 100], methods)
    compared = t[~t["method"].isin(REFERENCES)]
    if compared.empty:
        why = "no run of its methods with more than 100 transitions"
        return nothing(to, folder, "tradeoff_pareto_compact", why)
    methods = present(t, methods)
    extra = [Line2D([], [], color=MUTED, lw=1.6, label="envelope of all points"), *ring_handles()]
    extra += [method_handle(m, to.ours) for m in methods if m in REFERENCES]
    extra.append(Line2D([], [], ls="", marker="*", color=MUTED, ms=8, label="ideal"))
    fig = plt.figure(figsize=COMPACT_SIZE)
    ax = fig.add_axes(compact_frame(fig, len(extra) - 4))
    plane_on(ax, t, methods, to, labels=False, size=PAIR_MARKS)
    t = pareto_on(ax, t, compared, labels=False, size=PAIR_MARKS)
    ax.set_xlabel("instances solved (%)")
    ax.set_ylabel("instances reliable (%)")
    fig.legend(handles=extra, loc="lower center", bbox_to_anchor=(0.5, 0.0), **COMPACT_LEGEND)
    compact_canvas(fig)
    save_pooled(to, folder, fig, "tradeoff_pareto_compact", t)
    return True


def tradeoff_and_scalability(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """Two panels with ONE legend, for the text width of the paper, over several domains: (a) the plane of
    tradeoff_pareto as solved_and_tradeoff draws it (plane_on and pareto_on) and (b) solved_reliably_vs_states
    (states_on), each by the code of its own figure and from the same tables. The legend is the one of
    solved_and_tradeoff: every method of either panel once, then the envelope and the two rings of (a). The
    y name of (b) is on two lines (it is longer than the panel is high). Its csv is the index of the legend
    (method, panels); the numbers are in the csv files of the two figures."""
    name = "tradeoff_and_scalability"
    if d["domain"].nunique() < 2:
        return nothing(to, folder, name, "drawn over several domains only")
    plane_methods = to.methods(PAIR_PANELS["b"])
    left = shares(d[d["n_transitions"] > 100], plane_methods)
    compared = left[~left["method"].isin(REFERENCES)]
    right = reliably_by_states(d, methods) if STATES_BUDGET in set(d["n_transitions"]) else pd.DataFrame()
    if compared.empty or right.empty:
        return nothing(to, folder, name, "no run of the methods of one of its panels")
    plane_methods, methods = present(left, plane_methods), present(right, methods)
    listed = methods + [m for m in plane_methods if m not in methods]
    extra = [Line2D([], [], color=MUTED, lw=1.6, label="envelope of all points"), *ring_handles()]
    handles = [method_handle(m, to.ours) for m in listed] + extra
    fig = plt.figure(figsize=PAIR_SIZE)
    rows = math.ceil(len(handles) / PAIR_LEGEND["ncol"])
    foot = (0.115 * rows + 0.06) / PAIR_SIZE[1]  # the legend: its rows, in the share of the height
    top, bottom = 0.975, foot + 0.50 / PAIR_SIZE[1]
    side = (top - bottom) * PAIR_SIZE[1] / PAIR_SIZE[0]  # the plane is square: its width, in the figure's
    plane_ax = fig.add_axes([0.075, bottom, side, top - bottom])
    ax = fig.add_axes([0.075 + side + 0.115, bottom, 0.985 - (0.075 + side + 0.115), top - bottom])
    plane_on(plane_ax, left, plane_methods, to, labels=False, size=PAIR_MARKS)
    pareto_on(plane_ax, left, compared, labels=False, size=PAIR_MARKS)
    plane_ax.set_xlabel("instances solved (%)")
    plane_ax.set_ylabel("instances reliable (%)")
    states_on(ax, right, methods, to)
    ax.set_ylabel(STATES_AXIS.replace(") and", ")\nand"), linespacing=1.15)
    spacing = dict(handlelength=2.6, columnspacing=1.0, handletextpad=0.5, labelspacing=0.3, borderaxespad=0)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), **spacing, **PAIR_LEGEND)
    pair_letters(fig, (plane_ax, ax))
    panels = (("a", plane_methods), ("b", methods))
    index = pd.DataFrame(
        {"method": listed, "panels": ["+".join(x for x, ms in panels if m in ms) for m in listed]}
    )
    save_pooled(to, folder, fig, name, index)
    return True


def pair_legend(fig, handles: list) -> tuple:
    """The legend of a pair of panels (PAIR_SIZE) under them; returns (top, bottom) of the panels."""
    rows = math.ceil(len(handles) / PAIR_LEGEND["ncol"])
    foot = (0.115 * rows + 0.06) / PAIR_SIZE[1]  # the legend: its rows, in the share of the height
    spacing = dict(handlelength=2.6, columnspacing=1.0, handletextpad=0.5, labelspacing=0.3, borderaxespad=0)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), **spacing, **PAIR_LEGEND)
    return 0.975, foot + 0.50 / PAIR_SIZE[1]


def pair_index(listed: list, panels: tuple) -> pd.DataFrame:
    return pd.DataFrame({"method": listed, "panels": ["+".join(x for x, ms in panels if m in ms) for m in listed]})


def tradeoff_and_optimality(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """Two panels with ONE legend, over several domains: (a) the plane of tradeoff_pareto, as in
    tradeoff_and_scalability, and (b) solved_vs_optimality_fade_ends (optimality_on with FADED_ENDS)."""
    name = "tradeoff_and_optimality"
    if d["domain"].nunique() < 2:
        return nothing(to, folder, name, "drawn over several domains only")
    plane_methods = to.methods(PAIR_PANELS["b"])
    left = shares(d[d["n_transitions"] > 100], plane_methods)
    compared = left[~left["method"].isin(REFERENCES)]
    right = d[d["method"].isin(methods) & (d["n_transitions"] == OPTIMALITY_N)]
    if compared.empty or right.empty:
        return nothing(to, folder, name, "no run of the methods of one of its panels")
    plane_methods, methods = present(left, plane_methods), present(right, methods)
    listed = methods + [m for m in plane_methods if m not in methods]
    extra = [Line2D([], [], color=MUTED, lw=1.6, label="envelope of all points"), *ring_handles()]
    fig = plt.figure(figsize=PAIR_SIZE)
    top, bottom = pair_legend(fig, [method_handle(m, to.ours) for m in listed] + extra)
    side = (top - bottom) * PAIR_SIZE[1] / PAIR_SIZE[0]  # the plane is square: its width, in the figure's
    plane_ax = fig.add_axes([0.075, bottom, side, top - bottom])
    ax = fig.add_axes([0.075 + side + 0.10, bottom, 0.985 - (0.075 + side + 0.10), top - bottom])
    plane_on(plane_ax, left, plane_methods, to, labels=False, size=PAIR_MARKS)
    pareto_on(plane_ax, left, compared, labels=False, size=PAIR_MARKS)
    plane_ax.set_xlabel("instances solved (%)")
    plane_ax.set_ylabel("instances reliable (%)")
    optimality_on(ax, right, methods, to, faded=FADED_ENDS)
    pair_letters(fig, (plane_ax, ax))
    save_pooled(to, folder, fig, name, pair_index(listed, (("a", plane_methods), ("b", methods))))
    return True


# ---- the split figures of the paper: each panel, and the legend, in a file of its own ------------------
# The paper sets the panels side by side as \subfigure (a) and (b), and the legend under them with no
# letter. Every panel has the same size and the same margins, so the two panels of a figure align; the
# legend is as wide as the text. Saved without the tight bounding box, so the margins stay as set.
SPLIT_PANEL = (3.45, 2.15)  # a panel, in inches with its axis labels: half the text width of the paper
SPLIT_PLANE = (3.45, 2.45)  # a panel of the plane solved against reliable: a little taller
SPLIT_MARGINS = dict(left=0.185, right=0.985, bottom=0.19, top=0.955)  # of a panel, in its share
SPLIT_LEGEND = dict(ncol=6, fontsize=6.8, handlelength=2.6, columnspacing=1.0, handletextpad=0.5,
                    labelspacing=0.3, borderaxespad=0)  # fmt: skip
SPLIT_AXES = {
    "solved": "runs solved (%)",
    "reliable": "runs reliable (%)",
    "states": "runs solved and reliable\nin every state (%)",
}


def split_panel(size: tuple = SPLIT_PANEL):
    fig = plt.figure(figsize=size)
    margins = dict(SPLIT_MARGINS)
    if size[1] != SPLIT_PANEL[1]:  # a taller panel: the same margins in inches
        margins["bottom"] *= SPLIT_PANEL[1] / size[1]
        margins["top"] = 1 - (1 - margins["top"]) * SPLIT_PANEL[1] / size[1]
    box = [margins["left"], margins["bottom"], margins["right"] - margins["left"], margins["top"] - margins["bottom"]]
    return fig, fig.add_axes(box)


def save_split(to: FigureSet, folder: str, fig, name: str, data: pd.DataFrame) -> None:
    with mpl.rc_context({"savefig.bbox": "standard"}):
        save_pooled(to, folder, fig, name, data)


def split_legend(to: FigureSet, folder: str, name: str, handles: list, ncol: int | None = None) -> None:
    """The legend of a split figure, alone, as wide as the text, cropped to its rows (ncol: its columns,
    default SPLIT_LEGEND's)."""
    fig = plt.figure(figsize=(7.0, 0.3))
    legend = SPLIT_LEGEND | {"ncol": ncol or SPLIT_LEGEND["ncol"]}
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.5), **legend)
    labels = pd.DataFrame({"label": [h.get_label() for h in handles]})
    save_to(to.out, f"{to.name}/{folder}" if folder else to.name, fig, f"{name}_legend", labels,
            csv="_".join(filter(None, [to.name, folder, f"{name}_legend"])))  # fmt: skip


def save_trio(to: FigureSet, folder: str, name: str, panels: list, handles: list,
              size: tuple = SPLIT_PANEL, index: pd.DataFrame | None = None, ncol: int | None = None,
              panel_ncol: int = 2) -> None:
    """A figure of the paper with panels (a), (b), ..., in the files the paper and a reader need:
        <name>_a, _b, ...           each panel without legend: \\subfigure in LaTeX, which adds the letters
        <name>_legend               the legend alone, as wide as the text: the third figure, under them
        <name>_a_full, _b_full ...  each panel with its own legend under it: a figure that stands alone
    No file has the letters (a), (b). panels: functions draw(ax) -> DataFrame (the numbers of the panel,
    saved beside its file); every panel has the size and margins of split_panel(size), with or without
    its legend. index: the numbers of the whole figure, data/<set>_<name>.csv (default: the panels'
    numbers, with a column `panel`). ncol: the columns of the legend file (default SPLIT_LEGEND's);
    panel_ncol: those of the legend under a panel (fewer for long labels)."""
    numbers = []
    width, height = size
    rows = math.ceil(len(handles) / panel_ncol)
    strip = 0.125 * rows + 0.1  # the legend under a panel, in inches
    margins = dict(SPLIT_MARGINS)
    if height != SPLIT_PANEL[1]:  # as split_panel: the same margins in inches
        margins["bottom"] *= SPLIT_PANEL[1] / height
        margins["top"] = 1 - (1 - margins["top"]) * SPLIT_PANEL[1] / height
    total = height + strip
    for letter, draw in zip("abcdefgh", panels, strict=False):
        fig, ax = split_panel(size)
        data = draw(ax)
        save_split(to, folder, fig, f"{name}_{letter}", data.copy())  # save_pooled adds a column to it
        numbers.append(data.assign(panel=letter))
        fig = plt.figure(figsize=(width, total))  # the same panel, its legend under it
        across, up = margins["right"] - margins["left"], margins["top"] - margins["bottom"]
        box = [margins["left"], (strip + margins["bottom"] * height) / total, across, up * height / total]
        draw(fig.add_axes(box))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                   **(SPLIT_LEGEND | {"ncol": panel_ncol}))  # fmt: skip
        save_split(to, folder, fig, f"{name}_{letter}_full", data.copy())
    split_legend(to, folder, name, handles, ncol)
    whole = index.copy() if index is not None else pd.concat(numbers, ignore_index=True)
    if "method" in whole:
        whole.insert(1, "label", whole["method"].map(lambda m: label_of(m, to.ours)))
    data_dir = to.out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    whole.to_csv(data_dir / ("_".join(filter(None, [to.name, folder, name])) + ".csv"), index=False)


def solved_and_scalability(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """Figure 3 of the paper, as save_trio writes it, over several domains: (a) the % of instances
    solved against the transitions (solved_on) and (b) solved AND reliable in every state against the
    observed states, with STATES_BUDGET transitions, smoothed (states_on); one legend."""
    name = "solved_and_scalability"
    if d["domain"].nunique() < 2:
        return nothing(to, folder, name, "drawn over several domains only")
    left = shares(d, methods)
    budget = STATES_BUDGET in set(d["n_transitions"])
    right = reliably_by_states(d, methods, everywhere=True) if budget else pd.DataFrame()
    if left.empty or right.empty:
        return nothing(to, folder, name, "no run of the methods of one of its panels")
    left_methods, right_methods = present(left, methods), present(right, methods)
    listed = left_methods + [m for m in right_methods if m not in left_methods]

    def solved(ax):
        solved_on(ax, left, left_methods, to)
        ax.set_ylabel(SPLIT_AXES["solved"])
        return left

    def states(ax):
        lines = states_on(ax, right, right_methods, to)
        ax.set_ylabel(SPLIT_AXES["states"], linespacing=1.1, fontsize=8.8)
        return lines

    save_trio(to, folder, name, [solved, states], [method_handle(m, to.ours) for m in listed],
              index=pair_index(listed, (("a", left_methods), ("b", right_methods))))  # fmt: skip
    return True


def tradeoff_plane(ax, d: pd.DataFrame, methods: list, to: FigureSet) -> pd.DataFrame:
    """A panel of figure 5: the plane of tradeoff_pareto (runs solved against runs reliable, one path per
    method as the data grow, from more than 100 transitions) with the envelope of all points and no
    rings, wider than tall. Returns its numbers (empty: no run of its methods)."""
    t = shares(d[d["n_transitions"] > 100], methods)
    compared = t[~t["method"].isin(REFERENCES)]
    if compared.empty:
        return pd.DataFrame()
    plane_on(ax, t, present(t, methods), to, labels=False, size=PAIR_MARKS, square=False)
    t = pareto_on(ax, t, compared, labels=False, size=PAIR_MARKS, rings=False)
    ax.set_xlabel(SPLIT_AXES["solved"])
    ax.set_ylabel(SPLIT_AXES["reliable"])
    return t


def tradeoff_and_rewards(main: pd.DataFrame, rewards: pd.DataFrame, to: FigureSet, folder: str = "") -> bool:
    """Figure 5 of the paper, as save_trio writes it: (a) tradeoff_plane of the main runs and (b)
    that of the runs of the ablation of the rewards (both over several domains), with one legend. Drawn by
    ablations/ablation_rewards/plot.py --domain all_domains, which has the two tables of instances."""
    name = "tradeoff_and_rewards"
    methods = to.methods(PAIR_PANELS["b"])
    if main["domain"].nunique() < 2 or rewards["domain"].nunique() < 2:
        return nothing(to, folder, name, "drawn over several domains only, for the two tables")
    tables = [judged(main), judged(rewards)]
    drawn = [present(shares(t[t["n_transitions"] > 100], methods), methods) for t in tables]
    if not all(drawn):
        return nothing(to, folder, name, "no run of its methods in one of the two tables")
    listed = drawn[0] + [m for m in drawn[1] if m not in drawn[0]]
    panels = [lambda ax, t=t: tradeoff_plane(ax, t, methods, to) for t in tables]
    extra = [Line2D([], [], color=MUTED, lw=1.6, label="envelope of all points")]
    save_trio(to, folder, name, panels, [method_handle(m, to.ours) for m in listed] + extra, SPLIT_PLANE,
              index=pair_index(listed, (("a", drawn[0]), ("b", drawn[1]))))  # fmt: skip
    return True


def reach(path: pd.DataFrame, target: float) -> tuple[float, float] | None:
    """(transitions, % safe) where the path of a method (by number of transitions) first solves `target` %
    of the instances: between two data sizes, linear in the log of the transitions. None: never."""
    n = np.log10(path["n_transitions"].to_numpy(dtype=float))
    solved, safe = path["solved"].to_numpy(), path["safe"].to_numpy()
    for i in range(len(n)):
        if solved[i] >= target:
            if i == 0:
                return 10 ** n[0], safe[0]
            w = (target - solved[i - 1]) / (solved[i] - solved[i - 1])
            return 10 ** (n[i - 1] + w * (n[i] - n[i - 1])), safe[i - 1] + w * (safe[i] - safe[i - 1])
    return None


def tradeoff_data_needed(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """The transitions a method needs to solve each share of TARGETS of the instances (x, fewer to the
    right) against the % safe with that much data; in bold, what it solves with the most data of the run."""
    t = shares(d[d["n_transitions"] >= 100], methods)
    if t.empty:
        return nothing(to, folder, "tradeoff_data_needed", "no run of its methods from 100 transitions on")
    methods = present(t, methods)
    fewest, most = t["n_transitions"].min(), t["n_transitions"].max()
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.plot(fewest, 100, marker="*", color=MUTED, ms=11, ls="", zorder=1, clip_on=False)
    ax.annotate("ideal", (fewest, 100), xytext=(-7, -11), textcoords="offset points", ha="right", fontsize=7,
                color=MUTED)  # fmt: skip
    rows, finals = [], []
    for method in methods:
        path = t[t["method"] == method].sort_values("n_transitions")
        end = path.iloc[-1]  # the largest data size of the method
        points = [(target, reach(path, target)) for target in TARGETS]
        points = [(target, *at) for target, at in points if at and at[0] < 0.999 * end["n_transitions"]]
        points.append((round(float(end["solved"])), float(end["n_transitions"]), float(end["safe"])))
        xs, ys = [x for _, x, _ in points], [y for _, _, y in points]
        style = method_line(method, to.ours)
        ax.plot(xs, ys, **(style | {"marker": None}))
        scatter_path(ax, xs, ys, [9 + 0.3 * share for share, _, _ in points], style)
        graph, form = split_form(method)  # the labels of two methods of ours on opposite sides
        above = (graph == REFERENCE or (graph in to.ours and form == "MDP")) != (graph in to.ours[1:])
        for i, (share, x, y) in enumerate(points):
            final = i == len(points) - 1
            if final:
                finals.append((y, x, f"{share}%", style["color"]))
            else:
                ax.annotate(f"{share}%", (x, y), xytext=(0, 5 if above else -5), textcoords="offset points",
                            ha="center", va="bottom" if above else "top", fontsize=5.5, color=style["color"],
                            zorder=8)  # fmt: skip
            rows.append({"method": method, "point": "final" if final else "target", "solved": share,
                         "transitions": x, "safe": y})  # fmt: skip
    height, gap = np.inf, 3.6  # the final labels, in bold at the left of their points and kept apart
    for y, x, text, color in sorted(finals, reverse=True):
        height = min(y, height - gap)
        ax.annotate(text, (x, height), xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=6, fontweight="bold", color=color, zorder=8)  # fmt: skip
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(most * 3.2, fewest * 0.6)  # inverted: fewer transitions to the right
    ax.set_ylim(-4, 108)
    percent_axis(ax.yaxis)
    ax.set_xlabel("transitions needed to solve a share of the instances")
    ax.set_ylabel("instances reliable with that much data (%)")
    fig.tight_layout()
    pooled_legend(fig, methods, to)
    save_pooled(to, folder, fig, "tradeoff_data_needed", pd.DataFrame(rows))
    return True


def reported_vs_true(
    d: pd.DataFrame, methods: list, folder: str, to: FigureSet, common=False, top=None, name=None,
    share_label: str = "", every: bool = False,
) -> bool:
    """One dot per run, a panel per method (those of REPORTED the instances have, in its order; four to a
    row, or three when their number is a multiple of three only): the value the method reported (x)
    against the true value of its policy (y), both from the start and normalised (0: a random policy, 1:
    the optimal one), coloured by domain (DOMAIN_COLOURS). Under the diagonal (grey) the method reported
    more than the true value ("not reliable", in the legend); each panel has the share of its runs that
    did not ("reliability"). The interval MDP of ours reports a lower bound, the other methods their own
    estimate. Runs: those that finished and reported a value; their share of all the runs of the method
    (every instance and number of transitions it has, finished or not) follows its name in the title of
    the panel ("CD-IMDP (ours) – 99.1%"; on a second line when a title would not fit its panel). A run beyond
    REPORTED_X or REPORTED_Y is drawn at the border. With two discovery methods as ours (appendix) the
    panels are still those of the first. common (reported_vs_true_common): only the runs (RUN_KEY: domain,
    size, seed, transitions and, in an ablation, reward) that every method of the figure finished with a
    value, so every panel has the same runs; their number is in the legend, and the titles are the plain
    names (every method returned a value on every common run). top (reported_vs_true_top8): only the `top`
    most reliable methods of `methods` (most_reliable), in that order; the csv has every candidate, with
    its `rank` and whether it is `shown`."""
    name = name or "reported_vs_true" + ("_common" if common else f"_top{top}" if top else "")
    if "reported" not in d:
        return nothing(to, folder, name, "its instances have no reported value")
    if not every:  # with two discovery methods as ours, the panels of the first only
        methods = [m for m in methods if split_form(m)[0] not in to.ours[1:]]
    possible = d[d["method"].isin(methods)].groupby(["method", "domain"]).size()  # finished or not
    d = d[d["method"].isin(methods) & d["ok"].astype(bool)].dropna(subset=["reported", "true_value"])
    if d.empty:
        return nothing(to, folder, name, "no run of its methods reported a value")
    methods = present(d, methods)
    returned = d.groupby(["method", "domain"]).size()
    if common:
        d = common_runs(d, methods)
        if d.empty:
            return nothing(to, folder, name, "no run that every one of its methods finished with a value")
    shared = len(d) // len(methods)  # with `common`: the runs of every panel
    domains = [name for name in DOMAIN_COLOURS if name in set(d["domain"])]
    d = d[d["domain"].isin(domains)]
    hidden = []  # with `top`: the candidates that are not drawn (they are in the csv)
    if top:
        ranked = most_reliable(d, methods, returned, possible, to)
        methods, hidden = ranked[:top], ranked[top:]
        if REFERENCE in methods:  # the reference always first (top left), the others by reliability
            methods = [REFERENCE, *(m for m in methods if m != REFERENCE)]
    columns = 3 if len(methods) % 3 == 0 and len(methods) % 4 else min(4, len(methods))
    rows_n = math.ceil(len(methods) / columns)
    high_row = 1.62 if columns == 4 else 1.9  # inches of a row of panels: wider panels are a little taller
    high_row = 1.45 if top else high_row  # the paper's grid of two rows: a little lower
    fig, axes = plt.subplots(rows_n, columns, figsize=(7.0, high_row * rows_n + 0.55), sharex=True,
                             sharey=True, squeeze=False)  # fmt: skip
    (x0, x1), (y0, y1) = REPORTED_X, REPORTED_Y
    low, high = min(x0, y0) - 1, max(x1, y1) + 1
    rows, titles = [], []
    for panel, method in enumerate([*methods, *hidden], start=1):
        ax = axes.flat[panel - 1] if panel <= len(methods) else None  # None: a candidate that is not drawn
        at = d[d["method"] == method].sample(frac=1, random_state=0)  # random order: no domain hides another
        held = at["reported"] <= at["true_value"] + 1e-9
        if ax is not None:
            ax.fill_between([low, high], [low, high], low, color="#f0efec", lw=0, zorder=0)
            ax.plot([low, high], [low, high], color="#3a3a3a", lw=0.7, zorder=1)
            colours = at["domain"].map(DOMAIN_COLOURS)
            ax.scatter(at["reported"].clip(x0, x1), at["true_value"].clip(y0, y1), c=colours, s=3.2, lw=0,
                       alpha=0.55, zorder=2, rasterized=True)  # fmt: skip
            if len(at):
                ax.text(0.035, 0.955, f"reliability: {100 * held.mean():.1f}%", transform=ax.transAxes,
                        va="top", fontsize=7.2, color=INK, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))  # fmt: skip
            share = 100 * returned[method].sum() / possible[method].sum()
            titles.append((ax, label_of(method, to.ours), None if common else f"{share_label}{share:.1f}%"))
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0 - 0.08, y1 + REPORTED_HEAD)
            ax.set_xticks([-4, -2, 0, 2])
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=7.5)
        outside = ~(at["reported"].between(x0, x1) & at["true_value"].between(y0 - 1e-9, y1 + 1e-9))
        for where in [*domains, "all"] if len(domains) > 1 else domains:
            part = at if where == "all" else at[at["domain"] == where]
            rows.append({
                "panel": panel, "method": method, "domain": where, "runs": len(part),
                "reported_le_true": 100 * held[part.index].mean() if len(part) else np.nan,
                "mean_value": part["true_value"].mean(), "median_value": part["true_value"].median(),
                "possible_runs": int(scope(possible, method, where)),
                "returned_runs": int(scope(returned, method, where)),
                "outside": int(outside[part.index].sum()), "x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1,
            })  # fmt: skip
    for ax in axes.flat[len(methods) :]:
        ax.set_visible(False)
    bottom = 0.72 / fig.get_figheight()  # the x label and the legend, in inches under the panels
    fig.subplots_adjust(left=0.068, right=0.995, top=1 - 0.22 / fig.get_figheight(), bottom=bottom,
                        wspace=0.09, hspace=0.3)  # fmt: skip
    for ax, label, share in titles:
        ax.set_title(f"{label} – {share}" if share else label, pad=3, fontsize=REPORTED_TITLE, color=INK)
    fig.canvas.draw()
    wide = any(ax.title.get_window_extent().width > ax.get_window_extent().width for ax, _, _ in titles)
    if wide:  # a title would run into the next panel: the share on a second line, in every panel
        for ax, label, share in titles:
            ax.set_title(f"{label}\n{share}" if share else label, pad=3, fontsize=REPORTED_TITLE, color=INK,
                         linespacing=1.1)  # fmt: skip
        two = any(share for _, _, share in titles)
        fig.subplots_adjust(top=1 - (0.36 if two else 0.22) / fig.get_figheight(), hspace=0.4 if two else 0.3)
    fig.supxlabel("normalised reported value", fontsize=9, y=0.31 / fig.get_figheight())
    fig.supylabel("normalised true value", fontsize=9, x=0.004)
    handles = [Line2D([], [], ls="", marker="o", ms=5, mfc=DOMAIN_COLOURS[where], mec="none",
                      label=DOMAIN_NAMES.get(where, where)) for where in domains]  # fmt: skip
    handles.append(Line2D([], [], ls="", marker="s", ms=7, mfc="#f0efec", mec="#c3c2b7", mew=0.5,
                          label="not reliable"))  # fmt: skip
    if common:
        handles.append(Line2D([], [], ls="", marker="", label=f"N common runs: {shared:,}"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), fontsize=8,
               handletextpad=0.3, columnspacing=1.6, borderaxespad=0)  # fmt: skip
    table = pd.DataFrame(rows)
    table["returned_share"] = 100 * table["returned_runs"] / table["possible_runs"]
    if common:
        table["common_runs"] = shared
    if top:
        table = table.rename(columns={"panel": "rank"}).assign(shown=lambda t: t["rank"] <= len(methods))
    save_pooled(to, folder, fig, name, table)
    return True


def most_reliable(
    d: pd.DataFrame, methods: list, returned: pd.Series, possible: pd.Series, to: FigureSet
) -> list:
    """`methods` from the most to the least reliable on the runs `d` (those that returned a value): the
    share with a reported value at most the true one. Ties: ours first, then the higher share of runs
    that returned a value, then the order of `methods`."""
    held = (d["reported"] <= d["true_value"] + 1e-9).groupby(d["method"]).mean()
    share = {m: returned[m].sum() / possible[m].sum() for m in methods}
    return sorted(methods, key=lambda m: (-round(held[m], 9), not to.is_ours(m), -share[m], methods.index(m)))


def reported_vs_true_top8(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """reported_vs_true for the paper: the TOP_N most reliable methods of REPORTED_TOP (most_reliable), four
    to a row."""
    return reported_vs_true(d, methods, folder, to, top=TOP_N)


def reported_vs_true_selected(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """reported_vs_true for the eight methods of REPORTED_SELECTED, from the most to the least reliable."""
    name = "reported_vs_true_selected"
    return reported_vs_true(d, methods, folder, to, top=len(REPORTED_SELECTED), name=name,
                            share_label="runs returned: ")


def reported_vs_true_all(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """reported_vs_true for every method of REPORTED_ALL (both discovery methods), from the most to the
    least reliable; in the appendix set only (the paper has reported_vs_true_selected)."""
    name = "reported_vs_true_all"
    if to.name != "appendix":
        return nothing(to, folder, name, "drawn in the appendix set only")
    return reported_vs_true(d, methods, folder, to, top=len(methods), name=name,
                            share_label="runs returned: ", every=True)  # fmt: skip


def scope(counts: pd.Series, method: str, where: str) -> int:
    """A count per (method, domain) of one method, in one domain or in all ("all")."""
    of = counts[method]
    return of.sum() if where == "all" else of.get(where, 0)


def common_runs(d: pd.DataFrame, methods: list) -> pd.DataFrame:
    """The rows of `d` (one per method and run) of the runs that every method of `methods` has: a run is
    its RUN_KEY (the columns of it that `d` has)."""
    key = [column for column in RUN_KEY if column in d]
    d = d[d["method"].isin(methods)]
    count = d.groupby(key, dropna=False)["method"].nunique()
    kept = count[count == len(methods)].index
    return d[pd.MultiIndex.from_frame(d[key]).isin(kept)]


def reported_vs_true_common(d: pd.DataFrame, methods: list, folder: str, to: FigureSet) -> bool:
    """reported_vs_true on the common runs only: those that every method of the figure finished with a
    value (common_runs), so the panels are compared on the same runs."""
    return reported_vs_true(d, methods, folder, to, common=True)


# name -> (folder in the set, function, its methods); the file is the last part of the name
POOLED_FIGURES = {
    "imdp_by_graph/solved_vs_transitions": ("imdp_by_graph", solved_vs_transitions, IMDP_BY_GRAPH),
    "imdp_by_graph/solved_vs_optimality": ("imdp_by_graph", solved_vs_optimality, IMDP_BY_GRAPH),
    "mdp_by_graph/solved_vs_transitions": ("mdp_by_graph", solved_vs_transitions, MDP_BY_GRAPH),
    "mdp_by_graph/solved_vs_optimality": ("mdp_by_graph", solved_vs_optimality, MDP_BY_GRAPH),
    "vs_baselines/solved_vs_transitions": ("vs_baselines", solved_vs_transitions, VS_BASELINES),
    "vs_baselines/solved_vs_optimality": ("vs_baselines", solved_vs_optimality, VS_BASELINES),
    "vs_baselines/solved_vs_optimality_to90": ("vs_baselines", solved_vs_optimality_to90, VS_BASELINES),
    "vs_baselines/solved_vs_optimality_grey": ("vs_baselines", solved_vs_optimality_grey, VS_BASELINES),
    "vs_baselines/solved_vs_regret": ("vs_baselines", solved_vs_regret, VS_BASELINES),
    "vs_baselines/solved_vs_regret_grey": ("vs_baselines", solved_vs_regret_grey, VS_BASELINES),
    "vs_baselines/solved_vs_regret_fade": ("vs_baselines", solved_vs_regret_fade, VS_BASELINES),
    "vs_baselines/solved_vs_regret_fade90": ("vs_baselines", solved_vs_regret_fade90, VS_BASELINES),
    "vs_baselines/solved_vs_regret_fade_ends": ("vs_baselines", solved_vs_regret_fade_ends, VS_BASELINES),
    "vs_baselines/solved_vs_optimality_fade_ends": ("vs_baselines", solved_vs_optimality_fade_ends, VS_BASELINES),
    "vs_baselines/solved_vs_time": ("vs_baselines", solved_vs_time, VS_BASELINES),
    "tradeoff": ("", tradeoff, TRADEOFF),
    "tradeoff_pareto": ("", tradeoff_pareto, TRADEOFF),
    "tradeoff_data_needed": ("", tradeoff_data_needed, TRADEOFF),
    "solved_and_tradeoff": ("", solved_and_tradeoff, VS_BASELINES),
    "vs_baselines/reliable_vs_transitions": ("vs_baselines", reliable_vs_transitions, VS_BASELINES),
    "vs_baselines/reliable_everywhere_vs_transitions": (
        "vs_baselines", reliable_everywhere_vs_transitions, VS_BASELINES),
    "solved_and_reliable": ("", solved_and_reliable, VS_BASELINES),
    "solved_and_reliable_everywhere": ("", solved_and_reliable_everywhere, VS_BASELINES),
    "tradeoff_pareto_compact": ("", tradeoff_pareto_compact, TRADEOFF),
    "reported_vs_true": ("", reported_vs_true, REPORTED),
    "reported_vs_true_common": ("", reported_vs_true_common, REPORTED),
    "reported_vs_true_top8": ("", reported_vs_true_top8, REPORTED_TOP),
    "reported_vs_true_selected": ("", reported_vs_true_selected, REPORTED_SELECTED),
    "reported_vs_true_all": ("", reported_vs_true_all, REPORTED_ALL),
} | {
    "solved_reliably_vs_states": ("", solved_reliably_vs_states, VS_BASELINES),
    "solved_reliably_vs_states_everywhere": ("", solved_reliably_vs_states_everywhere, VS_BASELINES),
    "tradeoff_and_scalability": ("", tradeoff_and_scalability, VS_BASELINES),
    "tradeoff_and_optimality": ("", tradeoff_and_optimality, VS_BASELINES),
    "solved_and_scalability": ("", solved_and_scalability, VS_BASELINES),
}  # fmt: skip


def pooled_figures(instances: pd.DataFrame, out: Path, sets=None, only: list | None = None) -> list[str]:
    """Draw the pooled figures of a table of instances (see the top of this section), once per set of
    SETS (`sets`: some of their names; default all), into out/<set> with their numbers in out/data;
    `only`: names of POOLED_FIGURES. Returns the figures written, as <set>/<name>."""
    d = judged(instances)
    counts = d.groupby(["method", "n_transitions"]).size()
    print(f"  pooled figures: up to {int(counts.max()) if len(counts) else 0:,} instances per data size")
    written = []
    with mpl.rc_context(PAPER_STYLE):
        for to in (FigureSet(name, ours, out) for name, ours in SETS.items() if not sets or name in sets):
            for name, (folder, draw, methods) in POOLED_FIGURES.items():
                with references_named():
                    if (not only or name in only) and draw(d, to.methods(methods), folder, to):
                        written.append(f"{to.name}/{name}")
    return written


def pooled(name: str):
    """A pooled figure as a figure of one domain (FIGURES), from the instances of its runs."""

    def draw(dom: Domain) -> bool:
        to = FigureSet(dom.set, dom.ours, dom.out)
        if dom.name not in POOLED_DOMAINS:
            return nothing(to, "", name, f"drawn for {', '.join(POOLED_DOMAINS)} only")
        folder, function, methods = POOLED_FIGURES[name]
        with references_named():
            return function(judged(instances_of(dom.runs, dom.name)), to.methods(methods), folder, to)

    draw.__doc__ = POOLED_FIGURES[name][1].__doc__
    return draw


# ==== 7. exploratory figures =======================================================================
def name_of(model: str) -> str:
    return {"flat": "flat IMDP"}.get(model, model)


def line_of(model: str) -> dict:
    return {"color": COLOR[model], "ls": LINE.get(model, "-"), "marker": MARKER.get(model, "o")}


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
    draw(ax, grid, BASELINES, "baselines (their own value estimate as the reported value)")
    ax.set_xlabel("transitions in the dataset", fontsize=8)
    cax = fig.add_axes([0.42, 0.13, 0.2, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("share of runs where reported value ≤ true value  (no number: 100%; hatched: not run)",
                 fontsize=8, color=INK2)  # fmt: skip
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["0%", "50%", "100%"])
    fig.suptitle(f"Is the reported value reliable?  (sizes {dom.size_range()} pooled, finished runs)", x=0.1,
                 ha="left", y=0.97, fontsize=11)  # fmt: skip
    save(dom, "explore", fig, "4_promise_held", table(rows))


def promise_vs_true(dom: Domain) -> None:
    """Promise and true value of the certified policy at one size, one panel per graph source."""
    size, ns = dom.detail_size, budgets(dom)
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.9), sharex=True, sharey=True)
    rows = []
    quantities = [("value", BLUE, "-", "true value"), ("promise_value", RED, (0, (4, 2)), "lower bound")]
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
    fig.suptitle(f"Lower bound vs. true value of the certified policy ({dom.solver})  ·  {dom.title(size)} "
                 f"({dom.states[size]} states)", x=0.01, ha="left", y=1.08, fontsize=11)  # fmt: skip
    top_legend(fig, handles, ["true value of the policy", "certified lower bound"], ncol=2, y=0.98)
    fig.tight_layout(h_pad=1.2)
    save(dom, "explore", fig, "5_promise_vs_true", table(rows))


def solvers(dom: Domain) -> None:
    """The box and the exact set on the true graph: promise and value at one size, solve time by size (the
    exact set: runs before 2026-11-17, or ablations/ablation_box_exact)."""
    size, ns = dom.detail_size, budgets(dom)
    names = {"IMDP (IA)": BLUE, "IMDP (exact)": PURPLE}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    rows = []
    for solver, color in names.items():
        x = dom.pick("true graph", solver)
        if x.empty:
            continue
        for ax, col in zip(axes[:2], ["promise_value", "value"], strict=False):
            c = cells(x[x["size"] == size], col)
            ax.plot(c["n_transitions"], c["mean"], color=color, marker="o")
            rows.append(c.assign(method=solver, panel=col))
        y = x[x["ok"] & (x["n_transitions"] == N_FIXED)].groupby("size")["solve_s"].median().reindex(dom.kept)
        axes[2].plot([dom.states[s] for s in y.index], y.to_numpy(), color=color, marker="o")
        rows.append(y.reset_index().assign(method=solver, panel="solve_s"))
    for ax, text in zip(axes[:2], ["certified lower bound", "true value of the policy"], strict=False):
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
               ["interval arithmetic (box)", "exact (vertex products)"], 2, 0.98)  # fmt: skip
    fig.tight_layout()
    save(dom, "explore", fig, "6_solvers", table(rows))


def discovery(dom: Domain) -> None:
    """The learned graph against the true one: recall, precision and structural Hamming distance."""
    models = ["PCMCI+ (lagged)", "DYNOTEARS (lagged)", "Strehl (fixed hull)", "Met-Rmax (fixed hull)",
              "sparse", "full-fact"]  # fmt: skip
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


# ==== 8. the command line ==========================================================================
# name -> (group, function, the sources whose runs it needs: without them the figure is skipped); a paper
# figure also needs the sources of the discovery methods that are ours in the set being drawn (DISCOVERY)
GRAPHS_NEEDED = ["truegraph", "sparse", "fullfact", "flat"]
FIGURES = (
    {
        "value": ("paper", value, GRAPHS_NEEDED),
        "baselines": ("paper", baselines, ["truegraph", "sota"]),
        "imdp_value_held": ("paper", imdp_value_held, GRAPHS_NEEDED),
        "coverage": ("paper", coverage, GRAPHS_NEEDED),
        "promise_scatter": ("paper", promise_scatter, ["truegraph", "sota"]),
        "value_vs_safety": ("paper", value_vs_safety, GRAPHS_NEEDED + ["sota"]),
        "data_needed": ("paper", data_needed, GRAPHS_NEEDED),
        "graph_errors": ("paper", graph_errors, GRAPHS_NEEDED[:-1]),
    }
    # the pooled figures of a domain: each draws the methods the runs have
    | {name: ("paper", pooled(name), []) for name in POOLED_FIGURES}
    | {
        "1_value_graph_sources": ("explore", value_graph_sources, ["truegraph"]),
        "2_value_kwik": ("explore", value_kwik, ["truegraph", "strehl", "met"]),
        "3_value_vs_baselines": ("explore", value_vs_baselines, ["truegraph", "sota"]),
        "4_promise_held": ("explore", promise_held, ["truegraph", "sota"]),
        "5_promise_vs_true": ("explore", promise_vs_true, ["truegraph"]),
        "6_solvers": ("explore", solvers, ["truegraph"]),
        "7_discovery": ("explore", discovery, ["pcmci_lagged", "dynotears_lagged"]),
        "8_scalability": ("explore", scalability, ["truegraph"]),
        "9_transitions_needed": ("explore", transitions_needed, ["truegraph", "sota"]),
    }
)
GROUPS = {"pooled": list(POOLED_FIGURES)}  # a name of --only that selects several figures
STYLES = {"paper": PAPER_STYLE, "explore": EXPLORE_STYLE}


def chosen(only: list | None) -> tuple[list | None, list]:
    """--only as (the figures, the sets): a name of SETS selects that set of the paper figures (alone:
    all of them), a name of GROUPS its figures."""
    sets = [name for name in only or [] if name in SETS]
    figures = [f for name in only or [] if name not in SETS for f in GROUPS.get(name, [name])]
    if sets and not figures:
        figures = [name for name, (group, _, _) in FIGURES.items() if group == "paper"]
    return figures or None, sets or list(SETS)


def plot_domain(
    name: str, results: Path, out: Path, solver: str | None = None, only: list | None = None
) -> list[str]:
    """Draw the figures of one domain: the paper figures once per set of SETS (paper/, appendix/), the
    exploratory ones once. Returns the names of the figures written. solver None: the domain's
    (DOMAIN_SETTINGS), else DEFAULT_SOLVER. only: names of figures, of GROUPS or of SETS (chosen)."""
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
    only, sets = chosen(only)
    written = []
    for figure, (group, draw, required) in FIGURES.items():
        if only and figure not in only:
            continue
        for folder, ours in [(s, SETS[s]) for s in sets] if group == "paper" else [(group, [])]:
            missing = [s for s in required + [DISCOVERY[g] for g in ours] if s not in sources]
            if missing:
                print(f"  skipped {folder}/{figure}: no run of {', '.join(missing)} yet")
                continue
            with mpl.rc_context(STYLES[group]):
                done = draw(replace(dom, set=folder, ours=tuple(ours)))
            if done is not False and figure not in written:  # False: a pooled figure with nothing to draw
                written.append(figure)
    return written


# needed_by_states: the transitions needed to reach THRESHOLD against the observed states, a column per
# domain of POOLED_DOMAINS; top: the MDPs on a graph against the baselines; bottom: the interval MDPs (the
# box) across graphs. A point is the median over the seeds of a size (a seed that never reaches THRESHOLD
# within the data and the limits counts as infinite), drawn where the median is finite: most seeds reach it.
NEEDED_TOP = [f"{OURS} MDP", f"{OURS} SPIBB", "true graph MDP", "FOCUS", "DiFaRL", "SL-SPIBB", "flat SPIBB",
              "SLF-Rmax", "Met-Rmax", "offline DQN", "offline CQL"]  # fmt: skip
NEEDED_BOTTOM = [OURS, REFERENCE, "sparse", "full graph", "flat"]


def runs_needed(dom: Domain, method: str) -> pd.DataFrame:
    """The runs of a method at the kept sizes; SLF-Rmax and Met-Rmax: their arm at the papers' thresholds."""
    if method in KWIK_METHODS:
        arm = {v: k for k, v in KWIK_ARMS.items()}[method]
        d = dom.runs[(dom.runs["method"] == arm) & dom.runs["kwik_level"].eq("paper")]
        return d[d["size"].isin(dom.kept)]
    return kept(dom, method)


def needed_by_states(results: Path, out: Path, sets=None) -> list[str]:
    """The figure needed_by_states (above) of the main runs, into out/<set>/ with its numbers in out/data/;
    returns the figures written, as <set>/needed_by_states."""
    doms = []
    for name in POOLED_DOMAINS:
        runs = load(results, name)
        if runs.empty:
            print(f"  skipped needed_by_states: no runs of {name}")
            return []
        doms.append(describe(name, runs, "IMDP (IA)", out))
    written = []
    with mpl.rc_context(PAPER_STYLE):
        for set_name, ours in SETS.items():
            if sets and set_name not in sets:
                continue
            to = FigureSet(set_name, ours, out)
            fig, axes = plt.subplots(2, len(doms), figsize=(7.0, 4.9), sharey="row")
            rows, handles = [], {}
            for col, base in enumerate(doms):
                dom = replace(base, set=set_name, ours=tuple(ours))
                for row, family in enumerate((NEEDED_TOP, NEEDED_BOTTOM)):
                    ax = axes[row, col]
                    for method in to.methods(family):
                        r = runs_needed(dom, method)
                        if r.empty:
                            continue
                        t = needed(dom, r)
                        t = t[np.isfinite(t["median"].to_numpy())]
                        style = method_line(method, ours, ms=3)
                        handles.setdefault(style["label"], Line2D([], [], **(style | {"ms": 4})))
                        if len(t):
                            ax.plot(t["states"], t["median"], **style)
                        rows.append(t.assign(domain=dom.name, method=method, label=style["label"],
                                             row="mdp" if row == 0 else "imdp"))  # fmt: skip
                    states_axis(ax)
                    ax.set_yscale("log")
                    if row == 0:
                        ax.set_title(DOMAIN_NAMES[dom.name])
                    else:
                        ax.set_xlabel("observed states")
                axes[0, 0].set_ylabel(f"transitions to reach {THRESHOLD}")
                axes[1, 0].set_ylabel(f"transitions to reach {THRESHOLD}")
            fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.0)
            fig.legend(handles=list(handles.values()), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
                       fontsize=6.8, handlelength=2.6, columnspacing=1.0, borderaxespad=0)  # fmt: skip
            data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            save_to(out, set_name, fig, "needed_by_states", data, csv=f"{set_name}_needed_by_states")
            written.append(f"{set_name}/needed_by_states")
    return written


def discovery_errors(results: Path, out: Path, sets=None) -> list[str]:
    """The graphs learned by PCMCI+ and DYNOTEARS against the true one, a column per domain of
    POOLED_DOMAINS: the share of the true parents missed (top) and the parents added, as a share of the
    true parents (bottom), against the transitions; the mean over every size and seed whose discovery
    finished. Drawn in the appendix set (both methods are ours there)."""
    tables = []
    for name in POOLED_DOMAINS:
        runs = load(results, name)
        if runs.empty:
            print(f"  skipped discovery_errors: no runs of {name}")
            return []
        for method, model in (("PCMCI+", MODEL["PCMCI+"]), ("DYNOTEARS", MODEL["DYNOTEARS"])):
            d = runs[(runs["model"] == model) & runs["missing_share"].notna()]
            d = d.drop_duplicates(["size", "seed", "n_transitions"])
            d = d.assign(added=d["extra"] / d["n_true_edges"])
            g = d.groupby("n_transitions")[["missing_share", "added"]].mean().reset_index()
            tables.append(g.assign(domain=name, method=method))
    data = pd.concat(tables, ignore_index=True)
    written = []
    with mpl.rc_context(PAPER_STYLE):
        for set_name, ours in SETS.items():
            if (sets and set_name not in sets) or len(ours) < 2:
                continue
            fig, axes = plt.subplots(2, len(POOLED_DOMAINS), figsize=(7.0, 4.3), sharex=True, sharey="row")
            for col, name in enumerate(POOLED_DOMAINS):
                for row, column in enumerate(("missing_share", "added")):
                    ax = axes[row, col]
                    for method in ("PCMCI+", "DYNOTEARS"):
                        t = data[(data["domain"] == name) & (data["method"] == method)]
                        ax.plot(t["n_transitions"], 100 * t[column], **method_line(method, ours, ms=3))
                    paper_axis(ax)
                    ax.set_ylim(-4, 104)
                    if row == 0:
                        ax.set_title(DOMAIN_NAMES[name])
                    else:
                        ax.set_xlabel("transitions")
            axes[0, 0].set_ylabel("true parents\nmissed (%)")
            axes[1, 0].set_ylabel("parents added\n(% of the true ones)")
            fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.8)
            fig.legend(handles=[method_handle(m, ours) for m in ("PCMCI+", "DYNOTEARS")], loc="lower center",
                       bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=2.6, borderaxespad=0)  # fmt: skip
            save_to(out, set_name, fig, "discovery_errors", data, csv=f"{set_name}_discovery_errors")
            written.append(f"{set_name}/discovery_errors")
    return written


def plot_all_domains(results: Path, out: Path, only: list | None = None) -> list[str]:
    """Draw the pooled figures over the instances of POOLED_DOMAINS together, into out/all_domains (both
    sets, or those of `only`); returns the figures written, as <set>/<name>."""
    print(f"\n===== {ALL_DOMAINS} ({', '.join(POOLED_DOMAINS)}): {results} -> {out / ALL_DOMAINS} =====")
    tables = [instances_of(runs, name) for name in POOLED_DOMAINS if len(runs := load(results, name))]
    only, sets = chosen(only)
    written = pooled_figures(pd.concat(tables, ignore_index=True), out / ALL_DOMAINS, sets, only)
    if not only or "needed_by_states" in only:
        written += needed_by_states(results, out / ALL_DOMAINS, sets)
    if not only or "discovery_errors" in only:
        written += discovery_errors(results, out / ALL_DOMAINS, sets)
    return written


def domains_with_results(results: Path) -> list[str]:
    return (
        sorted(p.name for p in results.iterdir() if p.is_dir() and latest_runs(p, p.name))
        if results.is_dir()
        else []
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27: the figures of one or more domains.")
    parser.add_argument("domains", nargs="*", help=f"e.g. sysadmin sepsis drone; {ALL_DOMAINS}: "
                        f"the pooled figures over {', '.join(POOLED_DOMAINS)}")  # fmt: skip
    parser.add_argument("--list", action="store_true", help="print the figure names and exit")
    parser.add_argument("--only", nargs="+", metavar="FIGURE",
                        help="draw only these figures (pooled: the thirty-four pooled figures; paper or "
                        "appendix: that set of the paper figures)")  # fmt: skip
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
            print(f"{group:<8} {figure:<36} {(draw.__doc__ or figure).strip().splitlines()[0]}")
        for group, figures in GROUPS.items():
            print(f"{'':<8} {group:<36} with --only: the {len(figures)} figures from {figures[0]} on")
        for name, ours in SETS.items():
            print(f"{'':<8} {name:<36} with --only: the paper figures in {name}/ only (ours: "
                  f"{', '.join(ours)}); default: every set")  # fmt: skip
        return 0
    known = [*FIGURES, *GROUPS, *SETS, "needed_by_states", "discovery_errors"]  # these two: all_domains only
    wrong = [f for f in args.only or [] if f not in known]
    if wrong:
        parser.error(f"no figure {wrong}; the figures are: {', '.join(known)}")
    known = domains_with_results(args.results)
    if not args.domains:
        parser.error(f"give a domain; with results in {args.results}: {', '.join(known) or 'none'}")
    for name in POOLED_DOMAINS if ALL_DOMAINS in args.domains else []:
        if name not in known:
            parser.error(f"{ALL_DOMAINS} needs the results of {name!r}, and {args.results} has none")
    for name in args.domains:
        if name not in known and name != ALL_DOMAINS:
            parser.error(
                f"no results of {name!r} in {args.results}; with results: {', '.join(known) or 'none'}"
            )
    for name in args.domains:
        if name == ALL_DOMAINS:
            plot_all_domains(args.results, args.out, args.only)
        else:
            plot_domain(name, args.results, args.out, args.solver, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
