"""One look for every figure: a fixed colour and marker per method, paper-sized fonts, PDF + PNG.

Line style by role: our methods solid, oracle methods (given the true graph) dotted, baselines
dashed. Methods not listed in ARM_STYLE get the next free colour and marker."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to files, also from worker processes
import matplotlib.pyplot as plt  # noqa: E402

# colour-blind safe colours: Okabe & Ito (2008), and Paul Tol's indigo, olive and teal for the rest.
# Colours are unique within one experiment's figures (exp1, exp2), and some are reused across them
# for methods that never share a figure (e.g. orange: flat in exp1, DiFaRL in exp2); the marker and
# the line style (ROLE_LINE) tell methods apart as well. Green means "given the true graph".
ARM_STYLE = {
    "PCMCI+ -> IMDP": {"color": "#0072B2", "marker": "s"},  # blue
    "DYNOTEARS -> IMDP": {"color": "#D55E00", "marker": "p"},  # vermilion
    "sparse -> IMDP": {"color": "#CC79A7", "marker": "o"},  # reddish purple (exp1, sparse arm)
    "true graph -> IMDP": {"color": "#009E73", "marker": "d"},  # bluish green
    "Schnitzer exact": {"color": "#009E73", "marker": "d"},  # its name in results written before v1.33
    "Factored SPIBB (true graph)": {"color": "#009E73", "marker": "d"},  # given the true graph (exp2)
    "oracle -> IMDP": {"color": "#000000", "marker": "D"},  # the true observed MDP (baselines/true_model.py)
    "flat IMDP": {"color": "#E69F00", "marker": "v"},  # orange
    "flat MDP": {"color": "#E69F00", "marker": "v"},
    "full-fact -> IMDP": {"color": "#56B4E9", "marker": "^"},  # sky blue; no discovery: every input a parent
    "DiFaRL": {"color": "#E69F00", "marker": "P"},  # exp2
    "FOCUS": {"color": "#56B4E9", "marker": "h"},  # exp2
    "flat SPIBB": {"color": "#CC79A7", "marker": "<"},  # exp2
    "SL-SPIBB": {"color": "#332288", "marker": ">"},  # indigo (Tol), exp2
    "offline DQN": {"color": "#999933", "marker": "X"},  # olive (Tol), exp2
    "offline CQL": {"color": "#882255", "marker": "*"},  # wine (Tol)
    "BIC -> IMDP": {"color": "#44AA99", "marker": "o"},  # teal (Tol)
    "GES -> IMDP": {"color": "#CC79A7", "marker": "^"},
    "BIC -> IMDP (hull)": {"color": "#332288", "marker": "o"},
    "BIC -> IMDP (padded)": {"color": "#332288", "marker": "s"},
}
ARM_LABEL = {
    "BIC -> IMDP": "BIC → IMDP",
    "PCMCI+ -> IMDP": "PCMCI+ → IMDP",
    "GES -> IMDP": "GES → IMDP",
    "DYNOTEARS -> IMDP": "DYNOTEARS → IMDP",
    "BIC -> IMDP (hull)": "BIC → IMDP (hull)",
    "BIC -> IMDP (padded)": "BIC → IMDP (padded)",
    "true graph -> IMDP": "true graph → IMDP",
    "Schnitzer exact": "Schnitzer et al., exact (true graph)",
    "flat IMDP": "flat IMDP",
    "flat MDP": "flat MDP",
    "DiFaRL": "DiFaRL",
    "full-fact -> IMDP": "full-fact → IMDP",
    "full-fact -> MDP": "full-fact → MDP",
    "sparse -> IMDP": "sparse → IMDP",
    "sparse -> MDP": "sparse → MDP",
    "oracle -> IMDP": "oracle",  # the true observed MDP (± the solver's tolerance in an IMDP grid)
    "oracle -> MDP": "oracle",  # a grid has one of the two
}
FULL_WIDTH = 7.0  # inches, a two-column paper's text width
DPI = 500  # PNG resolution (the PDF is vector)
ROLE_LINE = {
    "ours": "-",
    "oracle": ":",
    "reference": ":",
    "control": "-.",
    "baseline": "--",
}  # "oracle": older results
NO_GUARANTEE_LINE = "-."  # ours without a guarantee (X -> MDP): the colour of its X -> IMDP twin
EXTRA_COLOURS = ["#F0E442", "#BCBD22", "#8C8C00", "#E377C2"]
EXTRA_MARKERS = ["h", "*", "p", "<", ">", "X"]


def method_styles(methods: dict[str, tuple[str, bool]]) -> dict[str, dict]:
    """{method: (role, guarantee)} -> {method: colour, marker, ls, label, role}. An "X -> MDP" method
    takes the colour and marker of "X -> IMDP" and a dash-dot line (no guarantee); the references (true
    graph) are dotted with a guarantee, dash-dot without."""
    styles, extra = {}, 0
    for name, (role, guarantee) in methods.items():
        twin = name.replace(" -> MDP", " -> IMDP")
        if name in ARM_STYLE or twin in ARM_STYLE:
            style = ARM_STYLE.get(name) or ARM_STYLE[twin]
            colour, marker = style["color"], style["marker"]
        else:
            colour = EXTRA_COLOURS[extra % len(EXTRA_COLOURS)]
            marker = EXTRA_MARKERS[extra % len(EXTRA_MARKERS)]
            extra += 1
        label = ARM_LABEL.get(name, name.replace("->", "→"))  # the line style tells the guarantee
        no_guarantee = role in ("ours", "reference", "oracle") and not guarantee  # X -> MDP
        ls = NO_GUARANTEE_LINE if no_guarantee else ROLE_LINE.get(role, "-")
        styles[name] = {"color": colour, "marker": marker, "ls": ls, "label": label, "role": role}
    return styles


def setup():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,  # editable text in the PDF
            "savefig.bbox": "tight",
        }
    )


def save(fig, stem: Path):
    """Write stem.pdf and stem.png."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=DPI)
    plt.close(fig)
