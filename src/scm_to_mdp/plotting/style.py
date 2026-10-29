"""One look for every figure: a fixed colour and marker per method, paper-sized fonts, PDF + PNG.

Line style by role: our methods solid, oracle methods (given the true graph) dotted, baselines
dashed. Methods not listed in ARM_STYLE get the next free colour and marker."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to files, also from worker processes
import matplotlib.pyplot as plt  # noqa: E402

# colour-blind safe colours (Okabe & Ito); the same method always looks the same. The line style
# comes from the role (ROLE_LINE), not from here.
ARM_STYLE = {
    "BIC -> IMDP": {"color": "#0072B2", "marker": "o"},
    "PCMCI+ -> IMDP": {"color": "#009E73", "marker": "s"},
    "GES -> IMDP": {"color": "#CC79A7", "marker": "^"},
    "true graph -> IMDP": {"color": "#8C564B", "marker": "d"},
    "Schnitzer exact": {"color": "#8C564B", "marker": "d"},  # its name in results written before v1.33
    "flat IMDP": {"color": "#E69F00", "marker": "v"},
    "flat MDP": {"color": "#7F7F7F", "marker": "v"},
    "DiFaRL": {"color": "#D55E00", "marker": "P"},
    "offline DQN": {"color": "#56B4E9", "marker": "X"},
    "offline CQL": {"color": "#000000", "marker": "*"},
    "flat SPIBB": {"color": "#9467BD", "marker": "<"},
    "SL-SPIBB": {"color": "#9467BD", "marker": ">"},
    "Factored SPIBB (true graph)": {"color": "#9467BD", "marker": "d"},  # the SPIBB family, given the graph
    "FOCUS": {"color": "#17BECF", "marker": "h"},
    "DYNOTEARS -> IMDP": {"color": "#882255", "marker": "p"},
    "BIC -> IMDP (hull)": {"color": "#332288", "marker": "o"},
    "BIC -> IMDP (padded)": {"color": "#332288", "marker": "s"},
    "oracle -> IMDP": {"color": "#404040", "marker": "D"},  # the true observed MDP (baselines/true_model.py)
    "full-fact -> IMDP": {"color": "#0072B2", "marker": "^"},  # no discovery: every input a parent
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
            "lines.linewidth": 1.4,
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
