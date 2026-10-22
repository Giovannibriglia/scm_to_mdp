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
    "BIC -> RMDP": {"color": "#0072B2", "marker": "o"},
    "PCMCI+ -> RMDP": {"color": "#009E73", "marker": "s"},
    "GES -> RMDP": {"color": "#CC79A7", "marker": "^"},
    "Schnitzer IA": {"color": "#8C564B", "marker": "D"},
    "Schnitzer exact": {"color": "#8C564B", "marker": "d"},
    "flat RMDP": {"color": "#E69F00", "marker": "v"},
    "flat MDP": {"color": "#7F7F7F", "marker": "v"},
    "DiFaRL": {"color": "#D55E00", "marker": "P"},
}
ARM_LABEL = {
    "BIC -> RMDP": "BIC → RMDP",
    "PCMCI+ -> RMDP": "PCMCI+ → RMDP",
    "GES -> RMDP": "GES → RMDP",
    "Schnitzer IA": "Schnitzer et al., IA (true graph)",
    "Schnitzer exact": "Schnitzer et al., exact (true graph)",
    "flat RMDP": "flat RMDP",
    "flat MDP": "flat MDP (no guarantee)",
    "DiFaRL": "DiFaRL (no guarantee)",
}
FULL_WIDTH = 7.0  # inches, a two-column paper's text width
ROLE_LINE = {"ours": "-", "oracle": ":", "baseline": "--"}
EXTRA_COLOURS = ["#56B4E9", "#F0E442", "#000000", "#9467BD", "#17BECF", "#BCBD22"]
EXTRA_MARKERS = ["h", "*", "p", "<", ">", "X"]


def method_styles(methods: dict[str, tuple[str, bool]]) -> dict[str, dict]:
    """{method: (role, guarantee)} -> {method: colour, marker, ls, label}."""
    styles, extra = {}, 0
    for name, (role, guarantee) in methods.items():
        if name in ARM_STYLE:
            colour, marker = ARM_STYLE[name]["color"], ARM_STYLE[name]["marker"]
        else:
            colour, marker = EXTRA_COLOURS[extra % 6], EXTRA_MARKERS[extra % 6]
            extra += 1
        label = ARM_LABEL.get(name, name if guarantee else f"{name} (no guarantee)")
        styles[name] = {"color": colour, "marker": marker, "ls": ROLE_LINE.get(role, "-"), "label": label}
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
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
