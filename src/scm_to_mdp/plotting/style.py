"""One look for every figure: a fixed colour and marker per method, paper-sized fonts, PDF + PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to files, also from worker processes
import matplotlib.pyplot as plt  # noqa: E402

# colour-blind safe colours (Okabe & Ito); the same method always looks the same
ARM_STYLE = {
    "BIC -> RMDP": {"color": "#0072B2", "marker": "o", "ls": "-"},
    "PCMCI+ -> RMDP": {"color": "#009E73", "marker": "s", "ls": "-"},
    "GES -> RMDP": {"color": "#CC79A7", "marker": "^", "ls": "-"},
    "Schnitzer IA": {"color": "#8C564B", "marker": "D", "ls": "-"},
    "Schnitzer exact": {"color": "#8C564B", "marker": "d", "ls": "--"},
    "flat RMDP": {"color": "#E69F00", "marker": "v", "ls": "-"},
    "flat MDP": {"color": "#7F7F7F", "marker": "v", "ls": "--"},
    "DiFaRL": {"color": "#D55E00", "marker": "P", "ls": "-"},
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
