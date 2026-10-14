"""One figure style for the paper: colour-blind-safe palette, column widths, vector output.

Figures are always drawn from result files produced by the experiment scripts; nothing is drawn by
hand. ``save`` writes a vector PDF (for the paper) and a 300-dpi PNG (for quick viewing).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe & Ito (2008), "Color Universal Design": distinguishable under the common colour-vision deficiencies.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}
SINGLE_COLUMN = 3.4   # inches (two-column paper layouts)
DOUBLE_COLUMN = 7.0

STYLE = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,           # embed TrueType fonts (editable text in the PDF)
    "ps.fonttype": 42,
}


def use_style() -> None:
    plt.rcParams.update(STYLE)


def save(fig, path: str | Path) -> list[Path]:
    """Write ``<path>.pdf`` and ``<path>.png`` (300 dpi); return the written paths."""
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    out = [base.with_suffix(".pdf"), base.with_suffix(".png")]
    fig.savefig(out[0])
    fig.savefig(out[1], dpi=300)
    return out
