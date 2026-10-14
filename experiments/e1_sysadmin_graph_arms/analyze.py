"""E1 figures and tables from the result files (no simulation here).

Reads ``results/e1/pilot_loggers.csv`` (written by ``pilot_loggers.py``) and writes

- ``results/e1/figures/pilot_loggers.{pdf,png}``: what each logger family's data cover and can tell;
- ``results/e1/tables/pilot_loggers.csv`` and ``.tex``: the same numbers, mean and 95% bootstrap
  interval over seeds.

    uv run python experiments/e1_sysadmin_graph_arms/analyze.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from scm_to_mdp.plotting import DOUBLE_COLUMN, OKABE_ITO, save, use_style
from scm_to_mdp.stats import bootstrap_mean_ci

ORDER = ["uniform", "specialist", "aliasing", "softmax_q", "lazy", "recent_failure", "confounded"]
LABELS = {"uniform": "uniform", "specialist": "region specialist", "aliasing": "aliasing rule",
          "softmax_q": "softmax over $Q^*$", "lazy": "lazy", "recent_failure": "recent failure",
          "confounded": "confounded"}
METRICS = [  # (column, short name, LaTeX header)
    ("joint_rows", "joint coverage", "joint"),
    ("factor_only_rows", "factor-only coverage", "factor only"),
    ("aliasing_max", "aliasing index", "aliasing"),
    ("aliasing_empty_cells", "empty parent cells", "empty cells"),
    ("concentrability_pi_star", "concentrability ($\\pi^*$)", "$C_{\\pi^*}$"),
    ("confounding_excess", "confounding excess", "confounding"),
]


def load(path: Path) -> dict[str, dict[str, list[float]]]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        bucket = out.setdefault(r["logger"], {})
        for col, *_ in METRICS:
            if col in r:
                bucket.setdefault(col, []).append(float(r[col]))
    return out


def summarise(data):
    return {lg: {col: bootstrap_mean_ci(v) for col, v in cols.items()} for lg, cols in data.items()}


def write_tables(summary, n_seeds: dict[str, int], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    loggers = [lg for lg in ORDER if lg in summary]
    csv_path = out_dir / "pilot_loggers.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["logger", "seeds"] + [f"{c}_{s}" for c, *_ in METRICS for s in ("mean", "lo", "hi")])
        for lg in loggers:
            w.writerow([lg, n_seeds[lg]] + [f"{x:.4g}" for c, *_ in METRICS for x in summary[lg][c]])

    def cell(stat):
        m, lo, hi = stat
        if np.isinf(m):
            return "$\\infty$"
        return f"{m:.2f}" if hi - lo < 5e-3 else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"

    tex_path = out_dir / "pilot_loggers.tex"
    lines = ["\\begin{tabular}{l" + "r" * len(METRICS) + "}", "\\toprule",
             "Logger & " + " & ".join(h for *_, h in METRICS) + " \\\\", "\\midrule"]
    for lg in loggers:
        lines.append(LABELS[lg] + " & " + " & ".join(cell(summary[lg][c]) for c, *_ in METRICS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")
    return [csv_path, tex_path]


def figure(data, summary, out_path: Path) -> list[Path]:
    """Three panels sharing the logger axis: coverage, aliasing, confounding."""
    import matplotlib.pyplot as plt

    use_style()
    loggers = [lg for lg in ORDER if lg in summary]
    y = np.arange(len(loggers))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.4, 1, 1], "wspace": 0.22})

    ax = axes[0]
    joint = np.array([summary[lg]["joint_rows"][0] for lg in loggers])
    fonly = np.array([summary[lg]["factor_only_rows"][0] for lg in loggers])
    ax.barh(y, joint, height=0.6, color=OKABE_ITO["blue"], label="visited jointly")
    ax.barh(y, fonly, left=joint, height=0.6, color=OKABE_ITO["sky"], label="covered factor by factor only")
    ax.set(xlim=(0, 1), xlabel="share of state-action rows")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)

    def dots(ax, col, color):
        for k, lg in enumerate(loggers):
            vals = data[lg][col]
            jitter = np.linspace(-0.15, 0.15, len(vals))
            ax.scatter(vals, k + jitter, s=7, color=color, alpha=0.6, linewidths=0)
            m, lo, hi = summary[lg][col]
            ax.errorbar(m, k, xerr=[[m - lo], [hi - m]], fmt="|", color="black", ms=8, lw=1, capsize=0)

    dots(axes[1], "aliasing_max", OKABE_ITO["orange"])
    axes[1].set(xlim=(-0.03, 1.03), xlabel="aliasing index (max over machines)")
    dots(axes[2], "confounding_excess", OKABE_ITO["vermillion"])
    axes[2].set(xlabel="confounding excess")
    axes[2].set_xlim(left=-0.005)

    axes[0].set(yticks=y, yticklabels=[LABELS[lg] for lg in loggers])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    for ax, tag in zip(axes, "abc", strict=True):
        ax.text(-0.02, 1.02, f"({tag})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")
    paths = save(fig, out_path)
    plt.close(fig)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/e1"))
    args = ap.parse_args()
    data = load(args.results / "pilot_loggers.csv")
    summary = summarise(data)
    n_seeds = {lg: len(next(iter(cols.values()))) for lg, cols in data.items()}
    written = figure(data, summary, args.results / "figures" / "pilot_loggers")
    written += write_tables(summary, n_seeds, args.results / "tables")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
