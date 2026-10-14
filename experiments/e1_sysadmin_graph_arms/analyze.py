"""E1 figures and tables from the result files (no simulation here).

Reads whichever of these exist and writes, under ``results/e1/figures`` and ``results/e1/tables``:

- ``pilot_loggers.csv`` (``pilot_loggers.py``) -> ``pilot_loggers.{pdf,png}``, ``pilot_loggers.{csv,tex}``:
  what each logger family's data cover and can tell;
- ``structure_arms.csv`` (``structure_arms.py``) -> ``structure_arms.{pdf,png}``,
  ``structure_arms.{csv,tex}``: S2/S3 graph arms against the true graph, and the S1 checks.

Every number is a mean over seeds with a 95% percentile-bootstrap interval.

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


ARM_STYLE = {"bic": ("one graph: BIC", OKABE_ITO["vermillion"]),
             "jpcmci": ("one graph: J-PCMCI+", OKABE_ITO["purple"]),
             "posterior": ("Bayesian credible family", OKABE_ITO["orange"]),
             "icp": ("ICP family", OKABE_ITO["blue"])}
SUITE_STYLE = {"S-A aliasing": OKABE_ITO["grey"], "S-B two aliasing": OKABE_ITO["blue"],
               "S-C softmax+lazy": OKABE_ITO["green"], "S-D two specialists": OKABE_ITO["sky"]}


def load_structure(path: Path):
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["episodes"] = int(r["episodes"])
        for k in ("valid", "family_size", "exact", "missed_parent", "gate_p", "markov_p", "stationarity_p"):
            r[k] = float(r[k])
    return rows


def _cell(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def structure_figure(rows, out_path: Path, alpha: float = 0.1) -> list[Path]:
    """(a) missed true parents of the one-graph arms; (b) ICP family size per suite; (c) validity."""
    import matplotlib.pyplot as plt

    use_style()
    episodes = sorted({r["episodes"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 3.1), gridspec_kw={"wspace": 0.42})

    def line(ax, xs, sets, color, label, ls="-"):
        stats = [bootstrap_mean_ci(v) for v in sets]
        m = [s[0] for s in stats]
        ax.plot(xs, m, ls, color=color, marker="o", ms=3, label=label)
        ax.fill_between(xs, [s[1] for s in stats], [s[2] for s in stats], color=color, alpha=0.15, lw=0)

    ax = axes[0]
    for arm in ("bic", "jpcmci"):
        if not _cell(rows, arm=arm):
            continue
        for edge, ls in (("weak", "-"), ("strong", "--")):
            sets = [[r["missed_parent"] for r in _cell(rows, arm=arm, edge=edge, episodes=e)]
                    for e in episodes]
            line(ax, episodes, sets, ARM_STYLE[arm][1], f"{ARM_STYLE[arm][0]}, {edge}", ls)
    ax.set(xscale="log", ylim=(-0.03, 1.03), ylabel="runs missing a true parent")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)

    ax = axes[1]
    for suite, color in SUITE_STYLE.items():
        sets = [[r["family_size"] for r in _cell(rows, arm="icp", edge="weak", suite=suite, episodes=e)]
                for e in episodes]
        if all(sets):
            line(ax, episodes, sets, color, suite)
    ax.set(xscale="log", yscale="log", ylabel="ICP family size per target")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)

    ax = axes[2]
    for arm in ("icp", "posterior"):
        sets = [[r["valid"] for r in _cell(rows, arm=arm, edge="weak", episodes=e)] for e in episodes]
        line(ax, episodes, sets, ARM_STYLE[arm][1], ARM_STYLE[arm][0])
    ax.axhline(1 - alpha, color="black", lw=0.8, ls=":", label=f"nominal level {1 - alpha:.1f}")
    ax.set(xscale="log", ylim=(-0.03, 1.03), ylabel="runs with a valid family")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1)
    from matplotlib.ticker import NullFormatter, NullLocator
    for a in axes:
        a.set_xticks(episodes)
        a.set_xticklabels([str(e) for e in episodes])
        a.xaxis.set_minor_locator(NullLocator())
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_xlabel("episodes per suite")
    for a, tag in zip(axes, "abc", strict=True):
        a.text(-0.02, 1.02, f"({tag})", transform=a.transAxes, ha="right", va="bottom", fontweight="bold")
    paths = save(fig, out_path)
    plt.close(fig)
    return paths


def structure_tables(rows, out_dir: Path, alpha: float = 0.1) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({(r["suite"], r["edge"], r["episodes"], r["arm"]) for r in rows if r["arm"] != "oracle"})
    csv_path = out_dir / "structure_arms.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "edge", "episodes", "arm", "runs", "valid_mean", "valid_lo", "valid_hi",
                    "family_size_mean", "missed_parent_mean"])
        for k in keys:
            sub = _cell(rows, suite=k[0], edge=k[1], episodes=k[2], arm=k[3])
            v = bootstrap_mean_ci([r["valid"] for r in sub])
            w.writerow([*k, len(sub), *(f"{x:.3f}" for x in v),
                        f"{np.mean([r['family_size'] for r in sub]):.2f}",
                        f"{np.mean([r['missed_parent'] for r in sub]):.2f}"])
    # Compact LaTeX table: weak edge, valid share and family size per arm.
    arms = [a for a in ("bic", "jpcmci", "posterior", "icp") if _cell(rows, arm=a)]
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{ARM_STYLE[a][0]}}}" for a in arms)
    lines = [r"\begin{tabular}{ll" + "rr" * len(arms) + "}", r"\toprule",
             "Suite & Episodes & " + head + r" \\",
             " & & " + " & ".join("valid & size" for _ in arms) + r" \\", r"\midrule"]
    for suite in SUITE_STYLE:
        for e in sorted({r["episodes"] for r in rows}):
            cells = []
            for a in arms:
                sub = _cell(rows, suite=suite, edge="weak", episodes=e, arm=a)
                if not sub:
                    cells += ["--", "--"]
                    continue
                cells += [f"{np.mean([r['valid'] for r in sub]):.2f}",
                          f"{np.mean([r['family_size'] for r in sub]):.1f}"]
            lines.append(f"{suite} & {e} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = out_dir / "structure_arms.tex"
    tex_path.write_text("\n".join(lines) + "\n")
    # S1 checks: false-alarm rates of the data-only tests (setting 1: none should reject).
    s1_path = out_dir / "s1_checks.csv"
    with s1_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "runs", "gate_reject", "markov_reject", "stationarity_reject"])
        for suite in SUITE_STYLE:
            sub = _cell(rows, suite=suite, arm="oracle")
            if sub:
                tests = ("gate_p", "markov_p", "stationarity_p")
                rate = [np.mean([r[k] <= alpha / 4 for r in sub]) for k in tests]
                w.writerow([suite, len(sub), *(f"{x:.3f}" for x in rate)])
    return [csv_path, tex_path, s1_path]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/e1"))
    args = ap.parse_args()
    written: list[Path] = []
    if (args.results / "pilot_loggers.csv").exists():
        data = load(args.results / "pilot_loggers.csv")
        summary = summarise(data)
        n_seeds = {lg: len(next(iter(cols.values()))) for lg, cols in data.items()}
        written += figure(data, summary, args.results / "figures" / "pilot_loggers")
        written += write_tables(summary, n_seeds, args.results / "tables")
    if (args.results / "structure_arms.csv").exists():
        rows = load_structure(args.results / "structure_arms.csv")
        written += structure_figure(rows, args.results / "figures" / "structure_arms")
        written += structure_tables(rows, args.results / "tables")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
