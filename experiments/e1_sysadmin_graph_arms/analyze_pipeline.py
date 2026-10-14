"""E1 stage 3 figures and tables, from ``results/e1/pipeline_arms.csv`` (no simulation here).

Writes under ``results/e1/figures`` and ``results/e1/tables``:

- ``pipeline_regret.{pdf,png}``: regret of the returned policy against total episodes, one panel per
  (edge strength, logger suite), one line per arm (plug-in controls dashed);
- ``pipeline_promise.{pdf,png}``: (a) share of runs where the promise held (promise <= true value;
  bars) and where the true kernel lay inside every box (diamonds: the event the guarantee rests on),
  (b) feasible rows with a set narrower than the simplex, (c) promise against true value;
- ``pipeline_s5.{pdf,png}``: S5, does graph disagreement matter: the largest plug-in regret among the
  sampled hypotheses and the share of states where they disagree, for multi-candidate arms;
- ``pipeline_arms.{csv,tex}``: per arm and total episodes (both edges and all suites pooled): regret,
  promise, sound rate, truth inside every box, rows with a set, valid structure.

Every number is a mean over runs with a 95% percentile-bootstrap interval.

    uv run python experiments/e1_sysadmin_graph_arms/analyze_pipeline.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scm_to_mdp.plotting import DOUBLE_COLUMN, OKABE_ITO, save, use_style
from scm_to_mdp.stats import bootstrap_mean_ci

ARM_STYLE = {  # arm -> (label, colour, linestyle)
    "oracle": ("oracle graph", OKABE_ITO["black"], "-"),
    "flat": ("flat (no structure)", OKABE_ITO["grey"], "-"),
    "bic": ("one learned graph (BIC)", OKABE_ITO["vermillion"], "-"),
    "posterior": ("Bayesian credible family", OKABE_ITO["purple"], "-"),
    "icp": ("ICP family", OKABE_ITO["sky"], "-"),
    "agreement": ("agreement rule", OKABE_ITO["green"], "-"),
    "icp_agreement": ("ICP + agreement", OKABE_ITO["blue"], "-"),
    "plugin_oracle": ("plug-in, oracle graph", OKABE_ITO["black"], "--"),
    "plugin_bic": ("plug-in, BIC graph", OKABE_ITO["vermillion"], "--"),
}
SET_ARMS = [a for a in ARM_STYLE if not a.startswith("plugin")]
MULTI = ["posterior", "icp", "agreement", "icp_agreement"]


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k in ("suite", "edge", "arm"):
                continue
            r[k] = float(v) if v not in ("", "nan") else np.nan
    return rows


def episode_axis(ax, episodes) -> None:
    """Log x axis labelled with the plain episode counts that were run."""
    ax.set_xscale("log")
    ax.set_xticks(episodes, [str(int(e)) for e in episodes])
    ax.xaxis.set_minor_locator(plt.NullLocator())


def pick(rows, **eq) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in eq.items())]


def ci(values) -> tuple[float, float, float]:
    v = [x for x in values if not np.isnan(x)]
    return bootstrap_mean_ci(v) if v else (np.nan, np.nan, np.nan)


def regret_figure(rows, out: Path) -> list[Path]:
    suites = sorted({r["suite"] for r in rows})
    edges = [e for e in ("strong", "weak") if any(r["edge"] == e for r in rows)]
    episodes = sorted({r["episodes"] for r in rows})
    fig, axes = plt.subplots(len(edges), len(suites), figsize=(DOUBLE_COLUMN, 1.75 * len(edges) + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    for i, edge in enumerate(edges):
        for j, suite in enumerate(suites):
            ax = axes[i, j]
            for arm, (label, colour, ls) in ARM_STYLE.items():
                stats = [ci([r["regret"] for r in pick(rows, suite=suite, edge=edge, arm=arm, episodes=e)])
                         for e in episodes]
                if all(np.isnan(s[0]) for s in stats):
                    continue
                m, lo, hi = (np.array(x) for x in zip(*stats, strict=True))
                ax.plot(episodes, m, ls, color=colour, label=label, marker="o", ms=2.5)
                ax.fill_between(episodes, lo, hi, color=colour, alpha=0.12, lw=0)
            episode_axis(ax, episodes)
            if i == 0:
                ax.set_title(suite)
            if j == 0:
                ax.set_ylabel(f"{edge} edge\nregret $V^*-V^{{\\hat\\pi}}$")
            if i == len(edges) - 1:
                ax.set_xlabel("total episodes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.22 if len(edges) > 1 else 0.35, hspace=0.25, wspace=0.12)
    paths = save(fig, out)
    plt.close(fig)
    return paths


def promise_figure(rows, out: Path) -> list[Path]:
    episodes = sorted({r["episodes"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.6),
                             gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1, 1]})
    y = np.arange(len(ARM_STYLE))[::-1]
    sound = [ci([r["sound"] for r in pick(rows, arm=a)]) for a in ARM_STYLE]
    m, lo, hi = (np.array(v) for v in zip(*sound, strict=True))
    inside = [ci([r["truth_in_boxes"] for r in pick(rows, arm=a)]) for a in ARM_STYLE]
    for k, (_, colour, ls) in enumerate(ARM_STYLE.values()):
        axes[0].barh(y[k], m[k], color=colour, hatch="////" if ls == "--" else None,
                     edgecolor="white" if ls == "--" else colour, lw=0,
                     xerr=[[m[k] - lo[k]], [hi[k] - m[k]]], error_kw={"lw": 0.6, "capsize": 1.5})
        if not np.isnan(inside[k][0]):
            axes[0].plot(inside[k][0], y[k], ls="none", marker="D", ms=3.5, mfc="white",
                         mec=OKABE_ITO["black"], mew=0.8)
    axes[0].set_yticks(y, [ARM_STYLE[a][0] for a in ARM_STYLE], fontsize=6)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("share of runs")
    axes[0].set_title("(a) bars: promise $\\leq$ true value\n$\\diamond$: truth inside every box")
    for arm in SET_ARMS:
        label, colour, ls = ARM_STYLE[arm]
        stats = [ci([r["rows_with_set"] for r in pick(rows, arm=arm, episodes=e)]) for e in episodes]
        mm, lo2, hi2 = (np.array(v) for v in zip(*stats, strict=True))
        axes[1].plot(episodes, mm, ls, color=colour, marker="o", ms=2.5, label=label)
        axes[1].fill_between(episodes, lo2, hi2, color=colour, alpha=0.12, lw=0)
    axes[1].axhline(80, color=OKABE_ITO["grey"], lw=0.6, ls=":")
    episode_axis(axes[1], episodes)
    axes[1].set(xlabel="total episodes", ylabel="feasible rows with a set (of 80)", title="(b) coverage")
    for arm, (label, colour, _) in ARM_STYLE.items():
        sel = pick(rows, arm=arm)
        axes[2].scatter([r["promise"] for r in sel], [r["true_value"] for r in sel], s=3, color=colour,
                        alpha=0.5, lw=0, label=label)
    lim = [0, max(r["v_star"] for r in rows) * 1.05]
    axes[2].plot(lim, lim, color=OKABE_ITO["grey"], lw=0.6, ls="--")
    axes[2].set(xlim=lim, ylim=lim, xlabel="promise (robust value)", ylabel="true value",
                title="(c) above the diagonal = sound")
    paths = save(fig, out)
    plt.close(fig)
    return paths


def s5_figure(rows, out: Path) -> list[Path]:
    episodes = sorted({r["episodes"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN * 0.75, 2.3), gridspec_kw={"wspace": 0.4})
    for arm in MULTI:
        label, colour, ls = ARM_STYLE[arm]
        for ax, col in zip(axes, ("s5_max_regret", "s5_disagree_share"), strict=True):
            stats = [ci([r[col] for r in pick(rows, arm=arm, episodes=e)]) for e in episodes]
            if all(np.isnan(s[0]) for s in stats):
                continue
            m, lo, hi = (np.array(v) for v in zip(*stats, strict=True))
            ax.plot(episodes, m, ls, color=colour, marker="o", ms=2.5, label=label)
            ax.fill_between(episodes, lo, hi, color=colour, alpha=0.12, lw=0)
    for ax in axes:
        episode_axis(ax, episodes)
    axes[0].set(xlabel="total episodes", ylabel="largest plug-in regret\namong hypotheses")
    axes[1].set(xlabel="total episodes", ylabel="share of states where\nthey disagree", ylim=(0, 1.02))
    axes[1].legend(loc="lower left")
    paths = save(fig, out)
    plt.close(fig)
    return paths


TABLE = [("regret", "regret", "{:.1f}"), ("promise", "promise", "{:.1f}"), ("sound", "sound", "{:.2f}"),
         ("truth_in_boxes", "truth in boxes", "{:.2f}"),
         ("rows_with_set", "rows with set", "{:.0f}"), ("valid_structure", "valid structure", "{:.2f}")]


def tables(rows, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = sorted({r["episodes"] for r in rows})
    body = []
    for arm in ARM_STYLE:
        for e in episodes:
            sel = pick(rows, arm=arm, episodes=e)
            if sel:
                body.append((arm, int(e), len(sel), {c: ci([r[c] for r in sel]) for c, *_ in TABLE}))
    csv_path = out_dir / "pipeline_arms.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        stats_cols = [f"{c}_{s}" for c, *_ in TABLE for s in ("mean", "lo", "hi")]
        w.writerow(["arm", "episodes", "runs", *stats_cols])
        for arm, e, n, st in body:
            w.writerow([arm, e, n] + [f"{v:.4g}" for c, *_ in TABLE for v in st[c]])
    tex_path = out_dir / "pipeline_arms.tex"
    lines = ["\\begin{tabular}{llr" + "c" * len(TABLE) + "}", "\\toprule",
             "arm & episodes & runs & " + " & ".join(h for _, h, _ in TABLE) + " \\\\", "\\midrule"]
    for arm, e, n, st in body:
        cells = []
        for c, _, fmt in TABLE:
            m, lo, hi = st[c]
            cells.append("--" if np.isnan(m) else f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]")
        lines.append(f"{ARM_STYLE[arm][0]} & {e} & {n} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex_path.write_text("\n".join(lines) + "\n")
    return [csv_path, tex_path]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/e1/pipeline_arms.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/e1"))
    args = ap.parse_args()
    use_style()
    rows = load(args.results)
    written = regret_figure(rows, args.out / "figures" / "pipeline_regret")
    written += promise_figure(rows, args.out / "figures" / "pipeline_promise")
    written += s5_figure(rows, args.out / "figures" / "pipeline_s5")
    written += tables(rows, args.out / "tables")
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
