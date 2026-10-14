"""E1 stage 3 figures and tables, from the pipeline and ablation result files (no simulation here).

Writes under ``results/e1/figures`` and ``results/e1/tables`` (each figure as vector PDF + 300-dpi PNG):

- ``pipeline_regret_eps{100 eps}`` (``eps02``, ``eps20``): regret of the returned policy against
  episodes per logger, one panel
  per (edge, suite), one line per arm (plug-in controls dashed); dotted grey: the best logger's own
  regret (its value is a reference, not an arm);
- ``pipeline_promise``: (a) share of runs where the promise held (bars) and where the true kernel lay
  inside every box (diamonds); (b) feasible rows with a set that the data never visited jointly
  (covered factor by factor only; flat has none by construction); (c) promise against true value;
- ``pipeline_s5``: S5 for multi-candidate arms: hypotheses before and after the merge (K, K'), the
  largest plug-in regret among sampled hypotheses, and the share of reached states where they disagree;
- ``pipeline_loggers``: each multi-logger suite against its loggers alone at the same total data;
- ``pipeline_gate``: the S1 gate (rejection rate per suite) and S-E (truth inside every box and regret
  for the arms, with and without following the gate);
- ``pipeline_ablations`` (if the ablation files exist): false bound d = 2, distractor bit, N = 6;
- tables ``pipeline_arms``, ``model_cards``, ``pipeline_loggers``, ``ablations`` (``.csv`` and
  ``.tex``).

Unless stated, setting-1 suites S-A..S-D with all loggers are pooled over edges and exploration.
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
    "oracle_gated": ("oracle, gated (S1)", OKABE_ITO["orange"], "-"),
    "plugin_oracle": ("plug-in, oracle graph", OKABE_ITO["black"], "--"),
    "plugin_bic": ("plug-in, BIC graph", OKABE_ITO["vermillion"], "--"),
}
SET_ARMS = [a for a in ARM_STYLE if not a.startswith("plugin")]
SHORT = {"oracle": "oracle", "flat": "flat", "bic": "BIC", "posterior": "credible", "icp": "ICP",
         "agreement": "agreement", "icp_agreement": "ICP+agr.", "oracle_gated": "oracle, gated"}
MULTI = ["posterior", "icp", "agreement", "icp_agreement"]
SETTING1 = ("S-A", "S-B", "S-C", "S-D")
TEXT = ("suite", "base_suite", "edge", "arm")


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in TEXT:
                r[k] = float(v) if v not in ("", "nan", None) else np.nan
    return rows


def pick(rows, **eq) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in eq.items())]


def core(rows) -> list[dict]:
    """Setting-1 suites, all loggers together."""
    return [r for r in rows if r["alone"] == 0 and r["base_suite"].startswith(SETTING1)]


def ci(values) -> tuple[float, float, float]:
    v = [x for x in values if not np.isnan(x)]
    return bootstrap_mean_ci(v) if v else (np.nan, np.nan, np.nan)


def episode_axis(ax, sizes) -> None:
    """Log x axis labelled with the plain sizes that were run."""
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{int(e) // 1000}k" if e >= 1000 else str(int(e)) for e in sizes])
    ax.xaxis.set_minor_locator(plt.NullLocator())


def line(ax, rows, col, sizes, colour, ls="-", label=None, by="per_logger"):
    stats = [ci([r[col] for r in rows if r[by] == e]) for e in sizes]
    if all(np.isnan(s[0]) for s in stats):
        return
    m, lo, hi = (np.array(v) for v in zip(*stats, strict=True))
    ax.plot(sizes, m, ls, color=colour, label=label, marker="o", ms=2.5)
    ax.fill_between(sizes, lo, hi, color=colour, alpha=0.12, lw=0)


# ---------------------------------------------------------------------------------------------- figures
def regret_figure(rows, eps: float, out: Path) -> list[Path]:
    rows = [r for r in rows if r["alone"] == 0 and r["eps"] == eps]
    suites = sorted({r["base_suite"] for r in rows})
    edges = [e for e in ("strong", "weak") if any(r["edge"] == e for r in rows)]
    sizes = sorted({r["per_logger"] for r in rows})
    fig, axes = plt.subplots(
        len(edges),
        len(suites),
        figsize=(DOUBLE_COLUMN, 1.75 * len(edges) + 0.7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, edge in enumerate(edges):
        for j, suite in enumerate(suites):
            ax = axes[i, j]
            sel = pick(rows, base_suite=suite, edge=edge)
            for arm, (label, colour, ls) in ARM_STYLE.items():
                line(ax, pick(sel, arm=arm), "regret", sizes, colour, ls, label)
            ref = [r["v_star"] - r["logger_value_best"] for r in pick(sel, arm="oracle")]
            if ref:
                ax.axhline(
                    np.mean(ref),
                    color=OKABE_ITO["grey"],
                    lw=0.8,
                    ls=":",
                    label="best logger's own regret" if (i, j) == (0, 0) else None,
                )
            episode_axis(ax, sizes)
            if i == 0:
                ax.set_title(suite.replace("specialist+confounded", "spec.+confounded"), fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{edge} edge\nregret $V^*-V^{{\\hat\\pi}}$")
    fig.supxlabel("episodes per logger", y=0.17, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.25, hspace=0.25, wspace=0.15)
    paths = save(fig, out)
    plt.close(fig)
    return paths


def promise_figure(rows, out: Path) -> list[Path]:
    rows = core(rows)
    sizes = sorted({r["per_logger"] for r in rows})
    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE_COLUMN, 2.7), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1, 1]}
    )
    arms = [a for a in ARM_STYLE if pick(rows, arm=a)]
    y = np.arange(len(arms))[::-1]
    for k, arm in enumerate(arms):
        _, colour, ls = ARM_STYLE[arm]
        m, lo, hi = ci([r["sound"] for r in pick(rows, arm=arm)])
        axes[0].barh(
            y[k],
            m,
            color=colour,
            hatch="////" if ls == "--" else None,
            edgecolor="white" if ls == "--" else colour,
            lw=0,
            xerr=[[m - lo], [hi - m]],
            error_kw={"lw": 0.6, "capsize": 1.5},
        )
        inside = ci([r["truth_in_boxes"] for r in pick(rows, arm=arm)])[0]
        if not np.isnan(inside):
            axes[0].plot(
                inside, y[k], ls="none", marker="D", ms=3.5, mfc="white", mec=OKABE_ITO["black"], mew=0.8
            )
    axes[0].set_yticks(y, [ARM_STYLE[a][0] for a in arms], fontsize=6)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("share of runs")
    axes[0].set_title("(a) bars: promise $\\leq$ true value\n$\\diamond$: truth inside every box")
    for arm in SET_ARMS:
        label, colour, ls = ARM_STYLE[arm]
        line(axes[1], pick(rows, arm=arm), "rows_set_factor_only", sizes, colour, ls, label)
    episode_axis(axes[1], sizes)
    axes[1].set(
        xlabel="episodes per logger",
        ylabel="rows with a set, never\nvisited jointly",
        title="(b) coverage beyond the data",
    )
    for arm in arms:
        _, colour, _ = ARM_STYLE[arm]
        sel = pick(rows, arm=arm)
        axes[2].scatter(
            [r["promise"] for r in sel], [r["true_value"] for r in sel], s=2, color=colour, alpha=0.35, lw=0
        )
    lim = [0, max(r["v_star"] for r in rows) * 1.05]
    axes[2].plot(lim, lim, color=OKABE_ITO["grey"], lw=0.6, ls="--")
    axes[2].set(
        xlim=lim,
        ylim=lim,
        xlabel="promise (robust value)",
        ylabel="true value",
        title="(c) above the diagonal = sound",
    )
    paths = save(fig, out)
    plt.close(fig)
    return paths


def s5_figure(rows, out: Path) -> list[Path]:
    rows = core(rows)
    sizes = sorted({r["per_logger"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.4), gridspec_kw={"wspace": 0.42})
    for arm in MULTI:
        label, colour, _ = ARM_STYLE[arm]
        sel = pick(rows, arm=arm)
        for r in sel:
            r["_logK"] = np.log10(r["n_hypotheses"]) if r["n_hypotheses"] > 0 else np.nan
            r["_logK2"] = np.log10(r["s5_k_prime"]) if r["s5_k_prime"] > 0 else np.nan
        line(axes[0], sel, "_logK", sizes, colour, ":", None)
        line(axes[0], sel, "_logK2", sizes, colour, "-", label)
        line(axes[1], sel, "s5_max_regret", sizes, colour, "-", label)
        line(axes[2], sel, "s5_disagree_share", sizes, colour, "-", label)
    for ax in axes:
        episode_axis(ax, sizes)
        ax.set_xlabel("episodes per logger")
    axes[0].set(ylabel="$\\log_{10}$ hypotheses", title="(a) $K$ (dotted) $\\to K'$ after merge")
    axes[1].set(ylabel="largest plug-in regret", title="(b) worst sampled hypothesis")
    axes[2].set(ylabel="share of reached states", ylim=(0, 1.02), title="(c) hypotheses disagree")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    fig.subplots_adjust(bottom=0.3)
    paths = save(fig, out)
    plt.close(fig)
    return paths


LOGGER_METRICS = [
    ("icp", "survivors", "ICP family size\n(per machine)"),
    ("icp_agreement", "regret", "regret,\nICP + agreement"),
    ("oracle", "rows_with_set", "rows with a set,\noracle graph"),
]


def loggers_figure(rows, out: Path) -> list[Path]:
    suites = sorted({r["base_suite"] for r in rows if r["alone"] == 1})
    if not suites:
        return []
    fig, axes = plt.subplots(
        len(LOGGER_METRICS),
        len(suites),
        figsize=(DOUBLE_COLUMN, 1.6 * len(LOGGER_METRICS) + 0.5),
        sharex=True,
        squeeze=False,
    )
    colours = [OKABE_ITO["black"], OKABE_ITO["orange"], OKABE_ITO["sky"]]
    for j, suite in enumerate(suites):
        labels = sorted({r["suite"] for r in rows if r["base_suite"] == suite}, key=lambda s: (" | " in s, s))
        totals = sorted({r["episodes"] for r in rows if r["base_suite"] == suite})
        for i, (arm, col, ylabel) in enumerate(LOGGER_METRICS):
            ax = axes[i, j]
            for k, lab in enumerate(labels):
                name = "both loggers" if " | " not in lab else lab.split("alone: ")[1] + " alone"
                line(
                    ax, pick(rows, suite=lab, arm=arm), col, totals, colours[k % 3], "-", name, by="episodes"
                )
            episode_axis(ax, totals)
            if i == 0:
                ax.set_title(suite, fontsize=7)
                ax.legend(fontsize=5.5, loc="best")
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=6.5)
            if i == len(LOGGER_METRICS) - 1:
                ax.set_xlabel("total episodes")
    fig.subplots_adjust(hspace=0.25, wspace=0.3)
    paths = save(fig, out)
    plt.close(fig)
    return paths


def gate_figure(rows, out: Path) -> list[Path]:
    rows = [r for r in rows if r["alone"] == 0 and not np.isnan(r["gate_p"])]
    suites = sorted({r["base_suite"] for r in rows})
    sizes = sorted({r["per_logger"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.4), gridspec_kw={"wspace": 0.45})
    colours = dict(
        zip(suites, [OKABE_ITO[c] for c in ("sky", "green", "purple", "vermillion", "black")], strict=False)
    )
    for s in suites:
        line(axes[0], pick(rows, base_suite=s, arm="oracle"), "gate_rejected", sizes, colours[s], "-", s)
    axes[0].axhline(0.1, color=OKABE_ITO["grey"], lw=0.6, ls=":")
    axes[0].set(ylim=(-0.02, 1.02), ylabel="gate rejection rate", title="(a) S1 gate (nominal 0.1)")
    axes[0].legend(fontsize=5.5, loc="center right")
    se = [r for r in rows if r["base_suite"].startswith("S-E")]
    for arm in ("oracle", "icp_agreement", "oracle_gated", "plugin_oracle"):
        label, colour, ls = ARM_STYLE[arm]
        line(axes[1], pick(se, arm=arm), "truth_in_boxes", sizes, colour, ls, label)
        line(axes[2], pick(se, arm=arm), "regret", sizes, colour, ls, label)
    axes[1].set(ylim=(-0.02, 1.02), ylabel="share of runs", title="(b) S-E: truth inside every box")
    axes[2].set(ylabel="regret", title="(c) S-E: regret")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.62, 0.0), fontsize=6)
    fig.subplots_adjust(bottom=0.3)
    for ax in axes:
        episode_axis(ax, sizes)
        ax.set_xlabel("episodes per logger")
    paths = save(fig, out)
    plt.close(fig)
    return paths


ABLATIONS = [
    ("ablation_d2.csv", "false bound $d=2$", "S-B two aliasing"),
    ("ablation_distractor.csv", "distractor bit", "S-D two specialists"),
    ("ablation_n6.csv", "$N=6$ machines", None),
]


def ablation_figure(rows, results_dir: Path, out: Path) -> tuple[list[Path], list]:
    found = [(results_dir / f, t, base) for f, t, base in ABLATIONS if (results_dir / f).exists()]
    if not found:
        return [], []
    fig, axes = plt.subplots(
        2,
        len(found),
        figsize=(DOUBLE_COLUMN, 3.2),
        squeeze=False,
        sharex=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.3},
    )
    arms = [a for a in SET_ARMS if a != "oracle_gated"]
    x = np.arange(len(arms))
    table = []
    for j, (path, title, base) in enumerate(found):
        abl = [r for r in load(path) if r["alone"] == 0]
        sizes = sorted({r["per_logger"] for r in abl})
        eps = sorted({r["eps"] for r in abl})
        ref = (
            [
                r
                for r in rows
                if r["alone"] == 0
                and r["base_suite"] == base
                and r["per_logger"] in sizes
                and r["eps"] in eps
            ]
            if base
            else []
        )
        for i, col in enumerate(("regret", "truth_in_boxes")):
            ax = axes[i, j]
            for k, (sel, colour, name) in enumerate(
                ((ref, OKABE_ITO["grey"], "core run"), (abl, OKABE_ITO["blue"], "ablation"))
            ):
                if not sel:
                    continue
                st = [ci([r[col] for r in pick(sel, arm=a)]) for a in arms]
                m, lo, hi = (np.array(v) for v in zip(*st, strict=True))
                ax.bar(
                    x + (k - 0.5) * 0.38,
                    m,
                    0.38,
                    color=colour,
                    label=name,
                    yerr=[m - lo, hi - m],
                    error_kw={"lw": 0.5, "capsize": 1},
                )
            ax.set_xticks(x, [SHORT[a] for a in arms], rotation=45, ha="right", fontsize=6)
            if j == 0:
                ax.set_ylabel("regret" if col == "regret" else "truth inside\nevery box")
            if i == 0:
                ax.set_title(title)
                if j == 0:
                    ax.legend(fontsize=6)
        for a in arms:
            sel = pick(abl, arm=a)
            table.append((title, a, len(sel), {c: ci([r.get(c, np.nan) for r in sel]) for c, *_ in ABL_COLS}))
    paths = save(fig, out)
    plt.close(fig)
    return paths, table


# ----------------------------------------------------------------------------------------------- tables
def _cell(st, fmt):
    m, lo, hi = st
    return "--" if np.isnan(m) else f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def write_table(out_dir: Path, name: str, head: list[str], cols, body) -> list[Path]:
    """``body``: rows of (key cells..., stats dict); ``cols``: (column, header, format)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head + [f"{c}_{s}" for c, *_ in cols for s in ("mean", "lo", "hi")])
        for *keys, st in body:
            w.writerow(list(keys) + [f"{v:.4g}" for c, *_ in cols for v in st[c]])
    tex = [
        "\\begin{tabular}{" + "l" * len(head) + "c" * len(cols) + "}",
        "\\toprule",
        " & ".join(head + [h for _, h, _ in cols]) + " \\\\",
        "\\midrule",
    ]
    for *keys, st in body:
        tex.append(" & ".join([str(k) for k in keys] + [_cell(st[c], f) for c, _, f in cols]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    return [csv_path, tex_path]


MAIN_COLS = [
    ("regret", "regret", "{:.1f}"),
    ("promise", "promise", "{:.1f}"),
    ("sound", "sound", "{:.2f}"),
    ("truth_in_boxes", "truth in boxes", "{:.2f}"),
    ("rows_set_joint", "rows (joint)", "{:.0f}"),
    ("rows_set_factor_only", "rows (factor only)", "{:.0f}"),
    ("valid_structure", "valid structure", "{:.2f}"),
    ("shd", "SHD", "{:.1f}"),
]
CARD_COLS = [
    ("survivors", "candidates per machine", "{:.1f}"),
    ("model_factor_rows", "factor rows", "{:.0f}"),
    ("model_coupling", "joint rows per factor row", "{:.1f}"),
    ("n_hypotheses", "hypotheses", "{:.3g}"),
    ("s5_k_prime", "after S5 merge", "{:.3g}"),
]
LOG_COLS = [
    ("survivors", "candidates per machine", "{:.1f}"),
    ("valid_structure", "valid structure", "{:.2f}"),
    ("regret", "regret", "{:.1f}"),
    ("rows_with_set", "rows with a set", "{:.0f}"),
]
ABL_COLS = [
    ("regret", "regret", "{:.1f}"),
    ("truth_in_boxes", "truth in boxes", "{:.2f}"),
    ("valid_structure", "valid structure", "{:.2f}"),
    ("assumption_refuted", "bound refuted", "{:.2f}"),
    ("empty_family", "empty family", "{:.2f}"),
    ("survivors", "candidates per machine", "{:.1f}"),
]


def tables(rows, out_dir: Path) -> list[Path]:
    c = core(rows)
    sizes = sorted({r["per_logger"] for r in c})
    main = [
        (
            a,
            int(e),
            len(pick(c, arm=a, per_logger=e)),
            {k: ci([r.get(k, np.nan) for r in pick(c, arm=a, per_logger=e)]) for k, *_ in MAIN_COLS},
        )
        for a in ARM_STYLE
        for e in sizes
        if pick(c, arm=a, per_logger=e)
    ]
    cards = [
        (a, int(e), {k: ci([r.get(k, np.nan) for r in pick(c, arm=a, per_logger=e)]) for k, *_ in CARD_COLS})
        for a in SET_ARMS
        for e in sizes
        if pick(c, arm=a, per_logger=e)
    ]
    written = write_table(out_dir, "pipeline_arms", ["arm", "per logger", "runs"], MAIN_COLS, main)
    written += write_table(out_dir, "model_cards", ["arm", "per logger"], CARD_COLS, cards)
    log_rows = [r for r in rows if r["base_suite"] in {x["base_suite"] for x in rows if x["alone"] == 1}]
    body = []
    for lab in sorted({r["suite"] for r in log_rows}):
        for arm in ("icp", "icp_agreement", "oracle"):
            for e in sorted({r["episodes"] for r in log_rows}):
                sel = pick(log_rows, suite=lab, arm=arm, episodes=e)
                if sel:
                    body.append(
                        (lab, arm, int(e), {k: ci([r.get(k, np.nan) for r in sel]) for k, *_ in LOG_COLS})
                    )
    written += write_table(out_dir, "pipeline_loggers", ["data", "arm", "total episodes"], LOG_COLS, body)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/e1/pipeline_arms.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/e1"))
    args = ap.parse_args()
    use_style()
    rows = load(args.results)
    figs = args.out / "figures"
    written = []
    for eps in sorted({r["eps"] for r in rows}):
        written += regret_figure(rows, eps, figs / f"pipeline_regret_eps{round(eps * 100):02d}")
    written += promise_figure(rows, figs / "pipeline_promise")
    written += s5_figure(rows, figs / "pipeline_s5")
    written += loggers_figure(rows, figs / "pipeline_loggers")
    written += gate_figure(rows, figs / "pipeline_gate")
    abl_paths, abl_table = ablation_figure(rows, args.results.parent, figs / "pipeline_ablations")
    written += abl_paths
    written += tables(rows, args.out / "tables")
    if abl_table:
        written += write_table(
            args.out / "tables", "ablations", ["ablation", "arm", "runs"], ABL_COLS, abl_table
        )
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
