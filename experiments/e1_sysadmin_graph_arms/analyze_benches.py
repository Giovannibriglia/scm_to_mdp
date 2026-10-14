"""E1 per-step bench figures and tables, from ``results/e1/bench_<step>.csv`` (no simulation here).

One figure and one table per step. Read every figure the same way: the **safety** panel first (does
the variant keep the guarantee it claims?), then **quality** among the safe variants (lower regret,
tighter sets, higher promise). Variants without a guarantee are drawn dashed.

- ``bench_s1``: the S1 split (discovery fraction 0.3 / 0.5 / 0.7): truth inside every box and regret,
  for ICP + agreement and for BIC;
- ``bench_s4``: the S4 set builder: coverage of the true factor probabilities (all rows at once;
  dotted: the 0.9 level), mean interval width on seen rows, regret;
- ``bench_s78``: S7/S8, product box (the pipeline) vs Fréchet box vs exact factored inner problem:
  promise, how much the box loses against the exact factored promise, regret;
- ``bench_s6``: the S6 criterion (worst case, Hurwicz 0.5, optimistic, nominal maximum likelihood):
  share of runs whose promise held, regret, promise.

Every number is a mean over runs (both suites and edges pooled) with a 95% percentile-bootstrap
interval.

    uv run python experiments/e1_sysadmin_graph_arms/analyze_benches.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scm_to_mdp.plotting import DOUBLE_COLUMN, OKABE_ITO, save, use_style
from scm_to_mdp.stats import bootstrap_mean_ci

TEXT = ("suite", "edge", "variant", "arm", "bench")
STYLE = {
    "s1": {"0.3": OKABE_ITO["sky"], "0.5": OKABE_ITO["blue"], "0.7": OKABE_ITO["black"]},
    "s4": {
        "l1": OKABE_ITO["black"],
        "kl": OKABE_ITO["blue"],
        "hoeffding": OKABE_ITO["sky"],
        "clopper_pearson": OKABE_ITO["green"],
        "dirichlet": OKABE_ITO["purple"],
    },
    "s78": {
        "box (pipeline)": OKABE_ITO["black"],
        "Frechet box": OKABE_ITO["grey"],
        "exact factored": OKABE_ITO["vermillion"],
    },
    "s6": {
        "worst case": OKABE_ITO["black"],
        "Hurwicz 0.5": OKABE_ITO["blue"],
        "optimistic": OKABE_ITO["vermillion"],
        "nominal (MLE)": OKABE_ITO["orange"],
    },
}
NO_GUARANTEE = {"dirichlet", "Hurwicz 0.5", "optimistic", "nominal (MLE)"}
LABEL = {
    "l1": "L1 ball (Weissman)",
    "kl": "KL intervals",
    "hoeffding": "Hoeffding intervals",
    "clopper_pearson": "Clopper–Pearson intervals",
    "dirichlet": "Dirichlet posterior box",
    "0.3": "discovery 30%",
    "0.5": "discovery 50%",
    "0.7": "discovery 70%",
}


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in TEXT:
                r[k] = float(v) if v not in ("", "nan", "None", None) else np.nan
    return rows


def ci(values):
    v = [x for x in values if not np.isnan(x)]
    return bootstrap_mean_ci(v) if v else (np.nan, np.nan, np.nan)


def axis(ax, sizes):
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{int(e) // 1000}k" if e >= 1000 else str(int(e)) for e in sizes])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("episodes per logger")


def lines(ax, rows, bench, col, sizes, **eq):
    for var, colour in STYLE[bench].items():
        sel = [r for r in rows if r["variant"] == var and all(r.get(k) == v for k, v in eq.items())]
        if not sel:
            continue
        st = [ci([r[col] for r in sel if r["per"] == e]) for e in sizes]
        m, lo, hi = (np.array(x) for x in zip(*st, strict=True))
        ls = "--" if var in NO_GUARANTEE else "-"
        ax.plot(sizes, m, ls, color=colour, marker="o", ms=2.5, label=LABEL.get(var, var))
        ax.fill_between(sizes, lo, hi, color=colour, alpha=0.12, lw=0)
    axis(ax, sizes)


def figure(bench: str, rows, out: Path) -> list[Path]:
    sizes = sorted({r["per"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN, 2.5), gridspec_kw={"wspace": 0.42})
    if bench == "s1":
        plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COLUMN, 2.5), gridspec_kw={"wspace": 0.5})
        lines(axes[0], rows, bench, "truth_in_boxes", sizes, arm="icp_agreement")
        lines(axes[1], rows, bench, "regret", sizes, arm="icp_agreement")
        lines(axes[2], rows, bench, "truth_in_boxes", sizes, arm="bic")
        lines(axes[3], rows, bench, "regret", sizes, arm="bic")
        axes[0].set(ylabel="share of runs", title="(a) ICP+agr.: truth in boxes", ylim=(-0.02, 1.02))
        axes[1].set(ylabel="regret", title="(b) ICP+agr.: regret")
        axes[2].set(ylabel="share of runs", title="(c) BIC: truth in boxes", ylim=(-0.02, 1.02))
        axes[3].set(ylabel="regret", title="(d) BIC: regret")
    elif bench == "s4":
        lines(axes[0], rows, bench, "factor_coverage", sizes)
        axes[0].axhline(0.9, color=OKABE_ITO["grey"], lw=0.6, ls=":")
        lines(axes[1], rows, bench, "width", sizes)
        lines(axes[2], rows, bench, "regret", sizes)
        axes[0].set(ylabel="share of runs", title="(a) true factors covered", ylim=(-0.02, 1.02))
        axes[1].set(ylabel="mean interval width", title="(b) width on seen rows")
        axes[2].set(ylabel="regret", title="(c) regret")
    elif bench == "s78":
        lines(axes[0], rows, bench, "promise", sizes)
        exact = {
            (r["suite"], r["per"], r["edge"], r["seed"]): r["promise"]
            for r in rows
            if r["variant"] == "exact factored"
        }
        gap = [
            dict(r, gap=exact[(r["suite"], r["per"], r["edge"], r["seed"])] - r["promise"])
            for r in rows
            if r["variant"] != "exact factored"
        ]
        lines(axes[1], gap, bench, "gap", sizes)
        lines(axes[2], rows, bench, "regret", sizes)
        axes[0].set(ylabel="promise", title="(a) promise (all sound)")
        axes[1].set(ylabel="exact promise $-$ promise", title="(b) what the relaxation loses")
        axes[2].set(ylabel="regret", title="(c) regret")
    elif bench == "s6":
        lines(axes[0], rows, bench, "sound", sizes)
        lines(axes[1], rows, bench, "regret", sizes)
        lines(axes[2], rows, bench, "promise", sizes)
        axes[0].set(ylabel="share of runs", title="(a) promise $\\leq$ true value", ylim=(-0.02, 1.02))
        axes[1].set(ylabel="regret", title="(b) regret")
        axes[2].set(ylabel="promise", title="(c) promise")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(5, len(labels)), bbox_to_anchor=(0.5, -0.02), fontsize=6
    )
    fig.subplots_adjust(bottom=0.32)
    paths = save(fig, out)
    plt.close(fig)
    return paths


COLS = {
    "s1": [
        ("regret", "regret", "{:.1f}"),
        ("truth_in_boxes", "truth in boxes", "{:.2f}"),
        ("valid_structure", "valid structure", "{:.2f}"),
        ("gate_rejected", "gate rejects", "{:.2f}"),
    ],
    "s4": [
        ("factor_coverage", "factors covered", "{:.2f}"),
        ("truth_in_boxes", "truth in boxes", "{:.2f}"),
        ("width", "width", "{:.3f}"),
        ("regret", "regret", "{:.1f}"),
        ("sound", "sound", "{:.2f}"),
    ],
    "s78": [
        ("promise", "promise", "{:.1f}"),
        ("regret", "regret", "{:.1f}"),
        ("sound", "sound", "{:.2f}"),
        ("seconds", "seconds", "{:.3f}"),
    ],
    "s6": [("sound", "sound", "{:.2f}"), ("regret", "regret", "{:.1f}"), ("promise", "promise", "{:.1f}")],
}


def table(bench: str, rows, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = COLS[bench]
    keys = ["arm", "variant", "per"] if bench == "s1" else ["variant", "per"]
    groups = sorted({tuple(r.get(k) for k in keys) for r in rows}, key=lambda g: tuple(str(x) for x in g))
    body = []
    for g in groups:
        sel = [r for r in rows if tuple(r.get(k) for k in keys) == g]
        body.append((g, len(sel), {c: ci([r.get(c, np.nan) for r in sel]) for c, *_ in cols}))
    csv_path = out_dir / f"bench_{bench}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*keys, "runs", *[f"{c}_{s}" for c, *_ in cols for s in ("mean", "lo", "hi")]])
        for g, n, st in body:
            w.writerow([*g, n, *[f"{v:.4g}" for c, *_ in cols for v in st[c]]])
    tex = [
        "\\begin{tabular}{" + "l" * (len(keys) + 1) + "c" * len(cols) + "}",
        "\\toprule",
        " & ".join([*keys, "runs", *[h for _, h, _ in cols]]) + " \\\\",
        "\\midrule",
    ]
    for g, n, st in body:
        cells = []
        for c, _, fmt in cols:
            m, lo, hi = st[c]
            cells.append("--" if np.isnan(m) else f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]")
        g = [LABEL.get(str(x), str(x)) if not isinstance(x, float) else str(int(x)) for x in g]
        tex.append(" & ".join([*g, str(n), *cells]) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    tex_path = out_dir / f"bench_{bench}.tex"
    tex_path.write_text("\n".join(tex) + "\n")
    return [csv_path, tex_path]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results/e1"))
    args = ap.parse_args()
    use_style()
    for bench in ("s1", "s4", "s78", "s6"):
        path = args.results / f"bench_{bench}.csv"
        if not path.exists():
            continue
        rows = load(path)
        for p in figure(bench, rows, args.results / "figures" / f"bench_{bench}") + table(
            bench, rows, args.results / "tables"
        ):
            print("wrote", p)


if __name__ == "__main__":
    main()
