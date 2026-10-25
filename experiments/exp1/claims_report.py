"""Claim-by-claim figures and tables of one experiment-1 grid (protocol.py: A, A2, A3, B1, B2, C, the
confounded control), from the grid's summary/runs.csv and its settings' results_transfer.csv and
results_shift.csv. Works on a finished grid or on a snapshot of a running one.

    python experiments/exp1/claims_report.py results/exp1/<grid folder>

Writes <grid>/summary/claims/: claims.md (the tables), fig_*.png/.pdf and fig_*.csv (the numbers each
figure draws). Few arms per panel: colour = where the structure comes from (true graph, BIC, none),
line style = the model (solid: interval model with a promise, dashed: point model)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from scm_to_mdp.tool.sweep import read_results  # noqa: E402

# ---- the arms and their encoding (the colour follows the structure, the same in every figure) ----------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CORE = {  # claim A: structure x model
    "true graph -> RMDP": (BLUE, "-", "o"),
    "BIC -> RMDP": (ORANGE, "-", "s"),
    "flat RMDP": (AQUA, "-", "^"),
    "true graph -> MDP": (BLUE, "--", "o"),
    "BIC -> MDP": (ORANGE, "--", "s"),
    "flat MDP": (AQUA, "--", "^"),
}
CONTROLS = {  # claim A3: is it the right structure or just a small model?
    "true graph -> RMDP": (BLUE, "-", "o"),
    "logger-Markov graph -> RMDP": (GREEN, "-", "D"),
    "random matched graph -> RMDP": (VIOLET, "-", "v"),
    "true graph + 1 -> RMDP": (RED, "-", "P"),
    "flat RMDP": (AQUA, "-", "^"),
}
DISCOVERY = {  # every discovery algorithm -> RMDP (BIC's widened variants: same graph, wider sets)
    "true graph -> RMDP": (BLUE, "-", "o"),
    "BIC -> RMDP": (ORANGE, "-", "s"),
    "BIC -> RMDP (hull)": (ORANGE, ":", "s"),
    "BIC -> RMDP (padded)": (ORANGE, "-.", "s"),
    "PCMCI+ -> RMDP": (YELLOW, "-", "X"),
    "GES -> RMDP": (MAGENTA, "-", "*"),
    "DYNOTEARS -> RMDP": (MUTED, "-", "h"),
}
BASELINES = ["offline CQL", "offline DQN", "flat SPIBB", "SL-SPIBB", "FOCUS", "DiFaRL"]
LOGGERS = ["uniform", "expert ε=0.05", "expert ε=0"]
DELTA = 0.05
YMAX = 1.6  # normalised regret above 1: worse than the random policy


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def load(grid_dir: Path):
    runs = pd.read_csv(grid_dir / "summary" / "runs.csv", dtype={"size": str, "knob": str, "policy": str})
    grid = json.loads((grid_dir / "grid.json").read_text())
    folders = [grid_dir / s["folder"] for s in grid["settings"]]
    transfer = pd.concat([read_results(f, True, "results_transfer.csv") for f in folders], ignore_index=True)
    shift = pd.concat([read_results(f, True, "results_shift.csv") for f in folders], ignore_index=True)
    for frame in (transfer, shift):
        if len(frame):
            frame["safe"] = frame["safe"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    loggers = [p for p in grid["policies"] if p in set(runs["policy"])]
    return grid, runs, transfer, shift, loggers


def lines(ax, frame, arms, y, n_grid, agg="median"):
    """One line per arm: agg of y over everything but the number of rollouts."""
    for arm, (colour, ls, marker) in arms.items():
        part = frame[frame["method"] == arm]
        if not len(part):
            continue
        curve = part.groupby("n_rollouts")[y].agg(agg).reindex(n_grid)
        ax.plot(
            n_grid,
            curve.values,
            color=colour,
            linestyle=ls,
            marker=marker,
            markersize=5,
            linewidth=2,
            label=arm,
        )
    ax.set_xscale("log")
    ax.set_xticks(n_grid)
    ax.set_xticklabels([str(n) for n in n_grid], rotation=0)
    ax.minorticks_off()


def figure_grid(rows, cols, w=3.0, h=2.4):
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols + 1.2, h * rows + 1.0), squeeze=False, sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        style_axes(ax)
    return fig, axes


def finish(fig, axes, out: Path, stem: str, title: str, data: pd.DataFrame):
    handles, labels = {}, []
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in handles:
                handles[lab] = h
                labels.append(lab)
    fig.legend(
        [handles[lab] for lab in labels],
        labels,
        loc="lower center",
        ncol=min(len(labels), 3),
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    fig.suptitle(title, color=INK, fontsize=11)
    rows = (len(labels) + 2) // 3
    fig.tight_layout(rect=(0, 0.035 * rows + 0.02, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    data.to_csv(out / f"{stem}.csv", index=False)


def pivot(frame, value, arms, ns, by="policy", agg="median"):
    part = frame[frame["method"].isin(arms) & frame["n_rollouts"].isin(ns)]
    table = part.groupby(["method", by, "n_rollouts"])[value].agg(agg).unstack([by, "n_rollouts"])
    order = {p: i for i, p in enumerate(LOGGERS)}
    table = table[sorted(table.columns, key=lambda c: (order.get(c[0], len(order)), c[1]))]
    return table.reindex([a for a in arms if a in table.index])


def order(table: pd.DataFrame, rows: list[str]) -> pd.DataFrame:
    """Rows in the given order; columns (a logger, or (metric, logger)) in the protocol's logger order."""
    rank = {p: i for i, p in enumerate(LOGGERS)}

    def key(c):
        return (c[0], rank.get(c[1], len(rank))) if isinstance(c, tuple) else (rank.get(c, len(rank)),)

    return table[sorted(table.columns, key=key)].reindex([r for r in rows if r in table.index])


def md(table: pd.DataFrame, fmt="{:.2f}") -> str:
    shown = table.map(lambda v: "–" if pd.isna(v) else fmt.format(v))
    if isinstance(shown.columns, pd.MultiIndex):
        shown.columns = [" ".join(str(c) for c in col) for col in shown.columns]
    head = [str(shown.index.name or "")] + [str(c) for c in shown.columns]
    rows = [[str(i)] + list(r) for i, r in zip(shown.index, shown.values, strict=True)]
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        + ["| " + " | ".join(r) + " |" for r in rows]
    )


def main(grid_dir: Path):
    grid, runs, transfer, shift, loggers = load(grid_dir)
    out = grid_dir / "summary" / "claims"
    out.mkdir(parents=True, exist_ok=True)
    n_grid, sizes, name = grid["n_rollouts"], [str(s) for s in grid["sizes"]], grid["name"]
    sizes = [s for s in sizes if s in set(runs["size"])]
    plain = [p for p in LOGGERS if p in loggers]
    confounded = [p for p in loggers if p.startswith("confounded")]
    hidden_knobs = sorted(set(shift["knob"])) if len(shift) else []
    text = [
        f"# {name}: the claims, figure by figure",
        "",
        f"Sizes {', '.join(sizes)}; knobs {', '.join(grid['knobs'])}; {runs['seed'].nunique()} seeds; "
        f"{len(runs):,} runs. Medians over seeds, knobs and (where pooled) sizes. Normalised regret = "
        "(V* - V^pi) / (V* - V_random) on the observed state (0 = optimal, 1 = random).",
        "",
    ]

    # ---- A: sample efficiency (true value), loggers x sizes ---------------------------------------------
    fig, axes = figure_grid(len(plain), len(sizes))
    for i, pol in enumerate(plain):
        for j, size in enumerate(sizes):
            ax = axes[i, j]
            lines(ax, runs[(runs["policy"] == pol) & (runs["size"] == size)], CORE, "norm_regret", n_grid)
            ax.set_ylim(-0.03, YMAX)
            if i == 0:
                ax.set_title(f"{grid['size_name']} = {size}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{pol}\nnormalised regret", fontsize=8, color=INK)
            if i == len(plain) - 1:
                ax.set_xlabel("rollouts", fontsize=8, color=INK)
    data = runs[runs["method"].isin(CORE) & runs["policy"].isin(plain)]
    data = data.groupby(["policy", "size", "method", "n_rollouts"])["norm_regret"].median().reset_index()
    finish(
        fig,
        axes,
        out,
        "fig_A_regret",
        f"{name} - A: true value of the policy (median over seeds and "
        "knobs); solid = interval model, dashed = point model",
        data,
    )
    keep = CORE | {b: None for b in BASELINES}
    text += [
        "## A - sample efficiency: normalised regret (median), pooled sizes",
        "",
        "![](fig_A_regret.png)",
        "",
        md(pivot(runs, "norm_regret", list(keep), [25, 100, 400])),
        "",
    ]

    # ---- validity: share of runs where the promise held -------------------------------------------------
    pols = plain + confounded
    fig, axes = figure_grid(1, len(pols))
    for j, pol in enumerate(pols):
        ax = axes[0, j]
        lines(ax, runs[runs["policy"] == pol], CORE, "safe", n_grid, agg="mean")
        ax.axhline(1 - DELTA, color=MUTED, linewidth=1, linestyle=":")
        ax.text(n_grid[0], 1 - DELTA - 0.06, "1 - δ", fontsize=7, color=MUTED)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(pol, fontsize=9, color=INK)
        ax.set_xlabel("rollouts", fontsize=8, color=INK)
    axes[0, 0].set_ylabel(
        "share of runs: promise <= true value\n(every observed state)", fontsize=8, color=INK
    )
    data = (
        runs[runs["method"].isin(CORE)]
        .groupby(["policy", "method", "n_rollouts"])["safe"]
        .mean()
        .reset_index()
    )
    finish(
        fig,
        axes,
        out,
        "fig_validity",
        f"{name} - does the promise hold? (point models: their own value estimate as the 'promise')",
        data,
    )
    text += [
        "## Validity: share of runs where the promise held (mean), pooled sizes",
        "",
        "![](fig_validity.png)",
        "",
        md(pivot(runs, "safe", list(keep), [25, 100, 400], agg="mean")),
        "",
    ]

    # ---- A3: the controls ----------------------------------------------------------------------------
    fig, axes = figure_grid(2, len(plain))
    for j, pol in enumerate(plain):
        part = runs[runs["policy"] == pol]
        lines(axes[0, j], part, CONTROLS, "norm_regret", n_grid)
        lines(axes[1, j], part, CONTROLS, "safe", n_grid, agg="mean")
        axes[1, j].axhline(1 - DELTA, color=MUTED, linewidth=1, linestyle=":")
        axes[0, j].set_title(pol, fontsize=9, color=INK)
        axes[1, j].set_xlabel("rollouts", fontsize=8, color=INK)
        axes[0, j].set_ylim(-0.03, YMAX)
        axes[1, j].set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("normalised regret (median)", fontsize=8, color=INK)
    axes[1, 0].set_ylabel("promise held (share)", fontsize=8, color=INK)
    data = runs[runs["method"].isin(CONTROLS) & runs["policy"].isin(plain)]
    data = (
        data.groupby(["policy", "method", "n_rollouts"])[["norm_regret", "safe"]]
        .agg({"norm_regret": "median", "safe": "mean"})
        .reset_index()
    )
    finish(
        fig, axes, out, "fig_A3_controls", f"{name} - A3: the right structure or just a small model?", data
    )
    struct = runs[runs["method"].isin(CONTROLS) & runs["policy"].isin(plain)]
    text += [
        "## A3 - controls: regret (median) and promise held (mean), pooled sizes and n",
        "",
        "![](fig_A3_controls.png)",
        "",
        md(struct.groupby(["method", "policy"])["norm_regret"].median().unstack().reindex(list(CONTROLS))),
        "",
        md(struct.groupby(["method", "policy"])["safe"].mean().unstack().reindex(list(CONTROLS))),
        "",
        "Structure recovered (mean SHD, missing true edges, action recall):",
        "",
        md(
            struct.groupby(["method", "policy"])[["shd", "missing", "action_recall"]]
            .mean()
            .unstack()
            .reindex(list(CONTROLS))
            .dropna(how="all")
        ),
        "",
    ]

    # ---- discovery: every algorithm (and BIC's widened variants) against the true graph -----------------
    arms = {a: s for a, s in DISCOVERY.items() if a in set(runs["method"])}
    fig, axes = figure_grid(2, len(pols))
    for j, pol in enumerate(pols):
        part = runs[runs["policy"] == pol]
        lines(axes[0, j], part, arms, "norm_regret", n_grid)
        lines(axes[1, j], part, arms, "safe", n_grid, agg="mean")
        axes[1, j].axhline(1 - DELTA, color=MUTED, linewidth=1, linestyle=":")
        axes[0, j].set_title(pol, fontsize=9, color=INK)
        axes[1, j].set_xlabel("rollouts", fontsize=8, color=INK)
        axes[0, j].set_ylim(-0.03, YMAX)
        axes[1, j].set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("normalised regret (median)", fontsize=8, color=INK)
    axes[1, 0].set_ylabel("promise held (share)", fontsize=8, color=INK)
    found = runs[runs["method"].isin(arms) & runs["policy"].isin(pols)]
    data = (
        found.groupby(["policy", "method", "n_rollouts"])[["norm_regret", "safe"]]
        .agg({"norm_regret": "median", "safe": "mean"})
        .reset_index()
    )
    finish(fig, axes, out, "fig_discovery", f"{name} - which discovery algorithm? (-> RMDP)", data)
    point_arms = [a.replace("-> RMDP", "-> MDP") for a in arms if a.endswith("-> RMDP")]
    every = list(arms) + [a for a in point_arms if a in set(runs["method"])]
    found = runs[runs["method"].isin(every) & runs["policy"].isin(pols)]
    by = found.groupby(["method", "policy"])
    text += [
        "## Discovery: every algorithm -> RMDP (and -> MDP), pooled sizes and n",
        "",
        "![](fig_discovery.png)",
        "",
        "Regret (median):",
        "",
        md(order(by["norm_regret"].median().unstack(), every)),
        "",
        "Promise held (mean):",
        "",
        md(order(by["safe"].mean().unstack(), every)),
        "",
        "Structure recovered: missing true edges (mean), extra edges (mean), SHD (mean):",
        "",
        md(
            order(by[["missing", "extra", "shd"]].mean().unstack(), [a for a in arms if a in every]), "{:.1f}"
        ),
        "",
    ]

    # ---- B1: reward transfer ----------------------------------------------------------------------------
    if len(transfer):
        tasks = sorted(set(transfer["task"]))
        fig, axes = figure_grid(1, len(plain))
        for j, pol in enumerate(plain):
            lines(axes[0, j], transfer[transfer["policy"] == pol], CORE, "norm_regret", n_grid)
            axes[0, j].set_ylim(-0.03, YMAX)
            axes[0, j].set_title(pol, fontsize=9, color=INK)
            axes[0, j].set_xlabel("rollouts", fontsize=8, color=INK)
        axes[0, 0].set_ylabel(
            "normalised regret on the new rewards\n(median over tasks)", fontsize=8, color=INK
        )
        data = transfer[transfer["method"].isin(CORE)]
        data = data.groupby(["policy", "task", "method", "n_rollouts"])["norm_regret"].median().reset_index()
        finish(
            fig,
            axes,
            out,
            "fig_B1_transfer",
            f"{name} - B1: the same models re-planned under {len(tasks)} new rewards ({', '.join(tasks)})",
            data,
        )
        gap = transfer[transfer["method"].isin(["true graph -> RMDP", "flat RMDP"])]
        gap = gap.pivot_table(
            index=["policy", "task", "n_rollouts", "knob", "size", "seed"],
            columns="method",
            values="norm_regret",
        ).dropna()
        gain = (gap["flat RMDP"] - gap["true graph -> RMDP"]).rename("gain").reset_index()
        cov = transfer[transfer["method"] == "true graph -> RMDP"].assign(
            only_factored=lambda f: f["cov_joint_1"] - f["cov_factor_1"]
        )
        cov = cov.groupby(["policy", "task"])[["only_factored", "cov_joint_1"]].mean()
        b1 = gain.groupby(["policy", "task"])["gain"].mean().to_frame().join(cov)
        b1.index = [f"{pol}, {task}" for pol, task in b1.index]
        text += [
            "## B1 - reward transfer: normalised regret on the new rewards (median), pooled sizes",
            "",
            "![](fig_B1_transfer.png)",
            "",
            md(pivot(transfer, "norm_regret", list(keep), [25, 100, 400])),
            "",
            "Gain of true graph -> RMDP over flat RMDP (regret difference, mean) against the task's "
            "occupancy that only a factored model can estimate (cov_joint_1 - cov_factor_1, mean):",
            "",
            md(b1, "{:.3f}"),
            "",
        ]

    # ---- B2: dynamics shift (robustness only) -----------------------------------------------------------
    if len(shift):
        arms = list(CORE) + BASELINES
        part = shift[shift["method"].isin(arms)]
        base = runs[runs["knob"].isin(hidden_knobs) & runs["method"].isin(arms)].assign(task="data (0.1)")
        both = pd.concat([base, part], ignore_index=True)
        table = (
            both.groupby(["method", "task"])[["norm_regret", "safe"]]
            .agg({"norm_regret": "median", "safe": "mean"})
            .unstack("task")
            .reindex(arms)
        )
        text += [
            f"## B2 - dynamics shift ({', '.join(hidden_knobs)}): the hidden variable more frequent at "
            "deployment; robustness, not causal transfer",
            "",
            "Regret (median) and promise held (mean), pooled sizes, loggers and n:",
            "",
            md(table),
            "",
        ]

    # ---- the confounded logger (control) ----------------------------------------------------------------
    if confounded and hidden_knobs:
        part = runs[runs["knob"].isin(hidden_knobs) & runs["method"].isin(list(CORE) + BASELINES)]
        part = part[part["policy"].isin(["expert ε=0.05"] + confounded)]
        table = (
            part.groupby(["method", "policy"])[["norm_regret", "safe"]]
            .agg({"norm_regret": "median", "safe": "mean"})
            .unstack("policy")
            .reindex(list(CORE) + BASELINES)
        )
        text += [
            f"## Control - the confounded logger, knobs with a hidden variable ({', '.join(hidden_knobs)}),"
            " against the same expert unconfounded",
            "",
            md(table),
            "",
        ]

    # ---- C: understanding (the learned model against the true kernel) ----------------------------------
    point = ["true graph -> MDP", "BIC -> MDP", "flat MDP"]
    interval = ["true graph -> RMDP", "BIC -> RMDP", "flat RMDP"]
    fig, axes = figure_grid(1, 3)
    unconf = runs[runs["policy"].isin(plain)]
    for ax, (col, arms, label) in zip(
        axes[0],
        [
            ("tv_uniform", point, "TV to the true kernel, uniform rows"),
            ("tv_data", point, "TV to the true kernel, logged rows"),
            ("contain_uniform", interval, "rows whose interval contains the truth\n(uniform weights)"),
        ],
        strict=True,
    ):
        lines(ax, unconf, {a: CORE[a] for a in arms}, col, n_grid)
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("rollouts", fontsize=8, color=INK)
        ax.set_ylim(-0.02, 1.02)
    data = unconf[unconf["method"].isin(point + interval)]
    data = data.groupby(["method", "n_rollouts"])[
        ["tv_uniform", "tv_data", "contain_uniform", "width_unseen"]
    ]
    finish(
        fig,
        axes,
        out,
        "fig_C_model",
        f"{name} - C: how well the model knows the environment (unconfounded loggers)",
        data.median().reset_index(),
    )
    cols = ["tv_uniform", "tv_data", "contain_uniform", "contain_data", "width_seen", "width_unseen"]
    c_table = unconf.groupby("method")[cols].median().reindex(point + interval + list(CONTROLS)[1:4])
    text += [
        "## C - understanding: the model against the true kernel (median, unconfounded loggers)",
        "",
        "![](fig_C_model.png)",
        "",
        md(c_table.drop_duplicates(), "{:.3f}"),
        "",
    ]

    (out / "claims.md").write_text("\n".join(text))
    print(f"claims: {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
