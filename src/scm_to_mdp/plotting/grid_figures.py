"""Figures of a grid (tool/report_grid.py): every setting (knob x behaviour policy) on one page.

    sample_efficiency_grid   rows = policies, columns = knobs; x = size, y = rollouts needed
    rollouts_vs_model_size   x = parent rows of the true factored model, y = rollouts needed
    promise_heatmap          share of finished runs whose promise holds, method x setting
    discovery_errors         missing / extra edges of the learned graph against the rollouts
    rollouts_vs_coverage     x = C* of the logger, y = rollouts needed, one panel per method
    confounding              x = confounding bias of the logger, y = promise - true value

Each function takes the tables it draws (built in report_grid) and returns nothing; report_grid
saves those tables next to the figure (figure_data/<name>.csv).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

from scm_to_mdp.plotting.style import FULL_WIDTH, save, setup

POLICY_COLOURS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]


def _legend(fig, handles, ncol=4, y=-0.01):
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(ncol, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, y),
    )


def _method_handles(methods, styles):
    return [
        Line2D(
            [],
            [],
            ls=styles[m]["ls"],
            marker=styles[m]["marker"],
            color=styles[m]["color"],
            label=styles[m]["label"],
        )
        for m in methods
    ]


def _log_ticks(axis, values, most: int = 7):
    """Log axis with ticks at (some of) the given values, labelled plainly (e.g. 12, 40, 160)."""
    values = sorted({float(v) for v in values if np.isfinite(v) and v > 0})
    if len(values) > most:
        values = [values[int(round(i))] for i in np.linspace(0, len(values) - 1, most)]
    axis.set_major_locator(plt.FixedLocator(values))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _rollout_axis(ax, grid, top):
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticks(list(grid) + [top], [str(g) for g in grid] + ["not\nreached"])
    ax.set_ylim(grid[0] / 1.4, top * 1.4)


def sample_efficiency_grid(
    summary, sizes, size_name, knobs, knob_name, policies, methods, styles, grid, target, title, stem
):
    """summary: rows (knob, policy, size, method, median, q25, q75)."""
    setup()
    top = grid[-1] * 1.8
    fig, axes = plt.subplots(
        len(policies),
        len(knobs),
        figsize=(FULL_WIDTH, 1.9 * len(policies) + 0.9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    position = {s: i for i, s in enumerate(sizes)}
    for r, policy in enumerate(policies):
        for c, knob in enumerate(knobs):
            ax = axes[r, c]
            cell = summary[(summary["policy"] == policy) & (summary["knob"] == knob)]
            if cell.empty:
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center", color="grey")
            for k, m in enumerate(methods):
                sub = cell[cell["method"] == m].copy()
                sub = sub[sub["size"].isin(position)].sort_values("size", key=lambda s: s.map(position))
                if sub.empty:
                    continue
                st = styles[m]
                x = sub["size"].map(position).to_numpy(float) + (k - (len(methods) - 1) / 2) * 0.04
                med = sub["median"].to_numpy(float)
                lo = np.where(np.isfinite(sub["q25"]), sub["q25"], top)
                hi = np.where(np.isfinite(sub["q75"]), sub["q75"], top)
                y = np.where(np.isfinite(med), med, top)
                ax.plot(x, y, ls=st["ls"], color=st["color"], lw=1.1)
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.08, lw=0)
                ok = np.isfinite(med)
                ax.plot(x[ok], y[ok], ls="none", marker=st["marker"], color=st["color"], markersize=3.5)
                ax.plot(
                    x[~ok],
                    y[~ok],
                    ls="none",
                    marker=st["marker"],
                    color=st["color"],
                    markersize=3.5,
                    markerfacecolor="white",
                )
            _rollout_axis(ax, grid, top)
            ax.set_xticks(range(len(sizes)), sizes)
            if r == 0:
                ax.set_title(f"{knob_name} = {knob}")
            if c == 0:
                ax.set_ylabel(f"{policy}\nrollouts", fontsize=7)
            if r == len(policies) - 1:
                ax.set_xlabel(size_name)
    fig.suptitle(f"{title}: rollouts for a {target:.0%}-optimal policy (median over seeds, quartile band)")
    band = plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.25, lw=0, label="first to third quartile")
    hollow = Line2D(
        [], [], ls="none", marker="o", color="k", markerfacecolor="white", label="median not reached"
    )
    _legend(fig, _method_handles(methods, styles) + [hollow, band])
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    save(fig, stem)


def rollouts_vs_model_size(points, policies, methods, styles, grid, title, stem):
    """points: rows (policy, knob, size, method, true_rows, median). One panel per policy."""
    setup()
    top = grid[-1] * 1.8
    fig, axes = plt.subplots(1, len(policies), figsize=(FULL_WIDTH, 2.9), sharey=True, squeeze=False)
    for ax, policy in zip(axes[0], policies, strict=True):
        cell = points[points["policy"] == policy]
        for m in methods:
            sub = cell[cell["method"] == m]
            if sub.empty:
                continue
            st = styles[m]
            y = np.where(np.isfinite(sub["median"]), sub["median"], top)
            ok = np.isfinite(sub["median"].to_numpy(float))
            ax.plot(
                sub["true_rows"][ok], y[ok], ls="none", marker=st["marker"], color=st["color"], markersize=4
            )
            ax.plot(
                sub["true_rows"][~ok],
                y[~ok],
                ls="none",
                marker=st["marker"],
                color=st["color"],
                markersize=4,
                markerfacecolor="white",
            )
        ax.set_xscale("log")
        _log_ticks(ax.xaxis, points["true_rows"])
        _rollout_axis(ax, grid, top)
        ax.set_title(policy)
    fig.supxlabel("parent rows of the true factored model (log scale)", fontsize=8, y=0.17)
    axes[0, 0].set_ylabel("rollouts for 95% (median)")
    fig.suptitle(f"{title}: data needed against the size of the true causal model (all knobs and sizes)")
    _legend(fig, _method_handles(methods, styles))
    fig.tight_layout(rect=(0, 0.18, 1, 0.95))
    save(fig, stem)


def promise_heatmap(held: pd.DataFrame, title, stem):
    """held: index = methods, columns = settings (labels), values = share of runs whose promise holds."""
    setup()
    # the setting labels are long (logger + knob): vertical, one line each, and room for them below
    labels = [str(c).replace("\n", ", ") for c in held.columns]
    below = 0.065 * max((len(t) for t in labels), default=10)  # inches of rotated label text
    width = max(FULL_WIDTH, 0.55 * held.shape[1] + 3.2)
    fig, ax = plt.subplots(figsize=(width, 0.42 * len(held) + 1.4 + below))
    values = held.to_numpy(float)
    image = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(held.shape[1]), labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(held.shape[0]), held.index)
    ax.grid(False)
    fig.colorbar(
        image, ax=ax, fraction=0.025, label="share of finished runs: promise <= true value in every state"
    )
    ax.set_title(f"{title}: where the promise holds (over every size and number of rollouts)")
    fig.tight_layout()
    save(fig, stem)


def discovery_errors(curves, knobs, knob_name, policies, methods, styles, column, ylabel, title, stem):
    """curves: rows (knob, policy, n_rollouts, method, <column>) averaged over sizes and seeds."""
    setup()
    fig, axes = plt.subplots(
        len(policies),
        len(knobs),
        figsize=(FULL_WIDTH, 1.8 * len(policies) + 0.9),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    top = (
        float(np.nanmax(curves[column])) * 1.08 if len(curves) and np.isfinite(curves[column]).any() else 1.0
    )
    for r, policy in enumerate(policies):
        for c, knob in enumerate(knobs):
            ax = axes[r, c]
            cell = curves[(curves["policy"] == policy) & (curves["knob"] == knob)]
            for m in methods:
                sub = cell[cell["method"] == m].sort_values("n_rollouts")
                if not sub.empty:
                    st = styles[m]
                    ax.plot(
                        sub["n_rollouts"],
                        sub[column],
                        ls=st["ls"],
                        marker=st["marker"],
                        color=st["color"],
                        markersize=3,
                    )
            ax.set_xscale("log")
            _log_ticks(ax.xaxis, curves["n_rollouts"])
            ax.set_ylim(0, max(top, 1e-3))
            if r == 0:
                ax.set_title(f"{knob_name} = {knob}")
            if c == 0:
                ax.set_ylabel(f"{policy}\n{ylabel}", fontsize=7)
            if r == len(policies) - 1:
                ax.set_xlabel("rollouts")
    fig.suptitle(f"{title}: {ylabel} (mean over sizes and seeds)")
    _legend(fig, _method_handles(methods, styles))
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save(fig, stem)


def rollouts_vs_coverage(points, policies, methods, styles, grid, title, stem):
    """points: rows (policy, method, C_star, n_star), one per (knob, size, seed). One panel per method."""
    setup()
    top = grid[-1] * 1.8
    cols = 4
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(FULL_WIDTH, 1.9 * rows + 0.8), sharex=True, sharey=True, squeeze=False
    )
    finite = points["C_star"][np.isfinite(points["C_star"])]
    right = finite.max() * 3 if len(finite) else 10.0
    for ax, m in zip(axes.flat, methods, strict=False):
        sub = points[points["method"] == m]
        for k, policy in enumerate(policies):
            p = sub[sub["policy"] == policy]
            x = np.where(np.isfinite(p["C_star"]), p["C_star"], right)
            y = np.where(np.isfinite(p["n_star"]), p["n_star"], top)
            ax.plot(x, y, ls="none", marker="o", markersize=2.5, alpha=0.6, color=POLICY_COLOURS[k % 6])
        ax.set_xscale("log")
        _rollout_axis(ax, grid, top)
        ax.set_title(styles[m]["label"], fontsize=7)
    for ax in axes.flat[len(methods) :]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("C* of the logger")
    handles = [
        Line2D([], [], ls="none", marker="o", color=POLICY_COLOURS[k % 6], label=p)
        for k, p in enumerate(policies)
    ]
    fig.suptitle(f"{title}: rollouts needed against coverage of the target policy (C* = inf: right edge)")
    _legend(fig, handles, ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save(fig, stem)


def confounding(points, policies, methods, styles, title, stem):
    """points: rows (policy, method, bias_mean, gap), one per (knob, size, seed) at the largest number
    of rollouts. gap > 0: the promise is above the true value (violated)."""
    setup()
    cols = min(3, len(methods))
    rows = int(np.ceil(len(methods) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(FULL_WIDTH, 2.1 * rows + 0.9), sharex=True, sharey=True, squeeze=False
    )
    for ax in axes.flat[len(methods) :]:
        ax.axis("off")
    for ax, m in zip(axes.flat, methods, strict=False):
        sub = points[points["method"] == m]
        for k, policy in enumerate(policies):
            p = sub[sub["policy"] == policy]
            ax.plot(
                p["bias_mean"],
                p["gap"],
                ls="none",
                marker="o",
                markersize=3,
                alpha=0.7,
                color=POLICY_COLOURS[k % 6],
            )
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(styles[m]["label"], fontsize=7)
    fig.supxlabel(
        "confounding bias of the data (mean total variation over the logged pairs)", fontsize=8, y=0.1
    )
    fig.supylabel("promise − true value, start (> 0: violated)", fontsize=8)
    handles = [
        Line2D([], [], ls="none", marker="o", color=POLICY_COLOURS[k % 6], label=p)
        for k, p in enumerate(policies)
    ]
    fig.suptitle(f"{title}: does confounding break the promise? (largest number of rollouts)")
    _legend(fig, handles, ncol=3)
    fig.tight_layout(rect=(0.02, 0.12, 1, 0.94))
    save(fig, stem)
