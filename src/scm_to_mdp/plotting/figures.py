"""The four result figures of an experiment, drawn from its table of runs (one row per run).

Columns read: "<arm>|<metric>" for the values, "<arm>|<step>_status" / "_s" / "_gb" for the steps
(step = discovery, learning, compile, solve). A point is the mean over seeds, with a band of one
standard error, and is drawn only when every seed finished; a step stopped by a limit is an X at
its measured time / memory, and the arm's line ends there (larger instances are not run).
The "compile" step (building the RMDP from the learned model) is not shown anywhere.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scm_to_mdp.plotting.style import ARM_LABEL, ARM_STYLE, FULL_WIDTH, save, setup

STEP_TITLES = {
    "discovery": "Causal discovery",
    "learning": "Parameter learning",
    "solve": "Solving (deployment)",
}
STOPPED = ("timeout", "memout")


class Axis:
    """The x axis of an experiment: the instance sizes in order, and their tick labels."""

    def __init__(self, res: pd.DataFrame, by: str, order: list, labels: list[str] | None, title: str):
        self.by, self.order, self.title = by, [o for o in order if o in set(res[by])], title
        names = dict(zip(order, labels or [str(o) for o in order], strict=True))
        self.labels = [names[o] for o in self.order]
        self.x = np.arange(len(self.order))
        self.n_seeds = res.groupby(by)["seed"].nunique().reindex(self.order).values

    def mean_sem(self, res: pd.DataFrame, column: str):
        """Mean and standard error over seeds; NaN where some seed has no value."""
        if column not in res:
            return None, None
        g = res.groupby(self.by)[column]
        mean, sem, count = (
            g.agg(f).reindex(self.order).values.astype(float) for f in ("mean", "sem", "count")
        )
        mean[count < self.n_seeds] = np.nan
        return mean, np.nan_to_num(sem)

    def any_true(self, res: pd.DataFrame, mask: pd.Series) -> np.ndarray:
        return mask.groupby(res[self.by]).any().reindex(self.order).fillna(False).values

    def finish(self, ax, ylabel: str):
        ax.set_xticks(self.x, self.labels)
        ax.set_xlabel(self.title)
        ax.set_ylabel(ylabel)


def _dodge(arm: str) -> float:
    """A small sideways shift per method, so that methods with equal values stay visible."""
    names = list(ARM_STYLE)
    return (names.index(arm) - (len(names) - 1) / 2) * 0.025


def _line(ax, axis: Axis, arm: str, mean, sem, hollow=None):
    st = ARM_STYLE[arm]
    x = axis.x + _dodge(arm)
    ax.plot(x, mean, ls=st["ls"], marker=st["marker"], color=st["color"], label=ARM_LABEL[arm])
    ax.fill_between(x, mean - sem, mean + sem, color=st["color"], alpha=0.15, lw=0)
    if hollow is not None and hollow.any():
        ax.plot(
            x[hollow],
            mean[hollow],
            ls="none",
            marker=st["marker"],
            color=st["color"],
            markerfacecolor="white",
            markeredgewidth=1.2,
            markersize=6,
        )


def _legend(fig, arms, extra=()):
    handles = [
        Line2D(
            [],
            [],
            ls=ARM_STYLE[a]["ls"],
            marker=ARM_STYLE[a]["marker"],
            color=ARM_STYLE[a]["color"],
            label=ARM_LABEL[a],
        )
        for a in arms
    ] + list(extra)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def performance(res, axis: Axis, arms, k: int, stem):
    """2 x 2: regret and promise - true value, at the nominal start and over k random starts."""
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.4), sharex=True)
    panels = [
        ("regret", "Regret, nominal start", "regret  V* − V^π"),
        ("gap", "Promise − true value, nominal start", "promise − V^π"),
        ("regret_K", f"Regret, {k} random starts", "regret  V* − V^π"),
        ("gap_K", f"Promise − true value, {k} random starts", "promise − V^π"),
    ]
    shown = []
    for ax, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        for arm in arms:
            mean, sem = axis.mean_sem(res, f"{arm}|{metric}")
            if mean is None or np.isnan(mean).all():
                continue
            unsafe = axis.any_true(res, res.get(f"{arm}|safe", pd.Series(True, index=res.index)) == False)  # noqa: E712
            _line(ax, axis, arm, mean, sem, hollow=unsafe if metric.startswith("gap") else None)
            shown.append(arm)
        if metric.startswith("gap"):
            ax.axhline(0, color="k", lw=0.8)
            low, high = ax.get_ylim()
            ax.axhspan(0, max(high, 0.05 * (high - low)), color="#d62728", alpha=0.06, lw=0)
            ax.set_ylim(low, max(high, 0.05 * (high - low)))
            ax.text(
                0.01, 0.97, "promise violated", transform=ax.transAxes, va="top", fontsize=7, color="#d62728"
            )
        ax.set_title(title)
        axis.finish(ax, ylabel)
    hollow = Line2D(
        [],
        [],
        ls="none",
        marker="o",
        color="k",
        markerfacecolor="white",
        label="promise > true value in some state",
    )
    _legend(fig, [a for a in arms if a in shown], extra=[hollow])
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, stem)


def validity(res, axis: Axis, arms, delta: float, stem):
    """1 x 3: coverage (share of runs whose promise holds in every state, target >= 1 - delta),
    violation rate (share of states where it fails) and worst violation (by how much)."""
    setup()
    data = res.copy()
    for arm in arms:  # True / False (or text from the CSV) -> 1.0 / 0.0
        if f"{arm}|safe" in data:
            data[f"{arm}|coverage"] = data[f"{arm}|safe"].map(
                {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
            )
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.7), sharex=True)
    panels = [
        ("coverage", "Coverage of the promise", "share of runs"),
        ("violation_rate", "Violation rate", "share of states"),
        ("worst_violation", "Worst violation", "max (promise − V^π)⁺"),
    ]
    shown = []
    for ax, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for arm in arms:
            mean, sem = axis.mean_sem(data, f"{arm}|{metric}")
            if mean is None or np.isnan(mean).all():
                continue
            _line(ax, axis, arm, mean, 0 * sem if metric == "coverage" else sem)
            if arm not in shown:
                shown.append(arm)
        if metric == "coverage":
            ax.axhline(1 - delta, color="k", ls="--", lw=0.9)
            ax.text(axis.x[0], 1 - delta, f" 1 − δ = {1 - delta:g}", va="top", fontsize=7)
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        axis.finish(ax, ylabel)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _legend(fig, shown)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, stem)


def structure(res, axis: Axis, arms, stem):
    """1 x 2: SHD and SID of the learned graph against the true one."""
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.8), sharex=True)
    for ax, (metric, title) in zip(
        axes,
        [("shd", "Structural Hamming distance (SHD)"), ("sid", "Structural intervention distance (SID)")],
        strict=True,
    ):
        for arm in arms:
            mean, sem = axis.mean_sem(res, f"{arm}|{metric}")
            if mean is not None:
                _line(ax, axis, arm, mean, sem)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        axis.finish(ax, metric.upper() + " (0 = true graph)")
    _legend(fig, arms)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, stem)


def resources(res, axis: Axis, arms_per_step: dict, unit: str, limits: dict, stem):
    """1 x 3: time (unit "s") or peak memory (unit "gb") of discovery, learning and solving.
    limits: {step: limit or None}, drawn as a dashed line."""
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.7), sharex=True)
    shown = set()
    for ax, (step, arms) in zip(axes, arms_per_step.items(), strict=True):
        for arm in arms:
            column = f"{arm}|{step}_{unit}"
            mean, sem = axis.mean_sem(res, column)
            if mean is None:
                continue
            status = res.get(f"{arm}|{step}_status", pd.Series("ok", index=res.index))
            stopped = status.isin(STOPPED)
            mean[axis.any_true(res, stopped)] = np.nan
            _line(ax, axis, arm, mean, sem)
            shown.add(arm)
            if stopped.any():  # an X at the measured value of every stopped step
                pos = res.loc[stopped, axis.by].map({o: i for i, o in enumerate(axis.order)}) + _dodge(arm)
                ax.plot(
                    pos,
                    res.loc[stopped, column],
                    ls="none",
                    marker="X",
                    markersize=7,
                    color=ARM_STYLE[arm]["color"],
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                )
        if limits.get(step) is not None:
            ax.axhline(limits[step], color="k", ls="--", lw=0.9)
            ax.text(
                axis.x[0],
                limits[step],
                f" limit {limits[step]:g} {'s' if unit == 's' else 'GB'}",
                va="bottom",
                fontsize=7,
            )
        ax.set_yscale("log")
        ax.set_title(STEP_TITLES[step])
        axis.finish(ax, "seconds" if unit == "s" else "peak memory (GB)")
    stop = Line2D(
        [],
        [],
        ls="none",
        marker="X",
        color="grey",
        markeredgecolor="k",
        markersize=7,
        label="stopped by the limit (not run on larger instances)",
    )
    _legend(fig, [a for a in ARM_STYLE if a in shown], extra=[stop])
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, stem)
