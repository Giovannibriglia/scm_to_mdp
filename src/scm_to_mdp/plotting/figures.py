"""Result figures, drawn from results.csv (one row per size, seed, number of rollouts and method).

Figures over the problem size, at one number of rollouts:
    performance   regret and promise - true value (nominal start, K random starts)
    validity      hold rate of the promise, violation rate, worst violation
    structure     SHD and SID of the selected graph
    resources     time or peak memory of discovery, learning and solving
A point is the mean over seeds, with a band of one standard error, drawn only when every seed of
that size finished; a step stopped by a limit is an X at its measured time / memory.

Figure over the problem size and the number of rollouts:
    sample_efficiency   rollouts needed for a policy that is at least 95% optimal
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scm_to_mdp.plotting.style import FULL_WIDTH, save, setup

STEP_TITLES = {"discovery": "Causal discovery", "learning": "Parameter learning", "solve": "Solving"}
STOPPED = ("timeout", "memout")


class Axis:
    """The x axis over problem sizes: the sizes in order, and their tick labels."""

    def __init__(self, res: pd.DataFrame, sizes: list[str], labels: list[str], title: str):
        present = set(res["size"])
        self.order = [s for s in sizes if s in present]
        names = dict(zip(sizes, labels, strict=True))
        self.labels = [names[s] for s in self.order]
        self.x = np.arange(len(self.order))
        self.title = title
        self.n_seeds = res.groupby("size")["seed"].nunique().reindex(self.order).values

    def mean_sem(self, res: pd.DataFrame, method: str, column: str):
        """Mean and standard error over seeds of one method; NaN where some seed has no value."""
        sub = res[res["method"] == method]
        if column not in sub or sub[column].isna().all():
            return None, None
        g = pd.to_numeric(sub[column], errors="coerce").groupby(sub["size"])
        mean, sem, count = (
            g.agg(f).reindex(self.order).values.astype(float) for f in ("mean", "sem", "count")
        )
        mean[count < self.n_seeds] = np.nan
        return mean, np.nan_to_num(sem)

    def any_true(self, res: pd.DataFrame, method: str, mask: pd.Series) -> np.ndarray:
        rows = res["method"] == method
        return (
            mask[rows]
            .groupby(res.loc[rows, "size"])
            .any()
            .reindex(self.order)
            .fillna(False)
            .values.astype(bool)
        )

    def finish(self, ax, ylabel: str):
        ax.set_xticks(self.x, self.labels)
        ax.set_xlabel(self.title)
        ax.set_ylabel(ylabel)


def _dodge(methods: list[str], method: str) -> float:
    """A small sideways shift per method, so that methods with equal values stay visible."""
    return (methods.index(method) - (len(methods) - 1) / 2) * 0.025


def _line(ax, axis: Axis, methods, styles, method, mean, sem, hollow=None):
    st = styles[method]
    x = axis.x + _dodge(methods, method)
    ax.plot(x, mean, ls=st["ls"], marker=st["marker"], color=st["color"], label=st["label"])
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


def _legend(fig, methods, styles, extra=()):
    handles = [
        Line2D(
            [],
            [],
            ls=styles[m]["ls"],
            marker=styles[m]["marker"],
            color=styles[m]["color"],
            label=styles[m]["label"],
        )
        for m in methods
    ] + list(extra)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def _as_float(series: pd.Series) -> pd.Series:
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)


# ==========================================================================================
# Over the problem size, at one number of rollouts
# ==========================================================================================
def performance(res, axis: Axis, methods, styles, k: int, title: str, stem):
    """2 x 2: regret and promise - true value, at the nominal start and over k random starts."""
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.6), sharex=True)
    panels = [
        ("regret", "Regret, nominal start", "regret  V*_obs − V^π"),
        ("gap", "Promise − true value, nominal start", "promise − V^π"),
        ("regret_K", f"Regret, {k} random starts", "regret  V*_obs − V^π"),
        ("gap_K", f"Promise − true value, {k} random starts", "promise − V^π"),
    ]
    shown = []
    unsafe = _as_float(res["safe"]) == 0.0
    for ax, (metric, panel, ylabel) in zip(axes.flat, panels, strict=True):
        for m in methods:
            mean, sem = axis.mean_sem(res, m, metric)
            if mean is None or np.isnan(mean).all():
                continue
            hollow = axis.any_true(res, m, unsafe) if metric.startswith("gap") else None
            _line(ax, axis, methods, styles, m, mean, sem, hollow)
            if m not in shown:
                shown.append(m)
        if metric.startswith("gap"):
            ax.axhline(0, color="k", lw=0.8)
            low, high = ax.get_ylim()
            top = max(high, 0.05 * (high - low))
            ax.axhspan(0, top, color="#d62728", alpha=0.06, lw=0)
            ax.set_ylim(low, top)
            ax.text(
                0.01, 0.97, "promise violated", transform=ax.transAxes, va="top", fontsize=7, color="#d62728"
            )
        ax.set_title(panel)
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
    _legend(fig, shown, styles, extra=[hollow])
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, stem)


def validity(res, axis: Axis, methods, styles, delta: float | None, title: str, stem):
    """1 x 3: hold rate (share of runs whose promise holds in every state; a valid method keeps it
    >= 1 - delta), violation rate (share of states where it fails) and worst violation."""
    setup()
    data = res.assign(coverage=_as_float(res["safe"]))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.9), sharex=True)
    panels = [
        ("coverage", "Hold rate of the promise", "share of runs"),
        ("violation_rate", "Violation rate", "share of states"),
        ("worst_violation", "Worst violation", "max (promise − V^π)⁺"),
    ]
    shown = []
    for ax, (metric, panel, ylabel) in zip(axes, panels, strict=True):
        for m in methods:
            mean, sem = axis.mean_sem(data, m, metric)
            if mean is None or np.isnan(mean).all():
                continue
            _line(ax, axis, methods, styles, m, mean, 0 * sem if metric == "coverage" else sem)
            if m not in shown:
                shown.append(m)
        if metric == "coverage":
            if delta is not None:
                ax.axhline(1 - delta, color="k", ls="--", lw=0.9)
                ax.text(axis.x[0], 1 - delta, f" 1 − δ = {1 - delta:g}", va="top", fontsize=7)
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(panel)
        axis.finish(ax, ylabel)
    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _legend(fig, shown, styles)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, stem)


def structure(res, axis: Axis, methods, styles, title: str, stem):
    """1 x 2: SHD and SID of the selected graph against the true one (methods that learn a graph)."""
    setup()
    methods = [m for m in methods if axis.mean_sem(res, m, "shd")[0] is not None]
    if not methods:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.0), sharex=True)
    for ax, (metric, panel) in zip(
        axes,
        [("shd", "Structural Hamming distance (SHD)"), ("sid", "Structural intervention distance (SID)")],
        strict=True,
    ):
        for m in methods:
            mean, sem = axis.mean_sem(res, m, metric)
            if mean is not None:
                _line(ax, axis, methods, styles, m, mean, sem)
        ax.set_title(panel)
        ax.set_ylim(bottom=0)
        axis.finish(ax, metric.upper() + " (0 = true graph)")
    _legend(fig, methods, styles)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, stem)


def resources(res, axis: Axis, methods, styles, unit: str, limits: dict, title: str, stem):
    """1 x 3: time (unit "s") or peak memory (unit "gb") of discovery, learning and solving.
    limits: {step: limit or None}, drawn as a dashed line. "solve" includes the relaxation step."""
    setup()
    data = res.copy()
    if "relax_s" in data:  # the relaxation (box of the factor sets) is part of solving
        relax = pd.to_numeric(data[f"relax_{unit}"], errors="coerce")
        solve = pd.to_numeric(data[f"solve_{unit}"], errors="coerce")
        data[f"solve_{unit}"] = solve + relax.fillna(0) if unit == "s" else np.fmax(solve, relax)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.9), sharex=True)
    shown = []
    for ax, step in zip(axes, STEP_TITLES, strict=True):
        column = f"{step}_{unit}"
        for m in methods:
            mean, sem = axis.mean_sem(data, m, column)
            if mean is None:
                continue
            stopped = data[f"{step}_status"].isin(STOPPED)
            if step == "solve" and "relax_status" in data:
                stopped |= data["relax_status"].isin(STOPPED)
            mean[axis.any_true(data, m, stopped)] = np.nan
            _line(ax, axis, methods, styles, m, mean, sem)
            if m not in shown:
                shown.append(m)
            hit = stopped & (data["method"] == m)
            if hit.any():  # an X at the measured value of every stopped step
                pos = data.loc[hit, "size"].map({o: i for i, o in enumerate(axis.order)}) + _dodge(methods, m)
                ax.plot(
                    pos,
                    pd.to_numeric(data.loc[hit, column]),
                    ls="none",
                    marker="X",
                    markersize=7,
                    color=styles[m]["color"],
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
        label="stopped by a limit (not run on larger instances)",
    )
    _legend(fig, shown, styles, extra=[stop])
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, stem)


# ==========================================================================================
# Rollouts needed for a 95%-optimal policy
# ==========================================================================================
def target_value(res: pd.DataFrame) -> pd.Series:
    """The value a policy is compared with: V_star_obs (the best over the observed state) when the
    results have it, else V_star (older results; the same when nothing is hidden)."""
    if "V_star_obs" in res and res["V_star_obs"].notna().any():
        return pd.to_numeric(res["V_star_obs"], errors="coerce").fillna(pd.to_numeric(res["V_star"]))
    return pd.to_numeric(res["V_star"], errors="coerce")


def rollouts_needed(res: pd.DataFrame, target: float = 0.95) -> pd.DataFrame:
    """For every (size, seed, method): n* = the smallest number of rollouts n of the grid such that the
    policy is at least `target` optimal (true >= target * V_star_obs at the nominal start) at n and
    at every larger n of the grid; inf if never. A run that stopped counts as not optimal. Sizes
    with V_star_obs <= 0 have no such ratio and are left out."""
    rows = []
    res = res.assign(_target=target_value(res))
    for (size, seed, method), g in res.groupby(["size", "seed", "method"]):
        g = g.sort_values("n_rollouts")
        v_star = g["_target"].iloc[0]
        if not np.isfinite(v_star) or v_star <= 0:
            continue
        good = (g["status"] == "ok") & (pd.to_numeric(g["true"], errors="coerce") >= target * v_star)
        from_here = good[::-1].cummin()[::-1].to_numpy()  # good at n and at every larger n
        grid = g["n_rollouts"].to_numpy()
        n_star = float(grid[from_here.argmax()]) if from_here.any() else np.inf
        rows.append(
            {
                "size": size,
                "n_states": g["n_states"].iloc[0],
                "seed": seed,
                "method": method,
                "n_star": n_star,
            }
        )
    return pd.DataFrame(rows, columns=["size", "n_states", "seed", "method", "n_star"])


def summarize_rollouts(needed: pd.DataFrame) -> pd.DataFrame:
    """Median and quartiles of n* over seeds (values of the grid; inf = not reached)."""

    def q(x, p):
        return float(np.quantile(np.sort(x.to_numpy()), p, method="inverted_cdf"))

    g = needed.groupby(["size", "n_states", "method"])["n_star"]
    out = g.agg(median=lambda x: q(x, 0.5), q25=lambda x: q(x, 0.25), q75=lambda x: q(x, 0.75))
    out["reached"] = g.agg(lambda x: float(np.isfinite(x).mean()))
    return out.reset_index()


def sample_efficiency(
    res, sizes, labels, xlabel, methods, styles, target: float, title: str, stem
) -> pd.DataFrame:
    """x: problem size, y: rollouts needed for a policy at least `target` optimal: the median over
    seeds, with a band from the first to the third quartile (not a standard deviation: a seed that
    never reaches the target has n* = inf). Not reached within the grid: an open marker on the line
    above the largest number of rollouts. Returns the table of medians and quartiles."""
    setup()
    summary = summarize_rollouts(rollouts_needed(res, target))
    grid = sorted(res["n_rollouts"].unique())
    top = grid[-1] * 1.8
    order = [s for s in sizes if s in set(summary["size"])]
    position = {s: i for i, s in enumerate(order)}
    names = dict(zip(sizes, labels, strict=True))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.6))
    shown = []
    for m in methods:
        sub = summary[summary["method"] == m]
        sub = sub[sub["size"].isin(position)].sort_values("size", key=lambda c: c.map(position))
        if sub.empty:
            continue
        st = styles[m]
        x = sub["size"].map(position).to_numpy(float) + 1.5 * _dodge(methods, m)
        med, lo, hi = (sub[c].to_numpy() for c in ("median", "q25", "q75"))
        reached = np.isfinite(med)
        y = np.where(reached, med, top)
        ax.plot(x, y, ls=st["ls"], color=st["color"], lw=1.4)
        # band: first to third quartile over seeds (a seed that never reaches the target is at the top)
        lower = np.where(np.isfinite(lo), lo, top)
        upper = np.where(np.isfinite(hi), hi, top)
        ax.fill_between(x, lower, upper, color=st["color"], alpha=0.15, lw=0)
        ax.plot(x[reached], med[reached], ls="none", marker=st["marker"], color=st["color"], markersize=6)
        if (~reached).any():
            ax.plot(
                x[~reached],
                y[~reached],
                ls="none",
                marker=st["marker"],
                color=st["color"],
                markerfacecolor="white",
                markersize=6,
            )
        shown.append(m)
    ax.axhline(top, color="grey", ls=":", lw=0.8)
    ax.text(-0.3, top, f" not reached with {grid[-1]} rollouts", va="bottom", fontsize=7.5, color="grey")
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)), [names[s] for s in order])
    ax.set_xlim(-0.4, len(order) - 0.6)
    ax.minorticks_off()
    ax.set_yticks(grid, [str(g) for g in grid])
    ax.set_ylim(grid[0] / 1.4, top * 1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"rollouts for a {target:.0%}-optimal policy")
    ax.set_title(title)
    hollow = Line2D(
        [], [], ls="none", marker="o", color="k", markerfacecolor="white", label="not reached (median)"
    )
    band = plt.Rectangle(
        (0, 0), 1, 1, color="grey", alpha=0.25, lw=0, label="first to third quartile over seeds"
    )
    _legend(fig, shown, styles, extra=[hollow, band])
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, stem)
    return summary
