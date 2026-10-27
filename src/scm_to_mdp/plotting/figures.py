"""Result figures, drawn from results.csv (one row per size, seed, number of transitions and method).

Figures over the problem size, at one number of transitions:
    performance   regret and promise - true value (nominal start, K random starts)
    performance_normalized   the normalised value of the policy, (V^pi - V_random) / (V*_obs - V_random)
                  (0 = random, 1 = optimal; D4RL's normalised score), nominal start and K random starts
    validity      hold rate of the promise, violation rate, worst violation
    structure     SHD and SID of the selected graph
    parameter_learning   the learned probability sets: width, coverage, unseen rows, model size
    resources     time or peak memory of discovery, learning and solving
A point is the mean over seeds, with a band of one standard error, drawn only when every seed of
that size finished; in the time / memory figures, a method stopped by that limit (or skipped
after such a stop) has its marker on the limit line, at every size concerned.

Figure over the problem size and the number of transitions:
    sample_efficiency   transitions (samples) needed for a policy that is at
                        least 95% optimal
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from scm_to_mdp.plotting.stats import iqm_ci
from scm_to_mdp.plotting.style import FULL_WIDTH, save, setup

RATES = {"safe", "safe_rate", "coverage", "cert_holds"}  # shares of runs: plain means, no band

STEP_TITLES = {"discovery": "Causal discovery", "learning": "Parameter learning", "solve": "Solving"}
STOPPED = ("timeout", "memout")


def observed_states(res: pd.DataFrame) -> pd.Series:
    """size -> number of observed states (what the learners see; the full states for older rows)."""
    for column in ("n_observed_states", "n_states"):
        if column in res and res[column].notna().any():
            return pd.to_numeric(res[column], errors="coerce").groupby(res["size"]).median()
    return pd.Series(np.nan, index=pd.Index(res["size"].unique()))  # no counts: the sizes' own labels


class Axis:
    """The x axis over problem sizes, placed by their number of observed states: on a log scale when
    every size has its own number of states, else one tick per size (in the grid's order) labelled
    with its states and its size (e.g. Taxi's two variants, both 500 states)."""

    def __init__(self, res: pd.DataFrame, sizes: list[str], labels: list[str], title: str):
        present = set(res["size"])
        order = [s for s in sizes if s in present]
        names = dict(zip(sizes, labels, strict=True))
        states = observed_states(res).reindex(order)
        if states.notna().all() and states.nunique() == len(order):
            self.order = sorted(order, key=lambda s: states[s])
            self.x = np.log2(states[self.order].to_numpy(float))
            self.labels = [f"{int(states[s]):,}" for s in self.order]
            self.title = "observed states (log scale)"
        else:
            self.order = order
            self.x = np.arange(len(order), dtype=float)
            self.labels = [
                f"{int(states[s]):,}\n{names[s]}" if np.isfinite(states[s]) else names[s] for s in order
            ]
            self.title = f"observed states ({title})"
        self.n_seeds = res.groupby("size")["seed"].nunique().reindex(self.order).values

    def mean_sem(self, res: pd.DataFrame, method: str, column: str):
        """The centre over seeds of one method and its band, per size: the IQM and the distances to the
        ends of its 95% bootstrap interval ([2, n]: below, above; plotting/stats.py), or for a rate
        (RATES) the mean and no band. NaN where some seed has no value."""
        sub = res[res["method"] == method]
        if column not in sub or sub[column].isna().all():
            return None, None
        values = pd.to_numeric(sub[column], errors="coerce")
        centre = np.full(len(self.order), np.nan)
        band = np.zeros((2, len(self.order)))
        for j, size in enumerate(self.order):
            x = values[sub["size"] == size].to_numpy(float)
            if len(x) == 0 or np.isfinite(x).sum() < self.n_seeds[j]:
                continue
            if column in RATES:
                centre[j] = x.mean()
            else:
                centre[j], lo, hi = iqm_ci(x)
                band[:, j] = centre[j] - lo, hi - centre[j]
        return centre, band

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
        many = len(self.x) > 10 or any("\n" in t for t in self.labels) and len(self.x) > 5
        ax.set_xticks(self.x, self.labels, rotation=45 if many else 0, ha="right" if many else "center")
        ax.minorticks_off()
        ax.set_xlabel(self.title)
        ax.set_ylabel(ylabel)


def _dodge(methods: list[str], method: str) -> float:
    """A small sideways shift per method, so that methods with equal values stay visible."""
    return (methods.index(method) - (len(methods) - 1) / 2) * 0.025


def _line(ax, axis: Axis, methods, styles, method, mean, sem, hollow=None):
    """sem: the band, [2, n] (distances below and above the centre) or [n] (symmetric)."""
    st = styles[method]
    x = axis.x + _dodge(methods, method)
    below, above = (sem[0], sem[1]) if np.ndim(sem) == 2 else (sem, sem)
    ax.plot(x, mean, ls=st["ls"], marker=st["marker"], color=st["color"], label=st["label"])
    ax.fill_between(x, mean - below, mean + above, color=st["color"], alpha=0.15, lw=0)
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
    if not handles:  # nothing drawn (e.g. every method failed): no legend
        return
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def _as_float(series: pd.Series) -> pd.Series:
    """0/1 from booleans (or their text); numbers (e.g. a share averaged over tasks) pass through."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return series.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)


# ==========================================================================================
# Over the problem size, at one number of transitions
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
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, stem)


def _span(best: pd.Series, rand: pd.Series) -> pd.Series:
    """best - rand, NaN where it is too small to divide by (random about as good as the optimum)."""
    span = best - rand
    return span.where(span.abs() > 1e-6 * np.maximum(best.abs(), 1.0))


def normalised(res: pd.DataFrame) -> pd.DataFrame:
    """res with the min-max normalised value of the policy (0 = the uniformly random policy, 1 = the
    best policy over the observed state): the normalisation of D4RL (Fu et al. 2020, Sec. 5) and of
    Atari's human-normalised score (Mnih et al. 2015), with the exact optimum and random policy as
    anchors. NaN where V*_obs - V_random is too small to divide by.
        norm_value     (V^pi - V_random) / (V*_obs - V_random) at the nominal start
        norm_value_K   the same on the averages over the k random start states (true_K = V_star_K -
                       regret_K; needs V_random_K, the random policy's average over the same starts)
    and the share of the oracle's optimum, V* by value iteration on the full ground-truth MDP (an agent
    that also sees the hidden variables):
        oracle_share     V^pi / V* at the nominal start (NaN if V* <= 0)
        oracle_share_K   the same on the averages over the k starts (needs V_oracle_K)
    and, on the same scale, what a method claims and how far its certificate reaches:
        norm_promise     (promise - V_random) / (V*_obs - V_random)
        norm_cert_gap    (V_opt - V_rob) / (V*_obs - V_random) (robust methods, evaluation.certificate)"""
    if "norm_value" in res.columns:  # already normalised (e.g. averaged over tasks: transfer.average)
        return res
    num = {c: pd.to_numeric(res[c], errors="coerce") for c in ("true", "V_star_obs", "V_random")}
    span = _span(num["V_star_obs"], num["V_random"])
    out = res.assign(norm_value=(num["true"] - num["V_random"]) / span)
    if "promise" in res.columns:
        promise = pd.to_numeric(res["promise"], errors="coerce")
        out = out.assign(norm_promise=(promise - num["V_random"]) / span)
    if "cert_gap" in res.columns:
        out = out.assign(norm_cert_gap=pd.to_numeric(res["cert_gap"], errors="coerce") / span)
    if {"V_star_K", "regret_K", "V_random_K"} <= set(res.columns):
        best_K, rand_K = (pd.to_numeric(res[c], errors="coerce") for c in ("V_star_K", "V_random_K"))
        true_K = best_K - pd.to_numeric(res["regret_K"], errors="coerce")
        out = out.assign(norm_value_K=(true_K - rand_K) / _span(best_K, rand_K))
        if "V_oracle_K" in res.columns:
            oracle_K = pd.to_numeric(res["V_oracle_K"], errors="coerce")
            out = out.assign(oracle_share_K=true_K / oracle_K.where(oracle_K > 0))
    if "V_star" in res.columns:
        oracle = pd.to_numeric(res["V_star"], errors="coerce")
        out = out.assign(oracle_share=num["true"] / oracle.where(oracle > 0))
    return out


def performance_normalized(res, axis: Axis, methods, styles, target: float, k: int, title: str, stem):
    """2 x 2, at the nominal start (left) and over the k random start states (right): top, the
    normalised value of the policy (0 = random, 1 = optimal over the observed state); bottom, the
    share of V*, the optimum of an agent that sees the full state (value iteration on the full
    ground-truth MDP; not the "oracle" method, which sees the observed state). A line at `target` in
    every panel (see normalised). Every panel shows 0% to 100% (a value outside is off the panel)."""
    setup()
    data = normalised(res)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.6), sharex=True)
    panels = [
        ("norm_value", "Normalised value, nominal start"),
        ("norm_value_K", f"Normalised value, {k} random starts"),
        ("oracle_share", "Share of V* (full state), nominal start"),
        ("oracle_share_K", f"Share of V* (full state), {k} random starts"),
    ]
    shown = []
    for ax, (metric, panel) in zip(axes.flat, panels, strict=True):
        for m in methods:
            mean, sem = axis.mean_sem(data, m, metric) if metric in data else (None, None)
            if mean is None or np.isnan(mean).all():
                continue
            _line(ax, axis, methods, styles, m, mean, sem)
            if m not in shown:
                shown.append(m)
        if metric.startswith("norm"):
            lines = [(1.0, "-", "optimal ", "bottom"), (target, "--", f"{target:.0%} ", "top")]
            lines.append((0.0, ":", "random ", "bottom"))
            ylabel = "(V^π − V_rand) / (V*_obs − V_rand)"
        else:
            lines = [(1.0, "-", "V* ", "bottom"), (target, "--", f"{target:.0%} ", "top")]
            ylabel = "V^π / V*  (V* = VI on the full MDP)"
        ax.set_ylim(-0.02, 1.02)  # 0% to 100% in every panel
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        for level, style, text, va in lines:
            ax.axhline(level, color="k", ls=style, lw=0.8)
            ax.text(axis.x[-1], level, text, va=va, ha="right", fontsize=7)
        ax.set_title(panel)
        axis.finish(ax, ylabel)
    for ax in axes[0]:
        ax.set_xlabel("")  # the x axis is shared: named on the bottom row only
    _legend(fig, shown, styles)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, stem)


def validity_gate(res: pd.DataFrame, delta: float) -> pd.Series:
    """Per row: does its method keep its promise in this size, i.e. a hold rate over the seeds of at
    least 1 - delta? A method without a promise (no 'safe' value) never passes."""
    hold = _as_float(res["safe"]) if "safe" in res else pd.Series(np.nan, index=res.index)
    rate = hold.groupby([res["method"], res["size"]]).transform("mean")
    return rate >= 1 - delta - 1e-9


def gated_comparison(res, axis: Axis, methods, styles, delta: float, title: str, stem):
    """2 x 2, the validity-gated comparison: top, the normalised true value and the normalised promise
    (the value a method guarantees) of the methods whose promise held in at least 1 - delta of the
    seeds at that size (lines; a method that failed the gate is drawn as faint crosses at its value);
    bottom left, the hold rate; bottom right, the certificate (V_opt - V_rob) / (V*_obs - V_rand) of
    the robust methods, an upper bound on their normalised regret when the truth is in the set."""
    setup()
    data = normalised(res)
    data = data.assign(safe_rate=_as_float(data["safe"]) if "safe" in data else np.nan)
    valid = validity_gate(data, delta)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.8), sharex=True)
    panels = [
        ("norm_value", "True value, methods whose promise holds", "(V^π − V_rand) / (V*_obs − V_rand)"),
        (
            "norm_promise",
            "Guaranteed value (promise), same methods",
            "(promise − V_rand) / (V*_obs − V_rand)",
        ),
        ("safe_rate", "Hold rate of the promise", "share of seeds"),
        (
            "norm_cert_gap",
            "Certificate V_opt − V_rob (robust methods)",
            "(V_opt − V_rob) / (V*_obs − V_rand)",
        ),
    ]
    shown = []
    for i, (ax, (metric, panel, ylabel)) in enumerate(zip(axes.flat, panels, strict=True)):
        for m in methods:
            if metric not in data:
                continue
            gated = i < 2
            rows = data.assign(**{metric: data[metric].where(valid)}) if gated else data
            mean, sem = axis.mean_sem(rows, m, metric)
            if gated:
                faint, _ = axis.mean_sem(data.assign(**{metric: data[metric].where(~valid)}), m, metric)
                if faint is not None and not np.isnan(faint).all():
                    st = styles[m]
                    ax.plot(axis.x + _dodge(methods, m), faint, ls="none", marker="x", color=st["color"],
                            alpha=0.35)  # fmt: skip
            if mean is None or np.isnan(mean).all():
                continue
            _line(ax, axis, methods, styles, m, mean, 0 * sem if metric == "safe_rate" else sem)
            if m not in shown:
                shown.append(m)
        if i < 2:
            for level, style, text in [(1.0, "-", "optimal "), (0.0, ":", "random ")]:
                ax.axhline(level, color="k", ls=style, lw=0.8)
                ax.text(axis.x[-1], level, text, va="bottom", ha="right", fontsize=7)
        elif metric == "safe_rate":
            ax.axhline(1 - delta, color="k", ls="--", lw=0.9)
            ax.text(axis.x[0], 1 - delta, f" 1 − δ = {1 - delta:g}", va="top", fontsize=7)
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        else:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)  # room above the largest marker
        ax.set_title(panel)
        axis.finish(ax, ylabel)
    for ax in axes[0]:
        ax.set_xlabel("")
    failed = "promise failed (not gated in)"
    extra = [Line2D([], [], ls="none", marker="x", color="grey", alpha=0.5, label=failed)]
    _legend(fig, shown, styles, extra)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, stem)


def decision_edges(res, axis: Axis, methods, styles, title: str, stem):
    """1 x 2: the share of the true action -> state edges the learned graph keeps (the edges a planner
    needs to tell actions apart) and the SHD (methods that learn a graph)."""
    setup()
    methods = [m for m in methods if axis.mean_sem(res, m, "action_recall")[0] is not None]
    if not methods:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.4), sharex=True)
    panels = [("action_recall", "Action edges kept", "share of true action → state edges"),
              ("shd", "Structural Hamming distance (SHD)", "SHD (0 = true graph)")]  # fmt: skip
    for ax, (metric, panel, ylabel) in zip(axes, panels, strict=True):
        for m in methods:
            mean, sem = axis.mean_sem(res, m, metric)
            if mean is not None:
                _line(ax, axis, methods, styles, m, mean, sem)
        if metric == "action_recall":
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(panel)
        axis.finish(ax, ylabel)
    _legend(fig, methods, styles)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    save(fig, stem)


def transfer_gap(
    gaps: pd.DataFrame,
    axis: Axis,
    methods,
    styles,
    title: str,
    stem,
    ylabel: str = "other rewards − original",
):
    """1 x 2: the normalised value under the alternative rewards (mean over them) minus under the
    original reward, per method, at the nominal start and over the k random starts (< 0: the method
    does worse on the other tasks than on the one its data were logged for)."""
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.8), sharex=True, sharey=True)
    panels = [("gap_value", "Transfer gap, nominal start"), ("gap_value_K", "Transfer gap, random starts")]
    shown = []
    for ax, (metric, panel) in zip(axes, panels, strict=True):
        for m in methods:
            mean, sem = axis.mean_sem(gaps, m, metric) if metric in gaps else (None, None)
            if mean is None or np.isnan(mean).all():
                continue
            _line(ax, axis, methods, styles, m, mean, sem)
            if m not in shown:
                shown.append(m)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(panel)
        axis.finish(ax, ylabel)
    _legend(fig, shown, styles)
    fig.tight_layout(rect=(0, 0.3, 1, 1))  # room for the legend below the x labels
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
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, stem)


def structure(res, axis: Axis, methods, styles, title: str, stem):
    """2 x 2: recall and precision of the selected graph against the true one (0-100%), then SHD and
    SID (methods that learn or are given a graph)."""
    setup()
    res = res.copy()
    if "recall" not in res and {"missing", "extra", "n_true_edges"} <= set(res):  # runs written before
        hits = pd.to_numeric(res["n_true_edges"], errors="coerce") - pd.to_numeric(
            res["missing"], errors="coerce"
        )
        found = hits + pd.to_numeric(res["extra"], errors="coerce")
        res["recall"] = hits / pd.to_numeric(res["n_true_edges"], errors="coerce")
        res["precision"] = (hits / found).where(found > 0)
    methods = [m for m in methods if axis.mean_sem(res, m, "shd")[0] is not None]
    if not methods:
        return
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.2), sharex=True)
    panels = [
        ("recall", "Recall (true edges found)", "recall"),
        ("precision", "Precision (found edges that are true)", "precision"),
        ("shd", "Structural Hamming distance (SHD)", "SHD (0 = true graph)"),
        ("sid", "Structural intervention distance (SID)", "SID (0 = true graph)"),
    ]
    for ax, (metric, panel, ylabel) in zip(axes.flat, panels, strict=True):
        for m in methods:
            mean, sem = axis.mean_sem(res, m, metric) if metric in res else (None, None)
            if mean is not None:
                _line(ax, axis, methods, styles, m, mean, sem)
        ax.set_title(panel)
        if metric in ("recall", "precision"):
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        else:
            ax.set_ylim(bottom=0)
        axis.finish(ax, ylabel)
    _legend(fig, methods, styles)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, stem)


PARAMETER_PANELS = [
    ("s4_width", "Interval width", "mean hi − lo (optimal policy's visits)"),
    ("s4_coverage", "Coverage of the true probabilities", "share inside the intervals"),
    ("s4_unseen", "Unseen parent rows", "share of rows (interval [0, 1])"),
    ("s4_rows", "Model size", "parent rows"),
]


def parameter_learning(res, axis: Axis, methods, styles, delta: float | None, title: str, stem):
    """2 x 2: the learned probability sets (S4) of the methods that have them (evaluation.interval_metrics):
    interval width weighted by the optimal policy's occupancy, coverage of the true per-factor
    probabilities (a valid S4 keeps it >= 1 - delta), share of parent rows with no data, number of
    parent rows. Flat RMDP's joint intervals are projected onto the factors."""
    setup()
    methods = [m for m in methods if axis.mean_sem(res, m, "s4_width")[0] is not None]
    if not methods:
        return
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 5.6), sharex=True)
    for ax, (metric, panel, ylabel) in zip(axes.flat, PARAMETER_PANELS, strict=True):
        for m in methods:
            mean, sem = axis.mean_sem(res, m, metric)
            if mean is not None and not np.isnan(mean).all():
                _line(ax, axis, methods, styles, m, mean, sem)
        if metric == "s4_coverage":
            if delta is not None:
                ax.axhline(1 - delta, color="k", ls="--", lw=0.9)
                ax.text(axis.x[0], 1 - delta, f" 1 − δ = {1 - delta:g}", va="top", fontsize=7)
            ax.set_ylim(top=1.01)
        if metric in ("s4_coverage", "s4_unseen"):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        if metric == "s4_rows":
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(panel)
        axis.finish(ax, ylabel)
    for ax in axes[0]:
        ax.set_xlabel("")  # the x axis is shared: named on the bottom row only
    _legend(fig, methods, styles)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, stem)


def _limit_marks(data, history, axis, method, step, kind) -> list[int]:
    """Sizes (their positions on the axis) where `method` was stopped by `kind` ("timeout" / "memout") in
    `step`, or skipped because of such a stop (the sweep's rule, tool/sweep.py: a stop at (size, n)
    skips every larger size with at least n transitions, and larger n of the same size and seed)."""
    rank = {s: i for i, s in enumerate(axis.order)}
    hist = history[history["method"] == method]
    if "failed_step" not in hist:  # rows written before steps were named: no stop to place
        return []
    stops = hist[hist["status"].isin(STOPPED)].assign(
        rank=lambda d: d["size"].map(rank), step=lambda d: d["failed_step"].replace({"relax": "solve"})
    )
    stops = stops[stops["rank"].notna()]
    rows = data[data["method"] == method]
    marks = set()
    for r in rows[(rows["status"] == kind)].itertuples():
        if str(r.failed_step).replace("relax", "solve") == step:
            marks.add(rank[r.size])
    mine = stops[(stops["status"] == kind) & (stops["step"] == step)]
    for r in rows[rows["status"] == "skipped"].itertuples():
        at = rank[r.size]
        earlier = (mine["rank"] < at) & (mine["n_transitions"] <= r.n_transitions)
        same = (mine["rank"] == at) & (mine["seed"] == r.seed) & (mine["n_transitions"] < r.n_transitions)
        if (earlier | same).any():
            marks.add(at)
    return [float(axis.x[i]) for i in sorted(marks)]  # positions on the axis


def resources(res, axis: Axis, methods, styles, unit: str, limits: dict, title: str, stem, history=None):
    """1 x 3: time (unit "s") or peak memory (unit "gb") of discovery, learning and solving.
    limits: {step: limit or None}, drawn as a dashed line; a limit may also be an array, one value
    per size of the axis (a relative solve limit, tool/sweep.py), drawn through the sizes. "solve"
    includes the relaxation step.
    At every size where a method was stopped by this figure's limit (timeouts in the time figure,
    memouts in the memory figure), or skipped because of such a stop at a smaller size or number of
    transitions, its marker sits on the dashed line. A step stopped by the other limit is an X at its
    measured value. history: the rows of every number of transitions (to find why a row was skipped);
    default res."""
    setup()
    history = res if history is None else history
    kind = "timeout" if unit == "s" else "memout"
    data = res.copy()
    if "relax_s" in data:  # the relaxation (box of the factor sets) is part of solving
        relax = pd.to_numeric(data[f"relax_{unit}"], errors="coerce")
        solve = pd.to_numeric(data[f"solve_{unit}"], errors="coerce")
        data[f"solve_{unit}"] = solve + relax.fillna(0) if unit == "s" else np.fmax(solve, relax)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.9), sharex=True)
    shown, other_shown = [], False
    for ax, step in zip(axes, STEP_TITLES, strict=True):
        column = f"{step}_{unit}"
        limit = limits.get(step)
        for m in methods:
            if step == "discovery" and styles[m].get("role") in ("oracle", "reference", "control"):
                continue  # given the true graph: its "discovery" only packs that graph, not discovery
            mean, sem = axis.mean_sem(data, m, column)
            marks = _limit_marks(data, history, axis, m, step, kind) if limit is not None else []
            if mean is None and not marks:
                continue
            stopped = data[f"{step}_status"].isin(STOPPED)
            if step == "solve" and "relax_status" in data:
                stopped |= data["relax_status"].isin(STOPPED)
            if mean is not None:
                mean[axis.any_true(data, m, stopped)] = np.nan
                _line(ax, axis, methods, styles, m, mean, sem)
            if m not in shown:
                shown.append(m)
            st = styles[m]
            if marks:  # on the limit line: stopped by it, or skipped after it
                at = dict(zip(axis.x, np.broadcast_to(np.asarray(limit, float), axis.x.shape), strict=True))
                ax.plot(
                    np.array(marks) + _dodge(methods, m),
                    np.array([at[x] for x in marks]),
                    ls="none",
                    marker=st["marker"],
                    markersize=7,
                    color=st["color"],
                    markeredgecolor="k",
                    markeredgewidth=0.6,
                    clip_on=False,
                    zorder=5,
                )
            other = stopped & (data["method"] == m) & (data["status"] != kind)
            values = pd.to_numeric(data.loc[other, column], errors="coerce")
            if values.notna().any():  # stopped by the other limit: an X at the measured value
                other_shown = True
                index = {o: x for o, x in zip(axis.order, axis.x, strict=True)}
                pos = data.loc[other, "size"].map(index) + _dodge(methods, m)
                ax.plot(
                    pos,
                    values,
                    ls="none",
                    marker="X",
                    markersize=7,
                    color=st["color"],
                    markeredgecolor="k",
                    markeredgewidth=0.5,
                )
        if limit is not None and np.ndim(limit) > 0:  # one limit per size (relative solve limit)
            ax.plot(axis.x, limit, color="k", ls="--", lw=0.9)
            ax.text(axis.x[0], limit[0], " limit (x oracle)", va="bottom", fontsize=7)
        elif limit is not None:
            ax.axhline(limit, color="k", ls="--", lw=0.9)
            label = f" limit {limit:g} {'s' if unit == 's' else 'GB'}"
            ax.text(axis.x[0], limit, label, va="bottom", fontsize=7)
        else:
            ax.text(0.02, 0.98, "no time limit" if unit == "s" else "no memory limit",
                    transform=ax.transAxes, va="top", fontsize=7)  # fmt: skip
        ax.set_yscale("log")
        ax.set_title(STEP_TITLES[step])
        axis.finish(ax, "seconds" if unit == "s" else "peak memory (GB)")
    on_line = Line2D(
        [], [], ls="none", marker="o", color="grey", markeredgecolor="k", markersize=7,
        label=f"on the dashed line: {kind} at that size, or skipped after one",
    )  # fmt: skip
    extra = [on_line]
    if other_shown:
        extra.append(
            Line2D([], [], ls="none", marker="X", color="grey", markeredgecolor="k", markersize=7,
                   label=f"stopped by the {'memory' if unit == 's' else 'time'} limit (measured value)")
        )  # fmt: skip
    _legend(fig, shown, styles, extra=extra)
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


def transitions_needed(res: pd.DataFrame, target: float = 0.95) -> pd.DataFrame:
    """For every (size, seed, method): n* = the smallest number of transitions n of the grid
    such that the policy is at least `target` optimal (true >= target * V_star_obs at the nominal start)
    at n and at every larger n of the grid; inf if never; t_star = the transitions (samples) of that
    dataset (= n*). A run that stopped counts as not optimal.
    Sizes with V_star_obs <= 0 have no such ratio and are left out. n_observed_states: what the
    learners see (n_states counts the hidden variables too)."""
    rows = []
    res = res.assign(_target=target_value(res), _obs=observed_states(res).reindex(res["size"]).to_numpy())
    if "transitions" not in res:
        res = res.assign(transitions=np.nan)
    for (size, seed, method), g in res.groupby(["size", "seed", "method"]):
        g = g.sort_values("n_transitions")
        v_star = g["_target"].iloc[0]
        if not np.isfinite(v_star) or v_star <= 0:
            continue
        good = (g["status"] == "ok") & (pd.to_numeric(g["true"], errors="coerce") >= target * v_star)
        from_here = good[::-1].cummin()[::-1].to_numpy()  # good at n and at every larger n
        grid = g["n_transitions"].to_numpy()
        samples = pd.to_numeric(g["transitions"], errors="coerce").to_numpy(float)
        n_star = float(grid[from_here.argmax()]) if from_here.any() else np.inf
        t_star = float(samples[from_here.argmax()]) if from_here.any() else np.inf
        rows.append(
            {
                "size": size,
                "n_states": g["n_states"].iloc[0],
                "n_observed_states": g["_obs"].iloc[0],
                "seed": seed,
                "method": method,
                "n_star": n_star,
                "t_star": t_star,
            }
        )
    columns = ["size", "n_states", "n_observed_states", "seed", "method", "n_star", "t_star"]
    return pd.DataFrame(rows, columns=columns)


def summarize_needed(needed: pd.DataFrame) -> pd.DataFrame:
    """Median and quartiles over seeds of n* (median, q25, q75: values of the grid of budgets) and of
    the transitions t* (median_t, q25_t, q75_t); inf = not reached."""

    def q(x, p):
        return float(np.quantile(np.sort(x.to_numpy()), p, method="inverted_cdf"))

    if "t_star" not in needed:  # an older table: budgets only
        needed = needed.assign(t_star=np.nan)
    if "n_observed_states" not in needed:
        needed = needed.assign(n_observed_states=needed["n_states"])
    needed = needed.assign(n_observed_states=needed["n_observed_states"].fillna(needed["n_states"]))
    keys = ["size", "n_states", "n_observed_states", "method"]
    g = needed.groupby(keys)
    out = g["n_star"].agg(median=lambda x: q(x, 0.5), q25=lambda x: q(x, 0.25), q75=lambda x: q(x, 0.75))
    out["reached"] = g["n_star"].agg(lambda x: float(np.isfinite(x).mean()))
    for name, p in (("median_t", 0.5), ("q25_t", 0.25), ("q75_t", 0.75)):
        out[name] = g["t_star"].agg(lambda x, p=p: q(x.fillna(np.inf), p))
    return out.reset_index()


def sample_grid(res: pd.DataFrame) -> list[float]:
    """The transitions of every number of transitions of the grid (median over the runs): the y ticks of
    the figures of samples needed."""
    if "transitions" not in res or res["transitions"].isna().all():
        return [float(n) for n in sorted(res["n_transitions"].unique())]
    t = pd.to_numeric(res["transitions"], errors="coerce").groupby(res["n_transitions"]).median()
    return [float(v) for v in t.sort_index().to_numpy()]


def samples_label(v: float) -> str:
    """12000 -> "12k", 1500000 -> "1.5M"."""
    for div, unit in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:.3g}{unit}"
    return f"{v:.0f}"


def sample_efficiency(
    res, sizes, labels, xlabel, methods, styles, target: float, title: str, stem
) -> pd.DataFrame:
    """x: observed states (Axis), y: transitions (samples) needed for a policy at least `target`
    optimal: the median over seeds, with a band from the first to the third quartile (not a standard
    deviation: a seed that never reaches the target has t* = inf). Not reached within the grid: an
    open marker on the line above the largest dataset. Returns the table of medians and quartiles
    (of the budget n* and of the transitions t*)."""
    setup()
    summary = summarize_needed(transitions_needed(res, target))
    if summary.empty:  # no run finished (or no size has V*_obs > 0): nothing to draw
        return summary
    grid = sample_grid(res)
    top = grid[-1] * 1.8
    axis = Axis(res[res["size"].isin(set(summary["size"]))], sizes, labels, xlabel)
    position = dict(zip(axis.order, axis.x, strict=True))
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.6))
    shown = []
    for m in methods:
        sub = summary[summary["method"] == m]
        sub = sub[sub["size"].isin(position)].sort_values("size", key=lambda c: c.map(position))
        if sub.empty:
            continue
        st = styles[m]
        x = sub["size"].map(position).to_numpy(float) + 1.5 * _dodge(methods, m)
        med, lo, hi = (sub[c].to_numpy(float) for c in ("median_t", "q25_t", "q75_t"))
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
    ax.text(axis.x[0], top, f" not reached with {samples_label(grid[-1])} transitions", va="bottom",
            fontsize=7.5, color="grey")  # fmt: skip
    ax.set_yscale("log")
    axis.finish(ax, f"transitions for a {target:.0%}-optimal policy")
    ax.set_xlim(axis.x[0] - 0.4, axis.x[-1] + 0.4)
    ax.set_yticks(grid, [samples_label(g) for g in grid])
    ax.set_ylim(grid[0] / 1.4, top * 1.5)
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
