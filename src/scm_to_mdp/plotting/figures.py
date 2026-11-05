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
    coverage_efficiency transitions needed for probability sets that contain >= 99% of the true
                        probabilities (left), and are also small: mean width <= 0.2 (right)
    guarantee_efficiency transitions needed for a promise that holds (left), and that also proves the
                        policy better than the random one (right)

One figure per size, over the number of transitions (figures/zoom_all_methods/):
    zoom_by_size  policy quality, intervals that contain the true kernel, true edges missed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from scm_to_mdp.plotting.stats import iqm_ci, median_ci
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


def transitions_needed(
    res: pd.DataFrame, target: float = 0.95, condition: pd.Series | None = None
) -> pd.DataFrame:
    """For every (size, seed, method): n* = the smallest number of transitions n of the grid
    such that the policy is at least `target` optimal (true >= target * V_star_obs at the nominal start)
    at n and at every larger n of the grid; inf if never; t_star = the transitions (samples) of that
    dataset (= n*). A run that stopped counts as not optimal.
    Sizes with V_star_obs <= 0 have no such ratio and are left out. n_observed_states: what the
    learners see (n_states counts the hidden variables too).
    condition: another success test, one boolean per row of res (NaN = not met), in place of the
    value test (`target` is then unused and no size is left out); a stopped run still fails it."""
    rows = []
    if condition is not None:
        res = res.assign(_good=condition.reindex(res.index).fillna(False).astype(bool))
    res = res.assign(_obs=observed_states(res).reindex(res["size"]).to_numpy())
    if condition is None:
        res = res.assign(_target=target_value(res))
    if "transitions" not in res:
        res = res.assign(transitions=np.nan)
    for (size, seed, method), g in res.groupby(["size", "seed", "method"]):
        g = g.sort_values("n_transitions")
        if condition is not None:
            good = (g["status"] == "ok") & g["_good"]
        else:
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
    """Median over seeds and its 95% confidence interval (plotting/stats.median_ci) of n* (median,
    ci_lo, ci_hi: values of the grid of budgets) and of the transitions t* (median_t, ci_lo_t,
    ci_hi_t); inf = not reached."""

    if "t_star" not in needed:  # an older table: budgets only
        needed = needed.assign(t_star=np.nan)
    if "n_observed_states" not in needed:
        needed = needed.assign(n_observed_states=needed["n_states"])
    needed = needed.assign(n_observed_states=needed["n_observed_states"].fillna(needed["n_states"]))
    keys = ["size", "n_states", "n_observed_states", "method"]
    g = needed.groupby(keys)

    def summary(column, names):
        return g[column].apply(lambda x: pd.Series(median_ci(x.fillna(np.inf)), index=names)).unstack()

    out = summary("n_star", ["median", "ci_lo", "ci_hi"])
    out["reached"] = g["n_star"].agg(lambda x: float(np.isfinite(x).mean()))
    return out.join(summary("t_star", ["median_t", "ci_lo_t", "ci_hi_t"])).reset_index()


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


def _needed_panel(ax, res, summary, sizes, labels, xlabel, methods, styles, ylabel: str) -> list[str]:
    """One panel of transitions needed (summarize_needed): x the observed states, y the median t* over
    seeds with its 95% CI as the band; not reached: an open marker on the line above the largest
    dataset. Returns the methods drawn."""
    grid = sample_grid(res)
    top = grid[-1] * 1.8
    axis = Axis(res[res["size"].isin(set(summary["size"]))], sizes, labels, xlabel)
    position = dict(zip(axis.order, axis.x, strict=True))
    shown = []
    for m in methods:
        sub = summary[summary["method"] == m]
        sub = sub[sub["size"].isin(position)].sort_values("size", key=lambda c: c.map(position))
        if sub.empty:
            continue
        st = styles[m]
        x = sub["size"].map(position).to_numpy(float) + 1.5 * _dodge(methods, m)
        med, lo, hi = (sub[c].to_numpy(float) for c in ("median_t", "ci_lo_t", "ci_hi_t"))
        reached = np.isfinite(med)
        y = np.where(reached, med, top)
        ax.plot(x, y, ls=st["ls"], color=st["color"], lw=1.6)
        # band: 95% CI of the median over seeds (an end at inf is drawn at the "not reached" line)
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
    axis.finish(ax, ylabel)
    ax.set_xlim(axis.x[0] - 0.4, axis.x[-1] + 0.4)
    ax.set_yticks(grid, [samples_label(g) for g in grid])
    ax.set_ylim(grid[0] / 1.4, top * 1.5)
    return shown


def _needed_extras() -> list:
    hollow = Line2D(
        [], [], ls="none", marker="o", color="k", markerfacecolor="white", label="not reached (median)"
    )
    band = plt.Rectangle(
        (0, 0), 1, 1, color="grey", alpha=0.25, lw=0, label="95% CI of the median over seeds"
    )
    return [hollow, band]


def sample_efficiency(
    res, sizes, labels, xlabel, methods, styles, target: float, title: str, stem
) -> pd.DataFrame:
    """x: observed states (Axis), y: transitions (samples) needed for a policy at least `target`
    optimal: the median over seeds, with its 95% confidence interval as the band (plotting/stats.median_ci;
    a seed that never reaches the target has t* = inf). Not reached within the grid: an open marker on
    the line above the largest dataset. Returns the table of medians and intervals (of the budget n*
    and of the transitions t*)."""
    setup()
    summary = summarize_needed(transitions_needed(res, target))
    if summary.empty:  # no run finished (or no size has V*_obs > 0): nothing to draw
        return summary
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.6))
    ylabel = f"transitions for a {target:.0%}-optimal policy"
    shown = _needed_panel(ax, res, summary, sizes, labels, xlabel, methods, styles, ylabel)
    ax.set_title(title)
    _legend(fig, shown, styles, extra=_needed_extras())
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, stem)
    return summary


COVERAGE_NAMES = {
    "s4_coverage_seen": "true probabilities inside the intervals, seen (s, a)",
    "s4_coverage": "true probabilities inside the intervals, every (s, a)",
}


def _set_methods(res: pd.DataFrame, methods, has: pd.Series) -> list[str]:
    """The methods with a value in `has` (a mask over res), the oracle left out (the true model:
    trivially valid, and of width 0)."""
    present = has.groupby(res["method"]).any()
    roles = res.drop_duplicates("method").set_index("method")["role"] if "role" in res else pd.Series()
    oracle = {m for m in methods if roles.get(m) == "oracle" or m.startswith("oracle")}
    return [m for m in methods if present.get(m, False) and m not in oracle]


def _two_needed_panels(rows, out, heads, sizes, labels, xlabel, methods, styles, suptitle, stem):
    """Left and right: transitions needed for out["valid"] and out["tight"] (_needed_panel)."""
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 4.0), sharey=True)
    shown = []
    for ax, key, head in zip(axes, ("valid", "tight"), heads, strict=True):
        drawn = _needed_panel(
            ax, rows, out[key], sizes, labels, xlabel, methods, styles, "transitions needed"
        )
        shown += [m for m in drawn if m not in shown]
        ax.set_title(head, fontsize=9)
        ax.tick_params(axis="x", labelsize=6.5, labelrotation=45)
    axes[1].set_ylabel("")
    axes[1].texts[0].remove()  # "not reached ..." once, on the left panel
    fig.suptitle(suptitle, fontsize=9.5)
    _legend(fig, shown, styles, extra=_needed_extras())
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    save(fig, stem)


def coverage_efficiency(
    res, sizes, labels, xlabel, methods, styles, column: str, title: str, stem,
    target: float = 0.99, eps: float = 0.2,
) -> dict[str, pd.DataFrame]:  # fmt: skip
    """Transitions needed for valid (left) and for valid and small (right) probability sets (S4), on the
    axes of sample_efficiency: left, until `column` (a share of the true factor probabilities inside
    their intervals, evaluation.interval_metrics) is >= target; right, until it is and the mean
    interval width s4_width (hi - lo on the probability scale, weighted by the best policy's
    occupancy; an unseen row counts 1) is <= eps. Both at n and at every larger n (transitions_needed).
    Only the methods with intervals; not the oracle (the true model, trivially valid and of width 0).
    Returns {"valid": summary, "tight": summary} (summarize_needed); empty when no method has `column`."""
    empty = {"valid": pd.DataFrame(), "tight": pd.DataFrame()}
    if column not in res or "s4_width" not in res:
        return empty
    value = pd.to_numeric(res[column], errors="coerce")
    width = pd.to_numeric(res["s4_width"], errors="coerce")
    kept = _set_methods(res, methods, value.notna())
    if not kept:
        return empty
    rows = res[res["method"].isin(kept)]
    valid = value.loc[rows.index] >= target - 1e-12
    tight = valid & (width.loc[rows.index] <= eps + 1e-12)
    out = {
        "valid": summarize_needed(transitions_needed(rows, condition=valid)),
        "tight": summarize_needed(transitions_needed(rows, condition=tight)),
    }
    if out["valid"].empty:
        return empty
    heads = (f"coverage \u2265 {target:.0%}", f"coverage \u2265 {target:.0%} and mean width \u2264 {eps:g}")
    suptitle = f"{title}\n({COVERAGE_NAMES.get(column, column)})"
    _two_needed_panels(rows, out, heads, sizes, labels, xlabel, kept, styles, suptitle, stem)
    return out


CERT_FLOOR = 0.01  # the certificate figures (log axis): a value at or below it is drawn on it


def _median_table(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per size and method: the median over seeds of `column` and its 95% CI (stats.median_ci)."""
    out = []
    for (size, method), x in rows.groupby(["size", "method"]):
        values = pd.to_numeric(x[column], errors="coerce").dropna().to_numpy(float)
        if len(values):
            med, lo, hi = median_ci(values)
            out.append({"size": size, "method": method, "median": med, "ci_lo": lo, "ci_hi": hi,
                        "seeds": len(values)})  # fmt: skip
    return pd.DataFrame(out, columns=["size", "method", "median", "ci_lo", "ci_hi", "seeds"])


def _certificate_panel(ax, rows, axis: Axis, methods, styles) -> tuple[list[str], pd.DataFrame]:
    """Proved vs actual regret of the finished runs in rows (one number of transitions), over the sizes
    of axis, both normalised by V*_obs - V_random (normalised): the certificate (V_opt - V_rob), a bound
    on the regret when the set contains the true MDP (filled markers, line, 95% CI of the median as the
    band), and the actual regret V*_obs - V^pi = 1 - norm_value (hollow markers). Medians over seeds;
    log scale, values <= CERT_FLOOR drawn on it; a dotted line at 1 (the whole gap from random to
    optimal; its label is in the caller's legend). Returns the methods drawn and the table (size,
    method, median, ci_lo, ci_hi, seeds, regret: the median actual regret)."""
    done = normalised(rows[rows["status"] == "ok"])
    cert = _median_table(done, "norm_cert_gap")
    regret = _median_table(done.assign(norm_regret_=1.0 - done["norm_value"]), "norm_regret_")
    cert = cert.merge(regret[["size", "method", "median"]].rename(columns={"median": "regret"}),
                      on=["size", "method"], how="left")  # fmt: skip
    position = dict(zip(axis.order, axis.x, strict=True))
    shown = []
    for m in methods:
        sub = cert[(cert["method"] == m) & cert["size"].isin(position)]
        sub = sub.sort_values("size", key=lambda c: c.map(position))
        if sub.empty:
            continue
        st = styles[m]
        x = sub["size"].map(position).to_numpy(float) + 1.5 * _dodge(methods, m)
        med, lo, hi, reg = (
            np.maximum(sub[c].to_numpy(float), CERT_FLOOR) for c in ("median", "ci_lo", "ci_hi", "regret")
        )
        ax.plot(x, med, ls=st["ls"], color=st["color"], marker=st["marker"], markersize=6, lw=1.6)
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0)
        ax.plot(x, reg, ls="none", marker=st["marker"], color=st["color"], markerfacecolor="white",
                markersize=5, markeredgewidth=1.0)  # fmt: skip
        shown.append(m)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"\u2264 {v:g}" if v <= CERT_FLOOR * 1.001 else f"{v:g}")
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(bottom=CERT_FLOOR / 1.3)
    axis.finish(ax, r"regret, as a share of $V^* - V_{\mathrm{rand}}$")
    ax.set_xlim(axis.x[0] - 0.4, axis.x[-1] + 0.4)
    return shown, cert


def certificate(res, axis: Axis, methods, styles, title: str, stem) -> pd.DataFrame:
    """At one number of transitions (res = its rows): proved vs actual regret over the problem size
    (_certificate_panel), for the methods with interval sets, not the oracle. Returns its table;
    nothing drawn without the columns."""
    if not {"cert_gap", "V_star_obs", "V_random"} <= set(res.columns):
        return pd.DataFrame()
    kept = _set_methods(res, methods, pd.to_numeric(res["cert_gap"], errors="coerce").notna())
    if not kept:
        return pd.DataFrame()
    rows = res[res["method"].isin(kept)]
    setup()
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.62, 4.2))
    shown, cert = _certificate_panel(ax, rows, axis, kept, styles)
    if cert.empty:
        plt.close(fig)
        return cert
    ax.set_title(f"{title}\nproved vs actual regret of the returned policy", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, labelrotation=45 if len(axis.x) > 6 else 0)
    band = plt.Rectangle(
        (0, 0), 1, 1, color="grey", alpha=0.25, lw=0, label="95% CI of the median over seeds"
    )
    proved = Line2D(
        [], [], ls="-", marker="o", color="k", label=r"proved: $V_{\mathrm{opt}} - V_{\mathrm{rob}}$"
    )
    actual = Line2D(
        [], [], ls="none", marker="o", color="k", markerfacecolor="white", label=r"actual: $V^* - V^\pi$"
    )
    whole = Line2D([], [], ls=":", color="grey", label="1 = the whole gap from random to optimal")
    handles = [Line2D([], [], ls=styles[m]["ls"], marker=styles[m]["marker"], color=styles[m]["color"],
                      label=styles[m]["label"]) for m in shown] + [proved, actual, whole, band]  # fmt: skip
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01)
    )
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    save(fig, stem)
    return cert


def guarantee_efficiency(
    res, sizes, labels, xlabel, methods, styles, title: str, stem, useful: float = 0.0
) -> dict[str, pd.DataFrame]:
    """The guarantee on the axes of sample_efficiency (S8 -> S9). Left: transitions needed until the
    promise holds (safe: promise <= true value in every observed live state; evaluation.evaluate), at
    n and at every larger n (transitions_needed): a run whose promise breaks at a larger n never
    counts. Right: until it holds and also proves the policy better than the random one: normalised
    promise (V_rob - V_random) / (V*_obs - V_random) >= `useful` (0: V_rob >= V_random). Only the
    methods with interval sets (contain_all), not the oracle. Returns {"valid": summary, "tight":
    summary} (summarize_needed), and with `contain`, "contain": until the set contains the true MDP
    (contain_all, the event the promise relies on); empty without the columns. stem None: no figure."""
    empty = {"valid": pd.DataFrame(), "tight": pd.DataFrame(), "contain": pd.DataFrame()}
    if not {"safe", "promise", "contain_all", "V_star_obs", "V_random"} <= set(res.columns):
        return empty
    contain = _as_float(res["contain_all"].astype(object).where(res["contain_all"].notna()))
    kept = _set_methods(res, methods, contain.notna())
    if not kept:
        return empty
    rows = res[res["method"].isin(kept)]
    safe = _as_float(rows["safe"].astype(object).where(rows["safe"].notna())) == 1.0
    promise = normalised(rows)["norm_promise"]
    out = {
        "valid": summarize_needed(transitions_needed(rows, condition=safe)),
        "tight": summarize_needed(transitions_needed(rows, condition=safe & (promise >= useful - 1e-12))),
        "contain": summarize_needed(transitions_needed(rows, condition=contain.loc[rows.index] == 1.0)),
    }
    if out["valid"].empty:
        return empty
    better = "random" if useful == 0 else f"{useful:g} of the gap from random to optimal"
    proof = "\n" + r"$V_{\mathrm{rob}} \geq V_{\mathrm{rand}}$" if useful == 0 else ""
    heads = (
        "the promise holds in every state",
        f"it holds and proves the policy better than {better}{proof}",
    )
    if stem is not None:  # None: the tables only
        _two_needed_panels(rows, out, heads, sizes, labels, xlabel, kept, styles, title, stem)
    return out


# ==========================================================================================
# Regret map: normalised regret over (problem size, number of transitions), one panel per method
# ==========================================================================================
AS_RANDOM = (0.95, 1.05)  # "about as good as the random policy" (e.g. a fallback to a uniform logger)
RANDOM_LIKE = "#9E9E9E"  # grey
WORSE_THAN_RANDOM = "#D55E00"  # vermilion (Okabe & Ito): regret above AS_RANDOM[1]
REACHED = "#F0F0F0"  # very light grey: the regret stays within the target from here on


def regret_map(
    res, sizes, labels, xlabel, methods, styles, title: str, stem, target: float = 0.05
) -> pd.DataFrame:
    """One panel per method (not the oracle): x the problem sizes (by observed states), y the numbers of
    transitions; a cell is the median over seeds of the normalised regret 1 - norm_value =
    (V*_obs - V^pi) / (V*_obs - V_random) at the nominal start (normalised), on a colour gradient
    (viridis) up to AS_RANDOM[0], then grey (about random) and vermilion (worse than random). Reached: per
    seed, the first n from which the normalised regret stays <= target at every larger n
    (transitions_needed's rule); per size, the median over seeds; the
    cells from there up are grey and a black line marks the frontier. Hatched: some seed stopped by a
    limit (time or memory) at that cell; x: no finished run. Returns the cells (size, n_transitions,
    method, regret, n_reached)."""
    if not {"true", "V_star_obs", "V_random"} <= set(res.columns):
        return pd.DataFrame()
    data = normalised(res)
    data = data.assign(regret=1.0 - pd.to_numeric(data["norm_value"], errors="coerce"))
    kept = _set_methods(data, methods, data["regret"].notna())
    if not kept:
        return pd.DataFrame()
    data = data[data["method"].isin(kept)]
    good = (data["regret"] <= target + 1e-12).fillna(False)
    reached = summarize_needed(transitions_needed(data, condition=good)).set_index(["size", "method"])[
        "median"
    ]
    finished = data[data["status"] == "ok"]
    cells = (
        finished.groupby(["size", "n_transitions", "method"])["regret"]
        .median()
        .rename("regret")
        .reset_index()
    )
    stopped = data[data["status"].isin(STOPPED)].groupby(["size", "n_transitions", "method"]).size()
    axis = Axis(data, sizes, labels, xlabel)
    grid = sorted(data["n_transitions"].unique())
    order = axis.order
    cmap = plt.get_cmap("viridis_r")  # colour-blind safe gradient, yellow = no regret
    norm = plt.Normalize(0.0, AS_RANDOM[0])

    def colour(v):
        if not np.isfinite(v):
            return None
        if v > AS_RANDOM[1]:
            return WORSE_THAN_RANDOM
        if v >= AS_RANDOM[0]:
            return RANDOM_LIKE
        return cmap(norm(max(v, 0.0)))

    setup()
    ncol = min(3, len(kept))
    nrow = int(np.ceil(len(kept) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH, 2.25 * nrow + 0.9), squeeze=False,
                             sharex=True, sharey=True)  # fmt: skip
    for k, m in enumerate(kept):
        ax = axes.flat[k]
        for i, s in enumerate(order):
            n_star = reached.get((s, m), np.inf)
            for j, n in enumerate(grid):
                if np.isfinite(n_star) and n >= n_star:
                    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=REACHED, lw=0))
                    continue
                row = cells[(cells["size"] == s) & (cells["n_transitions"] == n) & (cells["method"] == m)]
                c = colour(float(row["regret"].iloc[0])) if len(row) else None
                if c is None:
                    ax.plot(i, j, marker="x", color="grey", markersize=3, lw=0)
                else:
                    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=c, lw=0))
                if stopped.get((s, n, m), 0):
                    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, hatch="////",
                                               edgecolor="white", lw=0))  # fmt: skip
            if np.isfinite(n_star):  # the frontier: the lower edge of the first reached cell
                j = grid.index(n_star) if n_star in grid else int(np.searchsorted(grid, n_star))
                ax.plot([i - 0.5, i + 0.5], [j - 0.5, j - 0.5], color="black", lw=1.4)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(-0.5, len(grid) - 0.5)
        ax.set_title(styles[m]["label"], fontsize=8.5)
        ax.grid(False)
        ax.set_xticks(range(len(order)), axis.labels, rotation=60, fontsize=6.5)
        ax.set_yticks(range(len(grid)), [samples_label(g) for g in grid], fontsize=6.5)
        if k % ncol == 0:
            ax.set_ylabel("transitions")
        if k // ncol == nrow - 1 or k + ncol >= len(kept):
            ax.set_xlabel("observed states", fontsize=7.5)
    for k in range(len(kept), nrow * ncol):  # empty panels: the one above shows the x tick labels
        axes.flat[k].set_visible(False)
        above = axes.flat[k - ncol]
        above.xaxis.set_tick_params(labelbottom=True)
        above.set_xlabel("observed states", fontsize=7.5)
    bar = fig.add_axes((0.08, 0.045, 0.36, 0.014))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, orientation="horizontal")
    cb.set_label("normalised regret (median over seeds)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=RANDOM_LIKE, label=f"{AS_RANDOM[0]:g}–{AS_RANDOM[1]:g} (≈ random)"),
        plt.Rectangle((0, 0), 1, 1, color=WORSE_THAN_RANDOM, label=f"> {AS_RANDOM[1]:g} (worse than random)"),
        plt.Rectangle((0, 0), 1, 1, color=REACHED, label=f"reached: ≤ {target:g} from here on"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#888888", hatch="////", edgecolor="white", label="a seed stopped"
        ),
        Line2D([], [], ls="none", marker="x", color="grey", label="no finished run"),
    ]
    fig.legend(
        handles=handles, loc="lower right", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.99, 0.0)
    )
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.075 + 0.05 / nrow, 1, 1))
    save(fig, stem)
    out = cells.assign(
        n_reached=[reached.get((s, m), np.inf) for s, m in zip(cells["size"], cells["method"], strict=True)]
    )
    return out


# ==========================================================================================
# Regret gain: what one more step of data buys, over (problem size, step of the budget grid)
# ==========================================================================================
GAIN_LIMIT = 1.0  # colour scale of the gain per decade: [-GAIN_LIMIT, GAIN_LIMIT], beyond it the end colours
NOTHING_TO_GAIN = "#BDBDBD"  # mid grey, apart from the white middle (gain ~ 0) of the diverging scale


def _paired_gains(data: pd.DataFrame, grid: list) -> pd.DataFrame:
    """Per (size, seed, method) and step n_k -> n_{k+1} of the grid, both runs finished: the gain per decade
    (r(n_k) - r(n_{k+1})) / log10(n_{k+1} / n_k), r the normalised regret. Paired by seed: the datasets
    are nested (the smaller budget is a prefix of the larger), so the difference is the effect of the
    added data."""
    ok = data[data["status"] == "ok"].set_index(["size", "seed", "method", "n_transitions"])["regret"]
    ok = ok[~ok.index.duplicated()]
    rows = []
    for a, b in zip(grid[:-1], grid[1:], strict=True):
        try:
            ra, rb = ok.xs(a, level="n_transitions"), ok.xs(b, level="n_transitions")
        except KeyError:
            continue
        both = pd.concat([ra.rename("ra"), rb.rename("rb")], axis=1, join="inner").dropna()
        gain = (both["ra"] - both["rb"]) / np.log10(b / a)
        rows.append(
            pd.DataFrame({"gain": gain, "ra": both["ra"], "rb": both["rb"]})
            .reset_index()
            .assign(n_from=a, n_to=b)
        )
    if not rows:
        return pd.DataFrame(columns=["size", "seed", "method", "gain", "ra", "rb", "n_from", "n_to"])
    return pd.concat(rows, ignore_index=True)


def learning_rates(data: pd.DataFrame, target: float = 0.05, min_points: int = 3) -> pd.DataFrame:
    """Per (size, method): the learning rate b of r ~ n^-b, the slope of log r on log n (least squares,
    sign flipped) over the finished runs of each seed with target < r (before the regret is small; r > 0),
    with at least min_points budgets; the median over the seeds that have a fit, and how many have one."""
    out = []
    ok = data[(data["status"] == "ok") & (data["regret"] > target)]
    for (size, method), g in ok.groupby(["size", "method"]):
        rates = []
        for _, s in g.groupby("seed"):
            if s["n_transitions"].nunique() >= min_points:
                slope = np.polyfit(
                    np.log10(s["n_transitions"].astype(float)), np.log10(s["regret"].astype(float)), 1
                )[0]
                rates.append(-slope)
        if rates:
            out.append({"size": size, "method": method, "rate": float(np.median(rates)), "seeds": len(rates)})
    return pd.DataFrame(out, columns=["size", "method", "rate", "seeds"])


def regret_gain(res, sizes, labels, xlabel, methods, styles, title: str, stem, target: float = 0.05):
    """One panel per method (not the oracle), on the axes of regret_map: x the problem sizes, y the steps
    n_k -> n_{k+1} of the budget grid; a cell is the median over seeds of the paired gain per decade of
    data (_paired_gains) on a diverging colour-blind-safe scale (purple: the added data lowered the
    regret; orange: it raised it), clipped at +-GAIN_LIMIT. A dot: the 95% CI of the median
    (stats.median_ci) excludes 0. Grey: the median regret is <= target at both ends (nothing left to
    gain); x: no paired run. Returns (cells: size, n_from, n_to, method, gain, lo, hi, seeds;
    learning_rates)."""
    empty = (pd.DataFrame(), pd.DataFrame())
    if not {"true", "V_star_obs", "V_random"} <= set(res.columns):
        return empty
    data = normalised(res)
    data = data.assign(regret=1.0 - pd.to_numeric(data["norm_value"], errors="coerce"))
    kept = _set_methods(data, methods, data["regret"].notna())
    if not kept:
        return empty
    data = data[data["method"].isin(kept)]
    grid = sorted(data["n_transitions"].unique())
    pairs = _paired_gains(data, grid)
    if pairs.empty:
        return empty
    cells = []
    for (size, a, b, m), g in pairs.groupby(["size", "n_from", "n_to", "method"]):
        med, lo, hi = median_ci(g["gain"].to_numpy(float))
        done = g["ra"].median() <= target and g["rb"].median() <= target
        cells.append({"size": size, "n_from": a, "n_to": b, "method": m, "gain": med, "lo": lo, "hi": hi,
                      "seeds": len(g), "nothing_to_gain": bool(done)})  # fmt: skip
    cells = pd.DataFrame(cells)
    rates = learning_rates(data, target)

    axis = Axis(data, sizes, labels, xlabel)
    order = axis.order
    steps = list(zip(grid[:-1], grid[1:], strict=True))
    cmap = plt.get_cmap("PuOr")  # orange (negative: a loss) ... purple (positive: a gain); colour-blind safe
    norm = plt.Normalize(-GAIN_LIMIT, GAIN_LIMIT)
    setup()
    ncol = min(3, len(kept))
    nrow = int(np.ceil(len(kept) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL_WIDTH, 2.25 * nrow + 1.1), squeeze=False,
                             sharex=True, sharey=True)  # fmt: skip
    index = cells.set_index(["size", "n_from", "method"])
    for k, m in enumerate(kept):
        ax = axes.flat[k]
        for i, s in enumerate(order):
            for j, (a, _) in enumerate(steps):
                if (s, a, m) not in index.index:
                    ax.plot(i, j, marker="x", color="grey", markersize=3, lw=0)
                    continue
                c = index.loc[(s, a, m)]
                if c["nothing_to_gain"]:
                    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=NOTHING_TO_GAIN, lw=0))
                    continue
                ax.add_patch(
                    plt.Rectangle(
                        (i - 0.5, j - 0.5),
                        1,
                        1,
                        color=cmap(norm(np.clip(c["gain"], -GAIN_LIMIT, GAIN_LIMIT))),
                        lw=0,
                    )
                )
                if c["lo"] > 0 or c["hi"] < 0:
                    ax.plot(i, j, marker=".", color="black", markersize=2.5, lw=0)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_ylim(-0.5, len(steps) - 0.5)
        ax.set_title(styles[m]["label"], fontsize=8.5)
        ax.grid(False)
        ax.set_xticks(range(len(order)), axis.labels, rotation=60, fontsize=6.5)
        ax.set_yticks(
            range(len(steps)), [f"{samples_label(a)}→{samples_label(b)}" for a, b in steps], fontsize=6
        )
        if k % ncol == 0:
            ax.set_ylabel("more data: from → to")
        if k + ncol >= len(kept):
            ax.set_xlabel("observed states", fontsize=7.5)
    for k in range(len(kept), nrow * ncol):  # empty panels: the one above shows the x tick labels
        axes.flat[k].set_visible(False)
        above = axes.flat[k - ncol]
        above.xaxis.set_tick_params(labelbottom=True)
        above.set_xlabel("observed states", fontsize=7.5)
    bar = fig.add_axes((0.18, 0.045, 0.4, 0.014))
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, orientation="horizontal", extend="both"
    )
    cb.set_label("gain in normalised regret per decade of data (median over seeds)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    extra = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=NOTHING_TO_GAIN,
            label=f"already near-optimal (regret ≤ {target:g} before and after)",
        ),
        Line2D([], [], ls="none", marker=".", color="black", label="95% CI of the median excludes 0"),
        Line2D([], [], ls="none", marker="x", color="grey", label="no paired run"),
    ]
    fig.legend(
        handles=extra, loc="lower right", ncol=1, frameon=False, fontsize=7, bbox_to_anchor=(0.98, 0.0)
    )
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.09 + 0.05 / nrow, 1, 1))
    save(fig, stem)
    return cells, rates


# ==========================================================================================
# One size at a time, over the number of logged transitions
# ==========================================================================================
ZOOM_FOLDER = "zoom_all_methods"  # figures/<ZOOM_FOLDER>/observed_states=<n>.{pdf,png}


def _zoom_stems(res: pd.DataFrame, present: list[str], folder) -> dict:
    """size -> figure stem: observed_states=<n>, with the size's name appended when two sizes have
    the same number of observed states (e.g. Taxi's two variants)."""
    states = observed_states(res).reindex(present)
    twice = states.duplicated(keep=False)
    stems = {}
    for s in present:
        n = f"{int(states[s])}" if np.isfinite(states[s]) else "unknown"
        tag = f"_{s}" if twice[s] or n == "unknown" else ""
        stems[s] = folder / f"observed_states={n}{tag}".replace(" ", "_").replace("/", "-")
    return stems


def zoom_by_size(res, sizes, labels, methods, styles, target: float, name: str, folder) -> pd.DataFrame:
    """One figure per size, over the logged transitions (log x), every method but the oracle (the true
    model: optimal at every budget):
        top      the share of the optimum V^pi / V*_obs (target_value) at the nominal start: the median
                 over seeds and its 95% CI (plotting/stats.median_ci); a dashed line at `target`
        middle   the seeds (of all) whose intervals contain the true kernel in every (s, a) (contain_all;
                 the methods with probability sets only)
        bottom   the true edges the learned graph misses (missing): median and 95% CI (the methods with
                 a graph only)
    A panel no method has values for is left out (e.g. the middle one for point-estimate MDPs). Per
    method, a vertical line at the transitions it needs for a `target`-optimal policy (the median over
    seeds of transitions_needed, as sample_efficiency; none if not reached), also in the legend. A point
    is drawn only when every seed of that size and budget finished. Files: folder /
    observed_states=<n>.{pdf,png} (_zoom_stems). Returns the points drawn (size, method,
    n_transitions, transitions, share, share_lo, share_hi, contain, missing, missing_lo, missing_hi)."""
    columns = ["size", "method", "n_transitions", "transitions", "share", "share_lo", "share_hi",
               "contain", "missing", "missing_lo", "missing_hi"]  # fmt: skip
    if not {"true", "status"} <= set(res.columns) or res.empty:
        return pd.DataFrame(columns=columns)
    setup()
    data = res.assign(
        _share=pd.to_numeric(res["true"], errors="coerce") / target_value(res).where(lambda v: v > 0),
        _contain=_as_float(res["contain_all"].astype(object)) if "contain_all" in res else np.nan,
        _missing=pd.to_numeric(res["missing"], errors="coerce") if "missing" in res else np.nan,
    )
    kept = _set_methods(data, methods, data["_share"].notna())
    grid = sorted(data["n_transitions"].unique())
    samples = dict(zip(grid, sample_grid(data), strict=True))
    needed = summarize_needed(transitions_needed(data[data["method"].isin(kept)], target))
    needed = needed.set_index(["size", "method"])["median_t"] if len(needed) else pd.Series(dtype=float)
    present = [s for s in sizes if s in set(data["size"])]
    names = dict(zip(sizes, labels, strict=True))
    stems = _zoom_stems(data, present, folder)
    states = observed_states(data)
    points = []
    for size in present:
        rows = data[(data["size"] == size) & data["method"].isin(kept)]
        n_seeds = rows["seed"].nunique()
        drawn = {}  # method -> its points at this size
        for m in kept:
            g = rows[rows["method"] == m]
            out = []
            for n in grid:
                x = g[g["n_transitions"] == n]
                if len(x) < n_seeds or (x["status"] != "ok").any() or x["_share"].isna().any():
                    continue  # some seed did not finish (or has no value): no point
                share, lo, hi = median_ci(x["_share"].to_numpy(float))
                c = x["_contain"]
                miss = x["_missing"].dropna().to_numpy(float)
                mm = median_ci(miss) if len(miss) == len(x) else (np.nan, np.nan, np.nan)
                out.append({"size": size, "method": m, "n_transitions": n, "transitions": samples[n],
                            "share": share, "share_lo": lo, "share_hi": hi,
                            "contain": float((c == 1.0).sum()) if c.notna().all() else np.nan,
                            "missing": mm[0], "missing_lo": mm[1], "missing_hi": mm[2]})  # fmt: skip
            if out:
                drawn[m] = pd.DataFrame(out)
                points += out
        if not drawn:
            continue
        panels = ["share"]
        if any(d["contain"].notna().any() for d in drawn.values()):
            panels.append("contain")
        if any(d["missing"].notna().any() for d in drawn.values()):
            panels.append("missing")
        shown = list(drawn)
        ncol = 1 if len(shown) <= 6 else 2
        legend_rows = int(np.ceil(len(shown) / ncol))
        height = 2.3 * len(panels) + 0.75 + 0.17 * legend_rows
        fig, axes = plt.subplots(len(panels), 1, figsize=(0.68 * FULL_WIDTH, height), sharex=True)
        axes = np.atleast_1d(axes)
        handles = []
        for m in shown:
            st, d = styles[m], drawn[m]
            x = d["transitions"].to_numpy(float)
            t_star = needed.get((size, m), np.inf)
            reached = f"{samples_label(t_star)}" if np.isfinite(t_star) else "not reached"
            for ax, panel in zip(axes, panels, strict=True):
                if panel == "share":
                    y, lo, hi = (d[c].to_numpy(float) for c in ("share", "share_lo", "share_hi"))
                elif panel == "contain":
                    y, lo, hi = d["contain"].to_numpy(float), None, None
                else:
                    y, lo, hi = (d[c].to_numpy(float) for c in ("missing", "missing_lo", "missing_hi"))
                if np.isnan(y).all():
                    continue
                ax.plot(x, y, ls=st["ls"], marker=st["marker"], color=st["color"], markersize=4.5)
                if lo is not None:
                    ax.fill_between(x, lo, hi, color=st["color"], alpha=0.15, lw=0)
                if np.isfinite(t_star):
                    ax.axvline(t_star, color=st["color"], ls=st["ls"], lw=0.9, alpha=0.8)
            handles.append(Line2D([], [], ls=st["ls"], marker=st["marker"], color=st["color"],
                                  label=f"{st['label']}  ({target:.0%}-optimal: {reached})"))  # fmt: skip
        for ax, panel in zip(axes, panels, strict=True):
            if panel == "share":
                ax.axhline(target, color="grey", ls="--", lw=0.8)
                ax.set_title("policy quality (median, 95% CI over seeds)")
                ax.set_ylabel("optimal share\n(true value / $V^*_{obs}$)")
            elif panel == "contain":
                ax.set_ylim(-0.5, n_seeds + 0.5)
                ax.set_title("intervals contain the true kernel in every (s, a)")
                ax.set_ylabel(f"seeds (of {n_seeds})")
            else:
                ax.set_ylim(bottom=-0.5)
                ax.set_title("why: true edges the learned graph misses (median, 95% CI)")
                ax.set_ylabel("true edges missed")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("logged transitions (log scale)")
        axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: samples_label(v)))
        axes[-1].xaxis.set_minor_formatter(NullFormatter())
        n_states = states.get(size, np.nan)
        head = f"{int(n_states):,} observed states" if np.isfinite(n_states) else names[size]
        if np.isfinite(n_states) and stems[size].name != f"observed_states={int(n_states)}":
            head += f" ({names[size]})"  # two sizes with the same number of states
        fig.suptitle(f"{name}: {head}\nvertical lines: transitions for a {target:.0%}-optimal policy "
                     "(median over seeds)", fontsize=9)  # fmt: skip
        fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=7,
                   bbox_to_anchor=(0.5, 0.0))  # fmt: skip
        fig.tight_layout(rect=(0, (0.25 + 0.17 * legend_rows) / height, 1, 1))
        save(fig, stems[size])
    return pd.DataFrame(points, columns=columns)
