"""Paper figures of clear27 for one domain: Experiments 1-4 (clear2027.tex, Section 5), 3 panels each (S9).

    .venv/bin/python experiments/clear27/paper_figures.py \
        --results results/server_oxford_2026-11-08/clear27 --domain sysadmin \
        --out results/clear27_paper/sysadmin

Every panel has the number of observed states on the x axis, except the two panels whose finding lives
along the number of transitions (Exp. 2 (c), Exp. 3 (c)). Every y axis is linear.

Metric. Normalised regret r = (V* - V^pi) / (V* - V_rand) at the nominal start: 0 = optimal, 1 = the
uniform random policy (the behaviour policy), > 1 = worse than random; the usual offline-RL
normalisation (Fu et al. 2020, D4RL).
    (a) transitions until r <= 0.05 at that budget and at every larger one; median over seeds and its
        95% order-statistic interval (plotting/stats.median_ci); "n.r." = the median seed never gets
        there within the largest budget.
    (b) r at the largest budget (500k); median and 95% interval over the seeds that finished it (their
        number when < 10).
    (c) a robustness check, a plain share of runs (no band).

Runs that hit a limit. A step over its time or memory limit is a stop; the larger budgets of that size
are then skipped. Two kinds:
    genuine   the method's own cost. In (a) a stopped seed has not reached the target; a cell whose
              median is not reached because most seeds stopped is drawn as x (not as n.r.), as is a
              size never run because of a stop at a smaller size. In (b) x = no seed finished the
              largest budget.
    spurious  a memout at <= 50 transitions whose recorded memory is more than twice the median of the
              same step in the finished runs of the same method and size (a measurement artefact of
              the shared server: e.g. 13.9 GB for a step that needs 0.7 GB). Its seed is left out of
              that cell, and a size skipped only because of it is left empty (to be rerun).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scm_to_mdp.plotting.figures import transitions_needed  # noqa: E402
from scm_to_mdp.plotting.stats import median_ci  # noqa: E402

TARGET = 0.05  # (a): r <= TARGET
DELTA = 0.05  # the confidence level of the S4 sets
SPURIOUS_MAX_BUDGET = 50
SPURIOUS_RATIO = 2.0
WIDTH = 6.0  # inches: the PMLR text width (jmlr.cls without 7x10)
STOP = ("memout", "timeout")

# colour = the entity (fixed across the figures where it recurs); marker and line style as a second encoding.
# Okabe & Ito (2008) colours, plus Tol's wine.
BLACK, BLUE, VERM, GREEN, PURPLE, ORANGE, SKY, WINE, GREY = (
    "#000000",
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#882255",
    "#777777",
)


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "lines.linewidth": 1.1,
            "lines.markersize": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


# ----------------------------------------------------------------------------------------------------- data
def load(root: Path, domain: str) -> pd.DataFrame:
    parts = []
    for f in sorted(root.glob(f"{domain}_*/**/results.csv")):
        source = f.relative_to(root).parts[0].removeprefix(f"{domain}_").rsplit("_", 2)[0]
        parts.append(pd.read_csv(f, low_memory=False).assign(source=source))
    if not parts:
        raise SystemExit(f"no results.csv under {root}/{domain}_*")
    res = pd.concat(parts, ignore_index=True)
    oracle = res["method"].eq("oracle -> MDP")  # every script runs the oracle: keep one copy
    res = pd.concat(
        [res[oracle].drop_duplicates(["size", "seed", "n_transitions"]), res[~oracle]], ignore_index=True
    )
    gap = res["V_star_obs"] - res["V_random"]
    extra = {
        "r": (res["V_star_obs"] - res["true"]) / gap,
        "promise_gap": (res["V_star_obs"] - res["promise"]) / gap,
    }
    for c in ("safe", "contain_all", "bracket_holds"):
        extra[c] = res[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return pd.concat([res.drop(columns=list(extra), errors="ignore"), pd.DataFrame(extra)], axis=1)


def classify_stops(res: pd.DataFrame):
    """Drop the seeds of spurious stops (module docstring). Returns (res, never), with
    never = {(method, size)} not run
    because of a genuine stop at the smallest budget of a smaller size."""
    ok = res[res["status"] == "ok"]
    cand = res[(res["status"] == "memout") & (res["n_transitions"] <= SPURIOUS_MAX_BUDGET)]
    bad = []
    for _, row in cand.iterrows():
        col = f"{row['failed_step']}_gb"
        if col in res:
            ref = ok.loc[(ok["method"] == row["method"]) & (ok["size"] == row["size"]), col].median()
            if np.isfinite(ref) and row[col] > SPURIOUS_RATIO * ref:
                bad.append((row["method"], row["size"], row["seed"]))
    bad = set(bad)
    print(f"spurious stops: {len(bad)} (method, size, seed) left out")
    keep = [k not in bad for k in zip(res["method"], res["size"], res["seed"], strict=True)]
    res = res[keep].copy()
    first = res["n_transitions"].min()
    early = res[res["status"].isin(STOP) & (res["n_transitions"] == first)]
    never = set()
    for method, g in res.groupby("method"):
        small = early.loc[early["method"] == method, "size"]
        for size, gs in g.groupby("size"):
            if (gs["status"] == "skipped").all() and (small < size).any():
                never.add((method, size))
    return res, never


def needed(res, methods):
    """(a) per (method, size): median, interval, seeds, and whether the median is 'not reached'
    because of stops."""
    sub = res[res["method"].isin(methods)]
    t = transitions_needed(sub, condition=sub["r"] <= TARGET)
    stopped = sub[sub["status"].isin(STOP)].groupby(["method", "size"])["seed"].unique().to_dict()
    rows = []
    ran = set(
        map(tuple, sub.loc[sub["status"] != "skipped", ["method", "size"]].drop_duplicates().to_numpy())
    )
    for (method, size), g in t.groupby(["method", "size"]):
        if (method, size) not in ran:  # never run (drawn by `never`, or left empty if the stop was spurious)
            continue
        med, lo, hi = median_ci(g["t_star"].fillna(np.inf).to_numpy(float))
        n_stop = len(set(stopped.get((method, size), [])))
        rows.append(
            (method, size, med, lo, hi, g["seed"].nunique(), not np.isfinite(med) and 2 * n_stop >= len(g))
        )
    return pd.DataFrame(rows, columns=["method", "size", "med", "lo", "hi", "k", "stopped"])


def at_budget(res, methods, column, budget):
    """(b) per (method, size): median and interval over the seeds that finished `budget`."""
    sub = res[res["method"].isin(methods) & (res["n_transitions"] == budget)]
    rows = []
    for (method, size), g in sub.groupby(["method", "size"]):
        done = g[g["status"] == "ok"]
        med, lo, hi = median_ci(done[column].to_numpy(float)) if len(done) else (np.nan,) * 3
        rows.append((method, size, med, lo, hi, done["seed"].nunique()))
    return pd.DataFrame(rows, columns=["method", "size", "med", "lo", "hi", "k"])


def share(res, methods, column, by="size"):
    sub = res[res["method"].isin(methods) & (res["status"] == "ok") & res[column].notna()]
    return sub.groupby(["method", by])[column].mean().rename("v").reset_index()


# --------------------------------------------------------------------------------------------------- drawing
def nice(v):
    """The smallest 'round' number (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^k) at least v."""
    e = 10 ** np.floor(np.log10(v))
    return float(next(m * e for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * e >= v - 1e-12))


def x_axis(ax, states, label=True):
    ax.set_xscale("log", base=2)
    xs = sorted(states.values())
    ax.set_xlim(xs[0] / 1.4, xs[-1] * 1.4)
    ax.set_xticks(xs[::2], [f"{int(v):,}" for v in xs[::2]])
    ax.set_xticks(xs[1::2], minor=True)
    ax.set_xticklabels([], minor=True)
    if label:
        ax.set_xlabel("observed states")


def dodge(x, i, n):
    return x * 2 ** ((i - (n - 1) / 2) * 0.06)


def errbars(ax, x, lo, hi, colour):
    ax.vlines(x, lo, hi, color=colour, lw=0.6, alpha=0.55, zorder=1)


def x_mark(ax, x, y, colour):
    ax.plot(x, y, ls="none", marker="x", color=colour, mew=1.1, ms=4.5, clip_on=False, zorder=4)


def draw_needed(ax, table, lines, states, never):
    """(a): medians on a linear axis in thousands; n.r. (open marker) and stopped (x) on a band
    above the values."""
    finite = table.loc[~table["stopped"], ["med", "hi"]].replace(np.inf, np.nan).to_numpy(float)
    top = nice(np.nanmax(finite) / 1000) if np.isfinite(finite).any() else 1.0
    band = top * 1.13
    ax.axhspan(top * 1.05, top * 1.21, color="#eeeeee", lw=0, zorder=0)
    for i, (method, (_, colour, marker, ls)) in enumerate(lines.items()):
        sub = table[table["method"] == method].sort_values("size")
        x = np.array([dodge(states[s], i, len(lines)) for s in sub["size"]])
        med = sub["med"].to_numpy(float) / 1000
        stop = sub["stopped"].to_numpy(bool)
        y = np.where(np.isfinite(med), med, band)
        lo = np.minimum(sub["lo"].to_numpy(float) / 1000, band)
        hi = np.minimum(sub["hi"].to_numpy(float) / 1000, band)
        errbars(ax, x[~stop], lo[~stop], hi[~stop], colour)
        ax.plot(x, np.where(stop, np.nan, y), color=colour, ls=ls, zorder=2)
        reached = np.isfinite(med)
        ax.plot(
            x[reached], y[reached], ls="none", marker=marker, color=colour, mec="white", mew=0.4, zorder=3
        )
        nr = ~reached & ~stop
        ax.plot(x[nr], y[nr], ls="none", marker=marker, mfc="white", mec=colour, mew=0.9, zorder=3)
        for xx in x[stop]:
            x_mark(ax, xx, band, colour)
        for s in sorted(s for m, s in never if m == method):
            x_mark(ax, dodge(states[s], i, len(lines)), band, colour)
    ax.set_ylim(0, top * 1.21)
    ticks = [t for t in ax.get_yticks() if 0 <= t <= top + 1e-9]
    ax.set_yticks(ticks + [band], [f"{t:g}" for t in ticks] + ["n.r."])
    ax.set_ylabel(r"transitions to $\tilde r \leq 0.05$ ($\times 10^3$)")


def draw_points(ax, table, lines, states, never, ymax=None):
    """(b): median and 95% interval per size; x at the top where some seed ran but none finished,
    or where the method never ran."""
    if ymax is None:
        ymax = nice(np.nanmax(table["med"].to_numpy(float)) * 1.15)
    for i, (method, (_, colour, marker, ls)) in enumerate(lines.items()):
        sub = table[table["method"] == method].sort_values("size")
        x = np.array([dodge(states[s], i, len(lines)) for s in sub["size"]])
        done = sub["k"].to_numpy() > 0
        errbars(ax, x[done], sub["lo"][done], np.minimum(sub["hi"][done], ymax), colour)
        ax.plot(
            x, np.where(done, sub["med"], np.nan), color=colour, ls=ls, marker=marker, mec="white", mew=0.4
        )
        for xx, (_, r) in zip(x, sub.iterrows(), strict=True):
            if 0 < r["k"] < 10:
                ax.annotate(
                    f"{int(r['k'])}",
                    (xx, r["med"]),
                    xytext=(1.5, 1.5),
                    textcoords="offset points",
                    fontsize=5.5,
                    color=colour,
                )
        for xx in x[~done]:
            x_mark(ax, xx, 0.95 * ymax, colour)
        for s in sorted(s for m, s in never if m == method):
            x_mark(ax, dodge(states[s], i, len(lines)), 0.95 * ymax, colour)
    ax.set_ylim(0, ymax)


def draw_share(ax, table, lines, states=None):
    """(c): a share of runs per size (states given) or per number of transitions; the 1 - delta line."""
    for i, (method, (_, colour, marker, ls)) in enumerate(lines.items()):
        sub = table[table["method"] == method].sort_values(table.columns[1])
        x = sub.iloc[:, 1].to_numpy(float)
        if states is not None:
            x = np.array([dodge(states[s], i, len(lines)) for s in sub.iloc[:, 1]])
        ax.plot(x, 100 * sub["v"], color=colour, ls=ls, marker=marker, mec="white", mew=0.4)
    delta_line(ax)
    ax.set_ylim(-3, 104)


def delta_line(ax):
    ax.axhline(100 * (1 - DELTA), color=GREY, lw=0.7, ls=(0, (4, 2)), zorder=0)
    ax.text(
        0.99,
        100 * (1 - DELTA) - 2,
        r"$1-\delta$",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="top",
        fontsize=6.5,
        color=GREY,
    )


def random_line(ax, text="random policy", x=0.02, below=False):
    ax.axhline(1.0, color=GREY, lw=0.7, ls=(0, (4, 2)), zorder=0)
    ax.annotate(
        text,
        (x, 1.0),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, -2 if below else 1.5),
        textcoords="offset points",
        fontsize=6.5,
        color=GREY,
        va="top" if below else "bottom",
        ha="left",
    )


def letter(ax, text):
    ax.set_title(text, loc="left", pad=3)


def handle(label, colour, marker=None, ls="-", **kw):
    return Line2D([], [], color=colour, marker=marker, ls=ls, mec="white", mew=0.4, label=label, **kw)


def legend(fig, handles, ncol):
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
        handlelength=2.3,
        columnspacing=1.3,
        handletextpad=0.5,
    )


def marks():
    return [
        Line2D([], [], ls="none", marker="o", mfc="white", mec="#444444", label="n.r.: median not reached"),
        Line2D([], [], ls="none", marker="x", color="#444444", label="stopped: time or memory limit"),
    ]


def three():
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25))
    fig.subplots_adjust(wspace=0.45, left=0.075, right=0.99, top=0.9, bottom=0.22)
    return fig, axes


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


def k(v):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


# -------------------------------------------------------------------------------------------------- figures
def exp1(res, states, never, budget, out):
    a = {
        "true graph -> IMDP (IA)": ("IMDP, box", BLUE, "o", "-"),
        "true graph -> IMDP (exact)": ("IMDP, exact", VERM, "s", "-"),
        "true graph -> MDP (plug-in)": ("plug-in MDP", BLACK, "D", "--"),
        "true graph -> Factored SPIBB": ("Factored SPIBB", PURPLE, "^", "--"),
    }
    best = {"true graph -> MDP (best, IA)": ("best MDP, box", ORANGE, "v", ":")}
    imdp = list(a)[:2]
    fig, ax = three()
    draw_needed(ax[0], needed(res, list(a)), a, states, never)
    letter(ax[0], "(a) sample efficiency")
    # (b) how far the promise is below V*: at the largest budget (solid) and at 10k (dashed)
    gaps, tables = {}, []
    for b, ls in ((budget, "-"), (10_000, "--")):
        for m in imdp:
            lab, c, mk, _ = a[m]
            gaps[f"{m}@{b}"] = (lab, c, mk, ls)
            tables.append(at_budget(res, [m], "promise_gap", b).assign(method=f"{m}@{b}"))
    never_b = {(f"{m}@{b}", s) for m, s in never for b in (budget, 10_000)}
    draw_points(ax[1], pd.concat(tables, ignore_index=True), gaps, states, never_b)
    random_line(ax[1], "random policy", x=0.45)
    ax[1].set_ylabel(r"promise gap $(V^\star-\underline{V})\,/\,(V^\star-V_{\rm rand})$")
    letter(ax[1], "(b) tightness of the promise")
    # (c) the promise (or the method's own value estimate) <= the true value, all budgets pooled
    c = dict(a) | best
    draw_share(ax[2], share(res, list(c), "safe"), c, states)
    ax[2].set_ylabel(r"estimate $\leq$ true value (% of runs)")
    letter(ax[2], "(c) soundness")
    for x in ax:
        x_axis(x, states)
    hs = [handle(*v) for v in list(a.values()) + list(best.values())] + marks()
    hs += [
        handle(f"(b) {k(budget)} transitions", "#444444"),
        handle("(b) 10k transitions", "#444444", ls="--"),
    ]
    legend(fig, hs, ncol=3)
    save(fig, out, "exp1_true_graph")


GRAPHS = {
    "true graph": ("true graph", BLACK, "P"),
    "full-fact": ("full-fact", PURPLE, "D"),
    "sparse": ("sparse", GREEN, "^"),
    "flat": ("flat", SKY, "v"),
}


def exp2(res, states, never, budget, out):
    imdp, plug = {}, {}
    for g, (lab, c, m) in GRAPHS.items():
        imdp["flat IMDP" if g == "flat" else f"{g} -> IMDP (IA)"] = (lab, c, m, "-")
        plug["flat -> MDP (plug-in)" if g == "flat" else f"{g} -> MDP (plug-in)"] = (lab, c, m, "--")
    both = imdp | plug
    fig, ax = three()
    draw_needed(ax[0], needed(res, list(imdp)), imdp, states, never)
    letter(ax[0], "(a) sample efficiency, IMDP")
    draw_points(ax[1], at_budget(res, list(both), "r", budget), both, states, never)
    random_line(ax[1])
    ax[1].set_ylabel(rf"normalised regret $\tilde r$ at {k(budget)}")
    letter(ax[1], f"(b) regret at {k(budget)} transitions")
    for x in ax[:2]:
        x_axis(x, states)
    # (c) along the transitions, pooled over sizes: the event "the true kernel lies in the set",
    # which every promise rests on
    sp = "sparse -> IMDP (IA)"
    others = [m for m in imdp if m != sp]
    t = share(res, others, "contain_all", by="n_transitions").groupby("n_transitions")["v"].mean()
    ax[2].plot(t.index, 100 * t, color=GREY, lw=1.1)
    for col, ls, mk in (("safe", ":", None), ("bracket_holds", "-.", None), ("contain_all", "-", "^")):
        t = share(res, [sp], col, by="n_transitions")
        ax[2].plot(t["n_transitions"], 100 * t["v"], color=GREEN, ls=ls, marker=mk, mec="white", mew=0.4)
    delta_line(ax[2])
    ax[2].set_ylim(-3, 104)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("transitions")
    ax[2].set_ylabel("share of runs (%)")
    letter(ax[2], "(c) the event behind the promise")
    hs = [handle(lab, c, m) for lab, c, m in GRAPHS.values()]
    hs += [handle("IMDP (box)", "#444444"), handle("plug-in MDP", "#444444", ls="--")] + marks()
    hs += [
        handle("(c) sparse: truth inside the set", GREEN, "^"),
        handle("(c) sparse: value bracket holds", GREEN, ls="-."),
        handle("(c) sparse: promise ≤ true value", GREEN, ls=":"),
        handle("(c) other graphs: truth inside the set", GREY),
    ]
    legend(fig, hs, ncol=4)
    save(fig, out, "exp2_assumed_graphs")


def exp3(res, states, never, budget, out):
    lines = {
        "true graph -> IMDP (IA)": ("true graph", BLACK, "P", ":"),
        "full-fact -> IMDP (IA)": ("full-fact", PURPLE, "D", ":"),
        "DYNOTEARS -> IMDP (IA)": ("DYNOTEARS", BLUE, "o", "-"),
        "DYNOTEARS (lagged) -> IMDP (IA)": ("DYNOTEARS (lagged)", BLUE, "o", "--"),
        "PCMCI+ -> IMDP (IA)": ("PCMCI+", VERM, "s", "-"),
        "PCMCI+ (lagged) -> IMDP (IA)": ("PCMCI+ (lagged)", VERM, "s", "--"),
        "Strehl (fixed hull) -> IMDP (IA)": ("SLF-Rmax", ORANGE, "X", "-"),
        "Met-Rmax (fixed hull) -> IMDP (IA)": ("Met-Rmax", WINE, "*", "-"),
    }
    fig, ax = three()
    draw_needed(ax[0], needed(res, list(lines)), lines, states, never)
    letter(ax[0], "(a) sample efficiency, IMDP")
    draw_points(ax[1], at_budget(res, list(lines), "r", budget), lines, states, never)
    ax[1].set_ylabel(rf"normalised regret $\tilde r$ at {k(budget)}")
    letter(ax[1], f"(b) regret at {k(budget)} transitions")
    for x in ax[:2]:
        x_axis(x, states)
    # (c) graph recovery along the transitions, at one size
    size = 7 if 7 in states else sorted(states)[len(states) // 2]
    learners = [
        "DYNOTEARS (lagged) -> IMDP (IA)",
        "PCMCI+ (lagged) -> IMDP (IA)",
        "Strehl (fixed hull) -> IMDP (IA)",
        "Met-Rmax (fixed hull) -> IMDP (IA)",
    ]
    sub = res[(res["size"] == size) & res["method"].isin(learners) & (res["status"] == "ok")]
    for m in learners:
        g = sub[sub["method"] == m].groupby("n_transitions")[["recall", "precision"]].mean()
        _, c, mk, _ = lines[m]
        ax[2].plot(g.index, 100 * g["recall"], color=c, ls="-", marker=mk, mec="white", mew=0.4)
        ax[2].plot(g.index, 100 * g["precision"], color=c, ls=":")
    ax[2].set_xscale("log")
    ax[2].set_ylim(-3, 104)
    ax[2].set_xlabel("transitions")
    ax[2].set_ylabel("recall, precision of the edges (%)")
    letter(ax[2], f"(c) graph recovery, {int(states[size])} states")
    hs = [handle(*v) for v in lines.values()] + marks()
    hs += [handle("(c) recall", "#444444"), handle("(c) precision", "#444444", ls=":")]
    legend(fig, hs, ncol=4)
    save(fig, out, "exp3_learned_graphs")


def exp4(res, states, never, budget, out, best="DYNOTEARS (lagged)"):
    lines = {
        f"{best} -> IMDP (IA)": ("ours: IMDP (box)", BLACK, "P", "-"),
        f"{best} -> MDP (plug-in)": ("ours: plug-in MDP", BLACK, "o", "--"),
        "DiFaRL": ("DiFaRL", ORANGE, "X", "-"),
        "FOCUS": ("FOCUS", SKY, "h", "-"),
        "SL-SPIBB": ("SL-SPIBB", GREEN, "^", "-"),
        "flat SPIBB": ("flat SPIBB", PURPLE, "v", "-"),
        "offline DQN": ("offline DQN", VERM, "s", "-"),
        "offline CQL": ("offline CQL", BLUE, "D", "-"),
    }
    fig, ax = three()
    draw_needed(ax[0], needed(res, list(lines)), lines, states, never)
    letter(ax[0], "(a) sample efficiency")
    draw_points(ax[1], at_budget(res, list(lines), "r", budget), lines, states, never)
    random_line(ax[1])
    ax[1].set_ylabel(rf"normalised regret $\tilde r$ at {k(budget)}")
    letter(ax[1], f"(b) regret at {k(budget)} transitions")
    draw_share(ax[2], share(res, list(lines), "safe"), lines, states)
    ax[2].set_ylabel(r"estimate $\leq$ true value (% of runs)")
    letter(ax[2], "(c) soundness of the estimate")
    for x in ax:
        x_axis(x, states)
    legend(fig, [handle(*v) for v in lines.values()] + marks(), ncol=5)
    save(fig, out, "exp4_offline_rl")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=Path("results/clear27"))
    p.add_argument("--domain", default="sysadmin")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = a.out or Path("results/clear27_paper") / a.domain
    style()
    res, never = classify_stops(load(a.results, a.domain))
    states = res.groupby("size")["n_observed_states"].median().to_dict()
    budget = int(res["n_transitions"].max())
    for fig in (exp1, exp2, exp3, exp4):
        fig(res, states, never, budget, out)


if __name__ == "__main__":
    main()
