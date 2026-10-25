"""Chain picture of an MDP or interval MDP: nodes are states, edges are actions, one colour per action.

For every (state, action), the outcomes drawn are the most likely next states under the TRUE model
(probability >= threshold, at most max_outcomes): the most likely ones (ties included) as solid
edges, the less likely ones dashed. Every picture of one problem shows the same edges, labelled with
that model's probability (exact or estimated MDP) or interval [L, U] (interval MDP), so the pictures
can be compared. Nodes lie on a circle, or at given positions (e.g. the cells of a grid world).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from scm_to_mdp.core import IntervalMDP
from scm_to_mdp.plotting.style import save, setup

ACTION_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]


def draw(
    P_true: np.ndarray,
    model: IntervalMDP,
    state_labels: list[str],
    action_labels: list[str],
    title: str,
    stem,
    threshold: float = 0.05,
    max_outcomes: int = 3,
    positions: np.ndarray | None = None,
):
    """P_true [S, K, S'] over the model's actions; model.L == model.U for an exact MDP. positions
    [S, 2]: where the nodes go (scaled into the picture); default a circle."""
    setup()
    S, K, _ = P_true.shape
    exact = np.array_equal(model.L, model.U)
    if positions is None:
        angle = np.pi / 2 - 2 * np.pi * np.arange(S) / S
        pos = np.stack([np.cos(angle), np.sin(angle)], axis=1) * 1.0
    else:
        pos = np.asarray(positions, dtype=float)
        pos = pos - (pos.max(0) + pos.min(0)) / 2
        pos = pos / max(np.abs(pos).max(), 1e-12)  # inside [-1, 1]^2, centred
    size = 4.5 + 0.45 * S
    fig, ax = plt.subplots(figsize=(size, size))
    node_r = 0.055 + 0.6 / S
    points = node_r * size * 72 / 2.9  # node radius in points (the axes span 2.9 data units)
    for s in range(S):
        ax.add_patch(Circle(pos[s], node_r, fc="#dbe8f6", ec="k", lw=0.8, zorder=8))  # above edges and labels
        ax.text(*pos[s], state_labels[s], ha="center", va="center", fontsize=7, zorder=9)
    for s in range(S):
        for k in range(K):
            for t, solid in outcomes(P_true[s, k], threshold, max_outcomes):
                lo, hi = model.L[s, k, t], model.U[s, k, t]
                label = f"{lo:.2f}" if exact else f"[{lo:.2f}, {hi:.2f}]"
                style = dict(color=ACTION_COLOURS[k % len(ACTION_COLOURS)], lw=1.0, ls="-" if solid else "--")
                if t == s:
                    _self_loop(ax, pos[s], node_r, k, K, label, style)
                else:
                    _edge(ax, pos[s], pos[t], points, k, label, style)
    handles = [
        Line2D([], [], color=ACTION_COLOURS[k % len(ACTION_COLOURS)], lw=1.5, label=action_labels[k])
        for k in range(K)
    ]
    handles += [
        Line2D([], [], color="k", lw=1, label="most likely next state"),
        Line2D([], [], color="k", lw=1, ls="--", label="less likely next state"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    save(fig, stem)


def outcomes(row: np.ndarray, threshold: float, max_outcomes: int) -> list[tuple[int, bool]]:
    """The next states drawn for one (state, action): (index, solid), the likeliest first. Solid: the
    largest probability (every tie); dashed: the others, down to threshold, at most max_outcomes."""
    order = [int(t) for t in np.argsort(-row, kind="stable")[:max_outcomes] if row[t] >= threshold]
    top = row.max()
    return [(t, bool(np.isclose(row[t], top))) for t in order]


def _edge(ax, a, b, shrink_points, k, label, style):
    rad = 0.12 + 0.09 * k  # each action bends differently, so parallel edges stay apart
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=8,
            shrinkA=shrink_points,
            shrinkB=shrink_points,
            zorder=2,
            **style,
        )
    )
    mid = (a + b) / 2
    d = b - a
    control = mid + rad * np.array([d[1], -d[0]])  # the control point of matplotlib's arc3 curve
    point = 0.25 * a + 0.5 * control + 0.25 * b  # the middle of the curve
    ax.text(
        *point,
        label,
        fontsize=5,
        color=style["color"],
        ha="center",
        va="center",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
    )


def _self_loop(ax, centre, node_r, k, K, label, style):
    norm = np.linalg.norm(centre)
    outward = centre / norm if norm > 1e-9 else np.array([0.0, 1.0])  # the centre node of a grid: up
    spread = (k - (K - 1) / 2) * 0.75  # each action's loop at its own angle around the node
    c, s = np.cos(spread), np.sin(spread)
    direction = np.array([c * outward[0] - s * outward[1], s * outward[0] + c * outward[1]])
    loop_r = node_r * 0.45
    loop_centre = centre + direction * (node_r + loop_r * 0.7)
    ax.add_patch(Circle(loop_centre, loop_r, fill=False, zorder=1, **style))
    ax.text(
        *(loop_centre + direction * loop_r * 2.4),
        label,
        fontsize=5,
        color=style["color"],
        ha="center",
        va="center",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
    )
