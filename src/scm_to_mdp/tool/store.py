"""Saving the learned models of one seed, to look at them and to reuse them without a new run.

For the first seed of a sweep, at every size and at the largest number of rollouts (or at every
number: Experiment(save_models="all")), every method's model goes to

    <out_dir>/models/<size_label>/n=<n_rollouts>/<method>/
        model.npz      the model the solver used, the policy and its promise: the factor intervals
                       (lo, hi [S, K, n, C], FactorBounds) for a causal method, the transition
                       intervals (L, U [S, K, S], IntervalMDP) for flat RMDP and flat MDP, nothing
                       for DiFaRL (model-free: its action groups are in info.json); with the reward
                       (R, r_next, terminal), the action vectors and pi, V
        cbn.json       a causal method: the learned causal Bayesian network over the observed
                       columns (below), with the true parents for comparison
        cbn.{pdf,png}  its 2-slice graph: edges found (black), missed (red, dashed), extra (blue)
        mdp.{pdf,png}  a picture of the (interval) MDP, up to Experiment.pictures_max_states states
                       (edges: the likeliest outcomes under the true model; with a hidden variable,
                       under the model's own interval midpoints)
        info.json      method, size, seed, number of rollouts, status, promise at the start
    <out_dir>/models/<size_label>/ground_truth.{pdf,png}   the true MDP (small problems only)

load(folder) reads a folder back: {"model": FactorBounds | IntervalMDP | None, "solution": Solution,
"cbn": dict | None, "info": dict}; product_bounds.compile(model) gives the interval MDP of a causal
method's FactorBounds again.

What cbn.json is. The data determine a causal Bayesian network: the graph (the slice-t parents of
every next-state variable x') and the conditional distributions P(x' | parents). They do not
determine a structural causal model: many SCMs (structural equations with exogenous noise) induce
the same network, and the network is all the pipeline uses. cbn.json stores, for every x' and every
parent row that can occur (every combination of the state parents; for the action parents, the
action vectors of the data), the number of transitions n seen in that row, the estimate n_v / n
(None if n = 0) and the interval [lo, hi] of P(x' = v | row) that the method used.
"""

import json
from pathlib import Path

import numpy as np

from scm_to_mdp.core import NEXT_MARK, FactorBounds, IntervalMDP, Reward, Solution

MAX_DENSE_ENTRIES = 20_000_000  # S * K * S above which L and U of a flat model are not saved (~320 MB)


def method_folder(name: str) -> str:
    return name.replace(" -> ", "_").replace(" ", "_").replace("+", "plus")


# ---- the causal Bayesian network ------------------------------------------------------------------
def cbn(model: FactorBounds, data, parents: dict, true_parents: dict | None) -> dict:
    """The learned network as a dict (see the module docstring)."""
    df, inputs = data.df, data.inputs
    columns = {v: i for i, v in enumerate(data.state_vars)}
    S, K = model.lo.shape[:2]
    # the value of every input at every (s, a_k): state values from B, action values from data.actions
    values = np.concatenate(
        [np.repeat(model.B[:, None, :], K, axis=1), np.repeat(data.actions[None, :, :], S, axis=0)], axis=2
    )  # [S, K, n + n_a], columns in the order of data.inputs (state vars, then action vars)
    cpds = {}
    for v, pa in parents.items():
        i, card = columns[v], data.cards[v]
        idx = [inputs.index(p) for p in pa]
        rows = values[:, :, idx].reshape(S * K, len(idx))
        _, first = np.unique(rows, axis=0, return_index=True)
        counts = df.groupby(list(pa) + [v + NEXT_MARK]).size() if pa else df.groupby(v + NEXT_MARK).size()
        table = []
        for j in sorted(first):
            s, k = divmod(int(j), K)
            row = tuple(int(x) for x in rows[j])
            n_v = [int(counts.get(row + (val,) if pa else val, 0)) for val in range(card)]
            n = sum(n_v)
            table.append(
                {
                    "parents": dict(zip(pa, row, strict=True)),
                    "n": n,
                    "estimate": [c / n for c in n_v] if n else None,
                    "lo": model.lo[s, k, i, :card].round(6).tolist(),
                    "hi": model.hi[s, k, i, :card].round(6).tolist(),
                }
            )
        cpds[v + NEXT_MARK] = {"parents": list(pa), "rows": table}
    return {
        "variables": {c: int(data.cards[c]) for c in inputs},
        "state_vars": list(data.state_vars),
        "action_vars": list(data.action_vars),
        "parents": {v + NEXT_MARK: list(pa) for v, pa in parents.items()},
        "true_parents": None
        if true_parents is None
        else {v + NEXT_MARK: list(pa) for v, pa in true_parents.items()},
        "cpds": cpds,
    }


def draw_cbn(parents: dict, true_parents: dict | None, inputs: list[str], state_vars: list[str], title, stem):
    """The 2-slice graph: inputs (slice t) on the left, next-state variables on the right."""
    import matplotlib.pyplot as plt

    from scm_to_mdp.plotting.style import save, setup

    setup()
    true_parents = true_parents or {}
    left = {x: (0.0, -i) for i, x in enumerate(inputs)}
    shift = (len(inputs) - len(state_vars)) / 2
    right = {v: (1.0, -i - shift) for i, v in enumerate(state_vars)}
    fig, ax = plt.subplots(figsize=(4.2, 0.9 + 0.32 * len(inputs)))
    for v in state_vars:
        learned, true = set(parents.get(v, ())), {p for p in true_parents.get(v, ()) if p in left}
        for p in sorted(learned | true):
            if not true_parents or (p in learned and p in true):
                colour, style = "black", "-"  # found
            elif p in true:
                colour, style = "#CC0000", "--"  # missed
            else:
                colour, style = "#0072B2", "-"  # extra
            (x0, y0), (x1, y1) = left[p], right[v]
            ax.annotate(
                "",
                (x1 - 0.04, y1),
                (x0 + 0.04, y0),
                arrowprops={"arrowstyle": "->", "color": colour, "ls": style, "lw": 0.8},
            )
    for x, (px, py) in left.items():
        ax.text(px, py, x, ha="right", va="center", fontsize=7)
    for v, (px, py) in right.items():
        ax.text(px, py, v + NEXT_MARK, ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-len(inputs), 1)
    ax.axis("off")
    note = "black: found, red dashed: missed, blue: extra" if true_parents else ""
    ax.set_title(f"{title}\n{note}", fontsize=7.5)
    save(fig, Path(stem))


# ---- one method's folder ----------------------------------------------------------------------------
def save(folder: Path, result, data, truth, info: dict, pictures_max_states: int = 0) -> None:
    """Write model.npz, info.json and, when they apply, cbn.json, cbn.{pdf,png} and mdp.{pdf,png}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model, solution = result.model, result.solution
    arrays: dict = {}
    if solution is not None:
        arrays |= {"pi": solution.pi, "V": solution.V}
    reward = getattr(model, "reward", None)
    if reward is not None:
        arrays |= {"R": reward.R, "r_next": reward.r_next, "terminal": reward.terminal}
    arrays |= {"actions": data.actions, "B": data.B}
    observed = {
        "state_vars": list(data.state_vars),
        "action_vars": list(data.action_vars),
        "cards": {c: int(data.cards[c]) for c in data.inputs},
    }
    true_parents = None
    if truth.true_parents is not None:
        true_parents = {
            v: tuple(p for p in truth.true_parents[v] if p in data.cards) for v in data.state_vars
        }
    if isinstance(model, FactorBounds):
        arrays |= {"kind": "FactorBounds", "lo": model.lo, "hi": model.hi, "cards": np.array(model.cards)}
        if result.graph is not None:
            network = cbn(model, data, result.graph.parents, true_parents)
            (folder / "cbn.json").write_text(json.dumps(network, indent=1))
            draw_cbn(
                result.graph.parents,
                true_parents,
                data.inputs,
                data.state_vars,
                f"{info['method']}, {info['size_label']}, n={info['n_rollouts']}",
                folder / "cbn",
            )
    elif isinstance(model, IntervalMDP):
        exact = np.array_equal(model.L, model.U)
        if model.L.size <= MAX_DENSE_ENTRIES:
            arrays |= {"kind": "IntervalMDP", "L": model.L} | ({} if exact else {"U": model.U})
        else:
            arrays |= {"kind": "IntervalMDP (L, U not saved: too large)"}
            info = info | {"note": f"L and U have {model.L.size:,} entries each; not saved"}
    else:
        arrays |= {"kind": "none"}
    np.savez_compressed(folder / "model.npz", **arrays)
    (folder / "info.json").write_text(json.dumps(info | {"observed": observed}, indent=1, default=str))
    if data.B.shape[0] <= pictures_max_states and isinstance(model, (FactorBounds, IntervalMDP)):
        _draw_mdp(
            model,
            data,
            truth,
            f"{info['method']}, {info['size_label']}, n={info['n_rollouts']}",
            folder / "mdp",
        )


def _labels(truth, data):
    if truth.hidden:  # the observed states: their values, as truth.state_name writes them
        states = ["".join(str(int(x)) for x in row) for row in data.B]
    else:
        states = [truth.state_name(s) for s in range(truth.n_states)]
    actions = [
        "+".join(a for a, on in zip(data.action_vars, vec, strict=True) if on) or "none"
        for vec in data.actions
    ]
    return states, actions


def _draw_mdp(model, data, truth, title, stem):
    from scm_to_mdp.compilation import product_bounds
    from scm_to_mdp.plotting import mdp_graph

    states, actions = _labels(truth, data)
    if isinstance(model, FactorBounds):
        model, title = product_bounds.compile(model), title + "\n(box of the factor intervals)"
    elif np.array_equal(model.L, model.U):
        title += " (point estimates)"
    if truth.hidden:  # the true P is over the full states: pick the edges by the model's midpoints
        mid = (model.L + model.U) / 2
        P_ref = mid / np.maximum(mid.sum(-1, keepdims=True), 1e-12)
    else:
        P_ref = truth.P[:, truth.actions_of(data)]
    mdp_graph.draw(P_ref, model, states, actions, title, stem)


def draw_ground_truth(truth, data, title, stem):
    from scm_to_mdp.plotting import mdp_graph

    P_true = truth.P[:, truth.actions_of(data)]
    states, actions = _labels(truth, data)
    mdp_graph.draw(P_true, IntervalMDP(P_true, P_true, data.reward), states, actions, title, stem)


# ---- reading a folder back --------------------------------------------------------------------------
def load(folder) -> dict:
    """{"model", "solution", "cbn", "info"} from a folder written by save()."""
    folder = Path(folder)
    z = np.load(folder / "model.npz")
    info = json.loads((folder / "info.json").read_text())
    cbn_path = folder / "cbn.json"
    network = json.loads(cbn_path.read_text()) if cbn_path.exists() else None
    reward = Reward(z["R"], z["r_next"], z["terminal"]) if "R" in z else None
    kind = str(z["kind"])
    model = None
    if kind == "FactorBounds":
        model = FactorBounds(z["lo"], z["hi"], [int(c) for c in z["cards"]], z["B"], reward)
    elif kind == "IntervalMDP":
        model = IntervalMDP(z["L"], z["U"] if "U" in z else z["L"], reward)
    solution = Solution(z["V"], z["pi"]) if "pi" in z else None
    return {"model": model, "solution": solution, "cbn": network, "info": info}
