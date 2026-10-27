"""Saving the learned models of one seed, to look at them and to reuse them without a new run.

For the first seed of a sweep, at every size and at the largest number of rollouts (or at every
number: Experiment(save_models="all")), every method's model goes to

    <out_dir>/models/<size_label>/n=<n_rollouts>/<method>/
        model.npz      the model the solver used, the policy and its promise: the factor intervals
                       (lo, hi [S, K, n, C], FactorBounds) for a causal method, the transition
                       intervals (L, U [S, K, S], IntervalMDP) for flat IMDP and flat MDP, the
                       Q-table (Q [S, K]) for offline DQN and CQL, nothing for DiFaRL (its action
                       groups are in results.csv); with the reward
                       (R, r_next, terminal), the action vectors and pi, V
        cbn.json       a causal method: the learned causal Bayesian network over the observed
                       columns (below), with the true parents for comparison
        cbn.{pdf,png}  its 2-slice graph: edges found (black), missed (red, dashed), extra (blue)
        mdp.{pdf,png}  the chain picture of the (interval) MDP the model induces (plotting/mdp_graph.py:
                       nodes are states, one colour per action, the likeliest next states solid and
                       the less likely ones dashed, labelled with the model's probability or interval;
                       a causal method's FactorBounds as the box of its factor intervals), up to
                       Experiment.pictures_max_states observed states. The edges are chosen under the
                       true observed kernel, so every picture of one problem shows the same edges.
                       States with only a column and a row (one agent on a grid) are laid out on
                       their grid, the others on a circle.
        info.json      method, size, seed, number of rollouts, status, promise at the start
    <out_dir>/models/<size_label>/ground_truth.{pdf,png}   the true MDP over the observed states (the
                                                         observed kernel when a variable is hidden)

load(folder) reads a folder back: {"model": FactorBounds | IntervalMDP | None, "solution": Solution,
"cbn": dict | None, "info": dict}; product_bounds.compile(model) gives the interval MDP of a causal
method's FactorBounds again.

What cbn.json is. The data determine a causal Bayesian network: the graph (the slice-t parents of
every next-state variable x', and its same-step parents y') and the conditional distributions
P(x' | parents). They do not
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
    from scm_to_mdp.data import slice_inputs
    from scm_to_mdp.scm_learning.families import family
    from scm_to_mdp.scm_learning.families import rows as family_rows

    df = data.df
    columns = {v: i for i, v in enumerate(data.state_vars)}
    S, K = model.lo.shape[:2]
    X = slice_inputs(data)  # [S, K, n_inputs]: the value of every input at every (s, a_k)
    cpds = {}
    for v, given in parents.items():
        pa = family(data, given)  # slice-t parents, then same-step parents (their table's order)
        i, card = columns[v], data.cards[v]
        cards = [data.cards[p.removesuffix(NEXT_MARK)] for p in pa]
        code = family_rows(data, pa, X)  # [S, K, R] the table row at (s, k, same-step values r)
        lo_i, hi_i = (
            model.conditional[i][1:]
            if i in model.conditional
            else (model.lo[:, :, i, None], model.hi[:, :, i, None])
        )
        counts = df.groupby(list(pa) + [v + NEXT_MARK]).size() if pa else df.groupby(v + NEXT_MARK).size()
        table, done = [], set()
        for (s, k, r), row_index in np.ndenumerate(code):
            if row_index in done:
                continue
            done.add(row_index)
            row = tuple(int(x) for x in np.unravel_index(row_index, cards)) if pa else ()
            n_v = [int(counts.get(row + (val,) if pa else val, 0)) for val in range(card)]
            n = sum(n_v)
            table.append(
                {
                    "parents": dict(zip(pa, row, strict=True)),
                    "n": n,
                    "estimate": [c / n for c in n_v] if n else None,
                    "lo": lo_i[s, k, r, :card].round(6).tolist(),
                    "hi": hi_i[s, k, r, :card].round(6).tolist(),
                }
            )
        cpds[v + NEXT_MARK] = {
            "parents": list(pa),
            "rows": sorted(table, key=lambda t: list(t["parents"].values())),
        }
    return {
        "variables": {c: int(data.cards[c]) for c in data.inputs},
        "state_vars": list(data.state_vars),
        "action_vars": list(data.action_vars),
        "parents": {v + NEXT_MARK: list(pa) for v, pa in parents.items()},
        "true_parents": None
        if true_parents is None
        else {v + NEXT_MARK: list(pa) for v, pa in true_parents.items()},
        "cpds": cpds,
    }


def draw_cbn(parents: dict, true_parents: dict | None, inputs: list[str], state_vars: list[str], title, stem):
    """The 2-slice graph: inputs (slice t) on the left, next-state variables on the right; a same-step
    edge y' -> x' is an arc on the right (an undirected pair when compared with the truth, as SHD)."""
    import matplotlib.pyplot as plt

    from scm_to_mdp.plotting.style import save, setup

    setup()
    true_parents = true_parents or {}
    left = {x: (0.0, -i) for i, x in enumerate(inputs)}
    shift = (len(inputs) - len(state_vars)) / 2
    right = {v: (1.0, -i - shift) for i, v in enumerate(state_vars)}
    fig, ax = plt.subplots(figsize=(4.2, 0.9 + 0.32 * len(inputs)))

    def pair(p, v):  # a same-step edge as the pair it stands for (either direction)
        return frozenset((p.removesuffix(NEXT_MARK), v))

    learned_pairs = {pair(p, v) for v in state_vars for p in parents.get(v, ()) if p.endswith(NEXT_MARK)}
    true_pairs = {pair(p, v) for v in state_vars for p in true_parents.get(v, ()) if p.endswith(NEXT_MARK)}
    for v in state_vars:
        learned = {p for p in parents.get(v, ()) if p in left}
        true = {p for p in true_parents.get(v, ()) if p in left}
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
    for edge in sorted(learned_pairs | true_pairs, key=sorted):  # same-step edges: arcs on the right
        if not true_parents or (edge in learned_pairs and edge in true_pairs):
            colour, style = "black", "-"
        elif edge in true_pairs:
            colour, style = "#CC0000", "--"
        else:
            colour, style = "#0072B2", "-"
        a, b = sorted(edge)
        (x0, y0), (x1, y1) = right[a], right[b]
        ax.annotate(
            "",
            (x1 + 0.03, y1),
            (x0 + 0.03, y0),
            arrowprops={
                "arrowstyle": "-",
                "color": colour,
                "ls": style,
                "lw": 0.8,
                "connectionstyle": "arc3,rad=-0.6",
            },
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
        if solution.probs is not None:
            arrays["probs"] = solution.probs
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
        true_parents = {  # observed parents only (a same-step parent y' when y is observed)
            v: tuple(p for p in truth.true_parents[v] if p.removesuffix(NEXT_MARK) in data.cards)
            for v in data.state_vars
        }
    if isinstance(model, FactorBounds):
        arrays |= {"kind": "FactorBounds", "lo": model.lo, "hi": model.hi, "cards": np.array(model.cards)}
        for i, (js, lo_i, hi_i) in model.conditional.items():  # factors with same-step parents
            arrays |= {f"cond{i}_js": np.array(js), f"cond{i}_lo": lo_i, f"cond{i}_hi": hi_i}
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
    elif type(model).__name__ == "QTable":  # offline DQN / CQL: the learned Q on every observed state
        arrays |= {"kind": "QTable", "Q": model.Q}
    else:
        arrays |= {"kind": "none"}
    np.savez_compressed(folder / "model.npz", **arrays)
    (folder / "info.json").write_text(json.dumps(info | {"observed": observed}, indent=1, default=str))
    title = f"{info['method']}, {info['size_label']}, n={info['n_rollouts']}"
    if data.B.shape[0] <= pictures_max_states and isinstance(model, (FactorBounds, IntervalMDP)):
        _draw_mdp(model, data, truth, title, folder / "mdp")


def _labels(truth, data):
    if truth.hidden:  # the observed states: their values, as truth.state_name writes them
        states = ["".join(str(int(x)) for x in row) for row in data.B]
    else:
        states = [truth.state_name(s) for s in range(truth.n_states)]
    actions = [
        ", ".join(f"{a}={int(v)}" for a, v in zip(data.action_vars, vec, strict=True)) for vec in data.actions
    ]
    return states, actions


def positions(data) -> np.ndarray | None:
    """[S, 2] node positions on the grid when the observed state is one column x and one row y (in that
    order, row 0 at the top); None otherwise (a circle)."""
    names = [v.rstrip("0123456789") for v in data.state_vars]
    if names != ["x", "y"]:
        return None
    return np.stack([data.B[:, 0], -data.B[:, 1]], axis=1).astype(float)


def _point_or_box(model):
    """A causal method's FactorBounds as an interval MDP: the exact MDP of point estimates, or the box
    of its factor intervals; an IntervalMDP as it is. Returns (model, note for the title)."""
    from scm_to_mdp.compilation import product_bounds

    if isinstance(model, FactorBounds):
        point = np.array_equal(model.lo, model.hi) and all(
            np.array_equal(lo, hi) for _, lo, hi in model.conditional.values()
        )
        if point:
            return product_bounds.exact(model), "\n(the MDP of the point estimates)"
        return product_bounds.compile(model), "\n(box of the factor intervals)"
    return model, " (point estimates)" if np.array_equal(model.L, model.U) else ""


def _draw_mdp(model, data, truth, title, stem):
    from scm_to_mdp.plotting import mdp_graph

    states, actions = _labels(truth, data)
    model, note = _point_or_box(model)
    P_ref = truth.observed_kernel()[:, truth.actions_of(data)]  # the edges: the true observed kernel
    mdp_graph.draw(P_ref, model, states, actions, title + note, stem, positions=positions(data))


def draw_ground_truth(truth, data, title, stem):
    """The true MDP over the observed states (the observed kernel when a variable is hidden)."""
    from scm_to_mdp.plotting import mdp_graph

    P_true = truth.observed_kernel()[:, truth.actions_of(data)]
    states, actions = _labels(truth, data)
    exact = IntervalMDP(P_true, P_true, data.reward)
    mdp_graph.draw(P_true, exact, states, actions, title, stem, positions=positions(data))


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
        conditional = {
            int(key[4:-3]): (tuple(int(j) for j in z[key]), z[key[:-3] + "_lo"], z[key[:-3] + "_hi"])
            for key in z.files
            if key.startswith("cond") and key.endswith("_js")
        }
        model = FactorBounds(
            z["lo"], z["hi"], [int(c) for c in z["cards"]], z["B"], reward, conditional=conditional
        )
    elif kind == "IntervalMDP":
        model = IntervalMDP(z["L"], z["U"] if "U" in z else z["L"], reward)
    elif kind == "QTable":
        from scm_to_mdp.baselines.offline_q import QTable

        model = QTable(z["Q"], {})
    solution = Solution(z["V"], z["pi"], z["probs"] if "probs" in z else None) if "pi" in z else None
    return {"model": model, "solution": solution, "cbn": network, "info": info}
