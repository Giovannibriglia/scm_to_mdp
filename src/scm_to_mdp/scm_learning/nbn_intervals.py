"""Per-factor probability intervals with NBN (step S4).

For the chosen parents of every next-state variable, NBN counts the data and builds a Clopper-Pearson
interval for every value of every parent row. The confidence budget delta is split over all
intervals of all trajectories (union bound, sampling="trajectory"), so all intervals hold together
with probability >= 1 - delta. A parent row never seen in the data gets [0, 1]. The guarantee holds
when the parents are fixed before seeing the data (the true graph) and contain the true parents.
With a graph learned from the same data, the parent set is itself chosen by the data: the 1 - delta
is then conditional on that choice, not a guarantee (a union bound over every candidate parent set,
or a split of the data, would restore it). With a wrong graph the intervals bound a policy-dependent
average, not the mechanism.

Widening (still one graph G; `widen`, `k`): a family that misses a true parent is the failure that
breaks the promise, and a superset of the true parents is harmless (it has the same conditional).
So for every x_i, the inputs not in Pa_G(x_i) are ranked by the BIC of Pa_G(x_i) + {x} (best first,
ties in input order) and the first k are used:

    "pad"   one family per variable, Pa_G(x_i) + {x_1, ..., x_k}: one DAG, a superset of G. Its
            intervals bound the mechanism when the true parents are inside it (up to k missing
            parents, if they are the ones ranked first). Cost: more parent rows, so fewer visits
            per row (wider intervals, more unseen rows).
    "hull"  the families Pa_G(x_i), Pa_G(x_i) + {x_1}, ..., Pa_G(x_i) + {x_k}, each with its own
            intervals, and at every (s, a) the hull: min of the lower ends, max of the upper ends. It
            bounds the mechanism when one family contains the true parents (one missing parent among
            the first k). Each family keeps more visits per row than "pad"; the price is the union.

delta is split over every family (NBN's HypothesisSet.factor_sets), so all families' intervals hold
together. The ranking uses the same data as the intervals: as for G itself, the level is conditional
on that choice, unless n_candidates restores it.

Post-selection (`n_candidates`): when the parent sets were chosen from the same data, out of a family F_i
of candidate parent sets per variable fixed before seeing the data, the intervals are built at
delta / |F_i| (a union bound over the family: on that event the intervals of every candidate hold
together, so those of the selected one hold whatever the choice; Agarwal, Brantley, Jiang, Kakade, Sun,
"RL: Theory and Algorithms": a union bound over a family fixed in advance, as in Lemma A.20, p. 198, with
the random counts handled as in Lemma 6.2, p. 76, of which the delta / (n (n + 1)) split is a variant;
the guarantee concerns the candidates that contain the true parents). n_candidates: 1 (parents fixed
in advance: the true graph, the full graph), "inputs" (every subset of the slice-t inputs:
2^{number of inputs}), "inputs+next" (also the other next-state variables, for discoveries with
same-step edges), or a number.

FactorBounds.info["added_parents"] says which inputs were added; info also holds the
number of parent rows that can occur ("s4_rows") and the share of them with no data ("s4_unseen").
"""

import json

import numpy as np
from nbn.structure import HypothesisSet

from scm_to_mdp.core import Dataset, FactorBounds, Graph
from scm_to_mdp.data import slice_inputs
from scm_to_mdp.discovery.bic_search import bic, feasible_rows
from scm_to_mdp.scm_learning.families import family as _family
from scm_to_mdp.scm_learning.families import rows as family_rows
from scm_to_mdp.scm_learning.families import same_step, seen_info, seen_pairs

WIDEN = (None, "pad", "hull")


def ranked_additions(dataset: Dataset, graph: Graph, k: int) -> dict[str, list[str]]:
    """For every state variable x: the first k inputs not among the parents of x', ranked by the BIC
    of the family with that input added (best first; ties in the order of dataset.inputs)."""
    pairs = list(zip(dataset.state_vars, dataset.next_vars, strict=True))
    options = {v: [x for x in dataset.inputs if x not in graph.parents[v]] for v, _ in pairs}
    families = {v: [_family(dataset, (*graph.parents[v], x)) for x in options[v]] for v, _ in pairs}
    targets = {v_next: families[v] for v, v_next in pairs if families[v]}
    if not targets:
        return {v: [] for v, _ in pairs}
    counts = HypothesisSet(dataset.variables, targets).counts(dataset.tensors())
    out = {}
    for v, v_next in pairs:
        rows = [feasible_rows(f, dataset.cards, dataset.action_vars, dataset.actions) for f in families[v]]
        scores = [bic(counts[(v_next, f)].numpy(), n) for f, n in zip(families[v], rows, strict=True)]
        order = sorted(range(len(scores)), key=lambda j: (-scores[j], j))  # best first; ties: input order
        out[v] = [options[v][j] for j in order[:k]]
    return out


CANDIDATES = ("inputs", "inputs+next")


def candidate_count(dataset: Dataset, n_candidates: int | str = 1) -> int:
    """|F_i| of the module docstring: 1, 2^{n inputs} ("inputs"), 2^{n inputs + n state - 1}
    ("inputs+next"), or the number given."""
    if n_candidates == "inputs":
        return 2 ** len(dataset.inputs)
    if n_candidates == "inputs+next":
        return 2 ** (len(dataset.inputs) + len(dataset.state_vars) - 1)
    if isinstance(n_candidates, str) or int(n_candidates) < 1:
        raise ValueError(f"n_candidates must be >= 1 or one of {CANDIDATES}, not {n_candidates!r}")
    return int(n_candidates)


def learn(
    dataset: Dataset,
    graph: Graph,
    delta: float = 0.05,
    widen: str | None = None,
    k: int = 2,
    n_candidates: int | str = 1,
) -> FactorBounds:
    """lo, hi [S, K, n, C]: interval of P(x_i' = v | parents) at every state s and action k; a factor
    with same-step parents is a conditional (FactorBounds.conditional), and lo, hi hold its hull.
    widen: None (the parents of graph), "pad" or "hull" (see the module docstring), with k slice-t
    inputs added (the same-step parents stay those of graph). n_candidates: the union bound over the
    candidate parent sets when graph was chosen from the data (module docstring)."""
    if widen not in WIDEN:
        raise ValueError(f"widen must be one of {WIDEN}, not {widen!r}")
    if widen is not None and k < 1:
        raise ValueError(f"k must be >= 1 when widening, not {k}")
    pairs = list(zip(dataset.state_vars, dataset.next_vars, strict=True))
    added = ranked_additions(dataset, graph, k) if widen else {v: [] for v, _ in pairs}
    base = {v: _family(dataset, graph.parents[v]) for v, _ in pairs}
    if widen == "pad":
        families = {v: [_family(dataset, (*base[v], *added[v]))] for v, _ in pairs}
    elif widen == "hull":
        families = {v: [base[v]] + [_family(dataset, (*base[v], x)) for x in added[v]] for v, _ in pairs}
    else:
        families = {v: [base[v]] for v, _ in pairs}
    hs = HypothesisSet(dataset.variables, {v_next: families[v] for v, v_next in pairs})
    counts = hs.counts(dataset.tensors())
    n_family = candidate_count(dataset, n_candidates)
    sets = hs.factor_sets(
        counts,
        builder="interval",
        delta=delta / n_family,
        sampling="trajectory",
        method="clopper_pearson",
    )
    X = slice_inputs(dataset)  # [S, K, n_inputs]
    S, K = X.shape[:2]
    C = max(dataset.state_cards)
    lo, hi = np.ones((S, K, len(pairs), C)), np.zeros((S, K, len(pairs), C))
    conditional = {}
    for i, (v, v_next) in enumerate(pairs):
        card = dataset.cards[v]
        js = same_step(dataset, families[v][0])  # the same for every family of x_i (widening adds inputs)
        R = int(np.prod([dataset.state_cards[j] for j in js]))
        lo_i, hi_i = np.ones((S, K, R, C)), np.zeros((S, K, R, C))
        for parents in families[v]:  # the hull over the families (one family: its intervals)
            lower, upper = sets[(v_next, parents)].bounds()  # [parent rows, card], rows in C order
            row = family_rows(dataset, parents, X)  # [S, K, R]
            lo_i[..., :card] = np.minimum(lo_i[..., :card], lower.double().numpy()[row])
            hi_i[..., :card] = np.maximum(hi_i[..., :card], upper.double().numpy()[row])
        lo_i[..., card:] = 0.0  # padding values
        lo[:, :, i], hi[:, :, i] = lo_i.min(2), hi_i.max(2)  # a valid interval of the marginal
        if js:
            conditional[i] = (js, lo_i, hi_i)
    rows = seen = 0  # parent rows that can occur, and those with data (evaluation.interval_metrics)
    for v, v_next in pairs:
        for parents in families[v]:
            rows += feasible_rows(parents, dataset.cards, dataset.action_vars, dataset.actions)
            seen += int((counts[(v_next, parents)].sum(-1) > 0).sum())
    pairs_seen = seen_pairs(dataset, families, counts, X)
    info = {"s4_rows": rows, "s4_unseen": 1 - seen / rows, "s4_candidates": n_family, **seen_info(pairs_seen)}
    if widen:
        info |= {"widen": widen, "added_parents": json.dumps(added)}
    return FactorBounds(
        lo, hi, dataset.state_cards, dataset.B, dataset.reward, info=info, conditional=conditional,
        seen=pairs_seen,
    )
