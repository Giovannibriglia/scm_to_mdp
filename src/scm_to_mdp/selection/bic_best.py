"""Selection rule "bic_best" (step S3): keep the candidate graph with the best BIC on the data.

For scored parent sets (BIC search): the parent set of every x' with the best score, among the
combinations whose same-step edges (x_j' -> x_i') form a DAG. The total score is a sum over the
targets, so the best acyclic combination is found exactly by dynamic programming over the sets of
next-state variables (Silander & Myllymaki, UAI 2006): best(U) = max over the last variable v in U of
best(U - {v}) + the best score of v whose same-step parents lie in U - {v}. Cost 2^n * n for n state
variables (only when some candidate has a same-step parent; otherwise every target on its own).

For a class of graphs (PCMCI+, GES): some edges are directed, others are undirected (the data do
not tell their direction). The rule:

1. Enumerate every DAG of the class. The undirected edges split into connected "chain components";
   each component is oriented on its own, and an orientation comes from an order of its nodes.
   For a GES CPDAG, keep only orientations that create no new v-structure (a -> c <- b with a, b
   not adjacent). A component with more than MAX_ORDERS node orders is sampled, not enumerated.
2. Score every DAG with BIC on the data and keep the best.
3. With table CPDs and complete data BIC is score-equivalent: the DAGs of one Markov equivalence
   class get the same BIC (Koller & Friedman, 2009, Thm 18.3, p. 807). Our penalty counts only the
   parent rows that can occur (discovery/bic_search.py), so two DAGs of a class can differ by their
   penalty alone, and the best one is then kept. Exact ties are broken by the BIC of the transition
   model, sum_i BIC(x_i' | its slice-t parents), which is not invariant;
   scores within 1e-6 count as equal, and a remaining tie goes to the orientation whose sorted
   edge list comes first (so the choice does not depend on the order of Python sets).

The full BIC is a sum over families, and a family depends on the orientation of one component only,
so the best DAG is the best orientation of every component, chosen separately.
"""

import itertools

import numpy as np
import pandas as pd

from scm_to_mdp.core import NEXT_MARK, Candidates, Dataset, Graph, encode
from scm_to_mdp.discovery.bic_search import bic, feasible_rows

MAX_ORDERS = 20_000  # node orders tried per chain component


def select(candidates: Candidates, dataset: Dataset) -> Graph:
    """The graph with the best BIC among the candidates."""
    if candidates.kind == "scored_sets":
        parents = best_acyclic(candidates.scores)
        same_step = sum(p.endswith(NEXT_MARK) for ps in parents.values() for p in ps)
        return Graph(parents, report={**candidates.report, "intra_next": same_step})
    graph = choose_dag(
        set(candidates.directed), set(candidates.undirected), dataset, candidates.keep_vstructures
    )
    return Graph(graph.parents, report={**candidates.report, **graph.report})


def best_acyclic(scores: dict[str, list[tuple[tuple[str, ...], float]]]) -> dict[str, tuple[str, ...]]:
    """The parent set of every target (best total score) such that the same-step edges form a DAG
    (see the module docstring). Ties: the first candidate in the list order."""
    targets = list(scores)
    position = {v + NEXT_MARK: j for j, v in enumerate(targets)}

    def same_step_mask(parents):
        return sum(1 << position[p] for p in parents if p in position)

    if not any(same_step_mask(c) for sets in scores.values() for c, _ in sets):
        return {v: max(sets, key=lambda t: t[1])[0] for v, sets in scores.items()}
    n, full = len(targets), (1 << len(targets)) - 1
    best_within, choice_within = [], []  # [target][mask]: the best score whose same-step parents lie in mask
    for v in targets:
        score, choice = np.full(full + 1, -np.inf), np.full(full + 1, -1, dtype=np.int64)
        for c, (parents, value) in enumerate(scores[v]):
            m = same_step_mask(parents)
            if value > score[m]:
                score[m], choice[m] = value, c
        for bit in range(n):  # the best over subsets: max over masks contained in each mask
            for mask in range(full + 1):
                if mask >> bit & 1 and score[mask ^ (1 << bit)] > score[mask]:
                    score[mask], choice[mask] = score[mask ^ (1 << bit)], choice[mask ^ (1 << bit)]
        best_within.append(score)
        choice_within.append(choice)
    best, last = np.full(full + 1, -np.inf), np.full(full + 1, -1, dtype=np.int64)
    best[0] = 0.0
    for mask in range(1, full + 1):
        for j in range(n):
            if mask >> j & 1:
                value = best[mask ^ (1 << j)] + best_within[j][mask ^ (1 << j)]
                if value > best[mask]:
                    best[mask], last[mask] = value, j
    parents, mask = {}, full
    while mask:
        j = int(last[mask])
        rest = mask ^ (1 << j)
        parents[targets[j]] = scores[targets[j]][int(choice_within[j][rest])][0]
        mask = rest
    return {v: parents[v] for v in targets}


def family_bic(
    df: pd.DataFrame, node: str, parents: tuple[str, ...], cards: dict[str, int], n_rows: int | None = None
) -> float:
    """BIC of one family, from the data. cards: number of values of every column; n_rows: the rows of
    the parent table that can occur (bic_search.feasible_rows; default: all)."""
    y = df[node].values
    parent_cards = [cards[p] for p in parents]
    rows = encode(df[list(parents)].values, parent_cards) if parents else np.zeros(len(y), dtype=np.int64)
    counts = np.zeros((int(np.prod(parent_cards)), cards[node]))
    np.add.at(counts, (rows, y), 1)
    return bic(counts, n_rows)


def chain_components(undirected: set[frozenset]) -> list[set[str]]:
    """Connected components of the undirected edges."""
    neighbours: dict[str, set[str]] = {}
    for edge in undirected:
        a, b = tuple(edge)
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    components, seen = [], set()
    for start in neighbours:
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            v = stack.pop()
            if v not in component:
                component.add(v)
                stack.extend(neighbours[v] - component)
        seen |= component
        components.append(component)
    return components


def creates_new_vstructure(orientation, directed, adjacent) -> bool:
    """True if the new edges make some p -> child <- q with p, q not adjacent."""
    new_parents: dict[str, list[str]] = {}
    for a, b in orientation:
        new_parents.setdefault(b, []).append(a)
    for child, new in new_parents.items():
        every_parent = new + [a for a, b in directed if b == child]
        for p, q in itertools.combinations(every_parent, 2):
            if (p in new or q in new) and frozenset((p, q)) not in adjacent:
                return True
    return False


def orientations(component, undirected, directed, adjacent, check_vstructures, rng) -> list[frozenset]:
    """Every acyclic orientation of the component's edges (allowed by the v-structure rule)."""
    edges = [tuple(e) for e in undirected if set(e) <= component]
    nodes = sorted(component)
    if np.prod(range(1, len(nodes) + 1), dtype=float) <= MAX_ORDERS:
        orders = itertools.permutations(nodes)
    else:
        orders = (tuple(rng.permutation(nodes)) for _ in range(MAX_ORDERS))
    found = set()
    for order in orders:
        position = {v: p for p, v in enumerate(order)}
        orientation = frozenset((a, b) if position[a] < position[b] else (b, a) for a, b in edges)
        if orientation in found:
            continue
        if check_vstructures and creates_new_vstructure(orientation, directed, adjacent):
            continue
        found.add(orientation)
    return list(found)


def choose_dag(
    directed: set[tuple], undirected: set[frozenset], dataset: Dataset, check_vstructures: bool, seed: int = 0
) -> Graph:
    """The DAG of the class with the best BIC (ties: best transition BIC). Returns the parents of every
    x_i': its slice-t neighbours (whatever the direction of the edge), then its same-step parents x_j'
    (edges among next-state nodes, as oriented in the DAG), and a report: number of DAGs in the class,
    number tied, same-step edges."""
    df, rng = dataset.df, np.random.default_rng(seed)
    cards = {c: dataset.cards[c.removesuffix(NEXT_MARK)] for c in df.columns}
    adjacent = {frozenset(e) for e in directed} | set(undirected)
    next_nodes, actions = set(dataset.next_vars), dataset.action_vars

    def parents_of(node, edges):
        return tuple(sorted(a for a, b in edges if b == node))

    def slice_t_neighbours(node, edges):  # the time order: an input next to x' is a parent of x'
        return tuple(x for x in dataset.inputs if (x, node) in edges or (node, x) in edges)

    def family(v, parents):  # penalty on the rows that can occur (one-hot actions: bic_search)
        return family_bic(df, v, parents, cards, feasible_rows(parents, cards, actions, dataset.actions))

    def scores(edges, nodes):
        full = sum(family(v, parents_of(v, edges)) for v in nodes)
        transition = sum(family(v, slice_t_neighbours(v, edges)) for v in nodes if v in next_nodes)
        return full, transition

    chosen, n_dags, n_tied = set(directed), 1, 1
    for component in chain_components(undirected):
        options = orientations(component, undirected, directed, adjacent, check_vstructures, rng)
        if not options:  # a large component whose sampled orders all made a new v-structure
            options = orientations(component, undirected, directed, adjacent, False, rng)
        n_dags *= len(options)
        touched = component | {b for a, b in directed if a in component}  # families that change
        scored = [(scores(set(directed) | set(o), touched), o) for o in options]
        best_full = max(score[0] for score, _ in scored)
        tied = [(score, o) for score, o in scored if score[0] >= best_full - 1e-6]
        n_tied *= len(tied)
        best_transition = max(score[1] for score, _ in tied)  # tie-break: transition BIC, then
        still = [o for score, o in tied if score[1] >= best_transition - 1e-6]  # the first edge list
        chosen |= set(min(still, key=lambda o: sorted(o)))  # (deterministic, whatever the set order)

    # GES does not know the time order, so it may orient an edge x' -> x: an edge is a time step, and
    # every slice-t neighbour of x' is a parent of x' whatever its direction (PCMCI+ only has x -> x')
    same_step = acyclic_same_step({(a, b) for a, b in chosen if a in next_nodes and b in next_nodes})
    parents = {
        v: slice_t_neighbours(v_next, chosen)
        + tuple(u for u in dataset.next_vars if (u, v_next) in same_step)
        for v, v_next in zip(dataset.state_vars, dataset.next_vars, strict=True)
    }
    report = {"dags": n_dags, "tied": n_tied, "intra_next": len(same_step)}
    return Graph(parents, report=report)


def acyclic_same_step(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """The same-step edges without cycles: a class's DAG has none, but DYNOTEARS's thresholded weights
    may (its acyclicity is a penalty); an edge that closes a cycle is dropped, in sorted order."""
    kept: set[tuple[str, str]] = set()
    for a, b in sorted(edges):
        reach, stack = set(), [b]  # does b already reach a through the kept edges?
        while stack:
            u = stack.pop()
            if u not in reach:
                reach.add(u)
                stack.extend(y for x, y in kept if x == u)
        if a not in reach:
            kept.add((a, b))
    return kept
