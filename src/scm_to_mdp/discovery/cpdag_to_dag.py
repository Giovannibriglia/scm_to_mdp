"""How ONE graph is chosen when a discovery algorithm returns a class of graphs (step S3).

PCMCI+ and GES return a mixed graph: some edges are directed, others are undirected (the data do
not tell their direction). The rule used here:

1. Enumerate every DAG of the class. The undirected edges split into connected "chain components";
   each component is oriented on its own, and an orientation comes from an order of its nodes.
   For a GES CPDAG, keep only orientations that create no new v-structure (a -> c <- b with a, b
   not adjacent). A component with more than MAX_ORDERS node orders is sampled, not enumerated.
2. Score every DAG with BIC on the data and keep the best.
3. BIC is score-equivalent: the DAGs of one Markov equivalence class tie. Ties are broken by the
   BIC of the transition model, sum_i BIC(m_i' | its slice-t parents), which is not invariant.

The full BIC is a sum over families, and a family depends on the orientation of one component only,
so the best DAG is the best orientation of every component, chosen separately.
"""

import itertools

import numpy as np
import pandas as pd

from scm_to_mdp.core import NEXT_MARK, Dataset, Graph, encode
from scm_to_mdp.discovery.bic_search import bic

MAX_ORDERS = 20_000  # node orders tried per chain component


def family_bic(df: pd.DataFrame, node: str, parents: tuple[str, ...], cards: dict[str, int]) -> float:
    """BIC of one family, from the data. cards: number of values of every column."""
    y = df[node].values
    parent_cards = [cards[p] for p in parents]
    rows = encode(df[list(parents)].values, parent_cards) if parents else np.zeros(len(y), dtype=np.int64)
    counts = np.zeros((int(np.prod(parent_cards)), cards[node]))
    np.add.at(counts, (rows, y), 1)
    return bic(counts)


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
    """The DAG of the class with the best BIC (ties: best transition BIC). Returns the slice-t parents
    of every m_i' and a report: number of DAGs in the class, number tied, edges among next-state nodes."""
    df, rng = dataset.df, np.random.default_rng(seed)
    cards = {c: dataset.cards[c.removesuffix(NEXT_MARK)] for c in df.columns}
    adjacent = {frozenset(e) for e in directed} | set(undirected)
    next_nodes, inputs = set(dataset.next_vars), set(dataset.inputs)

    def parents_of(node, edges):
        return tuple(sorted(a for a, b in edges if b == node))

    def scores(edges, nodes):
        full = sum(family_bic(df, v, parents_of(v, edges), cards) for v in nodes)
        transition = sum(
            family_bic(df, v, tuple(p for p in parents_of(v, edges) if p in inputs), cards)
            for v in nodes
            if v in next_nodes
        )
        return full, transition

    chosen, n_dags, n_tied = set(directed), 1, 1
    for component in chain_components(undirected):
        options = orientations(component, undirected, directed, adjacent, check_vstructures, rng)
        n_dags *= max(len(options), 1)
        touched = component | {b for a, b in directed if a in component}  # families that change
        scored = [(scores(set(directed) | set(o), touched), o) for o in options]
        best_full = max(score[0] for score, _ in scored)
        tied = [(score, o) for score, o in scored if score[0] >= best_full - 1e-6]
        n_tied *= len(tied)
        chosen |= set(max(tied, key=lambda t: t[0][1])[1])  # tie-break: transition BIC

    parents = {
        v: tuple(x for x in dataset.inputs if (x, v_next) in chosen)
        for v, v_next in zip(dataset.state_vars, dataset.next_vars, strict=True)
    }
    intra_next = sum(1 for a, b in chosen if a in next_nodes and b in next_nodes)
    return Graph(parents, report={"dags": n_dags, "tied": n_tied, "intra_next": intra_next})
