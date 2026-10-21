import numpy as np
import pandas as pd

from scm_to_mdp.discovery import bic_search, cpdag_to_dag, ges, oracle, pcmci_plus
from scm_to_mdp.evaluation import structure_distances


def test_bic_search_recovers_the_ring(env, dataset):
    graph = bic_search.fit(dataset, max_parents=3)
    assert structure_distances(graph, env.true_parents(dataset), dataset)["shd"] == 0


def test_oracle(env, dataset):
    truth = env.true_parents(dataset)
    assert oracle.fit(dataset, truth).parents == truth


def test_pcmci_and_ges_return_a_graph(dataset):
    for graph in [pcmci_plus.fit(dataset, 0.01), ges.fit(dataset)]:
        assert set(graph.parents) == set(dataset.state_vars)
        assert graph.report["dags"] >= 1 and 1 <= graph.report["tied"] <= graph.report["dags"]


def test_orientations_of_an_undirected_chain():
    # x - y - z with nothing else: 3 DAGs without a new v-structure (x -> y <- z is excluded)
    undirected = {frozenset(("x", "y")), frozenset(("y", "z"))}
    adjacent = set(undirected)
    found = cpdag_to_dag.orientations(
        {"x", "y", "z"}, undirected, set(), adjacent, True, np.random.default_rng(0)
    )
    assert len(found) == 3
    assert frozenset({("x", "y"), ("z", "y")}) not in found


def test_chain_components():
    undirected = {frozenset(("a", "b")), frozenset(("c", "d"))}
    assert sorted(map(sorted, cpdag_to_dag.chain_components(undirected))) == [["a", "b"], ["c", "d"]]


def test_family_bic_prefers_the_true_parent():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 2000)
    y = np.where(rng.random(2000) < 0.9, x, 1 - x)
    df = pd.DataFrame({"x": x, "y": y, "z": rng.integers(0, 2, 2000)})
    cards = {"x": 2, "y": 2, "z": 2}
    assert cpdag_to_dag.family_bic(df, "y", ("x",), cards) > cpdag_to_dag.family_bic(df, "y", ("z",), cards)
