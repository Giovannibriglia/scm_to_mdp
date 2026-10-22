import numpy as np
import pandas as pd

from scm_to_mdp.discovery import bic_search, ges, oracle, pcmci_plus
from scm_to_mdp.evaluation import structure_distances
from scm_to_mdp.selection import bic_best, given
from scm_to_mdp.selection import bic_best as cpdag_to_dag  # the DAGs of a class: selection/bic_best.py


def test_bic_search_recovers_the_ring(env, dataset):
    candidates = bic_search.fit(dataset, max_parents=3)
    assert candidates.kind == "scored_sets" and candidates.report["candidates"] == 42
    graph = bic_best.select(candidates, dataset)
    assert structure_distances(graph, env.true_parents(), dataset)["shd"] == 0


def test_oracle(env, dataset):
    truth = env.true_parents()
    assert given.select(oracle.fit(dataset, truth), dataset).parents == truth


def test_pcmci_and_ges_return_a_graph(dataset):
    for candidates in [pcmci_plus.fit(dataset, 0.01), ges.fit(dataset)]:
        assert candidates.kind == "graph_class"
        graph = bic_best.select(candidates, dataset)
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


def test_given_refuses_a_class(dataset):
    import pytest

    from scm_to_mdp.core import Candidates

    with pytest.raises(ValueError, match="fully directed"):
        given.select(Candidates("graph_class", undirected=frozenset({frozenset(("m0", "a0"))})), dataset)


def test_bic_best_tie_break_does_not_depend_on_set_order(dataset):
    """Two orientations of one undirected edge between two state columns tie on every score; the
    choice must be the same whatever the order of the Python sets."""
    from scm_to_mdp.core import Candidates

    edge = frozenset(("m0", "m1"))
    picks = {
        tuple(
            sorted(
                bic_best.select(
                    Candidates("graph_class", frozenset(), frozenset({edge})), dataset
                ).parents.items()
            )
        )
        for _ in range(5)
    }
    assert len(picks) == 1


def test_feasible_rows_of_one_hot_actions():
    actions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # one action at a time, or none
    cards = {"x": 5, "a": 2, "b": 2, "c": 2}
    assert bic_search.feasible_rows(("x",), cards, ["a", "b", "c"], actions) == 5
    assert bic_search.feasible_rows(("x", "a"), cards, ["a", "b", "c"], actions) == 10
    assert bic_search.feasible_rows(("x", "a", "b", "c"), cards, ["a", "b", "c"], actions) == 20  # not 40


def test_bic_keeps_every_one_hot_action_parent():
    """x' = x moved by one of 4 one-hot moves (as the taxi's square). With the penalty on all 2^4
    action rows BIC keeps only 3 of the 4 parents here (checked); on the 5 rows that can occur it
    keeps them all."""
    from scm_to_mdp.core import Reward
    from scm_to_mdp.data import read_dataset

    rng = np.random.default_rng(0)
    n, card = 3000, 10
    x = rng.integers(0, card, n)
    k = rng.integers(0, 6, n)  # 4 moves and 2 actions that do nothing to x
    moves = np.eye(6, dtype=np.int64)[k]
    step = np.array([1, -1, 3, -3, 0, 0])[k]
    df = pd.DataFrame({"x": x, **{f"m{j}": moves[:, j] for j in range(6)}, "x'": (x + step) % card})
    reward = lambda B, A: Reward(np.zeros((len(B), len(A))), np.zeros(len(B)), np.zeros(len(B), bool))  # noqa: E731
    data = read_dataset(
        df, [f"m{j}" for j in range(6)], {"x": card, **{f"m{j}": 2 for j in range(6)}}, reward
    )
    graph = bic_best.select(bic_search.fit(data, max_parents=5), data)
    assert set(graph.parents["x"]) == {"x", "m0", "m1", "m2", "m3"}


def test_a_backward_edge_still_gives_a_parent(dataset):
    """GES does not know the time order and may return x' -> x; the edge is still a time step, so x
    is a parent of x' (as an undirected x - x' edge, whatever orientation wins)."""
    from scm_to_mdp.core import Candidates

    backward = Candidates(
        "graph_class", frozenset({("m0'", "m0"), ("m1'", "a1")}), frozenset({frozenset(("m2", "m2'"))})
    )
    parents = bic_best.select(backward, dataset).parents
    assert "m0" in parents["m0"] and "a1" in parents["m1"] and "m2" in parents["m2"]
