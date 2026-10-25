import numpy as np
import pandas as pd

from scm_to_mdp.discovery import bic_search, ges, oracle, pcmci_plus
from scm_to_mdp.evaluation import structure_distances
from scm_to_mdp.selection import bic_best, given
from scm_to_mdp.selection import bic_best as cpdag_to_dag  # the DAGs of a class: selection/bic_best.py


def test_bic_search_recovers_the_ring(env, dataset):
    candidates = bic_search.fit(dataset, max_parents=3)
    assert (
        candidates.kind == "scored_sets" and candidates.report["candidates"] == 42
    )  # 4 inputs + 2 other next-state variables, at most 3
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


def test_feasible_rows_count_the_action_values_seen():
    """BIC's penalty counts the parent rows that can occur: every value of the state parents (at t or
    t+1), times the action combinations of the data (here (1, 1) never happens)."""
    actions = np.array([[0, 0], [1, 0], [0, 1]])
    cards = {"x": 5, "y": 3, "a": 2, "b": 2}
    assert bic_search.feasible_rows(("x",), cards, ["a", "b"], actions) == 5
    assert bic_search.feasible_rows(("x", "a"), cards, ["a", "b"], actions) == 10
    assert bic_search.feasible_rows(("x", "a", "b"), cards, ["a", "b"], actions) == 15  # not 20
    assert bic_search.feasible_rows(("x", "y'"), cards, ["a", "b"], actions) == 15  # a same-step parent


def test_best_acyclic_choice_of_same_step_parents():
    """Each target prefers the other as a same-step parent; both together are a cycle, so the best
    acyclic combination keeps one edge, the one with the larger total."""
    scores = {
        "x": [(("x",), -10.0), (("x", "y'"), -4.0)],
        "y": [(("y",), -10.0), (("y", "x'"), -5.0)],
    }
    assert bic_best.best_acyclic(scores) == {"x": ("x", "y'"), "y": ("y",)}  # -14 beats -15
    no_same_step = {"x": [(("x",), -1.0), ((), -2.0)], "y": [((), -1.0)]}
    assert bic_best.best_acyclic(no_same_step) == {"x": ("x",), "y": ()}


def test_bic_finds_the_coupled_axes_of_the_lake():
    """Gym's slip moves the row OR the column: x' and y' are dependent given (state, action), and BIC
    (free to use same-step parents) links them, in either direction."""
    from scm_to_mdp.envs import FrozenLake

    env = FrozenLake("4x4", 1, 0, axes="joint")  # slippery, Gym's law
    data = env.ground_truth().learner_spec.read(env.sample(3000, 30, "random"))  # ~23k transitions; at
    # ~8k BIC still leaves the axes independent (its penalty: y0' multiplies x0's rows by 4)
    graph = bic_best.select(bic_search.fit(data), data)
    assert "y0'" in graph.parents["x0"] or "x0'" in graph.parents["y0"]
    assert graph.report["intra_next"] == 1


def test_fast_g2_equals_tigramite():
    from tigramite.independence_tests.gsquared import Gsquared

    rng = np.random.default_rng(0)
    for trial in range(100):
        n_z, T = int(rng.integers(0, 4)), int(rng.integers(5, 300))
        cards = rng.integers(2, 5, 2 + n_z)
        array = np.stack([rng.integers(0, c, T) for c in cards]).astype(float)
        if trial % 3 == 0:
            array[1] = (array[0] + rng.integers(0, 2, T)) % cards[1]  # a dependence
        xyz = np.array([0, 1] + [2] * n_z)
        slow, fast = Gsquared(), pcmci_plus.FastGsquared()
        g_slow, g_fast = slow.get_dependence_measure(array, xyz), fast.get_dependence_measure(array, xyz)
        assert slow._temp_dof == fast._temp_dof and np.isclose(g_slow, g_fast)


def test_a_backward_edge_still_gives_a_parent(dataset):
    """GES does not know the time order and may return x' -> x; the edge is still a time step, so x
    is a parent of x' (as an undirected x - x' edge, whatever orientation wins)."""
    from scm_to_mdp.core import Candidates

    backward = Candidates(
        "graph_class", frozenset({("m0'", "m0"), ("m1'", "reboot")}), frozenset({frozenset(("m2", "m2'"))})
    )
    parents = bic_best.select(backward, dataset).parents
    assert "m0" in parents["m0"] and "reboot" in parents["m1"] and "m2" in parents["m2"]
