"""Factors with same-step parents (FactorBounds.conditional): learned by S4 on a graph with y' -> x',
compiled by S7 into a box that contains the coupled true kernel, an exact MDP from point estimates,
refused by the vertex solver, and saved and loaded by the store."""

import numpy as np
import pytest

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import Graph
from scm_to_mdp.envs import FrozenLake
from scm_to_mdp.scm_learning import nbn_intervals, nbn_point
from scm_to_mdp.solving import vertex_robust_vi
from scm_to_mdp.tool import store


@pytest.fixture(scope="module")
def lake():
    env = FrozenLake("4x4", 1, 0, axes="joint")  # slippery, Gym's law: x0' and y0' are coupled
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(400, 30, "random"))
    return truth, data, Graph(dict(truth.true_parents))


def test_intervals_with_a_same_step_parent_contain_the_true_kernel(lake):
    truth, data, graph = lake
    bounds = nbn_intervals.learn(data, graph, delta=0.05)
    assert list(bounds.conditional) == [0]  # x0 (index 0) has the same-step parent y0'
    js, lo_x, hi_x = bounds.conditional[0]
    assert js == (1,) and lo_x.shape[2] == data.cards["y0"]  # one row per value of y0'
    box = product_bounds.compile(bounds)
    P = truth.P[:, truth.actions_of(data)]
    seen = np.zeros(box.L.shape[:2], dtype=bool)  # (s, a) rows with data
    s = data.state_index(data.df[data.state_vars].values)
    seen[s, data.action_index()] = True
    assert (box.L[seen] <= P[seen] + 1e-9).all() and (P[seen] <= box.U[seen] + 1e-9).all()


def test_without_the_same_step_edge_the_model_puts_mass_on_impossible_slips(lake):
    """Gym's slip never moves both axes; a model with independent axes (no y0' -> x0') puts mass on
    such diagonal moves, the model with the same-step edge none."""
    truth, data, graph = lake
    independent = Graph({v: tuple(p for p in ps if not p.endswith("'")) for v, ps in graph.parents.items()})
    P = truth.P[:, truth.actions_of(data)]
    seen = np.zeros(P.shape[:2], dtype=bool)
    seen[data.state_index(data.df[data.state_vars].values), data.action_index()] = True

    def impossible_mass(g):
        model = product_bounds.exact(nbn_point.learn(data, g, alpha=0.0))
        return (model.L[seen] * (P[seen] == 0)).sum(-1).mean()

    assert impossible_mass(graph) < 1e-12 and impossible_mass(independent) > 0.1


def test_model_errors_of_point_and_interval_models_with_a_same_step_parent(lake):
    """A point model with a conditional factor is scored as a point model (tv, kl), not as an interval
    model of width 0 (its lo/hi hold the hull over the same-step parent, so lo != hi there)."""
    from scm_to_mdp.evaluation import model_errors

    truth, data, graph = lake
    point = model_errors(nbn_point.learn(data, graph), truth, data, 0.9)
    assert {"tv_data", "tv_uniform", "kl_data"} <= set(point) and not any(
        k.startswith("contain") for k in point
    )
    assert 0.0 <= point["tv_data"] < 0.2
    bounds = model_errors(nbn_intervals.learn(data, graph, delta=0.05), truth, data, 0.9)
    assert "contain_data" in bounds and "tv_data" not in bounds and bounds["width_seen"] > 0


def test_point_estimates_give_an_exact_mdp_and_the_vertex_solver_refuses(lake):
    _, data, graph = lake
    point = nbn_point.learn(data, graph)
    mdp = product_bounds.exact(point)
    assert np.allclose(mdp.L.sum(-1), 1.0) and np.array_equal(mdp.L, mdp.U)
    with pytest.raises(ValueError, match="same-step"):
        vertex_robust_vi.solve(point, 0.9)


def test_the_store_keeps_the_conditional_factors(lake, tmp_path):
    truth, data, graph = lake
    from scm_to_mdp.tool.methods import Method, learn, solve

    method = Method("causal", "box_robust_vi", discovery="oracle", selection="given")
    result = solve(learn(data, method, None, truth.true_parents), data, 0.9)
    info = {"method": "true graph -> IMDP", "size_label": "4x4-1", "n_rollouts": 400}
    store.save(tmp_path, result, data, truth, info, pictures_max_states=16)
    back = store.load(tmp_path)["model"]
    assert list(back.conditional) == [0]
    assert np.array_equal(product_bounds.compile(back).L, product_bounds.compile(result.model).L)
    assert (tmp_path / "cbn.png").exists() and (tmp_path / "mdp.png").exists()
