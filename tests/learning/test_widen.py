"""S4 widening (scm_learning/nbn_intervals.py): with a graph that misses a true parent, the padded and
the hull intervals still contain the true factor probabilities; the plain intervals do not."""

import json

import numpy as np
import pytest

from scm_to_mdp.core import Graph
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.tool import Method, learn


@pytest.fixture(scope="module")
def setup():
    env = SysAdmin(3, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(1500, 20, "random"))
    return truth, data


def true_factors(truth, data) -> np.ndarray:
    """[S, K, n, C]: P(x_i' = v | s, a_k) at the data's states and actions, from the true model."""
    t_state = {tuple(b): j for j, b in enumerate(truth.B)}
    t_action = {tuple(a): j for j, a in enumerate(truth.actions)}
    S, K, n = len(data.B), len(data.actions), len(data.state_vars)
    out = np.zeros((S, K, n, max(data.state_cards)))
    for s, b in enumerate(data.B):
        for k, a in enumerate(data.actions):
            row = truth.P[t_state[tuple(b)], t_action[tuple(a)]]  # [S']
            for i in range(n):
                for v in range(data.state_cards[i]):
                    out[s, k, i, v] = row[truth.B[:, i] == v].sum()
    return out


def contained(bounds, P) -> bool:
    return bool(((bounds.lo <= P + 1e-9) & (P <= bounds.hi + 1e-9)).all())


def test_widening_recovers_a_missing_parent(setup):
    truth, data = setup
    P = true_factors(truth, data)
    v = data.state_vars[0]
    neighbour = next(p for p in truth.true_parents[v] if p != v and p in data.state_vars)
    wrong = Graph({x: tuple(truth.true_parents[x]) for x in data.state_vars})
    wrong = Graph({**wrong.parents, v: tuple(p for p in wrong.parents[v] if p != neighbour)})

    plain = nbn_intervals.learn(data, wrong)
    assert not contained(plain, P)  # the missing parent breaks the plain intervals
    for widen in ("pad", "hull"):
        bounds = nbn_intervals.learn(data, wrong, widen=widen, k=2)
        added = json.loads(bounds.info["added_parents"])
        assert neighbour in added[v], (widen, added)
        assert contained(bounds, P), widen


def test_widening_keeps_the_true_intervals_and_is_wider(setup):
    truth, data = setup
    true_graph = Graph({x: tuple(truth.true_parents[x]) for x in data.state_vars})
    plain = nbn_intervals.learn(data, true_graph)
    hull = nbn_intervals.learn(data, true_graph, widen="hull", k=2)
    assert (hull.lo <= plain.lo + 1e-12).all() and (hull.hi >= plain.hi - 1e-12).all()
    assert contained(hull, true_factors(truth, data))


def test_widened_methods_report_what_they_added(setup):
    _, data = setup
    method = Method(
        "causal",
        "box_robust_vi",
        discovery="bic",
        selection="bic_best",
        learning_options={"widen": "hull", "k": 1},
    )
    result = learn(data, method)
    assert result.ok and result.info["widen"] == "hull"
    assert all(len(x) == 1 for x in json.loads(result.info["added_parents"]).values())
    with pytest.raises(ValueError, match="widen"):
        nbn_intervals.learn(data, Graph({x: (x,) for x in data.state_vars}), widen="both")
