"""The near-optimality certificate (evaluation.certificate) and the action-edge recall (S9)."""

import numpy as np

from scm_to_mdp import evaluation
from scm_to_mdp.core import Graph
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Method, learn, solve
from scm_to_mdp.tool.evaluate import evaluate


def test_certificate_brackets_the_true_values():
    env = SysAdmin(3, 0, crash=2)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(200, 15, "random"))
    method = Method("causal", "box_robust_vi", discovery="oracle", selection="given")
    result = solve(learn(data, method, true_parents=truth.true_parents), data, 0.9)
    m = evaluate(result, data, truth, 0.9, np.arange(4))
    assert m["promise"] <= m["true"] + 1e-9  # the robust promise holds (true graph, enough data)
    best = float(truth.start @ truth.optimal_values(0.9))
    assert m["V_opt"] >= best - 1e-6  # the optimistic value is an upper bound (truth in the set)
    assert m["cert_gap"] >= best - m["true"] - 1e-6 and m["cert_holds"]


def test_optimistic_values_of_an_exact_model_are_its_values():
    env = SysAdmin(3, 0)
    truth = env.ground_truth()
    model = truth.as_model()
    V_opt = evaluation.optimistic_values(model, 0.9)
    assert np.allclose(V_opt, truth.optimal_values(0.9), atol=1e-5)


def test_optimistic_values_put_the_data_only_sinks_at_the_best_value():
    """R15: a state that is terminal in truth is never seen as a source, so the data-only model makes it
    a sink and does not know it ends the episode. With step rewards of -1 the cooperative model without
    the sinks keeps paying after the true end (V_opt = -10 < the true -1); with the sink at
    V_max = max(r_max, 0)/(1-gamma) = 0 it is an upper bound again."""
    from scm_to_mdp.core import IntervalMDP, Reward

    gamma = 0.9
    R, r_next = -np.ones((2, 1)), np.zeros(2)
    true_value = -1.0  # from state 0: pay -1, arrive in the terminal state 1, the episode ends
    L = np.zeros((2, 1, 2))
    L[0, 0, 1] = 1.0  # the data: 0 -> 1 every time
    U = L.copy()
    U[1, 0, :] = 1.0  # state 1 never a source: its row is free
    masks = {
        "unknown": np.zeros(2, bool),
        "never_source": np.array([False, True]),
        "doomed": np.zeros((2, 1), bool),
    }
    reward = Reward(R, r_next, np.zeros(2, bool))  # data-only: the terminal flags are not given
    without = evaluation.optimistic_values(IntervalMDP(L, U, reward), gamma)
    assert without[0] < true_value - 1  # the gap R15 is about
    V_opt = evaluation.optimistic_values(IntervalMDP(L, U, reward, data_only=masks), gamma)
    assert V_opt[0] >= true_value - 1e-6
    assert V_opt[1] == 0.0  # V_max


def test_action_recall():
    env = SysAdmin(3, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(20, 5, "random"))
    true = truth.parents_in(data)
    no_actions = Graph({v: tuple(p for p in ps if p not in data.action_vars) for v, ps in true.items()})
    assert evaluation.structure_distances(Graph(true), true, data)["action_recall"] == 1.0
    lost = evaluation.structure_distances(no_actions, true, data)
    assert lost["action_recall"] == 0.0 and lost["action_missing"] == lost["action_edges"] == 3
