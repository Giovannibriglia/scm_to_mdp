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


def test_action_recall():
    env = SysAdmin(3, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(20, 5, "random"))
    true = truth.parents_in(data)
    no_actions = Graph({v: tuple(p for p in ps if p not in data.action_vars) for v, ps in true.items()})
    assert evaluation.structure_distances(Graph(true), true, data)["action_recall"] == 1.0
    lost = evaluation.structure_distances(no_actions, true, data)
    assert lost["action_recall"] == 0.0 and lost["action_missing"] == lost["action_edges"] == 3
