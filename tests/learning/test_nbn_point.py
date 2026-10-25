"""X -> MDP (scm_learning/nbn_point.py and the causal learner with value iteration): Laplace point
estimates equal to NBN's own table mechanism, an exact MDP, and discovery shared between methods."""

import numpy as np
import torch
from nbn.mechanisms.parametric.categorical_table import CategoricalTableMechanism

from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import Graph
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.scm_learning import nbn_point
from scm_to_mdp.tool import Method, evaluate, learn, solve


def setup(n=200):
    env = SysAdmin(3, 0)
    truth = env.ground_truth()
    return truth, truth.learner_spec.read(env.sample(n, 20, "random"))


def test_laplace_estimates_match_nbn():
    truth, data = setup()
    graph = Graph({v: tuple(truth.true_parents[v]) for v in data.state_vars})
    model = nbn_point.learn(data, graph, alpha=1.0)
    assert np.array_equal(model.lo, model.hi) and np.allclose(model.lo.sum(-1), 1.0)
    v = data.state_vars[0]
    parents = tuple(x for x in data.inputs if x in truth.true_parents[v])
    mech = CategoricalTableMechanism(alpha=1.0)
    pa = torch.tensor(data.df[list(parents)].values)
    mech.fit_local(torch.tensor(data.df[v + "'"].values)[:, None].float(), pa.float(),
                   parent_cards=[data.cards[p] for p in parents], n_classes=data.cards[v])  # fmt: skip
    cpt = mech.cpt() if callable(mech.cpt) else mech.cpt  # [parent rows, card], Dirichlet(1) smoothed
    nbn_table = cpt.reshape(-1, data.cards[v]).double().numpy()
    ours = nbn_point.learn(data, Graph({**graph.parents}), alpha=1.0)
    # the same numbers at every (state, action): look up NBN's row of each
    from scm_to_mdp.core import encode
    from scm_to_mdp.data import slice_inputs

    X = slice_inputs(data)
    row = encode(X[..., [data.inputs.index(p) for p in parents]], [data.cards[p] for p in parents])
    assert np.allclose(ours.lo[:, :, 0, : data.cards[v]], nbn_table[row], atol=1e-6)


def test_exact_mdp_and_near_optimal_policy_with_the_true_graph():
    truth, data = setup(1500)
    graph = Graph({v: tuple(truth.true_parents[v]) for v in data.state_vars})
    mdp = product_bounds.exact(nbn_point.learn(data, graph))
    assert np.array_equal(mdp.L, mdp.U) and np.allclose(mdp.L.sum(-1), 1.0)
    method = Method("causal", "vi", discovery="oracle", selection="given")
    assert method.role == "reference" and not method.guarantee
    result = solve(learn(data, method, true_parents=truth.true_parents), data, 0.9)
    assert evaluate(result, data, truth, 0.9)["optimal_share"] > 0.95


def test_discovery_is_shared_and_gives_the_same_rows():
    _, data = setup()
    robust = Method("causal", "box_robust_vi", discovery="bic", selection="bic_best")
    point = Method("causal", "vi", discovery="bic", selection="bic_best")
    shared: dict = {}
    a, b = learn(data, robust, shared=shared), learn(data, point, shared=shared)
    assert len(shared) == 1 and a.graph.parents == b.graph.parents
    assert a.steps["discovery"] == b.steps["discovery"]  # one run, its record reused
    alone = learn(data, point)
    assert alone.graph.parents == b.graph.parents and np.array_equal(alone.model.lo, b.model.lo)
