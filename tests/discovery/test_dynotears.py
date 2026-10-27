"""DYNOTEARS (discovery/dynotears.py): it recovers a linear structural VAR, and SysAdmin's true parents."""

import numpy as np

from scm_to_mdp.discovery import dynotears
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.selection import bic_best


def test_recovers_a_linear_structural_var():
    rng = np.random.default_rng(0)
    n, d, m = 2000, 3, 3
    A = np.array([[0.8, 0.0, 0.0], [0.0, 0.0, -0.7], [0.0, 0.6, 0.0]])  # lagged input i -> target j
    W = np.zeros((d, d))
    W[0, 2] = 0.5  # intra-slice x0 -> x2
    Y = rng.normal(size=(n, m))
    X = np.zeros((n, d))
    for j in range(d):  # in the order of W (x0 before x2)
        X[:, j] = Y @ A[:, j] + X @ W[:, j] + 0.3 * rng.normal(size=n)
    W_hat, A_hat = dynotears.solve(X, Y, 0.01, 0.01)
    assert np.array_equal(np.abs(A_hat) > 0.1, A != 0)
    assert np.array_equal(np.abs(W_hat) > 0.1, W != 0)


def test_finds_sysadmin_parents_from_uniform_data():
    env = SysAdmin(4, 0)
    truth = env.ground_truth()
    data = truth.learner_spec.read(env.sample(250, 20, "random"))
    graph = bic_best.select(dynotears.fit(data), data)
    assert graph.parents == {v: tuple(truth.true_parents[v]) for v in data.state_vars}


def test_no_varying_next_state_gives_an_empty_graph():
    """A tiny dataset where no next-state variable changes (seen in the full runs at 10 transitions):
    nothing to fit, so no edge (the solver used to fail)."""
    env = SysAdmin(2, 0)
    truth = env.ground_truth()
    df = env.sample(1, 10, "random")
    for v in truth.learner_spec.state_vars:  # every machine stays as it is
        df[v + "'"] = df[v] = df[v].iloc[0]
    candidates = dynotears.fit(truth.learner_spec.read(df))
    assert not candidates.directed and not candidates.undirected
