"""The sparse graph (discovery/sparse.py): each next-state variable gets itself and its own actions."""

import numpy as np
import pandas as pd
import pytest

from scm_to_mdp.core import Reward
from scm_to_mdp.data import read_dataset
from scm_to_mdp.discovery import full_factored, sparse


def test_one_controller_owns_every_state_variable(dataset):
    """SysAdmin: one action variable (reboot), so x_i' <- x_i, reboot; no state-to-state edge."""
    parents = sparse.parents(dataset)
    assert parents == {v: (v, *dataset.action_vars) for v in dataset.state_vars}
    graph = sparse.fit(dataset)
    assert all(src in (dst[:-1], *dataset.action_vars) for src, dst in graph.directed)


def test_agents_own_the_variables_with_their_index():
    """Frozen Lake with two agents (a0, a1) and Drone with two drones (d0, d1): agent k moves only its own
    state variables."""
    lake = ["a0", "a1"]
    assert sparse.own_actions("x0", lake, "by_index") == ("a0",)
    assert sparse.own_actions("y1", lake, "by_index") == ("a1",)
    drones = ["d0", "d1"]
    assert [sparse.own_actions(v, drones, "by_index") for v in ("x1", "y1", "z1")] == [("d1",)] * 3
    assert sparse.own_actions("x1", lake, "all") == ("a0", "a1")
    with pytest.raises(ValueError, match="index"):
        sparse.own_actions("x2", lake, "by_index")  # no agent 2
    with pytest.raises(ValueError, match="index"):
        sparse.own_actions("wet", lake, "by_index")  # no index at all
    with pytest.raises(ValueError, match="actions"):
        sparse.own_actions("x0", lake, "some")


def test_it_equals_the_full_factored_graph_only_with_one_state_variable(dataset):
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({"x": rng.integers(0, 2, n), "a": rng.integers(0, 2, n), "x'": rng.integers(0, 2, n)})
    df.index = pd.MultiIndex.from_arrays([np.arange(n), np.zeros(n, dtype=int)], names=["episode", "t"])
    reward = lambda B, actions: Reward.immediate(np.zeros((len(B), len(actions))))  # noqa: E731
    one = read_dataset(df, ["a"], {"x": 2, "a": 2}, reward)
    assert sparse.parents(one) == full_factored.parents(one)
    assert sparse.parents(dataset) != full_factored.parents(dataset)  # SysAdmin: 3 machines
