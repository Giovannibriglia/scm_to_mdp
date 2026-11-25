"""Consistency family (discovery/consistency_family.py) on synthetic factors with known parents. Needs nbn
with agreement_set(extension_size=...) (NBN #333); skipped with an older nbn."""

import inspect

import numpy as np
import pandas as pd
import pytest
from nbn.structure import agreement_set

from scm_to_mdp.core import NEXT_MARK, Reward
from scm_to_mdp.data import read_dataset
from scm_to_mdp.discovery import consistency_family as cf
from scm_to_mdp.tool import methods

pytestmark = pytest.mark.skipif(
    "extension_size" not in inspect.signature(agreement_set).parameters,
    reason="nbn without agreement_set(extension_size=...)",
)


def _reward(B, A):
    return Reward(np.zeros((len(B), len(A))), np.zeros(len(B)), np.zeros(len(B), dtype=bool))


def synthetic(n_rows: int, rule, n_vars: int = 4, seed: int = 0, keep=None):
    """Binary state x0..x{n-1} drawn uniformly, one action; P(x0' = 1) = rule(x); the other next-state
    variables are fresh coin flips. keep(x) -> bool drops the rows it rejects (a logger that never visits
    them)."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_rows, n_vars))
    if keep is not None:
        X = X[[bool(keep(x)) for x in X]]
    n = len(X)
    cols = {f"x{j}": X[:, j] for j in range(n_vars)}
    cols["a"] = np.zeros(n, dtype=int)
    cols["x0'"] = (rng.random(n) < np.array([rule(x) for x in X])).astype(int)
    for j in range(1, n_vars):
        cols[f"x{j}'"] = rng.integers(0, 2, size=n)
    df = pd.DataFrame(cols, index=pd.MultiIndex.from_arrays([np.arange(n), np.zeros(n, int)],
                                                             names=["episode", "t"]))  # fmt: skip
    return read_dataset(df, ["a"], {f"x{j}": 2 for j in range(n_vars)} | {"a": 1}, _reward)


def xor(x):
    return 0.9 if x[0] ^ x[1] else 0.1


def test_the_pairwise_test_recovers_xor_parents_and_one_variable_extensions_do_not():
    data = synthetic(20000, xor)
    pair = cf.family(data, k=2)  # extension_size = k
    assert pair.params["extension_size"] == 2
    assert pair.survivors["x0"] == [("x0", "x1")] and pair.common["x0"] == ("x0", "x1")
    assert not any(pair.refuted.values())
    one = cf.family(data, k=2, extension_size=1)
    assert ("x2", "x3") in one.survivors["x0"]  # agrees with each one-variable extension
    assert ("x0", "x1") in one.survivors["x0"]  # a candidate with the true parents is never discarded
    assert set(one.common["x0"]) != {"x0", "x1"}


def test_views_union_is_a_superset_and_intersection_is_exact_when_the_wrong_sets_are_refuted():
    data = synthetic(20000, xor)
    fam = cf.family(data, k=2)
    assert fam.parents(data, "union")["x0"] == ("x0", "x1", "a")
    assert fam.parents(data, "intersection")["x0"] == ("x0", "x1", "a")
    # a coin flip has no parents: every candidate survives, the union is every state variable
    assert fam.parents(data, "union")["x1"] == ("x0", "x1", "x2", "x3", "a")
    assert fam.parents(data, "intersection")["x1"] == ("a",)
    with pytest.raises(ValueError, match="view"):
        fam.parents(data, "all")


def test_little_data_keeps_every_candidate_so_the_union_stays_sound():
    data = synthetic(60, xor)
    fam = cf.family(data, k=2)
    assert len(fam.survivors["x0"]) == 6
    assert set(fam.parents(data, "union")["x0"]) >= {"x0", "x1"}
    assert fam.parents(data, "intersection")["x0"] == ("a",)  # the intersection is not sound here


def test_a_bound_below_the_in_degree_is_reported_when_the_data_contradict_it():
    data = synthetic(40000, lambda x: 0.95 if (x[0] ^ x[1] ^ x[2]) else 0.05)  # three parents
    fam = cf.family(data, k=1, extension_size=3)
    assert fam.refuted["x0"]
    assert fam.parents(data, "union")["x0"] == ("x0", "x1", "x2", "x3", "a")  # falls back to every variable


def test_an_unvisited_setting_does_not_break_the_family():
    data = synthetic(20000, xor, keep=lambda x: not (x[0] == 1 and x[1] == 1))
    fam = cf.family(data, k=2)
    assert ("x0", "x1") in fam.survivors["x0"]


def test_fit_returns_the_graph_of_the_view_and_is_a_registered_discovery():
    data = synthetic(20000, xor)
    cand = methods.discovery_function("consistency")(data, k=2, view="intersection")
    assert cand.kind == "graph_class" and not cand.undirected
    assert {p for p, c in cand.directed if c == "x0" + NEXT_MARK} == {"x0", "x1", "a"}
    assert cand.report["view"] == "intersection" and cand.report["level"] == pytest.approx(0.95)
    assert cand.report["survivors"]["x0"] == 1 and cand.report["refuted"] == []
    with pytest.raises(ValueError, match="same_step"):
        cf.fit(data, k=2, same_step=True)
