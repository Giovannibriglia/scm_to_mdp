"""KWIK structure learners (discovery/kwik_structure.py): Strehl, Diuk, Littman (AAAI 2007) and the Adaptive
k-Meteorologists Algorithm of Diuk, Li, Leffler (ICML 2009, Alg. 1), on synthetic factors with known
parents."""

import numpy as np
import pandas as pd

from scm_to_mdp.core import Reward
from scm_to_mdp.data import read_dataset
from scm_to_mdp.discovery import kwik_structure as ks


def _reward(B, A):
    return Reward(np.zeros((len(B), len(A))), np.zeros(len(B)), np.zeros(len(B), dtype=bool))


def synthetic(n_rows: int, rule, n_vars: int = 3, seed: int = 0):
    """Binary state x0..x{n-1} drawn uniformly (one action, a = 0); y' = x0' follows P(x0' = 1) = rule(x);
    the other next-state variables are fresh coin flips."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n_rows, n_vars))
    cols = {f"x{j}": X[:, j] for j in range(n_vars)}
    cols["a"] = np.zeros(n_rows, dtype=int)
    p = np.array([rule(x) for x in X])
    cols["x0'"] = (rng.random(n_rows) < p).astype(int)
    for j in range(1, n_vars):
        cols[f"x{j}'"] = rng.integers(0, 2, size=n_rows)
    df = pd.DataFrame(cols, index=pd.MultiIndex.from_arrays([np.arange(n_rows), np.zeros(n_rows, int)],
                                                             names=["episode", "t"]))  # fmt: skip
    cards = {f"x{j}": 2 for j in range(n_vars)} | {"a": 1}
    return read_dataset(df, ["a"], cards, _reward)


def test_first_m_keeps_the_first_m_of_every_code():
    codes = np.array([3, 1, 3, 3, 1, 2, 3])
    assert ks._first_m(codes, 2).tolist() == [True, True, True, False, True, True, False]


def test_strehl_finds_an_xor_with_two_parents_and_fails_silently_below_the_in_degree():
    data = synthetic(4000, lambda x: 0.9 if x[0] ^ x[1] else 0.1)  # x0' depends on x0 XOR x1, not on x2
    two = ks.strehl_sl(data, k=2, m=30, eps1=0.2).factors[(0, 0)]
    assert two.known.all()
    for s, b in enumerate(data.B):  # the accepted candidate contains both parents, the prediction is accurate
        assert set(two.candidates[two.accepted[s]]) == {0, 1}
        assert abs(two.predict[s, 1] - (0.9 if b[0] ^ b[1] else 0.1)) <= 2 * 0.2
    # k = 1 < the true in-degree 2: the guarantee (Thm 1 assumes P depends on at most k inputs) is void.
    # Mostly the irrelevant x2 is accepted: its pair unions {x2, x0} and {x2, x1} each miss a parent, so
    # both estimate about 0.5 and agree; the prediction is wrong (error > 2 eps1), not "don't know"
    one = ks.strehl_sl(data, k=1, m=30, eps1=0.2).factors[(0, 0)]
    truth = np.array([0.9 if b[0] ^ b[1] else 0.1 for b in data.B])
    assert one.known.all() and (one.accepted == 2).sum() >= 6
    assert np.abs(one.predict[:, 1] - truth).max() > 2 * 0.2


def test_strehl_calibrated_m_follows_hoeffding_and_the_union_bound(dataset):
    m = ks.strehl_calibrated_m(dataset, 2, 0.2, 0.05)
    n = ks.strehl_events(dataset, 2)
    assert 2 * n * np.exp(-2 * m * 0.2**2) <= 0.05 < 2 * n * np.exp(-2 * (m - 1) * 0.2**2)


def test_met_rmax_eliminates_the_wrong_meteorologist_and_predicts_accurately():
    data = synthetic(6000, lambda x: 0.85 if x[1] else 0.15)  # one true parent: x1
    f = ks.met_rmax(data, D=1, m=20, eps=0.2).factors[(0, 0)]
    assert [f.candidates[h] for h in f.survivors] == [(1,)]
    assert f.known.all()
    truth = np.where(data.B[:, 1] == 1, 0.85, 0.15)
    assert np.abs(f.predict[:, 1] - truth).max() <= 0.2


def test_met_rmax_algorithm_1_by_hand():
    """Two meteorologists, x0 and x1, m_sub = 1 (each predicts after one sample of its setting), m = 2.
    Inputs and labels (z = x0'): x=(0,0) z=0, x=(1,1) z=1, x=(1,0) z=1, x=(0,1) z=0, x=(1,0) z=1, x=(0,1) z=0.
    The truth is x0' = x0, so x1's meteorologist must go after two disagreements."""
    X = np.array([[0, 0], [1, 1], [1, 0], [0, 1], [1, 0], [0, 1]])
    z = X[:, 0]
    cands = [(0,), (1,)]
    known, pred, _ = ks._meteorologist_predictions(X, z, cands, [2, 2], 2, m_sub=1)
    # t=0: both "don't know" (no sample yet); t=1: both see a new setting -> "don't know";
    # t=2: x0=1 seen at t=1 (z=1) -> 1.0; x1=0 seen at t=0 (z=0) -> 0.0: they disagree -> c=1, x1 worse
    assert known[:, 2].all() and pred[0, 2, 1] == 1.0 and pred[1, 2, 1] == 0.0
    f = ks._met_factor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), X, z, [2, 2], 2, 1, 2, 1, 0.2)
    assert f.survivors == [0]  # after t=2 and t=3 (x1=1 -> 1.0 vs x0=0 -> 0.0, z=0): x1 removed
    assert f.known.all() and f.predict[:, 1].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_met_rmax_exact_tie_removes_j():
    """Alg. 1 line 19: on Delta_ij = 0 the paper removes j; a float residue must not flip it. Predictions
    0.9 and 0.7 of value 0 and labels (values) 1, 0, 0, 0, 0: the squared errors sum to the same, but in
    floating point Delta is about 1e-16."""
    p_i, p_j = np.array([0.9, 0.1]), np.array([0.7, 0.3])
    d = 0.0
    for v in (1, 0, 0, 0, 0):
        z = np.eye(2)[v]
        d += ((p_i - z) ** 2).sum() - ((p_j - z) ** 2).sum()
    assert abs(d) < 1e-12 and d != 0.0  # a float residue
    i, j = 0, 1
    assert (i if d > ks.TIE else j) == j  # the rule of _met_factor, line 19


def test_transitions_follow_the_logged_order():
    data = synthetic(50, lambda x: 0.5)
    shuffled = data.df.sample(frac=1.0, random_state=0)
    from dataclasses import replace

    a = ks._transitions(data)[0][0]
    b = ks._transitions(replace(data, df=shuffled))[0][0]
    assert np.array_equal(a, b)


def test_parents_union_and_intersection_views():
    data = synthetic(6000, lambda x: 0.85 if x[1] else 0.15)
    s = ks.met_rmax(data, D=2, m=20, eps=0.2)
    union, inter = s.parents(data), s.parents(data, "intersection")
    assert "x1" in inter["x0"] and set(inter["x0"]) <= set(union["x0"])
