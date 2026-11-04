"""The post-selection union bound of S4 (scm_learning/nbn_intervals.py, n_candidates): the intervals are
built at delta / |F_i| when the parent sets were chosen from the data."""

import numpy as np
import pytest

from scm_to_mdp.discovery import oracle
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import given


def test_candidate_count(dataset):
    n_in, n_state = len(dataset.inputs), len(dataset.state_vars)
    assert nbn_intervals.candidate_count(dataset) == 1
    assert nbn_intervals.candidate_count(dataset, "inputs") == 2**n_in
    assert nbn_intervals.candidate_count(dataset, "inputs+next") == 2 ** (n_in + n_state - 1)
    assert nbn_intervals.candidate_count(dataset, 7) == 7
    for bad in ("all", 0):
        with pytest.raises(ValueError):
            nbn_intervals.candidate_count(dataset, bad)


def test_union_bound_widens_every_interval(env, dataset):
    graph = given.select(oracle.fit(dataset, env.true_parents()), dataset)
    plain = nbn_intervals.learn(dataset, graph, 0.05)
    split = nbn_intervals.learn(dataset, graph, 0.05, n_candidates="inputs")
    assert split.info["s4_candidates"] == 2 ** len(dataset.inputs) and plain.info["s4_candidates"] == 1
    assert np.all(split.lo <= plain.lo + 1e-12) and np.all(split.hi >= plain.hi - 1e-12)
    assert (split.hi - split.lo).sum() > (plain.hi - plain.lo).sum()  # strictly wider somewhere
    same = nbn_intervals.learn(dataset, graph, 0.05 / 2 ** len(dataset.inputs))  # = delta / |F| directly
    assert np.allclose(same.lo, split.lo) and np.allclose(same.hi, split.hi)
