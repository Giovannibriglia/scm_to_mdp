"""The baselines against their papers (2026-10-27 review): DiFaRL's groups and target, SPIBB's model
and baseline policy."""

import sys
from pathlib import Path

import numpy as np

from scm_to_mdp.baselines import difarl, spibb
from scm_to_mdp.core import Reward
from scm_to_mdp.data import read_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from domains import DOMAINS  # noqa: E402


def domain_data(name, size, knob, transitions, seed=0):
    env = DOMAINS[name].make_env(size, knob, seed)
    truth = env.ground_truth()
    df = env.sample(max(1, transitions // 5), 256, "random")
    return truth.learner_spec.read(df.iloc[:transitions])


def test_difarl_keeps_independent_agents_apart():
    """Frozen Lake with two agents: the agents' moves do not interact, so two groups (the chi-square
    G^2 test merged them: one group, i.e. flat FQI)."""
    data = domain_data("frozenlake", "4x4", "2", 20_000)
    assert sorted(map(sorted, difarl.groups(data))) == [[0], [1]]


def test_difarl_ignores_next_state_copies_of_the_actions():
    """Sepsis copies the treatments into the next state (abx_on', ...): conditioned on, every test would
    accept. Antibiotics and vasopressors both act on the blood pressure: one group; ventilation
    alone."""
    data = domain_data("sepsis", 2, "observed", 20_000)
    assert set(difarl.action_copies(data)) == {"abx_on'", "vaso_on'", "vent_on'"}
    names = [sorted(data.action_vars[i] for i in g) for g in difarl.groups(data)]
    assert sorted(names) == [["antibiotics", "vasopressors"], ["ventilation"]]


def test_difarl_detects_a_collider():
    """Two independent binary actions whose XOR is the next state: dependent given s'."""
    rng = np.random.default_rng(0)
    n = 4000
    a0, a1 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    import pandas as pd

    df = pd.DataFrame({"x": np.zeros(n, dtype=int), "a0": a0, "a1": a1, "x'": a0 ^ a1})
    df.index = pd.MultiIndex.from_arrays([np.arange(n), np.zeros(n, dtype=int)], names=["episode", "t"])
    reward = lambda B, actions: Reward.immediate(np.zeros((len(B), len(actions))))  # noqa: E731
    data = read_dataset(df, ["a0", "a1"], {"x": 2, "a0": 2, "a1": 2}, reward)
    assert sorted(map(sorted, difarl.groups(data))) == [[0, 1]]


def test_difarl_target_values_every_action_by_default(dataset):
    """Their code's target (max over every action vector, double Q, Polyak); seen_only is optional."""
    groups = difarl.groups(dataset)
    every = difarl.fqi(dataset, groups, 0.9)
    seen = difarl.fqi(dataset, groups, 0.9, seen_only=True)
    assert np.isfinite(every.V).all() and np.isfinite(seen.V).all()


def test_flat_spibb_keeps_the_estimates_of_bootstrapped_pairs(dataset):
    """Laroche et al.: the MLE model for every seen pair; only unseen pairs have no estimate."""
    model = spibb.flat_model(dataset, m=10_000)  # every pair bootstrapped
    s = dataset.state_index(dataset.df[dataset.state_vars].values)
    seen = np.zeros(model.bootstrapped.shape, dtype=bool)
    seen[s, dataset.action_index()] = True
    assert model.bootstrapped.all()
    assert np.allclose(model.P[seen].sum(1), 1.0) and np.allclose(model.P[~seen].sum(1), 0.0)


def test_spibb_baseline_policy_and_iterations(dataset):
    uniform = spibb.flat_model(dataset, m=10_000, behaviour="uniform")
    assert np.allclose(uniform.behaviour, 1.0 / len(dataset.actions))
    solution = spibb.solve(uniform, 0.9)  # every pair bootstrapped: the baseline itself
    assert np.allclose(solution.probs, uniform.behaviour) and solution.iterations == 1
