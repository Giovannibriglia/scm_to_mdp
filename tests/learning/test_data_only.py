"""Data-only S4 (learning option data_only=True, scm_learning/data_only.py): no value domains, support or terminal
states given; sinks at V_min, unseen rows sent to a sink, the unobserved mass free. The promise must stay a lower
bound on the true value, also when the true chain enters states the data never showed."""

import numpy as np
import pytest
import torch
from nbn.uncertainty.binomial import anytime_delta, binomial_interval

from scm_to_mdp.data import read_dataset
from scm_to_mdp.scm_learning import data_only
from scm_to_mdp.solving.stopping import data_only_values
from scm_to_mdp.tool import Method, evaluate, learn, solve

GAMMA = 0.9
TRUE = {"discovery": "oracle", "selection": "given"}
DATA_ONLY = {"data_only": True}


def test_the_weights_sum_to_at_most_one():
    assert data_only.value_weight(np.arange(100_000)).sum() <= 1.0
    assert data_only.value_weight(np.arange(100_000)).sum() > 0.9999


def test_intervals_are_clopper_pearson_on_observed_values_and_free_elsewhere():
    counts = np.array([[3.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    row_w, value_w = np.array([0.5, 0.25]), data_only.value_weight(np.arange(3))
    lo, hi = data_only.intervals(counts, row_w, value_w, 0.05)
    n = torch.tensor([4.0], dtype=torch.float64)
    level = anytime_delta(torch.tensor([0.05 * 0.5 * value_w[0]], dtype=torch.float64), n)
    ref_lo, ref_hi = binomial_interval(torch.tensor([3.0], dtype=torch.float64), n, level, method="clopper_pearson")
    assert lo[0, 0] == pytest.approx(float(ref_lo[0])) and hi[0, 0] == pytest.approx(float(ref_hi[0]))
    assert (lo[0, 1], hi[0, 1]) == (0.0, 1.0)  # not observed at the row: free (the residual mass)
    assert np.all(lo[1] == 0) and np.all(hi[1] == 1)  # an unseen row


@pytest.fixture(scope="module")
def unseen(env):
    """Two transitions: some values never appear, so the true chain can enter states the model calls unknown."""
    data = read_dataset(env.sample(2, 2, "repair_broken"), env.action_vars, env.cards, env.reward)
    assert data_only.sinks(data)[0].any()
    return data


@pytest.fixture(scope="module")
def scarce(env):
    """A few transitions: values, rows and source states the data never show."""
    return read_dataset(env.sample(3, 6, "repair_broken"), env.action_vars, env.cards, env.reward)


def test_sinks_are_the_unseen_values_and_the_states_never_left(scarce):
    unknown, never_source = data_only.sinks(scarce)
    seen = data_only.observed_values(scarce)
    for s in np.flatnonzero(unknown):
        assert any(not seen[v][scarce.B[s, j]] for j, v in enumerate(scarce.state_vars))
    sources = set(scarce.state_index(scarce.df[scarce.state_vars].values))
    assert all(s not in sources for s in np.flatnonzero(never_source))
    assert not (unknown & never_source).any()


def test_the_sink_values_are_below_every_value(scarce):
    reward = data_only.model_reward(scarce)
    assert not reward.terminal.any()
    _, never_source = data_only.sinks(scarce)
    doomed = np.zeros_like(scarce.reward.R, dtype=bool)
    v, q = data_only_values(reward, GAMMA, {"unknown": ~never_source, "never_source": never_source, "doomed": doomed})
    floor = min(reward.R.min() + reward.r_next.min(), 0.0) / (1 - GAMMA)
    assert np.all(v[never_source] == pytest.approx(floor))
    assert np.all(reward.r_next[~never_source] + GAMMA * v[~never_source] <= reward.r_next.min() + GAMMA * floor + 1e-9)


def test_the_optimistic_sink_values_are_above_every_value(scarce):
    """The same sinks for the certificate's upper bound (data_only_values(optimistic=True), bible R15)."""
    reward = data_only.model_reward(scarce)
    _, never_source = data_only.sinks(scarce)
    doomed = np.ones_like(scarce.reward.R, dtype=bool)
    masks = {"unknown": ~never_source, "never_source": never_source, "doomed": doomed}
    v, q = data_only_values(reward, GAMMA, masks, optimistic=True)
    ceiling = max(reward.R.max() + reward.r_next.max(), 0.0) / (1 - GAMMA)
    assert np.all(v[never_source] == pytest.approx(ceiling))
    arrival = reward.r_next[~never_source] + GAMMA * v[~never_source]
    assert np.all(arrival >= reward.r_next.max() + GAMMA * ceiling - 1e-9)
    lo_v, _ = data_only_values(reward, GAMMA, masks)
    assert np.all(v >= lo_v)
    assert np.isnan(q).all()  # every state is a sink here, so no pair is left to fix


ARMS = {
    "ia": Method("causal", "box_robust_vi", learning_options=DATA_ONLY, **TRUE),
    "exact": Method("causal", "exact_robust_vi", learning_options=DATA_ONLY, **TRUE),
    "plug-in": Method("causal", "vi", learning_options=DATA_ONLY, **TRUE),
    "full ia": Method("causal", "box_robust_vi", discovery="full", selection="given", learning_options=DATA_ONLY),
    "flat": Method("flat_rmdp", "robust_vi", learning_options=DATA_ONLY),
    "flat plug-in": Method("flat_mdp", "vi", learning_options=DATA_ONLY),
    "strehl fixed": Method("kwik", "box_robust_vi", discovery="strehl", discovery_options={"k": 2, "m": 20},
                           learning_options={"hull": "fixed", **DATA_ONLY}),
    "strehl accepted": Method("kwik", "exact_robust_vi", discovery="strehl", discovery_options={"k": 2, "m": 20},
                              learning_options=DATA_ONLY),
    "met accepted": Method("kwik", "box_robust_vi", discovery="met_rmax", discovery_options={"D": 2, "m": 7},
                           learning_options=DATA_ONLY),
}  # fmt: skip


@pytest.mark.parametrize("name", list(ARMS))
@pytest.mark.parametrize("which", ["unseen", "scarce", "dataset"])
def test_data_only_promises_are_lower_bounds(name, which, request, truth):
    data = request.getfixturevalue(which)
    result = solve(learn(data, ARMS[name], true_parents=truth.true_parents), data, GAMMA)
    assert result.ok, result.error
    assert 0 <= result.info["sink_states"] <= 1
    metrics = evaluate(result, data, truth, GAMMA)
    if ARMS[name].guarantee:
        assert metrics["safe"], (name, which, metrics["promise"], metrics["true"])


def test_data_only_is_more_pessimistic_than_the_given_support(scarce, truth):
    plain = solve(learn(scarce, Method("causal", "box_robust_vi", **TRUE), true_parents=truth.true_parents),
                  scarce, GAMMA)  # fmt: skip
    only = solve(learn(scarce, ARMS["ia"], true_parents=truth.true_parents), scarce, GAMMA)
    assert evaluate(only, scarce, truth, GAMMA)["promise"] <= evaluate(plain, scarce, truth, GAMMA)["promise"] + 1e-9


def test_data_only_is_off_by_default_and_checked():
    m = Method("causal", "box_robust_vi", **TRUE)
    assert "data_only" not in m.learning_options
    with pytest.raises(ValueError):
        Method("causal", "worst_mdp_vi", learning_options=DATA_ONLY, **TRUE)
    with pytest.raises(ValueError):
        Method("causal", "box_robust_vi", solver_options={"fixed_v": None}, **TRUE)


def test_the_kwik_own_modes_refuse_data_only(dataset):
    m = Method("kwik", "box_robust_vi", discovery="strehl", discovery_options={"k": 2, "m": 20},
               learning_options={"mode": "kwik", **DATA_ONLY})  # fmt: skip
    result = learn(dataset, m)
    assert not result.ok and "data_only" in (result.error or "")
