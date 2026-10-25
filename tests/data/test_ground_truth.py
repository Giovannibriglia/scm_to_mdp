import numpy as np
import pytest

from scm_to_mdp.data import DataError, DataSpec
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.ground_truth import MDPGroundTruth, ModelTooLarge, dense_model_gb


def test_env_ground_truth_matches_the_env(env, truth):
    assert truth.n_states == len(env.B) and np.array_equal(truth.P, env.P)
    assert np.array_equal(truth.reward.R, env.true_reward.R)
    assert truth.true_parents == env.true_parents()


def test_from_factored_equals_the_env(env, truth):
    built = MDPGroundTruth.from_factored(
        env.spec(), env.next_distributions, truth.actions, env.start, env.true_parents()
    )
    assert np.allclose(built.P, truth.P)


def test_a_too_large_model_is_refused_before_allocating():
    assert dense_model_gb(2**15, 16) > 100  # the size that froze the laptop
    with pytest.raises(ModelTooLarge, match="needs"):
        SysAdmin(15, seed=0)


def test_bad_arrays_are_refused(truth):
    with pytest.raises(ValueError, match="probability vector"):
        MDPGroundTruth(truth.spec, truth.P * 2, truth.actions, truth.start)
    with pytest.raises(ValueError, match="shape"):
        MDPGroundTruth(truth.spec, truth.P[:, :2], truth.actions, truth.start)
    with pytest.raises(ValueError, match="start"):
        MDPGroundTruth(truth.spec, truth.P, truth.actions, truth.start * 0)
    with pytest.raises(ValueError, match="true_parents"):
        MDPGroundTruth(truth.spec, truth.P, truth.actions, truth.start, true_parents={"m0": ("m0",)})


def test_data_checks(env, truth):
    df = env.sample(5, 3, "random")
    truth.check_data(df)
    with pytest.raises(DataError, match="missing"):
        truth.spec.check(df.drop(columns=["m0'"]))
    bad = df.copy()
    bad["m1"] = 2
    with pytest.raises(DataError, match="outside"):
        truth.spec.read(bad)
    no_such_action = df.copy()
    no_such_action["reboot"] = 4  # reboot has values 0..3 (no-op, three repairs)
    with pytest.raises(DataError, match="outside"):
        truth.check_data(no_such_action)


def test_spec_read_orders_the_columns(env):
    df = env.sample(5, 3, "random")
    spec = env.spec()
    shuffled = df[list(reversed(df.columns))].assign(note="ignored")
    data = spec.read(shuffled)
    assert list(data.df.columns) == spec.columns and data.inputs == spec.state_vars + spec.action_vars


def test_random_policy_value_is_below_optimal(truth):
    assert np.all(truth.random_policy_values(0.9) <= truth.optimal_values(0.9) + 1e-9)


def test_a_hand_made_two_state_mdp():
    """A user-made truth: one variable, one action column; 'stay' keeps the state, 'flip' flips it."""
    from scm_to_mdp.core import Reward

    spec = DataSpec(
        ["x"],
        ["flip"],
        {"x": 2, "flip": 2},
        lambda B, A: Reward.immediate(np.repeat(B[:, :1].astype(float), len(A), axis=1)),
    )
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[1, 0, 1] = 1.0  # stay
    P[0, 1, 1] = P[1, 1, 0] = 1.0  # flip
    truth = MDPGroundTruth(spec, P, np.array([[0], [1]]), np.array([1.0, 0.0]), {"x": ("x", "flip")})
    V = truth.optimal_values(0.5)
    assert np.isclose(V[1], 1 / (1 - 0.5)) and np.isclose(V[0], 0.5 * V[1])  # flip once, then stay
