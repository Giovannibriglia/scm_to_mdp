import numpy as np

from scm_to_mdp.data import read_dataset, slice_inputs
from scm_to_mdp.envs import SysAdmin


def test_kernel_is_a_distribution(env):
    assert env.P.shape == (8, 4, 8)
    assert np.allclose(env.P.sum(-1), 1.0)


def test_sample_table(env, dataset):
    df = dataset.df
    assert len(df) == 300 * 20
    assert df.index.names == ["episode", "t"]
    assert dataset.state_vars == ["m0", "m1", "m2"]
    assert dataset.action_vars == ["a0", "a1", "a2"]
    assert (df[dataset.action_vars].sum(axis=1) <= 1).all()  # one-hot or no-op


def test_same_seed_same_data():
    a, b = SysAdmin(3, seed=1), SysAdmin(3, seed=1)
    assert a.sample(10, 5, "random").equals(b.sample(10, 5, "random"))


def test_reward_and_inputs(env, dataset):
    assert dataset.reward.R.shape == (8, len(dataset.actions))
    X = slice_inputs(dataset)
    assert X.shape == (8, len(dataset.actions), 6)
    assert np.array_equal(X[:, 0, :3], dataset.B)


def test_true_parents_follow_the_hidden_mapping(env, dataset):
    parents = env.true_parents(dataset)
    for i in range(3):
        assert set(parents[f"m{i}"]) == {f"m{i}", f"m{(i - 1) % 3}", f"a{env.repairs.index(i)}"}


def test_env_actions_map_back(env, dataset):
    mapped = env.env_actions(dataset)
    for a, k in zip(dataset.actions, mapped, strict=True):
        assert k == (0 if a.sum() == 0 else env.repairs[int(np.argmax(a))] + 1)


def test_callable_policy(env):
    df = SysAdmin(3, seed=0).sample(5, 4, lambda s, rng: np.zeros(len(s), dtype=np.int64))
    data = read_dataset(df, env.action_vars, env.cards, env.reward)
    assert (data.df[data.action_vars].values == 0).all()
