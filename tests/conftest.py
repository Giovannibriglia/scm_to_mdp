import pytest

from scm_to_mdp.data import read_dataset
from scm_to_mdp.envs import SysAdmin


@pytest.fixture(scope="session")
def env():
    return SysAdmin(3, seed=0)


@pytest.fixture(scope="session")
def dataset(env):
    return read_dataset(env.sample(300, 20, "repair_broken"), env.action_vars, env.cards, env.reward)
