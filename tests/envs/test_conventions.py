"""The variable convention of every environment (envs/base.py): one variable per dimension of the state
and of the action, "do nothing" as a value of an action dimension, no done / terminated variable, and
exactly one next-state column per state variable."""

import itertools

import numpy as np
import pytest

from scm_to_mdp.core import NEXT_MARK
from scm_to_mdp.envs import (
    Aircraft,
    Chemistry,
    Coffee,
    Drone,
    FrozenLake,
    Sepsis,
    StockTrading,
    SysAdmin,
    Taxi,
)

ENVS = {
    "sysadmin": lambda: SysAdmin(3, 0, storm=0.2),
    "sepsis": lambda: Sepsis("small", 0),
    "chemistry": lambda: Chemistry(0.4, 0, n_objects=3),
    "coffee": lambda: Coffee("coffee"),
    "taxi": lambda: Taxi("noisy"),
    "aircraft": lambda: Aircraft(6, 4, 0),
    "stock": lambda: StockTrading(2, 2, 0),
    "lake two agents": lambda: FrozenLake("4x4", 2, 0, wind=0.2),
    "drone two": lambda: Drone(3, 2, 0),
}
DO_NOTHING = {"sysadmin": 0, "chemistry": 0, "aircraft": 0, "stock": 0}  # the value that does nothing


@pytest.fixture(scope="module", params=list(ENVS))
def case(request):
    return request.param, ENVS[request.param]()


def test_action_variables_are_independent_dimensions(case):
    """Every combination of the action variables' values is an action (no flag fixed by the others),
    each env action has its own combination, and each variable is categorical with >= 2 values."""
    _, env = case
    columns = env.action_columns(np.arange(env.n_actions))  # [A, n_a]
    cards = [env.cards[a] for a in env.action_vars]
    assert all(c >= 2 for c in cards)
    assert {tuple(c) for c in columns} == set(itertools.product(*(range(c) for c in cards)))
    assert len(columns) == env.n_actions


def test_do_nothing_is_a_value(case):
    name, env = case
    if name not in DO_NOTHING:
        pytest.skip("no do-nothing action in this domain")
    columns = env.action_columns(np.arange(env.n_actions))
    assert len(env.action_vars) == 1 and columns[0, 0] == DO_NOTHING[name]  # env action 0 does nothing


def test_no_done_variable_and_one_next_column_per_state(case):
    _, env = case
    assert not {v for v in env.state_vars if v.lower() in {"done", "terminated", "truncated", "terminal"}}
    df = env.sample(5, 5, "random")
    assert [c for c in df.columns if c.endswith(NEXT_MARK)] == [v + NEXT_MARK for v in env.state_vars]
    assert not set(df.columns) - set(
        env.state_vars + env.action_vars + [v + NEXT_MARK for v in env.state_vars]
    )
