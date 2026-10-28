"""The sparse graph (step S2, no discovery): every next-state variable x_i' gets itself and the action
variables it owns as parents, and nothing else (no state variable affects another one).

    pa(x_i') = (x_i, a_1, ..., a_m)      actions="all": every action variable (one controller)
    pa(x_i') = (x_i, a_k)                actions="by_index": the action variables with x_i's index
                                         (several agents: agent k's action moves only agent k)

"by_index" reads the trailing number of the names: Frozen Lake's x0, y0 own a0 and x1, y1 own a1;
Drone's x0, y0, z0 own d0. The domain says which rule applies (experiments/domains.py).

It is the opposite assumption to the full factored graph (full_factored.py): no interaction between
state variables. It contains the true graph only when the state variables do not interact, so its
interval set does not contain the true kernel where they do (SysAdmin's ring, Taxi's position); in
exchange each factor has |x_i| x |own actions| rows. It equals the full factored graph only with a
single state variable. Select it with the rule "given".
"""

import re

from scm_to_mdp.core import Candidates, Dataset
from scm_to_mdp.discovery import oracle

ACTIONS = ("all", "by_index")


def _index(name: str) -> str | None:
    match = re.search(r"(\d+)$", name)
    return match.group(1) if match else None


def own_actions(state_var: str, action_vars: list[str], actions: str = "all") -> tuple[str, ...]:
    """The action variables of state_var under the rule `actions` (see the module docstring)."""
    if actions not in ACTIONS:
        raise ValueError(f"actions must be one of {ACTIONS}, not {actions!r}")
    if actions == "all":
        return tuple(action_vars)
    index = _index(state_var)
    owned = tuple(a for a in action_vars if index is not None and _index(a) == index)
    if not owned:
        raise ValueError(
            f"actions='by_index': no action variable has the index of {state_var!r} (actions {action_vars})"
        )
    return owned


def parents(dataset: Dataset, actions: str = "all") -> dict[str, tuple[str, ...]]:
    """Itself and its own action variables, for every state variable."""
    return {v: (v, *own_actions(v, list(dataset.action_vars), actions)) for v in dataset.state_vars}


def fit(dataset: Dataset, actions: str = "all") -> Candidates:
    return oracle.fit(dataset, parents(dataset, actions))
