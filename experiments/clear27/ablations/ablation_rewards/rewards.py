"""The ladder of rewards of the ablation, from the easiest to the hardest, as rules in N (SysAdmin).

Every reward reads the machines only (never the hidden storm) and is given to the learner, as the
original is. Levels 1-4 at every N; levels 5 and 6 from N = FULL_LADDER_FROM, where some states are rare.

    level 1  additive    the original: the number of machines up, minus REPAIR_COST when a repair is made
    level 2  subset      only some machines count (scaled to the same maximum N): machine 0 alone (the
                         one the hidden storm hits), machine 1 alone, the first ceil(N/2), all but the last
    level 3  cost        the original with a repair cost of 0.1, 1.0 and 2.0: when to repair changes
    level 4  threshold   N when at least k machines are up, else 0: k = ceil(N/3), ceil(2N/3), N. A joint
                         event of several machines, not a sum
    level 5  goal        1 on arriving in ONE state, which ends the episode: all up, all down, alternating
                         up/down, the first half up. No repair cost
    level 6  trap        the original plus a crash: arriving with k or more machines down costs CRASH_COST
                         and ends the episode, k = ceil(N/2), ceil(2N/3); and "rare": 1 on arriving in the
                         1 or the 3 least-visited states of the data, which end the episode

10 rewards at N=3, 11 at N=4, 19 from N=5 (at N=3 two subsets coincide).
"""

import math
from dataclasses import dataclass

import numpy as np

from scm_to_mdp.reward_family import Task

REPAIR_COST = 0.4  # SysAdmin's (envs/sysadmin.py)
CRASH_COST = 10.0
FULL_LADDER_FROM = 5
LEVELS = {1: "additive", 2: "subset", 3: "cost", 4: "threshold", 5: "goal", 6: "trap"}


@dataclass(frozen=True)
class Rung:
    level: int
    name: str
    task: Task  # the reward over the observed states and every true action vector


def levels_of(machines: int) -> list[int]:
    return [1, 2, 3, 4] if machines < FULL_LADDER_FROM else [1, 2, 3, 4, 5, 6]


def subsets(machines: int) -> dict[str, list[int]]:
    """Level 2: the machines that count (duplicates dropped: at N=3 the half is all but the last)."""
    chosen = {
        "m0": [0],
        "m1": [1],
        "half": list(range(math.ceil(machines / 2))),
        "all_but_one": list(range(machines - 1)),
    }
    out, seen = {}, set()
    for name, which in chosen.items():
        if tuple(which) not in seen:
            out[name] = which
            seen.add(tuple(which))
    return out


def thresholds(machines: int) -> list[int]:
    return sorted({math.ceil(machines / 3), math.ceil(2 * machines / 3), machines})


def crashes(machines: int) -> list[int]:
    return sorted({math.ceil(machines / 2), math.ceil(2 * machines / 3)})


def names(machines: int) -> list[tuple[int, str]]:
    """(level, name) of every reward at N machines, in order (no data needed)."""
    out = [(1, "additive")]
    out += [(2, f"subset_{name}") for name in subsets(machines)]
    out += [(3, f"cost_{cost:g}") for cost in (0.1, 1.0, 2.0)]
    out += [(4, f"threshold_{k}") for k in thresholds(machines)]
    if machines >= FULL_LADDER_FROM:
        out += [(5, f"goal_{name}") for name in ("all_up", "all_down", "alternating", "half_up")]
        out += [(6, f"crash_{k}") for k in crashes(machines)]
        out += [(6, f"rare_{g}") for g in (1, 3)]
    return out


def ladder(B: np.ndarray, actions: np.ndarray, terminal: np.ndarray, visits: np.ndarray) -> list[Rung]:
    """The rewards of one (size, seed). B [S, N]: the observed states (0 = down, 1 = up); actions [A, 1]:
    the true action vectors (reboot: 0 = nothing, > 0 = a repair); terminal [S]: the original terminal
    states; visits [S]: how often the data visit each state (the "rare" rewards)."""
    S, machines = B.shape
    repairs = (actions[:, 0] > 0)[None, :]
    up = B.sum(1).astype(float)

    def on_state(value: np.ndarray, cost: float = REPAIR_COST, name: str = "") -> Task:
        return Task(name, value[:, None] - cost * repairs, np.zeros(S), terminal.copy())

    def on_arrival(goals: np.ndarray, name: str) -> Task:
        r_next, end = np.zeros(S), terminal.copy()
        r_next[goals] = 1.0
        end[goals] = True
        return Task(name, np.zeros((S, len(actions))), r_next, end)

    def state(pattern: list[int]) -> np.ndarray:
        return np.flatnonzero((np.array(pattern) == B).all(1))

    tasks: dict[str, Task] = {"additive": on_state(up, name="additive")}
    for name, which in subsets(machines).items():
        tasks[f"subset_{name}"] = on_state(machines / len(which) * B[:, which].sum(1), name="subset")
    for cost in (0.1, 1.0, 2.0):
        tasks[f"cost_{cost:g}"] = on_state(up, cost, name="cost")
    for k in thresholds(machines):
        tasks[f"threshold_{k}"] = on_state(machines * (up >= k).astype(float), name="threshold")
    if machines >= FULL_LADDER_FROM:
        half = math.ceil(machines / 2)
        goals = {
            "all_up": [1] * machines,
            "all_down": [0] * machines,
            "alternating": [(i + 1) % 2 for i in range(machines)],
            "half_up": [1] * half + [0] * (machines - half),
        }
        for name, pattern in goals.items():
            tasks[f"goal_{name}"] = on_arrival(state(pattern), "goal")
        for k in crashes(machines):
            crashed = (machines - up) >= k
            base = on_state(up)
            tasks[f"crash_{k}"] = Task("trap", base.R, -CRASH_COST * crashed, terminal | crashed)
        live = np.flatnonzero(~terminal)
        rarest = live[np.lexsort((live, visits[live]))]  # the least visited first; ties: the lower state
        for g in (1, 3):
            tasks[f"rare_{g}"] = on_arrival(rarest[:g], "trap")
    return [Rung(level, name, tasks[name]) for level, name in names(machines)]
