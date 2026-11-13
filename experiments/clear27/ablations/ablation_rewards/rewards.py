"""The ladders of rewards of the ablation, from the easiest to the hardest: one per domain, with the same
six levels. Every reward reads the observed state only (never a hidden variable), keeps the original
terminal states (the data stop there) and is given to the learner, as the original is.

    level 1  the environment's own reward
    level 2  subset      only some variables count
    level 3  cost        the original with a cost of acting: when to act changes
    level 4  threshold   a joint event of several variables, not a sum
    level 5  goal        1 on arriving in a chosen state (or small region), which ends the episode
    level 6  trap        a large penalty on the bad end, and "rare": 1 on arriving in the 1 or the 3
                         least-visited states of the data, which end the episode

SysAdmin, as rules in N (levels 1-4 at every N; 5 and 6 from N = FULL_LADDER_FROM, where some states are
rare):
    additive    the number of machines up, minus REPAIR_COST when a repair is made
    subset      only some machines count (scaled to the same maximum N): machine 0 alone (the one the hidden
                storm hits), machine 1 alone, the first ceil(N/2), all but the last
    cost        the original with a repair cost of 0.1, 1.0 and 2.0
    threshold   N when at least k machines are up, else 0: k = ceil(N/3), ceil(2N/3), N
    goal        all up, all down, alternating up/down, the first half up. No repair cost
    trap        the original plus a crash: arriving with k or more machines down costs CRASH_COST and ends
                the episode, k = ceil(N/2), ceil(2N/3); and the rare states
  10 rewards at N=3, 11 at N=4, 19 from N=5 (at N=3 two subsets coincide).

Sepsis (V vitals; the original: +1 on discharge, -1 on death, both end the episode):
    subset      on arrival, the share of the chosen vitals that are normal: heart rate alone, blood
                pressure alone, every vital (staying alive in a good state pays; a discharge ends it)
    cost        the original with a cost per treatment given: TREATMENT_COSTS
    threshold   on arrival, 1 when at least k vitals are normal: k = ceil(V/3), ceil(2V/3), V
    goal        every vital normal with only antibiotics on, with only ventilation on, with the three
                treatments on; heart rate high, the rest normal and no treatment (diabetic or not)
    trap        the original with death at -DEATH_COST; and the rare states
  16 rewards with 2 vitals, 17 with 3 or 4.

Drone (side L; the original: 1 on arriving in the target x high, y low, z high; the wall ends the episode
with nothing; "high" = the last cell, or the last two for L >= 5, as the target; "low" likewise):
    subset      the target on fewer axes: x high (any y, z); x and z high (any y)
    cost        the original with a cost per step: STEP_COSTS
    threshold   on arrival, 1 in a region set by two axes jointly: x and z high, x and y high, y and z high
    goal        the delivery in another corner: x1y1z1, x1y0z0, x0y0z1 (1 = high, 0 = low; the original is
                x1y0z1, which still ends the episode, with nothing)
    trap        the original with the wall at -WALL_COST; and the rare states
  15 rewards at every side.

The rare states of Sepsis and Drone are the live states in which the data ARRIVE least often, among those
in which they arrive at all: some states are never reached there (or are seen only as the start of an
episode), and a goal that cannot be reached is no task (SysAdmin visits every state).
"""

import math
from dataclasses import dataclass

import numpy as np

from scm_to_mdp.reward_family import Task

REPAIR_COST = 0.4  # SysAdmin's (envs/sysadmin.py)
CRASH_COST = 10.0
FULL_LADDER_FROM = 5
LEVELS = {1: "additive", 2: "subset", 3: "cost", 4: "threshold", 5: "goal", 6: "trap"}  # SysAdmin's
TREATMENT_COSTS = (0.01, 0.05, 0.2)  # Sepsis, per treatment given
DEATH_COST = 10.0
STEP_COSTS = (0.01, 0.03, 0.1)  # Drone, per step
WALL_COST = 10.0


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


# ---- Sepsis and Drone ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class World:
    """What a ladder reads of one (size, seed): the observed states and the environment's own reward."""

    B: np.ndarray  # [S, n] the observed states
    state_vars: list  # the names of the columns of B
    actions: np.ndarray  # [A, k] the true action vectors
    terminal: np.ndarray  # [S] bool, the original terminal states
    r_next: np.ndarray  # [S] the original reward of arriving
    visits: np.ndarray  # [S] how often the data visit each state (Sepsis, Drone: arrive in it)


def _rarest(w: World, g: int) -> np.ndarray:
    """The g least-visited states among the visited live ones (ties: the lower state)."""
    live = np.flatnonzero(~w.terminal & (w.visits > 0))
    return live[np.lexsort((live, w.visits[live]))][:g]


def _arriving(w: World, name: str, r_next: np.ndarray, goals: np.ndarray | None = None) -> Task:
    """A reward of arriving only. goals (a mask or indices): 1 there, and the episode ends there too."""
    r_next, end = np.asarray(r_next, dtype=float).copy(), w.terminal.copy()
    if goals is not None:
        if len(w.terminal[goals]) == 0 or w.terminal[goals].any():
            raise ValueError(f"the goal of {name!r} is empty or already a terminal state")
        r_next[goals] = 1.0
        end[goals] = True
    return Task(name, np.zeros((len(w.B), len(w.actions))), r_next, end)


SEPSIS_NORMAL = {"hr": 1, "sysbp": 1, "o2": 1, "glucose": 2}  # the vitals and their normal value
SEPSIS_FLAGS = ("abx_on", "vaso_on", "vent_on")  # the treatments given at the last step
SEPSIS_GOALS = ("on_abx", "on_vent", "on_all", "hr_high")


def sepsis_names(vitals: int) -> list[tuple[int, str]]:
    out = [(1, "original")]
    out += [(2, f"subset_{name}") for name in ("hr", "sysbp", "all")]
    out += [(3, f"cost_{cost:g}") for cost in TREATMENT_COSTS]
    out += [(4, f"threshold_{k}") for k in thresholds(vitals)]
    out += [(5, f"goal_{name}") for name in SEPSIS_GOALS]
    return out + [(6, f"death_{DEATH_COST:g}"), (6, "rare_1"), (6, "rare_3")]


def sepsis_ladder(w: World) -> list[Rung]:
    S, A = len(w.B), len(w.actions)
    column = {name: i for i, name in enumerate(w.state_vars)}
    vitals = [v for v in SEPSIS_NORMAL if v in column]
    normal = np.stack([w.B[:, column[v]] == SEPSIS_NORMAL[v] for v in vitals], axis=1)  # [S, V]
    flags = w.B[:, [column[f] for f in SEPSIS_FLAGS]]
    treatments = np.broadcast_to(w.actions.sum(1).astype(float)[None, :], (S, A))
    tasks: dict[str, Task] = {"original": _arriving(w, "original", w.r_next)}
    for name, chosen in {"hr": ["hr"], "sysbp": ["sysbp"], "all": vitals}.items():
        share = normal[:, [vitals.index(v) for v in chosen]].mean(1)
        tasks[f"subset_{name}"] = _arriving(w, "subset", share)
    for cost in TREATMENT_COSTS:
        tasks[f"cost_{cost:g}"] = Task("cost", -cost * treatments, w.r_next.astype(float), w.terminal.copy())
    for k in thresholds(len(vitals)):
        tasks[f"threshold_{k}"] = _arriving(w, "threshold", normal.sum(1) >= k)
    hr, rest = column["hr"], [i for i, v in enumerate(vitals) if v != "hr"]
    goals = {
        "on_abx": normal.all(1) & (flags == [1, 0, 0]).all(1),
        "on_vent": normal.all(1) & (flags == [0, 0, 1]).all(1),
        "on_all": normal.all(1) & (flags == [1, 1, 1]).all(1),
        "hr_high": (w.B[:, hr] == 2) & normal[:, rest].all(1) & (flags == 0).all(1),
    }
    for name in SEPSIS_GOALS:
        tasks[f"goal_{name}"] = _arriving(w, "goal", np.zeros(S), goals[name])
    death = w.terminal & (w.r_next < 0)
    tasks[f"death_{DEATH_COST:g}"] = _arriving(w, "trap", np.where(death, -DEATH_COST, w.r_next))
    for g in (1, 3):
        tasks[f"rare_{g}"] = _arriving(w, "trap", np.zeros(S), _rarest(w, g))
    return [Rung(level, name, tasks[name]) for level, name in sepsis_names(len(vitals))]


DRONE_GOALS = {"x1y1z1": (1, 1, 1), "x1y0z0": (1, 0, 0), "x0y0z1": (0, 0, 1)}  # 1 = high, 0 = low
DRONE_REGIONS = {"xz": (0, 2), "xy": (0, 1), "yz": (1, 2)}  # the two axes that must be high


def drone_names() -> list[tuple[int, str]]:
    out = [(1, "original"), (2, "subset_x"), (2, "subset_xz")]
    out += [(3, f"cost_{cost:g}") for cost in STEP_COSTS]
    out += [(4, f"threshold_{name}") for name in DRONE_REGIONS]
    out += [(5, f"goal_{name}") for name in DRONE_GOALS]
    return out + [(6, f"wall_{WALL_COST:g}"), (6, "rare_1"), (6, "rare_3")]


def drone_ladder(w: World) -> list[Rung]:
    S, A = len(w.B), len(w.actions)
    side = int(w.B.max()) + 1
    t = 1 if side >= 5 else 0  # the target's margin (envs/drone.py)
    high, low = w.B >= side - 1 - t, w.B <= t  # [S, 3] per axis x, y, z
    wall = w.terminal & (w.r_next == 0)
    tasks: dict[str, Task] = {"original": _arriving(w, "original", w.r_next)}
    for name, axes in {"x": [0], "xz": [0, 2]}.items():
        tasks[f"subset_{name}"] = _arriving(w, "subset", np.zeros(S), high[:, axes].all(1) & ~w.terminal)
        tasks[f"subset_{name}"].r_next[w.terminal & (w.r_next > 0)] = 1.0  # the original target is in it
    for cost in STEP_COSTS:
        tasks[f"cost_{cost:g}"] = Task(
            "cost", np.full((S, A), -cost), w.r_next.astype(float), w.terminal.copy()
        )
    for name, axes in DRONE_REGIONS.items():
        tasks[f"threshold_{name}"] = _arriving(w, "threshold", high[:, list(axes)].all(1) & ~wall)
    for name, sides in DRONE_GOALS.items():
        box = np.all([high[:, i] if s else low[:, i] for i, s in enumerate(sides)], axis=0)
        tasks[f"goal_{name}"] = _arriving(w, "goal", np.zeros(S), box)
    tasks[f"wall_{WALL_COST:g}"] = _arriving(w, "trap", np.where(wall, -WALL_COST, w.r_next))
    for g in (1, 3):
        tasks[f"rare_{g}"] = _arriving(w, "trap", np.zeros(S), _rarest(w, g))
    return [Rung(level, name, tasks[name]) for level, name in drone_names()]


# ---- every domain ------------------------------------------------------------------------------------
LEVEL_NAMES = {
    "sysadmin": LEVELS,
    "sepsis": LEVELS | {1: "original"},
    "drone": LEVELS | {1: "original"},
}


def names_of(domain: str, size: int) -> list[tuple[int, str]]:
    """(level, name) of every reward of a domain at a size (SysAdmin: machines; Sepsis: vitals; Drone: the
    side, which does not change the ladder), in order (no data needed)."""
    if domain == "sysadmin":
        return names(size)
    return sepsis_names(size) if domain == "sepsis" else drone_names()


def ladder_of(domain: str, w: World) -> list[Rung]:
    if domain == "sysadmin":
        return ladder(w.B, w.actions, w.terminal, w.visits)
    return sepsis_ladder(w) if domain == "sepsis" else drone_ladder(w)
