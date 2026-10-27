"""Grids of experiments: every combination of problem size x knob x behaviour policy x transitions.

A Grid names one problem and what to vary:

    Grid(
        name="SysAdmin",
        make_env=make_env,              # (size, knob, seed) -> Env, a top-level function
        sizes=[2, 3, 4], size_name="N",
        knobs=["ring", "star"], knob_name="topology",
        policies=[Policy("uniform"), Policy("expert", epsilon=0.05), Policy("confounded")],
        n_transitions=[10, 50, 200], seeds=[0, 1, 2], horizon=20,
        methods=standard_methods(...),
    )

sizes may also be a dict {knob: sizes}, when each knob has its own (e.g. Frozen Lake's maps for one and
for two agents: the knob is the number of agents). experiments(grid) turns it into one Experiment
(tool/sweep.py) per (knob, policy), each in its own
folder <out_dir>/<knob_name>=<knob>/<policy key>/ with its own results.csv, so a stopped grid
resumes setting by setting. The data of a setting are drawn by the behaviour policy (Env.behaviour:
"uniform", "expert", "confounded"), and every row gets the logger's coverage and confounding bias.
run_grid runs them all in one pool of workers (a free worker takes a job of any setting; each
setting keeps its own folder and its waves of sizes) and writes grid.json (the order of everything,
for the report);
report_grid (tool/report_grid.py) reads only the results.csv files and grid.json.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from scm_to_mdp.tool.methods import Limits, Method
from scm_to_mdp.tool.sweep import Experiment, run_sweeps


@dataclass(frozen=True)
class Policy:
    """A behaviour policy of Env.behaviour: "uniform", "expert" (epsilon: share of uniform actions)
    or "confounded" (the expert, plus the confounded action w.p. strength when the confounder is on)."""

    name: str
    epsilon: float = 0.05
    strength: float = 0.95

    def __post_init__(self):
        if self.name not in ("uniform", "expert", "confounded"):
            raise ValueError(f"policy must be 'uniform', 'expert' or 'confounded', not {self.name!r}")

    @property
    def label(self) -> str:
        if self.name == "uniform":
            return "uniform"
        if self.name == "expert":
            return f"expert ε={self.epsilon:g}"
        return f"confounded ε={self.epsilon:g} s={self.strength:g}"

    @property
    def key(self) -> str:
        """A folder name."""
        if self.name == "uniform":
            return "uniform"
        if self.name == "expert":
            return f"expert_eps{self.epsilon:g}"
        return f"confounded_eps{self.epsilon:g}_s{self.strength:g}"


@dataclass(frozen=True)
class Grid:
    name: str
    make_env: Callable[[Any, Any, int], Any]  # (size, knob, seed) -> Env; defined at module top level
    sizes: list | dict  # a list, or {knob: list} when each knob has its own sizes
    knobs: list
    policies: list[Policy]
    n_transitions: list[int]
    seeds: list[int]
    methods: dict[str, Method]
    horizon: int
    gamma: float = 0.9
    limits: Limits | None = field(default_factory=Limits)
    workers: int | str = "auto"  # parallel jobs: "auto" = what the machine allows (utils/compute.py)
    out_dir: Path = Path("results")
    size_name: str = "size"
    knob_name: str = "knob"
    n_starts: int = 20
    skip: Callable[[Any, Policy], bool] | None = None  # (knob, policy) -> True: not a valid setting
    constants: dict = field(default_factory=dict)  # the script's constants used by make_env (grid.json)
    save_models: str = "largest"  # first seed: save the models (tool/store.py): "largest" n, "all", "none"
    pictures_max_states: int = 16  # draw the MDPs (chain pictures) of problems with at most this many states
    reward_transfer: bool = False  # also evaluate under the environment's alternative rewards (sweep.py)
    reward_family: Any = None  # random rewards on the same dynamics (reward_family.RewardFamily; sweep.py)
    failure_horizon: int | None = None  # the failure probabilities: within this many steps (robustness.py)
    # dynamics shift (sweep.py): (size, knob, seed) -> {name: Env} with other dynamics, same observed states
    shifts: Callable[[Any, Any, int], dict] | None = None
    # relative solve limit (tool/sweep.py): max(solve_floor_s, solve_factor x the oracle's work at rate gamma)
    solve_factor: float | None = None
    solve_floor_s: float = 60.0


# ---- the pieces an Experiment needs (top-level, so that worker processes can receive them) ----------
def env_at(make_env, knob, size, seed):
    return make_env(size, knob, seed)


def logger_table(policy: Policy, gamma: float, env):
    """The behaviour policy of a setting as a table [S, A] over the full states."""
    return env.behaviour(policy.name, gamma, policy.epsilon, policy.strength)


def collect_with(policy: Policy, gamma: float, horizon: int, env, n_transitions: int):
    """At least n_transitions transitions of the setting's behaviour policy, without the hidden columns:
    whole episodes of at most `horizon` steps (fewer where an episode ends early), drawn in batches
    until there are enough; the episode ids run on across batches. A first batch of at most 32
    episodes measures the episode length; each next batch is sized from the mean length so far."""
    behaviour = logger_table(policy, gamma, env)
    batches, total, start = [], 0, 0
    length = min(horizon, 100)  # the expected episode length, updated after every batch
    while total < n_transitions:
        n_episodes = max(1, -(-(n_transitions - total) // length))
        if not batches:
            n_episodes = min(n_episodes, 32)  # a pilot: the episode length is not known yet
        df = env.sample(n_episodes, horizon, behaviour).reset_index()
        df["episode"] += start  # number the episodes on from the previous batch
        batches.append(df.set_index(["episode", "t"]))
        total += len(df)
        start = int(df["episode"].max()) + 1
        length = max(1, total // start)  # the mean episode length so far
    return env.hide(pd.concat(batches))


def sizes_of(grid: Grid, knob) -> list:
    """The sizes of one knob."""
    if isinstance(grid.sizes, dict):
        if knob not in grid.sizes:
            raise ValueError(f"no sizes for knob {knob!r}: sizes has {list(grid.sizes)}")
        return list(grid.sizes[knob])
    return list(grid.sizes)


def all_sizes(grid: Grid) -> list:
    """Every size of the grid, each once, in order (the knobs' own orders, one after the other)."""
    out = []
    for knob in grid.knobs:
        out += [s for s in sizes_of(grid, knob) if s not in out]
    return out


def settings(grid: Grid) -> list[tuple[Any, Policy]]:
    """Every valid (knob, policy), in the grid's order."""
    return [(k, p) for k in grid.knobs for p in grid.policies if grid.skip is None or not grid.skip(k, p)]


def folder(grid: Grid, knob, policy: Policy) -> Path:
    return Path(grid.out_dir) / f"{grid.knob_name}={knob}" / policy.key


def experiments(grid: Grid) -> list[Experiment]:
    return [
        Experiment(
            name=f"{grid.name}, {grid.knob_name}={knob}, {policy.label}",
            make_env=partial(env_at, grid.make_env, knob),
            collect=partial(collect_with, policy, grid.gamma, grid.horizon),
            sizes=sizes_of(grid, knob),
            n_transitions=list(grid.n_transitions),
            seeds=list(grid.seeds),
            methods=grid.methods,
            gamma=grid.gamma,
            n_starts=grid.n_starts,
            limits=grid.limits,
            workers=grid.workers,
            out_dir=folder(grid, knob, policy),
            size_name=grid.size_name,
            knob=str(knob),
            policy=policy.label,
            behaviour=partial(logger_table, policy, grid.gamma),
            horizon=grid.horizon,
            failure_horizon=grid.failure_horizon,
            settings={"policy": asdict(policy), **grid.constants},
            save_models=grid.save_models,
            pictures_max_states=grid.pictures_max_states,
            reward_transfer=grid.reward_transfer,
            reward_family=grid.reward_family,
            shifts=None if grid.shifts is None else partial(env_at, grid.shifts, knob),
            solve_factor=grid.solve_factor,
            solve_floor_s=grid.solve_floor_s,
        )
        for knob, policy in settings(grid)
    ]


def write_grid_file(grid: Grid) -> None:
    """grid.json: the names and the order of sizes, knobs and policies (read by report_grid)."""
    out = Path(grid.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    description = {
        "name": grid.name,
        "size_name": grid.size_name,
        "sizes": [str(s) for s in all_sizes(grid)],
        "sizes_by_knob": {str(k): [str(s) for s in sizes_of(grid, k)] for k in grid.knobs},
        "knob_name": grid.knob_name,
        "knobs": [str(k) for k in grid.knobs],
        "policies": [p.label for p in grid.policies],
        "n_transitions": sorted(grid.n_transitions),
        "seeds": list(grid.seeds),
        "horizon": grid.horizon,
        "gamma": grid.gamma,
        "constants": grid.constants,
        "settings": [
            {"knob": str(k), "policy": p.label, "folder": str(folder(grid, k, p).relative_to(out))}
            for k, p in settings(grid)
        ],
    }
    (out / "grid.json").write_text(json.dumps(description, indent=2))


def run_grid(grid: Grid, resume: bool = True) -> None:
    """Run every setting of the grid in one pool of grid.workers processes."""
    write_grid_file(grid)
    exps = experiments(grid)
    print(f"\n===== {grid.name}: {len(exps)} settings -> {grid.out_dir} =====")
    for exp in exps:
        print(f"  {exp.name}")
    run_sweeps(exps, resume=resume, title=grid.name)
