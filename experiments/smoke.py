"""Smoke versions of the experiments (experiments/exp1_smoke, experiments/exp2_smoke): the same
methods, domains and evaluation as experiments 1 and 2, on a tiny grid, to check in minutes that a
script runs end to end and to look at its final tables and figures. Their numbers are NOT results.

What shrinks (everything else is the real experiment's, from its protocol.py and common.py):
    sizes       the SIZES smallest of every domain (per knob)
    data        N_TRANSITIONS transitions per dataset, SEEDS seeds
    deep nets   offline DQN / CQL train for DEEP_OPTIONS["steps"] steps only
    rewards     where the real experiment re-plans under random rewards (experiment 1, Sepsis):
                REWARDS (one random reward per family) at every budget
Output: results/<experiment>_smoke/<domain>_<kind>_<date>_<time>/.
"""

import dataclasses
import importlib.util
from pathlib import Path

from scm_to_mdp.reward_family import RewardFamily
from scm_to_mdp.tool import Grid

HERE = Path(__file__).resolve().parent
SIZES = 2  # the smallest sizes of every domain (per knob)
N_TRANSITIONS = [100, 500, 2000]
SEEDS = [0, 1, 2]
DEEP_OPTIONS = {"steps": 500, "hidden": 32}  # offline DQN / CQL: only that they train
REWARDS = RewardFamily(per_family=1)  # one random reward per family


def load_protocol(experiment: str):
    """The real experiment's protocol.py (loaded by path: every experiment folder has one)."""
    spec = importlib.util.spec_from_file_location(f"{experiment}_protocol", HERE / experiment / "protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shrink(grid: Grid, experiment: str) -> Grid:
    """The real grid on the smoke settings above; results under results/<experiment>/."""
    sizes = (
        {k: v[:SIZES] for k, v in grid.sizes.items()} if isinstance(grid.sizes, dict) else grid.sizes[:SIZES]
    )
    methods = {
        name: dataclasses.replace(m, solver_options={**m.solver_options, **DEEP_OPTIONS})
        if m.learner in ("offline_dqn", "offline_cql")
        else m
        for name, m in grid.methods.items()
    }
    return dataclasses.replace(
        grid,
        sizes=sizes,
        n_transitions=N_TRANSITIONS,
        seeds=SEEDS,
        methods=methods,
        reward_family=REWARDS if grid.reward_family is not None else None,
        save_models="largest",
        out_dir=grid.out_dir.parents[1] / experiment / grid.out_dir.name,
    )
