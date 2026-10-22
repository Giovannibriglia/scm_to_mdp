"""The coffee robot: a classic factored MDP with a hand-written causal graph.

A robot buys coffee at the shop and delivers it to the user in the office; moving in the rain without
an umbrella makes it wet (Dearden & Boutilier, UAI 1994; src/scm_to_mdp/envs/coffee.py says which
numbers are from the paper and which are ours). Sizes: "coffee" (6 binary variables, 64 states,
4 actions) and "coffee+snack" (8 variables, 256 states, 6 actions).

What it shows: context-specific structure (an umbrella matters only when it rains and the robot
moves; delivering matters only in the office with coffee) in a problem small enough to read the
learned graph by hand (results/.../tables.md, SHD per method).

Run from the repository root (about 15 minutes):
    .venv/bin/python experiments/coffee.py [--resume | --report-only] [--workers N] [--device D]
Output: results/experiments/coffee_<date>_<time>/ (one folder per run; tool/runs.py)
"""

import dataclasses
import warnings
from pathlib import Path

from scm_to_mdp.envs import Coffee
from scm_to_mdp.tool import Experiment, Limits, report, run_experiment, run_folder, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["coffee", "coffee+snack"]
N_ROLLOUTS = [5, 10, 25, 50, 100, 200, 400, 800]  # coffee+snack needs more than 200
SEEDS = [0, 1, 2]
HORIZON, GAMMA, DELTA = 20, 0.9, 0.05
LIMITS = Limits(mem_gb=6.0, timeout_s={"discovery": 60.0, "selection": 60.0, "solve": 60.0})
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "coffee"


def make_env(size: str, seed: int) -> Coffee:
    return Coffee(size, seed)


def collect(env: Coffee, n_rollouts: int):
    """A robot acting at random; every rollout starts in the office, dry, without coffee."""
    return env.sample(n_rollouts, HORIZON, policy="random")


METHODS = standard_methods(max_parents=4, delta=DELTA, exact=True)  # true in-degree: up to 4

EXPERIMENT = Experiment(
    name="Coffee robot",
    make_env=make_env,
    collect=collect,
    sizes=SIZES,
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    methods=METHODS,
    gamma=GAMMA,
    limits=LIMITS,
    workers="auto",  # what the machine allows (utils/compute.py); --workers N overrides
    out_dir=OUT_DIR,
    settings={"horizon": HORIZON},  # checked when a run resumes
    size_name="domain",
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    if not report_only:
        run_experiment(dataclasses.replace(EXPERIMENT, out_dir=out_dir))
    report(out_dir, target=0.95, delta=DELTA)
