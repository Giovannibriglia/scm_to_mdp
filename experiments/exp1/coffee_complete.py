"""The coffee robot, complete grid: both sizes x behaviour policy x number of rollouts (protocol.py: the
claims, the loggers, the seeds and the expectations common to every domain).

A robot buys coffee at the shop and delivers it to the user in the office; moving in the rain without an
umbrella makes it wet (Dearden & Boutilier, UAI 1994; src/scm_to_mdp/envs/coffee.py says which numbers
are from the paper and which are ours).

    size        "coffee" (6 binary variables, 64 states, 4 actions), "coffee+snack" (8 variables, 256
                states, 6 actions)
    policies    uniform, expert eps 0.05, expert eps 0 (no hidden variable: no confounded logger, no shift)
    rollouts    5 ... 800 (horizon 20, gamma 0.9), 5 seeds, the standard methods with the per-(s,a)
                product set (every variable is 0/1)

What it adds: the only graph small enough to read by hand (claim C: summary/edges.csv, the cbn pictures),
context-specific structure (the umbrella matters only when it rains and the robot moves), and a 0/1
domain for the cost of the box (box_looseness.csv). Its alternative rewards (coffee_only, dry_delivery,
snack_only) need rows the original policy rarely uses (the umbrella in the rain).

Expectations for this domain, written 2026-10-24 before the runs:
    H1  the true graph is recovered by BIC at a few hundred rollouts under the uniform logger (C);
    H2  dry_delivery needs get_umbrella in the rain, which the original expert rarely does: under eps = 0
        cov_joint > cov_factor and the factored arms transfer better (B1);
    H3  the per-(s,a) product set beats the box by little (few variables per factor) (A, M4).

Run from the repository root (about half an hour):
    .venv/bin/python experiments/exp1/coffee_complete.py [--resume | --report-only] [--workers N]
Output: results/exp1/coffee_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, SEEDS, policies

from scm_to_mdp.envs import Coffee
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["coffee", "coffee+snack"]
N_ROLLOUTS = [5, 10, 25, 50, 100, 200, 400, 800]  # coffee+snack needs more than 200
HORIZON, GAMMA = 20, 0.9
LIMITS = Limits(
    mem_gb=6.0,
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 60.0, "solve": 60.0, "training": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "coffee_complete"


def make_env(size: str, knob: str, seed: int) -> Coffee:
    return Coffee(size, seed)


GRID = Grid(
    name="Coffee robot",
    make_env=make_env,
    sizes=SIZES,
    size_name="domain",
    knobs=["original"],
    knob_name="variant",
    policies=policies(confounded=False),
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=4, delta=DELTA, exact=True, deep=DEEP_BASELINES, state_parents=3),
    limits=LIMITS,
    workers="auto",
    out_dir=OUT_DIR,
    reward_transfer=True,
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
