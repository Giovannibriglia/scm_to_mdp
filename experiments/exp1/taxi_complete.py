"""Taxi, complete grid: both navigation settings x behaviour policy x number of rollouts (protocol.py: the
claims, the loggers, the seeds and the expectations common to every domain). The declared LIMIT case.

A taxi on a 5 x 5 grid picks up a passenger at one landmark and puts the passenger down at another
(Dietterich, JAIR 2000; src/scm_to_mdp/envs/taxi.py: x, y, passenger, destination, Gymnasium's 500 states,
one action variable with 6 values; the episode ends when the passenger is at the destination).

    size        navigation: "deterministic" (Dietterich's Sec. 3.1) or "noisy" (0.8 intended, 0.1 to each
                side, his Fickle Taxi, Sec. 7.1, with our independent axes)
    policies    uniform, expert eps 0.05, expert eps 0 (no hidden variable: no confounded logger, no shift)
    rollouts    25 ... 800 (horizon 50, gamma 0.99), 5 seeds, the standard methods (not 0/1)
    transfer    pickup_only, costly_moves (ours)

Why it is here: context-specific structure. The passenger changes only when the taxi is on the right
square and picks up or puts down, so its factor has about 1,800 parent rows that can occur, nearly as many
as the flat model's (s, a) rows: structure buys little, and the robust promise stays near "never
delivered" (bible 5.7: where the SCM view does not help). gamma 0.99: with -1 per step and +20 at
delivery, V* at the start is positive only for a long enough horizon.

Expectations for this domain, written 2026-10-24 before the runs:
    H1  the causal and flat robust arms are close, both far from V* at every n (A does not hold here);
    H2  BIC drops the square and the destination from the passenger's parents (rare rows), and the
        causal promise is then not valid;
    H3  cov_joint - cov_factor is small for both tasks (B1 cannot show).

Run from the repository root (a few hours; gamma 0.99 makes value iteration slow):
    .venv/bin/python experiments/exp1/taxi_complete.py [--resume | --report-only] [--workers N]
Output: results/exp1/taxi_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, SEEDS, policies

from scm_to_mdp.envs import Taxi
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["deterministic", "noisy"]
N_ROLLOUTS = [25, 50, 100, 200, 400, 800]
HORIZON, GAMMA = 50, 0.99
LIMITS = Limits(
    mem_gb=6.0,
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 120.0, "solve": 300.0, "training": 1800.0},
)  # gamma 0.99: slow value iteration
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "taxi_complete"


def make_env(size: str, knob: str, seed: int) -> Taxi:
    return Taxi(size, seed, axes="independent")


GRID = Grid(
    name="Taxi",
    make_env=make_env,
    sizes=SIZES,
    size_name="navigation",
    knobs=["original"],
    knob_name="variant",
    policies=policies(confounded=False),
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=5, delta=DELTA, exact=False, deep=DEEP_BASELINES, state_parents=3),
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
