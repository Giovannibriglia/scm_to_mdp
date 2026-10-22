"""Taxi (Dietterich, JAIR 2000): a familiar benchmark with context-specific structure.

A taxi on a 5 x 5 grid picks up a passenger at one landmark and puts the passenger down at another
(src/scm_to_mdp/envs/taxi.py; 600 states with the "delivered" end state, 6 one-hot actions).
Sizes: "deterministic" moves (Dietterich's Sec. 3.1) and "noisy" moves (0.8 intended, 0.1 to each
side, as in his Fickle Taxi, Sec. 7.1).

Things to know: - the taxi's square has 25 values and depends on 4 one-hot move columns: of the 25 * 2^4
rows of its table only 25 * 5 can occur, and BIC's penalty counts only those (discovery/bic_search.py); -
the passenger changes only when the taxi is on the right square and picks up or puts down: these rows are
rare, so BIC drops the taxi's square and the destination from the passenger's parents (a context-specific
effect) and the promise of the causal methods is then not valid; - the promise of every robust method stays
near -1/(1 - gamma) (= "never delivered"), even with the true graph: the passenger's table has about 1,800
rows that can occur, their intervals stay wide (median total width 1.5 at 3,200 rollouts in an earlier run),
and the box leaves the adversary room to block the delivery. A smaller discount does not fix this (at 0.95
the promise is about -17 to -20, and with noisy moves V* = 0.05, so "95% of V*" would mean nothing). Taxi
shows where the guarantee is empty; - the discount is 0.99: with -1 per step and +20 at delivery, V* at the
start is positive only for a long enough horizon (3.7 with noisy moves at 0.99), and the "95% of V*" ratio
needs V* > 0. With V* this close to 0, 95% of V* is a near-optimal bar (the random driver has a value near
-370); - DiFaRL's action groups are about actions that act together; here exactly one action is taken.

Run from the repository root (about 1 hour; gamma 0.99 makes value iteration slow):
    .venv/bin/python experiments/taxi.py [--resume | --report-only]
Output: results/experiments/taxi_<date>_<time>/ (one folder per run; tool/runs.py)
"""

import dataclasses
import warnings
from pathlib import Path

from scm_to_mdp.envs import Taxi
from scm_to_mdp.tool import Experiment, Limits, report, run_experiment, run_folder, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["deterministic", "noisy"]
N_ROLLOUTS = [25, 50, 100, 200, 400, 800]
SEEDS = [0, 1, 2]
HORIZON, GAMMA, DELTA = 50, 0.99, 0.05
LIMITS = Limits(
    mem_gb=6.0, timeout_s={"discovery": 60.0, "selection": 60.0, "solve": 120.0}
)  # gamma 0.99: slow VI
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "experiments" / "taxi"


def make_env(size: str, seed: int) -> Taxi:
    return Taxi(size, seed)


def collect(env: Taxi, n_rollouts: int):
    """A driver who picks one of the six actions at random; a rollout ends at a delivery."""
    return env.sample(n_rollouts, HORIZON, policy="random")


# true in-degree: 5 (the taxi's square: itself and the 4 moves; the passenger: itself, the square,
# the destination, pickup and putdown); variables are not 0/1
METHODS = standard_methods(max_parents=5, delta=DELTA, exact=False)

EXPERIMENT = Experiment(
    name="Taxi",
    make_env=make_env,
    collect=collect,
    sizes=SIZES,
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    methods=METHODS,
    gamma=GAMMA,
    limits=LIMITS,
    workers=3,
    out_dir=OUT_DIR,
    settings={"horizon": HORIZON},  # checked when a run resumes
    size_name="navigation",
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    if not report_only:
        run_experiment(dataclasses.replace(EXPERIMENT, out_dir=out_dir))
    report(out_dir, target=0.95, delta=DELTA)
