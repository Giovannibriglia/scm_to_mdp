"""Chemistry, complete grid: every size x graph density x behaviour policy x number of rollouts
(protocol.py: the claims, the loggers, the seeds, the shifts and the expectations common to every domain).

    size        objects: 4, 5, 6, 7 (3 colours: 81 ... 2187 observed states)
    knob        density of the reaction graph: 0.0 (no reaction), 0.4, 1.0 (at most 2 parents)
    policies    uniform (push a random object or nothing), expert eps 0.05, expert eps 0, confounded (the
                expert, but while the hidden catalyst is on it waits w.p. 0.95)
    hidden      the catalyst (src/scm_to_mdp/envs/chemistry.py) drives the reaction of one object (w.p.
                0.95 with it, 0.3 without), on w.p. 0.1 in the data, 0.3 and 0.6 at deployment; with
                density 0 there is no reaction and it changes nothing (a negative control)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.9), 5 seeds (the seed also draws the graph and
                the target
                colours), the standard methods (no per-(s,a) product set: 3 colours)
8 objects are left out: with the catalyst the true model has 13,122 states and needs about 25 GB.

Expectations for this domain, written 2026-10-24 before the runs:
    H1  structure pays off most with a sparse graph, less as it gets denser (A);
    H2  the expert pushes few objects: the push edges of the others are thin under eps = 0 (A3);
    H3  target_1..3 recolour objects the expert never pushes for the original task: cov_joint -
        cov_factor is large, and so is the transfer gain of the factored arms (B1);
    H4  confounded: the promise fails once the intervals are narrower than the bias; never at density
        0 (control).

Run from the repository root (many hours; a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1/chemistry_complete.py                 # a new run
    .venv/bin/python experiments/exp1/chemistry_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/chemistry_complete.py --report-only   # tables and figures only
    .venv/bin/python experiments/exp1/chemistry_complete.py --workers 8 --device cpu  # own plan
Output: results/exp1/chemistry_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, HORIZON, SAVE_MODELS, SEEDS, SHIFT_HIDDEN, TRAIN_HIDDEN, policies

from scm_to_mdp.envs import Chemistry
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = [4, 5, 6, 7]
DENSITIES = ["0.0", "0.4", "1.0"]
# of HORIZON steps: 200 ... 32k transitions (was 10 ... 1600 rollouts of 20)
N_EPISODES = [1, 2, 4, 8, 16, 32, 64, 128]
GAMMA = 0.9
LIMITS = Limits(
    mem_gb=8.0,
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 120.0, "solve": 120.0, "training": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "chemistry_complete"


def chemistry(n_objects, density: str, seed: int, catalyst: float) -> Chemistry:
    return Chemistry(
        float(density), seed, n_objects=int(n_objects), n_colours=3, max_parents=2, catalyst=catalyst
    )


def make_env(n_objects, density: str, seed: int) -> Chemistry:
    return chemistry(n_objects, density, seed, TRAIN_HIDDEN)


def shifts(n_objects, density: str, seed: int) -> dict:
    """The same reactions where the catalyst is on more often (the learner never sees it)."""
    return {f"catalyst={p:g}": chemistry(n_objects, density, seed, p) for p in SHIFT_HIDDEN}


GRID = Grid(
    name="Chemistry",
    make_env=make_env,
    sizes=SIZES,
    size_name="objects",
    knobs=DENSITIES,
    knob_name="density",
    policies=policies(),
    n_rollouts=N_EPISODES,  # one rollout = one episode of HORIZON steps
    save_models=SAVE_MODELS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=4, delta=DELTA, exact=False, deep=DEEP_BASELINES, state_parents=3),
    limits=LIMITS,
    workers="auto",  # what the machine allows (utils/compute.py); --workers N overrides
    out_dir=OUT_DIR,
    reward_transfer=True,
    shifts=shifts,
    constants={"catalyst": TRAIN_HIDDEN, "shift_catalyst": SHIFT_HIDDEN, "n_colours": 3, "max_parents": 2},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
