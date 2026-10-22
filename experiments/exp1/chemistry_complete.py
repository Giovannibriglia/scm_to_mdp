"""Chemistry, complete grid: every size x graph density x behaviour policy x number of rollouts.

    size        objects: 4, 5, 6, 7 (3 colours: 81 ... 2187 observed states)
    knob        density of the reaction graph: 0.0 (no reaction), 0.4, 1.0 (at most 2 parents)
    policies    uniform      push a random object or nothing
                expert       the best policy over the observed state, a uniform action w.p. 0.05
                confounded   the expert, but while a hidden catalyst is on it waits (does nothing)
                             w.p. 0.95
    rollouts    10 ... 1600 (horizon 20), 3 seeds, 7 standard methods (no exact solver: 3 colours)

The catalyst (src/scm_to_mdp/envs/chemistry.py) is in every setting: drawn afresh at every step
(p = 0.3), it drives the reaction of one object (w.p. 0.95 with it, 0.3 without), and it is never in
the data. With density 0 there is no reaction: the catalyst changes nothing (a negative control).
8 objects are left out: with the catalyst the true model has 13,122 states and needs about 25 GB.

Expectations, written before the run:
    H1  structure pays off most with a sparse graph and less as it gets denser;
    H2  expert data: robust methods beat plug-in at small budgets; flat MDP's promise fails;
    H3  expert data: discovery loses the push edges the expert rarely varies;
    H4  confounded data: the promise fails once the intervals are narrower than the bias; never at
        density 0 (no bias there).

Run from the repository root (many hours; a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1/chemistry_complete.py                 # a new run
    .venv/bin/python experiments/exp1/chemistry_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/chemistry_complete.py --report-only   # tables and figures only
Output: results/exp1/chemistry_complete_<date>_<time>/ (one folder per run): summary/ for the whole
grid, models/ of the first seed and one folder per setting (tool/runs.py, tool/store.py).
"""

import dataclasses
import warnings
from pathlib import Path

from scm_to_mdp.envs import Chemistry
from scm_to_mdp.tool import Grid, Limits, Policy, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = [4, 5, 6, 7]
DENSITIES = ["0.0", "0.4", "1.0"]
POLICIES = [
    Policy("uniform"),
    Policy("expert", epsilon=0.05),
    Policy("confounded", epsilon=0.05, strength=0.95),
]
N_ROLLOUTS = [10, 25, 50, 100, 200, 400, 800, 1600]
SEEDS = [0, 1, 2]  # the seed also draws the graph and the target colours
HORIZON, GAMMA, DELTA = 20, 0.9, 0.05
CATALYST = 0.3  # P(catalyst on) at every step
LIMITS = Limits(mem_gb=8.0, timeout_s={"discovery": 60.0, "selection": 60.0, "solve": 120.0})
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "chemistry_complete"


def make_env(n_objects, density: str, seed: int) -> Chemistry:
    return Chemistry(
        float(density), seed, n_objects=int(n_objects), n_colours=3, max_parents=2, catalyst=CATALYST
    )


GRID = Grid(
    name="Chemistry",
    make_env=make_env,
    sizes=SIZES,
    size_name="objects",
    knobs=DENSITIES,
    knob_name="density",
    policies=POLICIES,
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=4, delta=DELTA, exact=False),  # itself, its push, 2 parents
    limits=LIMITS,
    workers=3,
    out_dir=OUT_DIR,
    constants={"catalyst": CATALYST, "n_colours": 3, "max_parents": 2},  # checked when a run resumes
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
