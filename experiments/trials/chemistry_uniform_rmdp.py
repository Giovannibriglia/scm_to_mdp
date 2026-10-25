"""Chemistry, uniform logger: every causal discovery -> RMDP method against flat RMDP and SPIBB (S2-S9).

    size        objects: 4, 5, 6, 7 (3 colours: 81 ... 2187 observed states)
    knob        density of the reaction graph: 1.0, at most 2 parents (experiment 1 also runs 0.0 and
                0.4); the seed also draws the graph and the target colours
    logger      uniform (every action tried in every visited state: no positivity problem)
    rollouts    50, 250, 750 (horizon 20, gamma 0.9), 5 seeds (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> RMDP with the plain discovered graph (no widening of
                the families: no "support", "hull" or "pad"), flat RMDP, flat SPIBB and SL-SPIBB (safe
                policy improvement: a guarantee over the behaviour policy, not a value bound; SL-SPIBB
                with candidate parent sets of 3 state variables), and true graph -> RMDP as the
                reference (protocol.py, RMDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      as in experiment 1: the catalyst drives the reaction of one object (w.p. 0.95 with it, 0.3
                without), on w.p. 0.1 at every step and never in the data

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/chemistry_uniform_rmdp.py                # a new run
    .venv/bin/python experiments/trials/chemistry_uniform_rmdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/chemistry_uniform_rmdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/chemistry_uniform_rmdp.py --workers 6    # own plan
Output: results/trials/chemistry_uniform_rmdp_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DELTA, N_ROLLOUTS, POLICIES, RMDP_METHODS, SEEDS, TRAIN_HIDDEN, methods

from scm_to_mdp.envs import Chemistry
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = [4, 5, 6, 7]
DENSITY = "1.0"
HORIZON, GAMMA = 20, 0.9
LIMITS = Limits(
    mem_gb=5.0,  # peak without BIC 3.7 GB (flat RMDP learning, 7 objects, 750 rollouts)
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 120.0, "training": 600.0, "solve": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "chemistry_uniform_rmdp"
METHODS = methods(RMDP_METHODS, state_parents=3)


def make_env(n_objects, density: str, seed: int) -> Chemistry:
    return Chemistry(
        float(density), seed, n_objects=int(n_objects), n_colours=3, max_parents=2, catalyst=TRAIN_HIDDEN
    )


GRID = Grid(
    name="Chemistry",
    make_env=make_env,
    sizes=SIZES,
    size_name="objects",
    knobs=[DENSITY],
    knob_name="density",
    policies=POLICIES,
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=METHODS,
    limits=LIMITS,
    workers="auto",
    out_dir=OUT_DIR,
    reward_transfer=False,
    shifts=None,
    constants={"catalyst": TRAIN_HIDDEN, "n_colours": 3, "max_parents": 2},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
