"""Chemistry, uniform logger: every causal discovery -> IMDP method against flat IMDP and SPIBB (S2-S9).

    size        objects: 2 ... 7 (3 colours: 9 ... 2,187 observed states; 8 objects needs ~25 GB for the
                dense true model)
    knob        density of the reaction graph: 1.0, at most 2 parents (experiment 1 also runs 0.0 and
                0.4); the seed also draws the graph and the target colours
    logger      uniform (every action tried in every visited state: no positivity problem)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.9), 5 seeds; solving limited to 3 x the oracle
                (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> IMDP with the plain discovered graph (no widening of
                the families: no "support", "hull" or "pad"), flat IMDP, flat SPIBB and SL-SPIBB (safe
                policy improvement: a guarantee over the behaviour policy, not a value bound; SL-SPIBB
                with candidate parent sets of 3 state variables), and true graph -> IMDP as the
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

from protocol import (
    DELTA,
    HORIZON,
    LIMITS,
    POLICIES,
    RMDP_METHODS,
    SAVE_MODELS,
    SEEDS,
    SOLVE_FACTOR,
    SOLVE_FLOOR_S,
    TRAIN_HIDDEN,
    methods,
)

from scm_to_mdp.envs import Chemistry
from scm_to_mdp.tool import Grid, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = [2, 3, 4, 5, 6, 7]
N_EPISODES = [1, 4, 16, 64, 256, 1024]  # of HORIZON steps: 256 ... 262k transitions
DENSITY = "1.0"
GAMMA = 0.9
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
    n_rollouts=N_EPISODES,  # one rollout = one episode of HORIZON steps
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=METHODS,
    limits=LIMITS,
    solve_factor=SOLVE_FACTOR,
    solve_floor_s=SOLVE_FLOOR_S,
    save_models=SAVE_MODELS,
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
