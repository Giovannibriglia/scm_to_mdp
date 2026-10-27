"""Drone delivery, uniform logger: every causal discovery -> IMDP method against flat IMDP and SPIBB (S2-S9).

    size        grid side - drones: 3-1 ... 12-1 (one drone: 27 ... 1,728 observed states, 6 actions)
    knob        gust: "gust" (experiment 1 also runs "calm")
    logger      uniform (every action tried in every visited state: no positivity problem)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.95), 5 seeds; solving limited to 3 x the oracle
                (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> IMDP with the plain discovered graph (no widening of
                the families: no "support", "hull" or "pad"), flat IMDP, flat SPIBB and SL-SPIBB (safe
                policy improvement: a guarantee over the behaviour policy, not a value bound; SL-SPIBB
                with candidate parent sets of 3 state variables), and true graph -> IMDP as the
                reference (protocol.py, RMDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      as in experiment 1: a gust, w.p. 0.1 at every step, pushes drone 0 one cell further in -y
                w.p. 0.5; never in the data

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/drone_uniform_rmdp.py                # a new run
    .venv/bin/python experiments/trials/drone_uniform_rmdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/drone_uniform_rmdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/drone_uniform_rmdp.py --workers 6    # own plan
Output: results/trials/drone_uniform_rmdp_<date>_<time>/.
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

from scm_to_mdp.envs import Drone
from scm_to_mdp.tool import Grid, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["3-1", "4-1", "5-1", "6-1", "7-1", "8-1", "9-1", "10-1", "11-1", "12-1"]  # grid side - drones
N_EPISODES = [1, 4, 16, 64, 256, 1024]  # of HORIZON steps: 256 ... 262k transitions
GUST = "gust"
GAMMA = 0.95
GUST_EFFECT = 0.5
MODEL_GB = 12.0
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "drone_uniform_rmdp"
METHODS = methods(RMDP_METHODS, state_parents=3)


def make_env(size: str, gust: str, seed: int) -> Drone:
    side, drones = (int(v) for v in str(size).split("-"))
    gust_p = TRAIN_HIDDEN if gust == "gust" else None
    return Drone(side, drones, seed, gust=gust_p, gust_effect=GUST_EFFECT, max_model_gb=MODEL_GB)


GRID = Grid(
    name="Drone",
    make_env=make_env,
    sizes=SIZES,
    size_name="side-drones",
    knobs=[GUST],
    knob_name="gust",
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
    constants={"gust": TRAIN_HIDDEN, "gust_effect": GUST_EFFECT},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
