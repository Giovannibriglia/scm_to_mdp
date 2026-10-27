"""Aircraft collision avoidance, uniform logger: every causal discovery -> MDP method against the baselines
without a guarantee (S2-S9).

    size        width x height: 4x3 ... 18x13 (10 sizes, 80 ... 3,724 observed states)
    knob        proximity "on": an intended climb or descent succeeds w.p. 0.5 instead of 0.8 while the
                intruder is near (experiment 1 also runs "off")
    logger      uniform (every action tried in every visited state: no positivity problem)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.95), 5 seeds; solving limited to 3 x the oracle
                (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> MDP (point estimates, value iteration; no guarantee),
                flat MDP, DiFaRL, FOCUS, offline DQN, offline CQL, and true graph -> MDP as the reference
                (protocol.py, MDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      as in experiment 1: the turbulence halves the success of an intended climb or descent (the
                own altitude only) w.p. 0.1 at every step and is never in the data

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/aircraft_uniform_mdp.py                # a new run
    .venv/bin/python experiments/trials/aircraft_uniform_mdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/aircraft_uniform_mdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/aircraft_uniform_mdp.py --workers 6    # own plan
Output: results/trials/aircraft_uniform_mdp_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import (
    DELTA,
    HORIZON,
    LIMITS,
    MDP_METHODS,
    POLICIES,
    SAVE_MODELS,
    SEEDS,
    SOLVE_FACTOR,
    SOLVE_FLOOR_S,
    TRAIN_HIDDEN,
    methods,
)

from scm_to_mdp.envs import Aircraft
from scm_to_mdp.tool import Grid, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["4x3", "5x4", "6x4", "8x5", "8x6", "10x7", "12x8", "14x10", "16x12", "18x13"]  # width x height
N_EPISODES = [1, 4, 16, 64, 256, 1024]  # of HORIZON steps: 256 ... 262k transitions
PROXIMITY = "on"
GAMMA = 0.95
TURBULENCE_EFFECT = 0.5
MODEL_GB = 12.0
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "aircraft_uniform_mdp"
METHODS = methods(MDP_METHODS)


def make_env(size: str, proximity: str, seed: int) -> Aircraft:
    width, height = (int(v) for v in str(size).split("x"))
    return Aircraft(width, height, seed, proximity=proximity == "on", turbulence=TRAIN_HIDDEN,
                    turbulence_effect=TURBULENCE_EFFECT, max_model_gb=MODEL_GB)  # fmt: skip


GRID = Grid(
    name="Aircraft",
    make_env=make_env,
    sizes=SIZES,
    size_name="width x height",
    knobs=[PROXIMITY],
    knob_name="proximity",
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
    constants={"turbulence": TRAIN_HIDDEN, "turbulence_effect": TURBULENCE_EFFECT},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
