"""Frozen Lake (slippery), uniform logger: every causal discovery -> IMDP method against flat IMDP and SPIBB
(S2-S9).

    size        the map, per number of agents (the knob): one agent on 4x4, 8x8 (Gym's maps), 16x16,
                32x32 (random maps, envs/frozen_lake.py: 16 ... 1,024 observed states); two agents on 4x4,
                5x5, 6x6, 7x7 (256 ... 2,401 observed states, 16 joint actions; 8x8 does not fit: the dense
                true model needs ~17 GB); independent axes
    knob        agents: 1 or 2; the wind is on in both (experiment 1 also runs "calm")
    logger      uniform (every action tried in every visited state: no positivity problem)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.95), 5 seeds; solving limited to 3 x the oracle
                (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> IMDP with the plain discovered graph (no widening of
                the families: no "support", "hull" or "pad"), flat IMDP, flat SPIBB and SL-SPIBB (safe
                policy improvement: a guarantee over the behaviour policy, not a value bound; SL-SPIBB
                with candidate parent sets of 2 state variables), and true graph -> IMDP as the
                reference (protocol.py, RMDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      as in experiment 1: a wind, w.p. 0.1 at every step, pushes agent 0's row down w.p. 0.5; never
                in the data

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/frozenlake_uniform_rmdp.py                # a new run
    .venv/bin/python experiments/trials/frozenlake_uniform_rmdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/frozenlake_uniform_rmdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/frozenlake_uniform_rmdp.py --workers 6    # own plan
Output: results/trials/frozenlake_uniform_rmdp_<date>_<time>/.
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

from scm_to_mdp.envs import FrozenLake
from scm_to_mdp.tool import Grid, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

AGENTS = ["1", "2"]  # the knob
SIZES = {"1": ["4x4", "8x8", "16x16", "32x32"], "2": ["4x4", "5x5", "6x6", "7x7"]}  # maps, per knob
N_EPISODES = [1, 4, 16, 64, 256, 1024]  # of HORIZON steps: 256 ... 262k transitions
GAMMA = 0.95
WIND_EFFECT = 0.5
MODEL_GB = 12.0
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "frozenlake_uniform_rmdp"
METHODS = methods(RMDP_METHODS, state_parents=2)


def make_env(map_name: str, agents: str, seed: int) -> FrozenLake:
    return FrozenLake(str(map_name), int(agents), seed, slippery=True, wind=TRAIN_HIDDEN,
                      wind_effect=WIND_EFFECT, axes="independent", max_model_gb=MODEL_GB)  # fmt: skip


GRID = Grid(
    name="Frozen Lake",
    make_env=make_env,
    sizes=SIZES,
    size_name="map",
    knobs=AGENTS,
    knob_name="agents",
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
    constants={"wind": TRAIN_HIDDEN, "wind_effect": WIND_EFFECT},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
