"""Frozen Lake (slippery), uniform logger: every causal discovery -> MDP method against the baselines without
a guarantee (S2-S9).

    size        map - agents: 4x4-1, 8x8-1 (Gym's maps, one agent: 16, 64 states), 4x4-2 (two agents: 256
                states; experiment 1's 8x8-2 needs a server); independent axes
    knob        wind: "wind" (experiment 1 also runs "calm")
    logger      uniform (every action tried in every visited state: no positivity problem)
    rollouts    50, 250, 750 (horizon 50, gamma 0.95), 5 seeds (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> MDP (point estimates, value iteration; no guarantee),
                flat MDP, DiFaRL, FOCUS, offline DQN, offline CQL, and true graph -> MDP as the reference
                (protocol.py, MDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      as in experiment 1: a wind, w.p. 0.1 at every step, pushes agent 0's row down w.p. 0.5; never
                in the data

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/frozenlake_uniform_mdp.py                # a new run
    .venv/bin/python experiments/trials/frozenlake_uniform_mdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/frozenlake_uniform_mdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/frozenlake_uniform_mdp.py --workers 6    # own plan
Output: results/trials/frozenlake_uniform_mdp_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DELTA, MDP_METHODS, N_ROLLOUTS, POLICIES, SEEDS, TRAIN_HIDDEN, methods

from scm_to_mdp.envs import FrozenLake
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["4x4-1", "8x8-1", "4x4-2"]  # map - number of agents
WIND = "wind"
HORIZON, GAMMA = 50, 0.95
WIND_EFFECT = 0.5
MODEL_GB = 12.0
LIMITS = Limits(
    mem_gb=2.5,  # peak 0.5 GB (offline CQL, 4x4-2, 750 rollouts)
    timeout_s={"discovery": 120.0, "selection": 120.0, "learning": 120.0, "training": 1800.0, "solve": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "frozenlake_uniform_mdp"
METHODS = methods(MDP_METHODS)


def make_env(size: str, wind: str, seed: int) -> FrozenLake:
    map_name, agents = str(size).split("-")
    wind_p = TRAIN_HIDDEN if wind == "wind" else None
    return FrozenLake(map_name, int(agents), seed, slippery=True, wind=wind_p, wind_effect=WIND_EFFECT,
                      axes="independent", max_model_gb=MODEL_GB)  # fmt: skip


GRID = Grid(
    name="Frozen Lake",
    make_env=make_env,
    sizes=SIZES,
    size_name="map-agents",
    knobs=[WIND],
    knob_name="wind",
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
    constants={"wind": TRAIN_HIDDEN, "wind_effect": WIND_EFFECT},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
