"""Frozen Lake (slippery), complete grid: every size x wind setting x behaviour policy x rollouts
(protocol.py: the claims, the loggers, the seeds, the shifts and the expectations common to every domain).

    size        map - agents: 4x4-1, 8x8-1 (Gym's maps, one agent: 16, 64 states), 4x4-2, 8x8-2 (two agents
                with their own actions: 256, 4096 states; the two-agent lake of Schnitzer, Abate, Parker,
                AAAI 2026, App. E.3, with Gym's maps and our independent axes; 8x8-2 needs a server)
    knob        wind: "calm" (no wind in the data) or "wind" (a hidden wind, w.p. 0.1, pushes agent 0's
                row down w.p. 0.5; never in the data)
    policies    uniform, expert eps 0.05, expert eps 0; confounded (agent 0 moves up during wind w.p.
                0.95) only with wind
    shift       wind w.p. 0.3 and 0.6 at deployment (also after calm data)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.95), 5 seeds, the standard methods (no per-(s,a)
                product set: the cells are not 0/1)

Every move is slippery, with our independent axes (envs/frozen_lake.py, axes "independent"): each of x and
y follows its marginal under Gym's slip (intended or perpendicular, 1/3 each), drawn independently, so the
true graph has no same-step edge. With two agents, the true graph has no edge between the agents.

Expectations for this domain, written 2026-10-24 before the runs (after a one-seed pilot: 8x8-1, wind,
expert eps 0.05, 100 rollouts, true graph -> IMDP at normalised regret 0.00, PCMCI+ 0.16, BIC 1.01):
    H1  one agent: little structure (both axes need the whole cell): causal and flat IMDP close (A);
        two agents: the factored arms pass the gate at fewer rollouts than flat IMDP;
    H2  holes are traps: under eps = 0 the point models' Laplace guess for the moves the expert never
        tries spreads mass onto the goal (A2);
    H3  under eps = 0 the logger-Markov graph drops y from x' (seen in a check): its promise can fail
        (A3); BIC drops action edges at small n;
    H4  goal_top_right and goal_bottom_left lie off the expert's path: two agents, cov_joint -
        cov_factor large (B1).

Run from the repository root (8x8-2: about 5 GB for the true model and more for the robust model):
    .venv/bin/python experiments/exp1/frozenlake_complete.py                 # a new run
    .venv/bin/python experiments/exp1/frozenlake_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/frozenlake_complete.py --report-only   # tables and figures only
Output: results/exp1/frozenlake_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, HORIZON, SAVE_MODELS, SEEDS, SHIFT_HIDDEN, TRAIN_HIDDEN, policies

from scm_to_mdp.envs import FrozenLake
from scm_to_mdp.tool import Grid, Limits, Policy, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["4x4-1", "8x8-1", "4x4-2", "8x8-2"]  # map - number of agents
WINDS = ["calm", "wind"]
# of HORIZON steps: 500 ... 80k transitions (was 10 ... 1600 rollouts of 50)
N_EPISODES = [2, 4, 8, 16, 32, 64, 128, 256]
GAMMA = 0.95
WIND_EFFECT = 0.5
MODEL_GB = 12.0
LIMITS = Limits(
    mem_gb=16.0,
    timeout_s={"discovery": 120.0, "selection": 120.0, "learning": 120.0, "solve": 600.0, "training": 1800.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "frozenlake_complete"


def lake(size: str, wind: float | None, seed: int) -> FrozenLake:
    map_name, agents = str(size).split("-")
    return FrozenLake(map_name, int(agents), seed, slippery=True, wind=wind, wind_effect=WIND_EFFECT,
                      axes="independent", max_model_gb=MODEL_GB)  # fmt: skip


def make_env(size: str, wind: str, seed: int) -> FrozenLake:
    return lake(size, TRAIN_HIDDEN if wind == "wind" else None, seed)


def shifts(size: str, wind: str, seed: int) -> dict:
    """The same lake where the wind blows more often (the learner never sees it)."""
    return {f"wind={p:g}": lake(size, p, seed) for p in SHIFT_HIDDEN}


def no_confounder(wind: str, policy: Policy) -> bool:
    """Without wind the confounded logger is the expert: not run."""
    return wind == "calm" and policy.name == "confounded"


GRID = Grid(
    name="Frozen Lake",
    make_env=make_env,
    sizes=SIZES,
    size_name="map-agents",
    knobs=WINDS,
    knob_name="wind",
    policies=policies(),
    n_rollouts=N_EPISODES,  # one rollout = one episode of HORIZON steps
    save_models=SAVE_MODELS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=5, delta=DELTA, exact=False, deep=DEEP_BASELINES, state_parents=2),
    limits=LIMITS,
    workers="auto",
    out_dir=OUT_DIR,
    reward_transfer=True,
    shifts=shifts,
    constants={
        "wind": TRAIN_HIDDEN,
        "shift_winds": SHIFT_HIDDEN,
        "wind_effect": WIND_EFFECT,
        "model_gb": MODEL_GB,
    },
    skip=no_confounder,
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
