"""The coffee robot, uniform logger: every causal discovery -> IMDP method against flat IMDP and SPIBB
(S2-S9).

    size        "coffee" (6 binary variables, 64 states, 4 actions), "coffee+snack" (8 variables, 256
                states, 6 actions)
    knob        variant: original (Dearden & Boutilier, UAI 1994; src/scm_to_mdp/envs/coffee.py)
    logger      uniform (every action tried in every visited state: no positivity problem)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.9), 5 seeds; solving limited to 3 x the oracle
                (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> IMDP with the plain discovered graph (no widening of
                the families: no "support", "hull" or "pad"), flat IMDP, flat SPIBB and SL-SPIBB (safe
                policy improvement: a guarantee over the behaviour policy, not a value bound; SL-SPIBB
                with candidate parent sets of 3 state variables), and true graph -> IMDP as the
                reference (protocol.py, RMDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      none (this domain has no hidden variable)

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/coffee_uniform_rmdp.py                # a new run
    .venv/bin/python experiments/trials/coffee_uniform_rmdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/coffee_uniform_rmdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/coffee_uniform_rmdp.py --workers 6    # own plan
Output: results/trials/coffee_uniform_rmdp_<date>_<time>/.
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
    methods,
)

from scm_to_mdp.envs import Coffee
from scm_to_mdp.tool import Grid, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["coffee", "coffee+snack"]
N_EPISODES = [1, 4, 16, 64, 256]  # of HORIZON steps: 256 ... 65.5k transitions
VARIANT = "original"
GAMMA = 0.9
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "coffee_uniform_rmdp"
METHODS = methods(RMDP_METHODS, state_parents=3)


def make_env(size: str, knob: str, seed: int) -> Coffee:
    return Coffee(size, seed)


GRID = Grid(
    name="Coffee robot",
    make_env=make_env,
    sizes=SIZES,
    size_name="domain",
    knobs=[VARIANT],
    knob_name="variant",
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
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
