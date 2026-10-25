"""The coffee robot, uniform logger: every causal discovery -> MDP method against the baselines without a
guarantee (S2-S9).

    size        "coffee" (6 binary variables, 64 states, 4 actions), "coffee+snack" (8 variables, 256
                states, 6 actions)
    knob        variant: original (Dearden & Boutilier, UAI 1994; src/scm_to_mdp/envs/coffee.py)
    logger      uniform (every action tried in every visited state: no positivity problem)
    rollouts    50, 250, 750 (horizon 20, gamma 0.9), 5 seeds (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> MDP (point estimates, value iteration; no guarantee),
                flat MDP, DiFaRL, FOCUS, offline DQN, offline CQL, and true graph -> MDP as the reference
                (protocol.py, MDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      none (this domain has no hidden variable)

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/coffee_uniform_mdp.py                # a new run
    .venv/bin/python experiments/trials/coffee_uniform_mdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/coffee_uniform_mdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/coffee_uniform_mdp.py --workers 6    # own plan
Output: results/trials/coffee_uniform_mdp_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DELTA, MDP_METHODS, N_ROLLOUTS, POLICIES, SEEDS, methods

from scm_to_mdp.envs import Coffee
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["coffee", "coffee+snack"]
VARIANT = "original"
HORIZON, GAMMA = 20, 0.9
LIMITS = Limits(
    mem_gb=2.5,  # peak without BIC 0.5 GB (offline CQL, coffee+snack, 750 rollouts)
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 60.0, "training": 600.0, "solve": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "coffee_uniform_mdp"
METHODS = methods(MDP_METHODS)


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
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
