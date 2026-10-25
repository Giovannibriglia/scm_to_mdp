"""Taxi, uniform logger: every causal discovery -> RMDP method against flat RMDP and SPIBB (S2-S9).

    size        navigation: "deterministic" (Dietterich, JAIR 2000, Sec. 3.1) or "noisy" (0.8 intended, 0.1
                to each side; independent axes); 500 states, 6 actions
    knob        variant: original
    logger      uniform (every action tried in every visited state: no positivity problem)
    rollouts    50, 250, 750 (horizon 50, gamma 0.99), 5 seeds (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> RMDP with the plain discovered graph (no widening of
                the families: no "support", "hull" or "pad"), flat RMDP, flat SPIBB and SL-SPIBB (safe
                policy improvement: a guarantee over the behaviour policy, not a value bound; SL-SPIBB
                with candidate parent sets of 3 state variables), and true graph -> RMDP as the
                reference (protocol.py, RMDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      none (this domain has no hidden variable)

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/taxi_uniform_rmdp.py                # a new run
    .venv/bin/python experiments/trials/taxi_uniform_rmdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/taxi_uniform_rmdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/taxi_uniform_rmdp.py --workers 6    # own plan
Output: results/trials/taxi_uniform_rmdp_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DELTA, N_ROLLOUTS, POLICIES, RMDP_METHODS, SEEDS, methods

from scm_to_mdp.envs import Taxi
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["deterministic", "noisy"]
VARIANT = "original"
HORIZON, GAMMA = 50, 0.99  # gamma 0.99: with -1 per step and +20 at delivery (exp1/taxi_complete.py)
LIMITS = Limits(
    mem_gb=2.5,  # peak 0.4 GB (smoke run; the steps scale with the 500 states, not the rollouts)
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 120.0, "training": 1800.0, "solve": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "taxi_uniform_rmdp"
METHODS = methods(RMDP_METHODS, state_parents=3)


def make_env(size: str, knob: str, seed: int) -> Taxi:
    return Taxi(size, seed, axes="independent")


GRID = Grid(
    name="Taxi",
    make_env=make_env,
    sizes=SIZES,
    size_name="navigation",
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
