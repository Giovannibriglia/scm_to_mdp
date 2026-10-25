"""SysAdmin ring, uniform logger: every causal discovery -> RMDP method against flat RMDP (S2-S9).

    size        N = 2 ... 8 machines on the one-way ring (2^N observed states, N + 1 actions)
    logger      uniform (every action tried in every visited state: no positivity problem)
    rollouts    50, 250, 750 (horizon 20, gamma 0.9), 5 seeds
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> RMDP with the plain discovered graph (no widening of
                the families: no "support", "hull" or "pad"), flat RMDP, and true graph -> RMDP as the
                reference (delete its line in METHODS to drop it)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      as in experiment 1: the storm hits machine 0 w.p. 0.1 at every step and is never in the
                data (the same environment for data and evaluation)

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/discovery_rmdp/sysadmin_uniform_rmdp.py                 # a new run
    .venv/bin/python experiments/discovery_rmdp/sysadmin_uniform_rmdp.py --resume        # continue the latest
    .venv/bin/python experiments/discovery_rmdp/sysadmin_uniform_rmdp.py --report-only   # tables and figures
    .venv/bin/python experiments/discovery_rmdp/sysadmin_uniform_rmdp.py --workers 6     # own plan
Output: results/discovery_rmdp/sysadmin_uniform_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Grid, Limits, Policy, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = [2, 3, 4, 5, 6, 7, 8]
N_ROLLOUTS = [50, 250, 750]
SEEDS = [0, 1, 2, 3, 4]
HORIZON, GAMMA = 20, 0.9
DELTA = 0.05
STORM, STORM_EFFECT = 0.1, 0.8  # experiment 1's training environment (experiments/exp1/protocol.py)
LIMITS = Limits(
    mem_gb=8.0,
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 60.0, "solve": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "discovery_rmdp" / "sysadmin_uniform_mdp"

_ALL = standard_methods(max_parents=None, delta=DELTA, exact=False, deep=False)
METHODS = {
    name: _ALL[name]
    for name in (
        "BIC -> MDP",
        "PCMCI+ -> MDP",
        "GES -> MDP",
        "DYNOTEARS -> MDP",
        "flat MDP",
        "true graph -> MDP",  # reference
    )
}


def make_env(n, knob: str, seed: int) -> SysAdmin:
    return SysAdmin(int(n), seed, topology=str(knob), storm=STORM, storm_effect=STORM_EFFECT)


GRID = Grid(
    name="SysAdmin",
    make_env=make_env,
    sizes=SIZES,
    size_name="N",
    knobs=["ring"],
    knob_name="topology",
    policies=[Policy("uniform")],
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
    constants={"storm": STORM, "storm_effect": STORM_EFFECT},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
