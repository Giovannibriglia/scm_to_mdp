"""SysAdmin, complete grid: every size x topology x behaviour policy x number of rollouts (protocol.py:
the claims, the loggers, the seeds, the shifts and the expectations common to every domain).

    size        N = 2 ... 10 machines (2^N observed states; N >= 9 needs a server)
    knob        topology: ring (in-degree 3), bidirectional (4), star (hub); ring+crash: the ring where
                N // 2 machines down at once ends the episode with -10 (a trap, from the old exp3)
    policies    uniform, expert eps 0.05, expert eps 0, confounded (repairs machine 0 during storms
                w.p. 0.95)
    hidden      the storm (src/scm_to_mdp/envs/sysadmin.py) hits machine 0 only (a machine that is not
                repaired fails more: P(up) x 0.2), w.p. 0.1 in the data, 0.3 and 0.6 at deployment
    rollouts    10 ... 1600 (horizon 20, gamma 0.9), 5 seeds, the standard methods with the per-(s,a)
                product set (every variable is 0/1)

Expectations for this domain, written 2026-10-24 before the runs:
    H1  structure pays off more as N grows (A), less with the denser bidirectional ring;
    H2  ring+crash: the point models' optimism can lead into the crash under the eps = 0 expert; the
        robust arms keep their promise (A2);
    H3  under eps = 0 the expert repairs as a function of the states: the logger-Markov graph replaces
        reboot by a neighbour for some machines (seen in a check: m1 <- m0, m1, m3 on the 4-ring), and
        BIC may drop the same action edges (A3);
    H4  machine0_only and all_up_bonus need repairs the expert rarely makes: the largest
        cov_joint - cov_factor, the largest transfer gain (B1).

Run from the repository root (several hours; a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1/sysadmin_complete.py                 # a new run
    .venv/bin/python experiments/exp1/sysadmin_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/sysadmin_complete.py --report-only   # tables and figures only
    .venv/bin/python experiments/exp1/sysadmin_complete.py --workers 8 --device cpu  # own plan
Output: results/exp1/sysadmin_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, N_ROLLOUTS, SEEDS, SHIFT_HIDDEN, TRAIN_HIDDEN, policies

from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
TOPOLOGIES = ["ring", "bidirectional", "star", "ring+crash"]
HORIZON, GAMMA = 20, 0.9
STORM_EFFECT, CRASH_COST = 0.8, 10.0  # a storm multiplies machine 0's P(up) by 1 - 0.8
LIMITS = Limits(
    mem_gb=8.0,
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 60.0, "solve": 60.0, "training": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "sysadmin_complete"


def sysadmin(n, knob: str, seed: int, storm: float) -> SysAdmin:
    topology, _, crash = str(knob).partition("+")
    return SysAdmin(int(n), seed, topology=topology, storm=storm, storm_effect=STORM_EFFECT,
                    crash=int(n) // 2 if crash else None, crash_cost=CRASH_COST)  # fmt: skip


def make_env(n, knob: str, seed: int) -> SysAdmin:
    return sysadmin(n, knob, seed, TRAIN_HIDDEN)


def shifts(n, knob: str, seed: int) -> dict:
    """The same network where storms come more often (the learner never sees them)."""
    return {f"storm={p:g}": sysadmin(n, knob, seed, p) for p in SHIFT_HIDDEN}


GRID = Grid(
    name="SysAdmin",
    make_env=make_env,
    sizes=SIZES,
    size_name="N",
    knobs=TOPOLOGIES,
    knob_name="topology",
    policies=policies(),
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=4, delta=DELTA, exact=True, deep=DEEP_BASELINES, state_parents=3),
    limits=LIMITS,
    workers="auto",  # what the machine allows (utils/compute.py); --workers N overrides
    out_dir=OUT_DIR,
    reward_transfer=True,
    shifts=shifts,
    constants={
        "storm": TRAIN_HIDDEN,
        "shift_storms": SHIFT_HIDDEN,
        "storm_effect": STORM_EFFECT,
        "crash_cost": CRASH_COST,
    },  # fmt: skip
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
