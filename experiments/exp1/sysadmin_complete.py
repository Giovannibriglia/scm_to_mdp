"""SysAdmin, complete grid: every size x topology x behaviour policy x number of rollouts.

    size        N = 2 ... 9 machines (2^N observed states)
    knob        topology: ring (in-degree 3), bidirectional (4), star (hub)
    policies    uniform      every action equally likely: broad coverage, the action is randomised
                expert       the best policy over the observed state, a uniform action w.p. 0.05:
                             partial coverage (the expert's actions are frequent, the others rare)
                confounded   the expert, but during a hidden storm it repairs machine 0 w.p. 0.95
    rollouts    10 ... 1600 (horizon 20), 3 seeds, the 13 standard methods
                (README; offline DQN and CQL included: DEEP_BASELINES)

The storm (src/scm_to_mdp/envs/sysadmin.py) is in every setting: drawn afresh at every step
(p = 0.3), it hits machine 0 only (a machine that is not repaired fails more: P(up) x 0.2), and it
is never in the data. The uniform and expert loggers ignore it, so their data show the right
observed kernel; only the confounded logger makes the data lie (the repairs of machine 0 happen
in storms, so machine 0 looks healthier when it is not repaired than it is).

Expectations, written before the run:
    H1  learning the structure pays off more as N grows (BIC needs fewer rollouts than flat MDP
        and DiFaRL, and flat RMDP never gets there), and less with denser graphs (bidirectional);
    H2  expert data: robust methods do better than plug-in at small budgets; flat MDP's promise fails;
    H3  expert data: discovery loses action edges (the actions follow the state, rarely varied),
        more for PCMCI+ and GES than for BIC; then the causal methods' promise fails;
    H4  confounded data: the promise of every method fails once the intervals are narrower than
        the bias (small N or many rollouts), even with the true graph; uniform / expert: it holds
        with the true graph.

Run from the repository root (several hours; a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1/sysadmin_complete.py                 # a new run
    .venv/bin/python experiments/exp1/sysadmin_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/sysadmin_complete.py --report-only   # tables and figures only
Output: results/exp1/sysadmin_complete_<date>_<time>/ (one folder per run): summary/ for the whole
grid, models/ of the first seed and one folder per setting (tool/runs.py, tool/store.py).
"""

import dataclasses
import warnings
from pathlib import Path

from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Grid, Limits, Policy, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
TOPOLOGIES = ["ring", "bidirectional", "star"]
POLICIES = [
    Policy("uniform"),
    Policy("expert", epsilon=0.05),
    Policy("confounded", epsilon=0.05, strength=0.95),
]
N_ROLLOUTS = [10, 25, 50, 100, 200, 400, 1600]
SEEDS = [0, 1, 2]
HORIZON, GAMMA, DELTA = 20, 0.9, 0.05
DEEP_BASELINES = True  # offline DQN and offline CQL (about 15 s each per run on a laptop, ~45 s on a server)
STORM, STORM_EFFECT = 0.3, 0.8  # P(storm) at every step; a storm multiplies machine 0's P(up) by 1 - 0.8
LIMITS = Limits(
    mem_gb=8.0, timeout_s={"discovery": 60.0, "selection": 60.0, "solve": 60.0, "training": 600.0}
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "sysadmin_complete"


def make_env(n, topology: str, seed: int) -> SysAdmin:
    return SysAdmin(int(n), seed, topology=topology, storm=STORM, storm_effect=STORM_EFFECT)


GRID = Grid(
    name="SysAdmin",
    make_env=make_env,
    sizes=SIZES,
    size_name="N",
    knobs=TOPOLOGIES,
    knob_name="topology",
    policies=POLICIES,
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(
        max_parents=4, delta=DELTA, exact=True, deep=DEEP_BASELINES, state_parents=3
    ),  # true in-degree <= 4
    limits=LIMITS,
    workers=3,
    out_dir=OUT_DIR,
    constants={"storm": STORM, "storm_effect": STORM_EFFECT},  # checked when a run resumes
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
