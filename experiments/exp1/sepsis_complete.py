"""Sepsis, complete grid: every size x diabetes setting x behaviour policy x number of rollouts.

    size        vitals: 2 (hr, sysbp), 3 (+ o2), 4 (+ glucose): 72, 144, 720 observed states
                (twice as many with diabetes observed)
    knob        diabetes: absent (not in the model), observed (in the model and in the data),
                hidden (in the model, not in the data: the observed process is not Markov)
    policies    uniform      every treatment combination equally likely
                expert       the best policy over the observed state, a uniform action w.p. 0.05
                confounded   the expert, but a diabetic gets vasopressors w.p. 0.95 (the clinician
                             sees diabetes); skipped when diabetes is absent (it would be the expert)
    rollouts    100 ... 4000 patients (at most 20 steps), 3 seeds, 10 standard methods (no exact
                solver: the vitals are not 0/1; offline DQN and CQL too with DEEP_BASELINES)

Diabetes changes how blood pressure and glucose respond to vasopressors (Oberst & Sontag's
simulator, src/scm_to_mdp/envs/sepsis.py). DIABETES_EFFECT scales that difference (ours; 1 = the
original). With diabetes observed, the confounded clinician is no problem (the learner sees what
the clinician sees): it is the control for "hidden".

Expectations, written before the run:
    H1  structure pays off more with more vitals;
    H2  expert data: robust methods beat plug-in at small budgets; flat MDP's promise fails;
    H3  expert data: discovery loses the action edges that the expert rarely varies;
    H4  diabetes hidden: the promise can fail with every logger, because diabetes is a hidden cause
        of two vitals and persists over the episode (no product of factors over the observed
        variables is right), and more so with the confounded clinician;
    H5  diabetes observed: the confounded clinician behaves like the expert (no failure).

Run from the repository root (several hours; a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1/sepsis_complete.py                 # a new run
    .venv/bin/python experiments/exp1/sepsis_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/sepsis_complete.py --report-only   # tables and figures only
Output: results/exp1/sepsis_complete_<date>_<time>/ (one folder per run): summary/ for the whole
grid, models/ of the first seed and one folder per setting (tool/runs.py, tool/store.py).
"""

import dataclasses
import warnings
from pathlib import Path

from scm_to_mdp.envs import Sepsis
from scm_to_mdp.tool import Grid, Limits, Policy, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

VITALS = {2: ("hr", "sysbp"), 3: ("hr", "sysbp", "o2"), 4: ("hr", "sysbp", "o2", "glucose")}
DIABETES = ["absent", "observed", "hidden"]
POLICIES = [
    Policy("uniform"),
    Policy("expert", epsilon=0.05),
    Policy("confounded", epsilon=0.05, strength=0.95),
]
N_ROLLOUTS = [100, 250, 500, 1000, 2000, 4000]
SEEDS = [0, 1, 2]
HORIZON, GAMMA, DELTA = 20, 0.9, 0.05
DEEP_BASELINES = False  # True: also offline DQN and offline CQL (about 15 s each per run on a laptop)
DIABETES_EFFECT = 1.5  # the diabetic response, scaled away from the non-diabetic one (1 = original)
LIMITS = Limits(
    mem_gb=8.0, timeout_s={"discovery": 60.0, "selection": 60.0, "solve": 120.0, "training": 600.0}
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "sepsis_complete"


def make_env(n_vitals, diabetes: str, seed: int) -> Sepsis:
    return Sepsis("custom", seed, vitals=VITALS[int(n_vitals)], diabetes=diabetes != "absent",
                  hide_diabetes=diabetes == "hidden", diabetes_effect=DIABETES_EFFECT)  # fmt: skip


def no_confounder(diabetes: str, policy: Policy) -> bool:
    """Without diabetes the confounded clinician is the expert: not run."""
    return diabetes == "absent" and policy.name == "confounded"


GRID = Grid(
    name="Sepsis",
    make_env=make_env,
    sizes=list(VITALS),
    size_name="vitals",
    knobs=DIABETES,
    knob_name="diabetes",
    policies=POLICIES,
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(
        max_parents=6, delta=DELTA, exact=False, deep=DEEP_BASELINES, state_parents=4
    ),  # true in-degree <= 6
    limits=LIMITS,
    workers=3,
    out_dir=OUT_DIR,
    constants={"vitals": VITALS, "diabetes_effect": DIABETES_EFFECT},  # checked when a run resumes
    skip=no_confounder,
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
