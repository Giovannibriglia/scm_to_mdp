"""Sepsis, complete grid: every size x diabetes setting x behaviour policy x number of rollouts
(protocol.py: the claims, the loggers, the seeds and the expectations common to every domain).

    size        vitals: 2 (hr, sysbp), 3 (+ o2), 4 (+ glucose): 72, 144, 720 observed states (twice as
                many with diabetes observed)
    knob        diabetes: absent (not in the model), observed (in the model and in the data), hidden (in
                the model, not in the data)
    policies    uniform, expert eps 0.05, expert eps 0, confounded (a diabetic gets vasopressors w.p.
                0.95: the clinician sees diabetes; not run without diabetes)
    hidden      unlike the other domains, diabetes is PERSISTENT (drawn once per patient, w.p. 0.2), not
                drawn afresh at every step: with it hidden the observed process is not Markov, and no
                product of factors over the observed variables is right (the declared limit case of
                this grid). Shift: diabetic w.p. 0.5 and 0.8 at deployment (with diabetes observed, a
                change of the cause's distribution only: the causal kernel is unchanged, ECI pp. 77-79)
    rollouts    100 ... 4000 patients (at most 20 steps, gamma 0.9), 5 seeds, the standard methods (no
                per-(s,a) product set: the vitals are not 0/1)

Diabetes changes how blood pressure and glucose respond to vasopressors (Oberst & Sontag's simulator,
src/scm_to_mdp/envs/sepsis.py). DIABETES_EFFECT scales that difference (ours; 1 = the original).

Expectations for this domain, written 2026-10-24 before the runs:
    H1  structure pays off more with more vitals (A);
    H2  the expert never ventilates in some settings: under eps = 0 the logger-Markov graph drops
        ventilation from vent_on' (seen in a check), the action edges are thin (A3);
    H3  diabetes hidden: the promise can fail with every logger, even with the true graph (persistent
        hidden cause: no Markov factored model is right), more with the confounded clinician;
    H4  diabetes observed + prevalence shift: the causal models stay correct (a shift of a cause),
        the policies lose only by the change of patients (B2).

Run from the repository root (several hours; a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1/sepsis_complete.py                 # a new run
    .venv/bin/python experiments/exp1/sepsis_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/sepsis_complete.py --report-only   # tables and figures only
    .venv/bin/python experiments/exp1/sepsis_complete.py --workers 8 --device cpu  # own plan
Output: results/exp1/sepsis_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, SEEDS, policies

from scm_to_mdp.envs import Sepsis
from scm_to_mdp.tool import Grid, Limits, Policy, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

VITALS = {2: ("hr", "sysbp"), 3: ("hr", "sysbp", "o2"), 4: ("hr", "sysbp", "o2", "glucose")}
DIABETES = ["absent", "observed", "hidden"]
N_ROLLOUTS = [100, 250, 500, 1000, 2000, 4000]
HORIZON, GAMMA = 20, 0.9
DIABETES_EFFECT = 1.5  # the diabetic response, scaled away from the non-diabetic one (1 = original)
P_DIABETES, SHIFT_DIABETES = 0.2, [0.5, 0.8]
LIMITS = Limits(
    mem_gb=8.0,
    timeout_s={"discovery": 60.0, "selection": 60.0, "learning": 120.0, "solve": 120.0, "training": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "sepsis_complete"


def sepsis(n_vitals, diabetes: str, seed: int, p_diabetes: float) -> Sepsis:
    return Sepsis("custom", seed, vitals=VITALS[int(n_vitals)], diabetes=diabetes != "absent",
                  hide_diabetes=diabetes == "hidden", diabetes_effect=DIABETES_EFFECT,
                  p_diabetes=p_diabetes)  # fmt: skip


def make_env(n_vitals, diabetes: str, seed: int) -> Sepsis:
    return sepsis(n_vitals, diabetes, seed, P_DIABETES)


def shifts(n_vitals, diabetes: str, seed: int) -> dict:
    """More diabetic patients at deployment (none without diabetes)."""
    if diabetes == "absent":
        return {}
    return {f"diabetic={p:g}": sepsis(n_vitals, diabetes, seed, p) for p in SHIFT_DIABETES}


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
    policies=policies(),
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=6, delta=DELTA, exact=False, deep=DEEP_BASELINES, state_parents=4),
    limits=LIMITS,
    workers="auto",  # what the machine allows (utils/compute.py); --workers N overrides
    out_dir=OUT_DIR,
    reward_transfer=True,
    shifts=shifts,
    constants={
        "vitals": VITALS,
        "diabetes_effect": DIABETES_EFFECT,
        "p_diabetes": P_DIABETES,
        "shift_diabetes": SHIFT_DIABETES,
    },  # fmt: skip
    skip=no_confounder,
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
