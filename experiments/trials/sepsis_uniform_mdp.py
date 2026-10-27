"""Sepsis, uniform logger: every causal discovery -> MDP method against the baselines without a guarantee
(S2-S9).

    size        vitals: 2 (hr, sysbp), 3 (+ o2), 4 (+ glucose): 144, 288, 1440 observed states (diabetes
                observed)
    knob        diabetes: "observed", in the model and in the data (experiment 1 also runs "absent", and
                "hidden": the non-Markov limit case)
    logger      uniform (every action tried in every visited state: no positivity problem)
    episodes    N_EPISODES of HORIZON = 256 steps (gamma 0.9), 5 seeds; solving limited to 3 x the oracle
                (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> MDP (point estimates, value iteration; no guarantee),
                flat MDP, DiFaRL, FOCUS, offline DQN, offline CQL, and true graph -> MDP as the reference
                (protocol.py, MDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      none: diabetes (persistent, drawn once per patient w.p. 0.2; DIABETES_EFFECT 1.5 as in
                experiment 1) is observed; a rollout is a patient

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/sepsis_uniform_mdp.py                # a new run
    .venv/bin/python experiments/trials/sepsis_uniform_mdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/sepsis_uniform_mdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/sepsis_uniform_mdp.py --workers 6    # own plan
Output: results/trials/sepsis_uniform_mdp_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import (
    DELTA,
    HORIZON,
    LIMITS,
    MDP_METHODS,
    POLICIES,
    SAVE_MODELS,
    SEEDS,
    SOLVE_FACTOR,
    SOLVE_FLOOR_S,
    methods,
)

from scm_to_mdp.envs import Sepsis
from scm_to_mdp.tool import Grid, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

VITALS = {2: ("hr", "sysbp"), 3: ("hr", "sysbp", "o2"), 4: ("hr", "sysbp", "o2", "glucose")}
SIZES = list(VITALS)
N_EPISODES = [1, 4, 16, 64, 256, 1024]  # of HORIZON steps: 256 ... 262k transitions
DIABETES = "observed"
GAMMA = 0.9
DIABETES_EFFECT = 1.5  # the diabetic response, scaled away from the non-diabetic one (1 = original)
P_DIABETES = 0.2
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "sepsis_uniform_mdp"
METHODS = methods(MDP_METHODS)


def make_env(n_vitals, diabetes: str, seed: int) -> Sepsis:
    return Sepsis("custom", seed, vitals=VITALS[int(n_vitals)], diabetes=diabetes != "absent",
                  hide_diabetes=diabetes == "hidden", diabetes_effect=DIABETES_EFFECT,
                  p_diabetes=P_DIABETES)  # fmt: skip


GRID = Grid(
    name="Sepsis",
    make_env=make_env,
    sizes=SIZES,
    size_name="vitals",
    knobs=[DIABETES],
    knob_name="diabetes",
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
    constants={"p_diabetes": P_DIABETES, "diabetes_effect": DIABETES_EFFECT},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
