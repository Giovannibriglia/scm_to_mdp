"""Drone delivery, complete grid: every size x gust setting x behaviour policy x number of rollouts
(protocol.py: the claims, the loggers, the seeds, the shifts and the expectations common to every domain).

    size        grid side - drones: 3-1, 4-1, 5-1, 6-1 (one drone: 27 ... 216 states), 3-2, 4-2 (two drones:
                729, 4096 states, 36 joint actions; Schnitzer, Abate, Parker, AAAI 2026, App. E.5, use
                two drones in 5x5x5, beyond a dense true model; 4-2 needs a server)
    knob        gust: "calm" (no gust in the data) or "gust" (a hidden gust, w.p. 0.1, pushes drone 0 one
                cell further in -y w.p. 0.5; never in the data)
    policies    uniform, expert eps 0.05, expert eps 0; confounded (drone 0 climbs during gusts w.p.
                0.95) only with gusts
    shift       gusts w.p. 0.3 and 0.6 at deployment (also after calm data)
    rollouts    10 ... 1600 (horizon 50, gamma 0.95), 5 seeds, the standard methods (no per-(s,a)
                product set)

Wind drift p = 0.2 (theirs), with their per-axis marginals drawn independently per axis
(src/scm_to_mdp/envs/drone.py). True graph: x_i' <- x_i, d_i; y_i' <- y_i, d_i (and the gust for drone
0); z_i' <- z_i, d_i: small families, no edge between the drones.

Expectations for this domain, written 2026-10-24 before the runs:
    H1  structure pays off strongly: 3 families of 2 parents per drone against 27 ... 4096 joint states (A);
    H2  two drones: the joint action vectors the expert never combines are in no model (actions_absent
        under eps = 0), the per-drone factors still know every move (B1);
    H3  target_far_y and target_low ask for regions the expert's path avoids: large cov_joint -
        cov_factor (B1);
    H4  confounded (gust): the promise fails for drone 0 (control).

Run from the repository root (4-2 needs about 10 GB for the true model and as much for the robust model):
    .venv/bin/python experiments/exp1/drone_complete.py                 # a new run
    .venv/bin/python experiments/exp1/drone_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/drone_complete.py --report-only   # tables and figures only
Output: results/exp1/drone_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, N_ROLLOUTS, SEEDS, SHIFT_HIDDEN, TRAIN_HIDDEN, policies

from scm_to_mdp.envs import Drone
from scm_to_mdp.tool import Grid, Limits, Policy, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["3-1", "4-1", "5-1", "6-1", "3-2", "4-2"]  # grid side - number of drones
GUSTS = ["calm", "gust"]
HORIZON, GAMMA = 50, 0.95
GUST_EFFECT = 0.5
MODEL_GB = 12.0
LIMITS = Limits(
    mem_gb=16.0,
    timeout_s={"discovery": 120.0, "selection": 120.0, "learning": 120.0, "solve": 600.0, "training": 1800.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "drone_complete"


def drone(size: str, gust: float | None, seed: int) -> Drone:
    side, drones = (int(v) for v in str(size).split("-"))
    return Drone(side, drones, seed, gust=gust, gust_effect=GUST_EFFECT, max_model_gb=MODEL_GB)


def make_env(size: str, gust: str, seed: int) -> Drone:
    return drone(size, TRAIN_HIDDEN if gust == "gust" else None, seed)


def shifts(size: str, gust: str, seed: int) -> dict:
    """The same grid where gusts come more often (the learner never sees them)."""
    return {f"gust={p:g}": drone(size, p, seed) for p in SHIFT_HIDDEN}


def no_confounder(gust: str, policy: Policy) -> bool:
    """Without gusts the confounded logger is the expert: not run."""
    return gust == "calm" and policy.name == "confounded"


GRID = Grid(
    name="Drone",
    make_env=make_env,
    sizes=SIZES,
    size_name="side-drones",
    knobs=GUSTS,
    knob_name="gust",
    policies=policies(),
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=3, delta=DELTA, exact=False, deep=DEEP_BASELINES, state_parents=3),
    limits=LIMITS,
    workers="auto",
    out_dir=OUT_DIR,
    reward_transfer=True,
    shifts=shifts,
    constants={
        "gust": TRAIN_HIDDEN,
        "shift_gusts": SHIFT_HIDDEN,
        "gust_effect": GUST_EFFECT,
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
