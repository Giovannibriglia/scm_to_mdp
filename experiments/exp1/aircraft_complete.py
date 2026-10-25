"""Aircraft collision avoidance, complete grid: every size x proximity x behaviour policy x rollouts
(protocol.py: the claims, the loggers, the seeds, the shifts and the expectations common to every domain).

    size        width x height: 8x6, 12x8, 15x15 (441, 1053, 4096 observed states; 15x15 is the learning
                instance of Schnitzer, Abate, Parker, AAAI 2026, App. E.1; a server)
    knob        proximity: "on" (an intended climb or descent succeeds w.p. 0.5 instead of 0.8 while the
                intruder is near: their Sec. 2 example) or "off"
    policies    uniform, expert eps 0.05, expert eps 0, confounded (stays during hidden turbulence w.p.
                0.95)
    hidden      the turbulence (src/scm_to_mdp/envs/aircraft.py) halves the success of an intended climb
                or descent (the own altitude only), w.p. 0.1 in the data, 0.3 and 0.6 at deployment
    rollouts    10 ... 1600 (horizon 20, gamma 0.95), 5 seeds, the standard methods (no per-(s,a)
                product set)

Expectations for this domain, written 2026-10-24 before the runs:
    H1  proximity on: the own altitude has 5 parents (x, y, iy, up, down), the hardest family to learn;
        discovery that misses x or iy over-promises near the intruder (A);
    H2  the expert rarely climbs far from the intruder: under eps = 0 the action edges are thin and the
        point models can be optimistic about a late manoeuvre (A2: the collision is a trap);
    H3  shift: manoeuvres fail more often than in the data; a policy that relies on a late manoeuvre
        loses more (B2, robustness), as long as the robust set is wide enough (small n).

Run from the repository root (15x15: about 3 GB for the true model with turbulence):
    .venv/bin/python experiments/exp1/aircraft_complete.py                 # a new run
    .venv/bin/python experiments/exp1/aircraft_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/aircraft_complete.py --report-only   # tables and figures only
Output: results/exp1/aircraft_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, N_ROLLOUTS, SEEDS, SHIFT_HIDDEN, TRAIN_HIDDEN, policies

from scm_to_mdp.envs import Aircraft
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["8x6", "12x8", "15x15"]  # width x height
PROXIMITY = ["on", "off"]
HORIZON, GAMMA = 20, 0.95
TURBULENCE_EFFECT = 0.5
MODEL_GB = 12.0
LIMITS = Limits(
    mem_gb=16.0,
    timeout_s={"discovery": 120.0, "selection": 120.0, "learning": 120.0, "solve": 600.0, "training": 1800.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "aircraft_complete"


def aircraft(size: str, proximity: str, seed: int, turbulence: float) -> Aircraft:
    width, height = (int(v) for v in str(size).split("x"))
    return Aircraft(width, height, seed, proximity=proximity == "on", turbulence=turbulence,
                    turbulence_effect=TURBULENCE_EFFECT, max_model_gb=MODEL_GB)  # fmt: skip


def make_env(size: str, proximity: str, seed: int) -> Aircraft:
    return aircraft(size, proximity, seed, TRAIN_HIDDEN)


def shifts(size: str, proximity: str, seed: int) -> dict:
    """The same airspace with more turbulence (the learner never sees it)."""
    return {f"turbulence={p:g}": aircraft(size, proximity, seed, p) for p in SHIFT_HIDDEN}


GRID = Grid(
    name="Aircraft",
    make_env=make_env,
    sizes=SIZES,
    size_name="width x height",
    knobs=PROXIMITY,
    knob_name="proximity",
    policies=policies(),
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=5, delta=DELTA, exact=False, deep=DEEP_BASELINES, state_parents=3),
    limits=LIMITS,
    workers="auto",
    out_dir=OUT_DIR,
    reward_transfer=True,
    shifts=shifts,
    constants={
        "turbulence": TRAIN_HIDDEN,
        "shift_turbulence": SHIFT_HIDDEN,
        "turbulence_effect": TURBULENCE_EFFECT,
        "model_gb": MODEL_GB,
    },  # fmt: skip
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
