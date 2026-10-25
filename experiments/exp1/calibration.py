"""Calibration of the promises: 100 seeds on one small setting per domain, to test "the promise holds
w.p. >= 1 - delta" (protocol.py, claim A), which the 5 seeds of the complete grids can only describe.

For each domain: its smallest size and one knob, the uniform and the eps = 0.05 expert loggers, 25, 100
and 400 rollouts (Sepsis: 100, 250, 500 patients), 100 seeds, four arms:
    true graph -> RMDP       the guarantee with the right structure (1 - delta by the union bound)
    BIC -> RMDP              the graph chosen from the same data: the 1 - delta is conditional on it
    flat RMDP                no structure (1 - delta by the union bound)
    flat MDP                 no guarantee: the contrast (its "promise" is its own value estimate)
No transfer, no shift, no saved models.

summary/validity.csv: per (setting, size, n, method) the share of seeds whose promise held in every state,
its two-sided 95% Clopper-Pearson interval, the verdict against 1 - delta ("confirmed": lower end >=
0.95, which with 100 seeds needs 100/100: lower end 0.964; "refuted": upper end < 0.95, from about 11
failures on; else "undecided"), and contain_all_share: how often every learned row contained the
true one (the event the union bound guarantees; it implies the promise).

Run from the repository root (a few hours):
    .venv/bin/python experiments/exp1/calibration.py [--resume | --report-only] [--workers N]
Output: results/exp1/calibration_<date>_<time>/<domain>/.
"""

import dataclasses
import importlib
import sys
import warnings
from pathlib import Path

from protocol import DELTA

from scm_to_mdp.tool import Policy, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

SEEDS = list(range(100))
ARMS = ["true graph -> RMDP", "BIC -> RMDP", "flat RMDP", "flat MDP"]
POLICIES = [Policy("uniform"), Policy("expert", epsilon=0.05)]
# domain script -> (size, knob, numbers of rollouts)
SETTINGS = {
    "sysadmin_complete": (3, "ring", [25, 100, 400]),
    "sepsis_complete": (2, "observed", [100, 250, 500]),
    "chemistry_complete": (4, "0.4", [25, 100, 400]),
    "aircraft_complete": ("8x6", "on", [25, 100, 400]),
    "drone_complete": ("3-1", "gust", [25, 100, 400]),
    "stock_complete": ("2x2", 0.8, [25, 100, 400]),
    "frozenlake_complete": ("4x4-1", "wind", [25, 100, 400]),
    "coffee_complete": ("coffee", "original", [25, 100, 400]),
}
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "calibration"


def grid_of(script: str, out_dir: Path):
    m = importlib.import_module(script)
    size, knob, n_rollouts = SETTINGS[script]
    return dataclasses.replace(
        m.GRID,
        sizes=[size],
        knobs=[knob],
        policies=POLICIES,
        n_rollouts=n_rollouts,
        seeds=SEEDS,
        methods={name: m.GRID.methods[name] for name in ARMS},
        reward_transfer=False,
        shifts=None,
        save_models="none",
        out_dir=out_dir / script.removesuffix("_complete"),
    )


if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)
    for script in SETTINGS:
        grid = grid_of(script, out_dir)
        if not report_only:
            run_grid(grid)
        report_grid(grid.out_dir, target=0.95, delta=DELTA, settings_too=False)
