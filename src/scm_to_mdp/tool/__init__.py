"""The tool: from offline data to a policy with a guarantee, and how to evaluate it.

Three parts (each in its own module):

    1. learn + solve   data (DataSpec.read(df)) -> learned model -> policy and promise   (methods.py)
    2. evaluate        a solved result against a ground truth (MDPGroundTruth), or alone (evaluate.py)
    3. report          tables and figures from results.csv                              (report.py)

and the sweep that runs them over problem sizes, numbers of transitions and seeds (sweep.py), and grids
of sweeps over a knob and behaviour policies (grid.py, report_grid.py); the saved models of one
seed (store.py); one timestamped folder per run (runs.py). See experiments/exp1/ and
experiments/exp2/ for grids (experiments/common.py, domains.py).
"""

from scm_to_mdp.data import DataSpec
from scm_to_mdp.ground_truth import MDPGroundTruth, ModelTooLarge
from scm_to_mdp.tool.evaluate import evaluate
from scm_to_mdp.tool.grid import Grid, Policy, experiments, run_grid
from scm_to_mdp.tool.methods import (
    BASELINES,
    Limits,
    Method,
    Result,
    learn,
    oracle_methods,
    solve,
    standard_methods,
)
from scm_to_mdp.tool.report import report
from scm_to_mdp.tool.report_grid import report_grid
from scm_to_mdp.tool.runs import run_folder
from scm_to_mdp.tool.store import load as load_model
from scm_to_mdp.tool.sweep import Experiment, read_results, run_experiment

__all__ = [
    "BASELINES",
    "DataSpec",
    "Experiment",
    "Grid",
    "Limits",
    "MDPGroundTruth",
    "Method",
    "ModelTooLarge",
    "Policy",
    "Result",
    "evaluate",
    "experiments",
    "learn",
    "load_model",
    "read_results",
    "report",
    "report_grid",
    "run_experiment",
    "run_folder",
    "run_grid",
    "solve",
    "standard_methods",
    "oracle_methods",
]
