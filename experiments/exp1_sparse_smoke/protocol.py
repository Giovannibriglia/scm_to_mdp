"""Smoke version of experiment 1's sparse-graph run (experiments/exp1_sparse): its methods on a tiny grid
(experiments/smoke.py), to check that every script runs and to see its final tables and figures.
The numbers are not results; run experiments/exp1_sparse/ for those.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, domains.py, smoke.py

from common import main  # noqa: E402, F401
from smoke import load_protocol, shrink  # noqa: E402

EXPERIMENT = "exp1_sparse_smoke"
FULL = load_protocol("exp1_sparse")  # the real experiment: its methods, rewards and settings
METHODS = FULL.METHODS


def experiment_grid(domain: str, kind: str):
    return shrink(FULL.experiment_grid(domain, kind), EXPERIMENT)
