"""Experiment 1, sparse graph: the arms "sparse -> MDP" and "sparse -> IMDP" of experiment 1, run apart
because they were added after experiment 1 started (experiments/merge_runs.py joins the two runs).

The sparse graph (discovery/sparse.py) gives each next-state variable itself and its own action
variables as parents: every action variable (one controller), or, for the multi-agent domains (Frozen
Lake, Drone: domains.Domain.sparse_actions = "by_index"), only its agent's. Everything else is
experiment 1's (exp1/protocol.py): the same data (seeds fix the environment and the logger), budgets,
evaluation and random rewards. The oracle runs too, so the relative solve limit is measured in this run;
the merge keeps experiment 1's oracle rows and drops these.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, domains.py

from common import grid, main  # noqa: E402, F401

_spec = importlib.util.spec_from_file_location(
    "exp1_protocol", Path(__file__).resolve().parents[1] / "exp1" / "protocol.py"
)
EXP1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EXP1)

EXPERIMENT = "exp1_sparse"
REWARD_DOMAINS = EXP1.REWARD_DOMAINS
METHODS = {
    "mdp": ("sparse -> MDP", "oracle -> MDP"),
    "imdp": ("sparse -> IMDP", "oracle -> IMDP"),
}


def experiment_grid(domain: str, kind: str):
    return grid(EXPERIMENT, domain, kind, METHODS[kind], rewards=domain in REWARD_DOMAINS)
