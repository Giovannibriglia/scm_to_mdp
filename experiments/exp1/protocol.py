"""Experiment 1: is causal discovery better than assuming the full factored graph? (S2-S9)

Per domain, two scripts on the same data (experiments/common.py: uniform logger, 10 ... 10k
transitions, 10 seeds, evaluation at the nominal start and from 20 random starts):

    <domain>_mdp.py    point estimates and value iteration, no guarantee:
                       PCMCI+ -> MDP, DYNOTEARS -> MDP (causal discovery), full-fact -> MDP (every slice-t
                       input a parent of every next-state variable, no discovery), true graph -> MDP (the
                       oracle graph), flat MDP (no structure); oracle -> MDP (the true observed MDP: V*_obs
                       and the solve budget's yardstick)
    <domain>_imdp.py   interval sets (Clopper-Pearson, delta 0.05), box relaxation, robust VI: a promise
                       that is a lower bound on the true value w.h.p.: the same five arms -> IMDP, flat IMDP;
                       oracle -> IMDP

The discovered graph is also scored against the true one (recall: true edges found; precision: found
edges that are true; SHD, SID); full-fact has recall 1 by construction.

In REWARD_DOMAINS (Sepsis: nothing hidden, structure that matters per reward, a failure state), every
learned model is also re-planned, not re-learned, under 20 random rewards (common.REWARD_FAMILY) and
the domain's 3 alternative rewards: does the promise hold for every objective at once?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, domains.py

from common import grid, main  # noqa: E402, F401

EXPERIMENT = "exp1"
REWARD_DOMAINS = ("sepsis",)  # the domains evaluated under many rewards
METHODS = {
    "mdp": (
        "PCMCI+ -> MDP",
        "DYNOTEARS -> MDP",
        "full-fact -> MDP",
        "true graph -> MDP",
        "flat MDP",
        "oracle -> MDP",
    ),
    "imdp": (
        "PCMCI+ -> IMDP",
        "DYNOTEARS -> IMDP",
        "full-fact -> IMDP",
        "true graph -> IMDP",
        "flat IMDP",
        "oracle -> IMDP",
    ),
}


def experiment_grid(domain: str, kind: str):
    return grid(EXPERIMENT, domain, kind, METHODS[kind], rewards=domain in REWARD_DOMAINS)
