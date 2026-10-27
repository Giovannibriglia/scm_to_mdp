"""Experiment 2: our method against the state of the art (S2-S9).

Per domain, two scripts on the same data as experiment 1 (experiments/common.py: uniform logger,
1 ... 10k episodes of 256 steps, 10 seeds, evaluation at the nominal start and from 20 random starts).
Ours with both discovery methods of experiment 1 (PCMCI+ and DYNOTEARS), against

    DiFaRL        Ehrlichman, Tang, Dykstra, Makar (CLeaR 2026): action groups from causal discovery,
                  fitted Q iteration (baselines/difarl.py)
    FOCUS         Zhu, Chen, Tian, Zhang, Yu: CI-test structure + a MOPO-penalised factored model
                  (baselines/focus.py)
    SL-SPIBB      Simao & Spaan 2019: structure learning + safe policy improvement over the behaviour
                  policy (baselines/spibb.py)
    offline DQN   offline deep Q-learning on the logged data (baselines/offline_q.py)

    <domain>_mdp.py    ours as point estimates and value iteration (PCMCI+ / DYNOTEARS -> MDP)
    <domain>_imdp.py   ours as interval sets and robust VI (PCMCI+ / DYNOTEARS -> IMDP: a promise)

and the oracle (the true observed MDP) of the kind, for V*_obs and the solve budget.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, domains.py

from common import grid, main  # noqa: E402, F401

EXPERIMENT = "exp2"
BASELINES = ("DiFaRL", "FOCUS", "SL-SPIBB", "offline DQN")
METHODS = {
    "mdp": ("PCMCI+ -> MDP", "DYNOTEARS -> MDP", *BASELINES, "oracle -> MDP"),
    "imdp": ("PCMCI+ -> IMDP", "DYNOTEARS -> IMDP", *BASELINES, "oracle -> IMDP"),
}


def experiment_grid(domain: str, kind: str):
    return grid(EXPERIMENT, domain, kind, METHODS[kind])
