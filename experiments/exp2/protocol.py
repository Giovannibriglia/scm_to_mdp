"""Experiment 2: our method against the state of the art (S2-S9).

Per domain, two scripts on the same data as experiment 1 (experiments/common.py: uniform logger,
10 ... 10k transitions, 10 seeds, evaluation at the nominal start and from 20 random starts).
Ours with both discovery methods of experiment 1 (PCMCI+ and DYNOTEARS), against

    DiFaRL        Ehrlichman, Tang, Dykstra, Makar (CLeaR 2026): action groups from causal discovery,
                  fitted Q iteration (baselines/difarl.py)
    FOCUS         Zhu, Chen, Tian, Zhang, Yu: CI-test structure + a MOPO-penalised factored model
                  (baselines/focus.py)
    flat SPIBB    Pi_b-SPIBB of Laroche, Trichelair, Tachet des Combes (ICML 2019): safe policy
                  improvement over the behaviour policy on the flat MLE model (baselines/spibb.py)
    SL-SPIBB      Simao & Spaan (IJCAI 2019): the same with structure learning (baselines/spibb.py)
    offline DQN   offline deep Q-learning on the logged data (baselines/offline_q.py)

and, as a reference that is given the structure, Factored SPIBB (Simao & Spaan, AAAI 2019): the same
safe policy improvement on the true graph instead of a learned one (baselines/spibb.py).

    <domain>_mdp.py    ours as point estimates and value iteration (PCMCI+ / DYNOTEARS -> MDP)
    <domain>_imdp.py   ours as interval sets and robust VI (PCMCI+ / DYNOTEARS -> IMDP: a promise)

and the oracle (the true observed MDP) of the kind, for V*_obs and the solve budget.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, domains.py

from common import grid, main  # noqa: E402, F401

EXPERIMENT = "exp2"
BASELINES = ("DiFaRL", "FOCUS", "flat SPIBB", "SL-SPIBB", "offline DQN")
REFERENCES = ("Factored SPIBB (true graph)",)  # given the true graph
METHODS = {
    "mdp": ("PCMCI+ -> MDP", "DYNOTEARS -> MDP", *BASELINES, *REFERENCES, "oracle -> MDP"),
    "imdp": ("PCMCI+ -> IMDP", "DYNOTEARS -> IMDP", *BASELINES, *REFERENCES, "oracle -> IMDP"),
}


def experiment_grid(domain: str, kind: str):
    return grid(EXPERIMENT, domain, kind, METHODS[kind])
