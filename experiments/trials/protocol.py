"""Settings shared by the trial grids (one domain per pair of scripts: <domain>_uniform_rmdp.py and
<domain>_uniform_mdp.py).

Every trial runs one knob of its domain (experiment 1's training environment), the uniform logger only,
50 / 250 / 750 rollouts and 5 seeds, and evaluates only in the MDP the data come from (no reward
transfer, no dynamics shift). The discovered graphs are used as found (max_parents=None: no cap on the
parents BIC searches) and every robust arm uses the box relaxation (exact=False).

    *_rmdp.py   the methods with a formal promise: causal discovery -> RMDP, flat RMDP (a lower bound on
                the value), flat SPIBB and SL-SPIBB (an improvement over the behaviour policy w.h.p.,
                not a value bound); true graph -> RMDP as the reference
    *_mdp.py    the methods without one: causal discovery -> MDP (point estimates), flat MDP, DiFaRL,
                FOCUS, offline DQN, offline CQL; true graph -> MDP as the reference

To drop a method from every trial, delete its line below; for one domain, pass that script its own names.
"""

from scm_to_mdp.tool import Policy, standard_methods

N_ROLLOUTS = [50, 250, 750]
SEEDS = [0, 1, 2, 3, 4]
DELTA = 0.05
TRAIN_HIDDEN = 0.1  # P(hidden variable on) at every step: experiment 1's data (experiments/exp1/protocol.py)
POLICIES = [Policy("uniform")]

RMDP_METHODS = (
    "BIC -> RMDP",
    "PCMCI+ -> RMDP",
    "GES -> RMDP",
    "DYNOTEARS -> RMDP",
    "flat RMDP",
    "true graph -> RMDP",  # reference
    "flat SPIBB",  # safe improvement over the behaviour policy (not a value bound)
    "SL-SPIBB",
)
MDP_METHODS = (
    "BIC -> MDP",
    "PCMCI+ -> MDP",
    "GES -> MDP",
    "DYNOTEARS -> MDP",
    "flat MDP",
    "true graph -> MDP",  # reference
    "DiFaRL",
    "FOCUS",
    "offline DQN",
    "offline CQL",
)
DEEP = ("offline DQN", "offline CQL")


def methods(names, state_parents: int = 3) -> dict:
    """The named methods of tool.standard_methods. state_parents: SL-SPIBB's candidate parent sets (the
    domain's value in experiment 1)."""
    every = standard_methods(
        max_parents=None,
        delta=DELTA,
        exact=False,
        deep=any(name in DEEP for name in names),
        state_parents=state_parents,
    )
    return {name: every[name] for name in names}
