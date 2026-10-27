"""Settings shared by the trial grids (one domain per pair of scripts: <domain>_uniform_rmdp.py and
<domain>_uniform_mdp.py).

Every trial runs one knob of its domain (experiment 1's training environment), the uniform logger only,
episodes of HORIZON steps (each script lists its numbers of episodes, N_EPISODES; the figures count the
transitions, episodes x HORIZON, less where an episode ends early) and 5 seeds, and evaluates only in
the MDP the data come from (no reward transfer, no dynamics shift).

The budget is common to every domain (LIMITS, per step of a method). Solving is limited relative to the
oracle: on every dataset the oracle (the true observed MDP: oracle -> IMDP, within +-1e-6, robust VI;
oracle -> MDP, VI) runs first, with the memory limit only, and every other method may solve for
max(SOLVE_FLOOR_S, SOLVE_FACTOR x the oracle's time per value-iteration sweep x the sweeps the stopping
rule may need at the rate gamma): SOLVE_FACTOR times the oracle's work at the worst rate (tool/sweep.py;
not its total time, which can be a few sweeps where episodes end, e.g. Taxi). Every size of a grid must be
solvable by the oracle within the memory limit (calibrate_sizes.py measures it).

The discovered graphs are used as found (max_parents=None: no cap on the
parents BIC searches) and every robust arm uses the box relaxation (exact=False).

    *_rmdp.py   the methods with a formal promise: causal discovery -> IMDP, flat IMDP (a lower bound on
                the value), flat SPIBB and SL-SPIBB (an improvement over the behaviour policy w.h.p.,
                not a value bound); true graph -> IMDP as the reference; oracle -> IMDP
    *_mdp.py    the methods without one: causal discovery -> MDP (point estimates), flat MDP, DiFaRL,
                FOCUS, offline DQN, offline CQL; true graph -> MDP as the reference; oracle -> MDP

To drop a method from every trial, delete its line below; for one domain, pass that script its own names.
"""

from scm_to_mdp.tool import Limits, Policy, oracle_methods, standard_methods

HORIZON = 256  # steps per episode (rollout), every domain
SEEDS = [0, 1, 2, 3, 4]
LIMITS = Limits(  # per step of a method, every domain; "solve": the fallback when no oracle finished
    mem_gb=8.0,
    timeout_s={"discovery": 1800.0, "selection": 1800.0, "learning": 1800.0, "training": 3600.0,
               "solve": 3600.0},
)  # fmt: skip
SOLVE_FACTOR = 3.0  # solve limit = max(SOLVE_FLOOR_S, SOLVE_FACTOR x the oracle's work at the rate gamma)
SOLVE_FLOOR_S = 300.0
SAVE_MODELS = "all"  # the models (and pictures) of the first seed at every number of episodes
DELTA = 0.05
TRAIN_HIDDEN = 0.1  # P(hidden variable on) at every step: experiment 1's data (experiments/exp1/protocol.py)
POLICIES = [Policy("uniform")]

RMDP_METHODS = (
    "BIC -> IMDP",
    "PCMCI+ -> IMDP",
    "GES -> IMDP",
    "DYNOTEARS -> IMDP",
    "flat IMDP",
    "true graph -> IMDP",  # reference
    "flat SPIBB",  # safe improvement over the behaviour policy (not a value bound)
    "SL-SPIBB",
    "oracle -> IMDP",  # the true observed MDP +-1e-6: the upper reference and the budget's yardstick
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
    "oracle -> MDP",  # the true observed MDP: its value is V*_obs
)
DEEP = ("offline DQN", "offline CQL")


def methods(names, state_parents: int = 3) -> dict:
    """The named methods of tool.standard_methods. state_parents: SL-SPIBB's candidate parent sets (the
    domain's value in experiment 1)."""
    every = (
        standard_methods(
            max_parents=None,
            delta=DELTA,
            exact=False,
            deep=any(name in DEEP for name in names),
            state_parents=state_parents,
        )
        | oracle_methods()
    )
    return {name: every[name] for name in names}
