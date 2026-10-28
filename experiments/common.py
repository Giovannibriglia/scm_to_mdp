"""What experiments 1 and 2 share (S1, S9): the data, the budget, the evaluation and how a script runs.

    logger      the uniform-random behaviour policy only (every action tried in every visited state)
    data        a budget of N_TRANSITIONS logged transitions per dataset (the first n transitions of one
                table, so the datasets are nested); the logger plays whole episodes, until the task
                ends or for a few steps in domains without an end (domains.Domain.horizon), and the
                episode where the budget ends is cut. SEEDS seeds
    evaluation  in the MDP the data come from, with the same reward: at the nominal start and from
                N_STARTS random starts (no reward transfer, no dynamics shift)
    budget      LIMITS per step of a method (memory 8 GB); solving is limited relative to the oracle:
                on every dataset the oracle (the true observed MDP; oracle -> MDP by VI, oracle -> IMDP
                within +-1e-6 by robust VI) runs first with the memory limit only, and every other method
                may solve for max(SOLVE_FLOOR_S, SOLVE_FACTOR x the oracle's time per sweep x the sweeps
                the stopping rule may need at the rate gamma) (tool/sweep.py)
    robustness  for the methods with a promise (scm_to_mdp/robustness.py): the value bracket, the
                optimistic bound, a one-sweep check of the promise and, where a domain has failure
                states, the probability of failing within FAILURE_HORIZON steps
    rewards     where an experiment asks for it (experiment 1, Sepsis): every method re-planned under
                REWARD_FAMILY's random rewards and the domain's own alternative rewards
                (scm_to_mdp/reward_family.py: does the promise hold for every reward at once?)
    report      tool.report_grid: values as the IQM over seeds with a 95% bootstrap interval
                (plotting/stats.py); the transitions needed for a 95%-optimal policy as the median and
                its exact 95% interval (some seeds may never get there)

A script (experiments/exp<k>/<domain>_<kind>.py) builds its grid with grid(...) and calls main(...).
"""

import dataclasses
import warnings
from pathlib import Path

from domains import DOMAINS

from scm_to_mdp.reward_family import RewardFamily
from scm_to_mdp.tool import (
    Grid,
    Limits,
    Policy,
    oracle_methods,
    report_grid,
    run_folder,
    run_grid,
    standard_methods,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
# the data budgets: up to 100k, so that the largest problems (SysAdmin N = 11: 24.6k state-action
# pairs) can visit every pair
N_TRANSITIONS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
FAILURE_HORIZON = 256  # the failure probabilities: "fail within 256 steps", in every domain
SEEDS = list(range(10))
N_STARTS = 20  # random starts of the evaluation (besides the nominal start)
DELTA = 0.05  # confidence of the interval sets (S4)
TARGET = 0.95  # "a 95%-optimal policy": the threshold of the transitions-needed tables
LIMITS = Limits(  # per step of a method, every domain; "solve": the fallback when no oracle finished
    mem_gb=8.0,
    timeout_s={"discovery": 1800.0, "selection": 1800.0, "learning": 1800.0, "training": 3600.0,
               "solve": 3600.0},
)  # fmt: skip
SOLVE_FACTOR = 3.0
SOLVE_FLOOR_S = 300.0
SAVE_MODELS = "all"  # the models (and pictures) of the first seed at every budget
POLICIES = [Policy("uniform")]
# random rewards on the same dynamics, for the grids that ask for them (grid(..., rewards=True)):
# per_family rewards of each family (dense, sparse, noise, stress), every learned model re-planned
# under each (one solve per reward: about 55 s per IMDP solve at 2,048 states). n_transitions: the
# budgets evaluated (None: every one of N_TRANSITIONS); max_states: skip larger problems
# (None: every size)
REWARD_FAMILY = RewardFamily(per_family=5, n_transitions=None, max_states=None)
KINDS = ("mdp", "imdp")  # the scripts' two families: no guarantee (-> MDP) and with a promise (-> IMDP)
DEEP = ("offline DQN", "offline CQL")


def methods(names, domain) -> dict:
    """The named methods of tool.standard_methods and tool.oracle_methods, in the order given, with the
    domain's SPIBB settings and sparse-graph actions (domains.Domain) and the uniform logger as SPIBB's
    baseline policy."""
    every = (
        standard_methods(
            max_parents=None,
            delta=DELTA,
            exact=False,
            deep=any(name in DEEP for name in names),
            state_parents=domain.sl_spibb_parents,
            spibb_options={"m": domain.spibb_m, "behaviour": "uniform"},
            sl_spibb_options={"m": domain.sl_spibb_m, "eps": domain.sl_spibb_eps, "behaviour": "uniform"},
            sparse_actions=domain.sparse_actions,
        )
        | oracle_methods()
    )
    return {name: every[name] for name in names}


def grid(experiment: str, domain: str, kind: str, names, rewards: bool = False) -> Grid:
    """The grid of one script: experiment "exp1" | "exp2", a domain of domains.DOMAINS, kind "mdp" |
    "imdp", and its methods (names). rewards: also re-plan every method under REWARD_FAMILY's random
    rewards and the domain's alternative rewards (Env.alternative_rewards; results_transfer.csv).
    Results go to results/<experiment>/<domain>_<kind>_<date>_<time>."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    d = DOMAINS[domain]
    return Grid(
        name=d.title,
        make_env=d.make_env,
        sizes=d.sizes,
        size_name=d.size_name,
        knobs=d.knobs,
        knob_name=d.knob_name,
        constants=d.constants,
        policies=POLICIES,
        n_transitions=N_TRANSITIONS,
        seeds=SEEDS,
        horizon=d.horizon,
        failure_horizon=FAILURE_HORIZON,
        gamma=d.gamma,
        n_starts=N_STARTS,
        methods=methods(names, d),
        limits=LIMITS,
        solve_factor=SOLVE_FACTOR,
        solve_floor_s=SOLVE_FLOOR_S,
        save_models=SAVE_MODELS,
        workers="auto",
        out_dir=ROOT / "results" / experiment / f"{domain}_{kind}",
        reward_transfer=rewards,
        reward_family=REWARD_FAMILY if rewards else None,
        shifts=None,
    )


def main(base: Grid) -> None:
    """Run (or --resume, or --report-only) the grid, then write its tables and figures."""
    out_dir, report_only = run_folder(base.out_dir)  # a new <out_dir>_<date>_<time>, or an earlier one
    g = dataclasses.replace(base, out_dir=out_dir)
    if not report_only:
        run_grid(g)
    report_grid(g.out_dir, target=TARGET, delta=DELTA)
