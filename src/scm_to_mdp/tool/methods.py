"""Part 1 of the tool: learn a model from data (S2-S7), then solve it (S8).

A Method names one choice per step. The learner is one of:

    "causal"     S2 discovery -> S3 selection -> S4 parameter learning -> a factored interval MDP
                 discovery: "bic" | "pcmci+" | "ges" | "oracle" (the true graph, given)
                 selection: "bic_best" | "given" (for the oracle)
    "flat_rmdp"  baseline: one interval per (s, a, s'), no structure      -> an interval MDP
    "flat_mdp"   baseline: the plug-in MDP from counts, no guarantee      -> an exact MDP (L == U)
    "difarl"     baseline: DiFaRL's action groups, model-free, no guarantee -> action groups
    "offline_dqn", "offline_cql"
                 baselines: offline deep Q-learning on the logged data (baselines/offline_q.py),
                 model-free, no guarantee; nothing to learn before the solver, which trains
    "spibb", "sl_spibb"
                 baselines: safe policy improvement over the (estimated) behaviour policy, flat
                 (Laroche et al.) or with structure learning (Simao & Spaan; baselines/spibb.py)
    "focus"      baseline: FOCUS's CI-test structure + a MOPO-penalised factored model (discrete
                 version, baselines/focus.py) -> an exact MDP with a penalised reward

and the solver one of (which ones fit which learner: SOLVERS_FOR):

    "box_robust_vi"     relax the factored set to its box (Schnitzer et al.'s IA), then robust VI
    "vertex_robust_vi"  exact robust VI over the factored set (0/1 state variables only)
    "robust_vi"         robust VI on an interval MDP
    "vi"                value iteration on an exact MDP
    "fqi"               DiFaRL's decomposed fitted Q iteration (uses the data again)
    "greedy_q"          train the Q-network on the data (step "training"), then its greedy policy
    "spibb"             Pi_b-SPIBB policy iteration (a stochastic policy)

Every step runs either in this process (limits=None) or in its own process with a memory and time
limit (Limits). A step that fails stops the method: Result.status says how ("memout", "timeout",
"error") and Result.failed_step where.
"""

import inspect
import math
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from scm_to_mdp.baselines import difarl, flat_mdp, flat_rmdp, focus, offline_q, spibb
from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import ActionGroups, Dataset, Graph, Solution
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.selection import bic_best, given
from scm_to_mdp.solving import robust_vi, value_iteration, vertex_robust_vi
from scm_to_mdp.utils.limits import run_limited

LEARNERS = (
    "causal",
    "flat_rmdp",
    "flat_mdp",
    "difarl",
    "offline_dqn",
    "offline_cql",
    "spibb",
    "sl_spibb",
    "focus",
)
BASELINES = LEARNERS[1:]
OFFLINE_Q = ("offline_dqn", "offline_cql")
CQL_ALPHA = 0.1  # offline CQL's penalty weight (baselines/offline_q.py says how it was chosen)
DISCOVERY = ("bic", "pcmci+", "ges", "oracle")
SELECTION = {"bic_best": bic_best.select, "given": given.select}
SOLVERS_FOR = {
    "causal": ("box_robust_vi", "vertex_robust_vi"),
    "flat_rmdp": ("robust_vi",),
    "flat_mdp": ("vi", "robust_vi"),
    "difarl": ("fqi",),
    "offline_dqn": ("greedy_q",),
    "offline_cql": ("greedy_q",),
    "spibb": ("spibb",),
    "sl_spibb": ("spibb",),
    "focus": ("vi",),
}
ROBUST_SOLVERS = ("box_robust_vi", "vertex_robust_vi", "robust_vi")
STEPS = ("discovery", "selection", "learning", "relax", "training", "solve")  # every step a method can have
STOPPED = ("timeout", "memout")


def discovery_function(name: str):
    """fit(dataset, ...) -> Candidates of a discovery method (heavy libraries imported on use)."""
    if name == "bic":
        from scm_to_mdp.discovery import bic_search

        return bic_search.fit
    if name == "pcmci+":
        from scm_to_mdp.discovery import pcmci_plus

        return pcmci_plus.fit
    if name == "ges":
        from scm_to_mdp.discovery import ges

        return ges.fit
    from scm_to_mdp.discovery import oracle

    return oracle.fit


def _check_options(fn, options: dict, n_inputs: int, what: str):
    """Raise ValueError for an option that fn does not take (fn's first n_inputs are its inputs)."""
    allowed = list(inspect.signature(fn).parameters)[n_inputs:]
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise ValueError(f"{what}: unknown options {unknown}; allowed: {allowed}")


# ==========================================================================================
# The choice of a method
# ==========================================================================================
@dataclass(frozen=True)
class Method:
    """One choice per step. Example:
        Method("causal", discovery="pcmci+", selection="bic_best", solver="box_robust_vi",
               discovery_options={"alpha": 0.01}, learning_options={"delta": 0.05})
    Options go to the function of their step (unknown names are an error)."""

    learner: str
    solver: str
    discovery: str | None = None
    selection: str | None = None
    discovery_options: dict = field(default_factory=dict)
    learning_options: dict = field(default_factory=dict)
    solver_options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}, not {self.learner!r}")
        if self.solver not in SOLVERS_FOR[self.learner]:
            raise ValueError(f"learner {self.learner!r} is solved with one of {SOLVERS_FOR[self.learner]}")
        if self.learner == "causal":
            if self.discovery not in DISCOVERY:
                raise ValueError(f"the causal learner needs discovery in {DISCOVERY}")
            if self.selection not in SELECTION:
                raise ValueError(f"the causal learner needs selection in {list(SELECTION)}")
            if (self.discovery == "oracle") != (self.selection == "given"):
                raise ValueError("selection 'given' goes with discovery 'oracle', and only with it")
            if self.discovery != "oracle":
                _check_options(discovery_function(self.discovery), self.discovery_options, 1, "discovery")
            _check_options(nbn_intervals.learn, self.learning_options, 2, "learning")
        else:
            if self.discovery is not None or self.selection is not None:
                raise ValueError(f"learner {self.learner!r} has no discovery or selection to choose")
            if self.learner == "difarl":
                _check_options(difarl.groups, self.discovery_options, 1, "discovery (DiFaRL groups)")
            elif self.discovery_options:
                raise ValueError(f"learner {self.learner!r} takes no discovery options")
            learn_fn = {
                "flat_rmdp": flat_rmdp.learn,
                "flat_mdp": flat_mdp.learn,
                "spibb": spibb.flat_model,
                "sl_spibb": spibb.sl_model,
            }.get(self.learner)
            if self.learner == "focus":
                learn_fn = focus.penalised_mdp
                _check_options(learn_fn, self.learning_options, 2, "learning (FOCUS)")
                learn_fn = None
            if learn_fn is not None:
                _check_options(learn_fn, self.learning_options, 1, "learning")
            elif self.learning_options and self.learner != "focus":
                raise ValueError(f"{self.learner} takes no learning options (its options are solver options)")
        solver_fn = {
            "box_robust_vi": robust_vi.solve,
            "vertex_robust_vi": vertex_robust_vi.solve,
            "robust_vi": robust_vi.solve,
            "vi": value_iteration.solve,
        }.get(self.solver)
        if solver_fn is not None:
            _check_options(solver_fn, self.solver_options, 3 if self.solver == "vi" else 2, "solver")
        elif self.solver == "spibb":
            _check_options(spibb.solve, self.solver_options, 2, "solver (SPIBB)")
        elif self.solver == "greedy_q":
            _check_options(offline_q.learn, self.solver_options, 2, "solver (offline Q-learning)")
            if self.learner == "offline_dqn" and "cql_alpha" in self.solver_options:
                raise ValueError("offline DQN has no CQL penalty (use learner 'offline_cql')")
        else:
            _check_options(difarl.fqi, self.solver_options, 3, "solver (FQI)")

    @property
    def role(self) -> str:
        """ "baseline" (no causal structure learned), "oracle" (given the true graph) or "ours"."""
        if self.learner in BASELINES:
            return "baseline"
        return "oracle" if self.discovery == "oracle" else "ours"

    @property
    def guarantee(self) -> bool:
        """Whether the method's promise is meant as a lower bound on the true value (for "causal":
        when the selected graph contains the true parents)."""
        return self.solver in ROBUST_SOLVERS and self.learner in ("causal", "flat_rmdp")

    def describe(self) -> dict:
        return {
            "learner": self.learner,
            "discovery": self.discovery,
            "selection": self.selection,
            "solver": self.solver,
            "role": self.role,
            "guarantee": self.guarantee,
        }


@dataclass(frozen=True)
class Limits:
    """Memory limit of every step (GB) and time limits (seconds) of some steps, by step name."""

    mem_gb: float = 8.0
    timeout_s: dict = field(default_factory=lambda: {"discovery": 60.0, "selection": 60.0, "solve": 60.0})


# ==========================================================================================
# What a method produced
# ==========================================================================================
@dataclass
class Result:
    method: Method
    status: str = "ok"  # "ok", or how the failed step stopped: "memout", "timeout", "error"
    failed_step: str | None = None
    error: str | None = None
    steps: dict = field(default_factory=dict)  # step -> {"status", "seconds", "gb"}
    graph: Graph | None = None  # causal learner: the selected graph (S3)
    model: Any = None  # FactorBounds | IntervalMDP | ActionGroups (the learned model)
    solution: Solution | None = None  # after solve(): the policy and its promise
    info: dict = field(default_factory=dict)  # e.g. size of the graph class, DiFaRL's groups

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def row(self) -> dict:
        """Flat record of this result (one row of results.csv, without the evaluation)."""
        row = {**self.method.describe(), "status": self.status, "failed_step": self.failed_step}
        row["error"] = self.error
        for name in STEPS:
            if name in self.steps:
                for key, value in self.steps[name].items():
                    row[f"{name}_{'s' if key == 'seconds' else key}"] = value
        return {**row, **self.info}


def _run_step(result: Result, name: str, fn, x, limits: Limits | None):
    """fn(x) as step `name` of result; returns its output, or None when the step failed."""
    if limits is None:
        start = time.monotonic()
        try:
            value, status = fn(x), "ok"
        except MemoryError:
            value, status = None, "memout"
        except Exception as err:  # noqa: BLE001 - reported in the result, and as a warning
            value, status = repr(err), "error"
        seconds, gb = time.monotonic() - start, math.nan
    else:
        start = "spawn" if name == "training" else "fork"  # a network trains in a fresh process (limits.py)
        out = run_limited(
            fn, x, mem_gb=limits.mem_gb, timeout_s=limits.timeout_s.get(name), start_method=start
        )
        value, status, seconds, gb = out.value, out.status, out.seconds, out.peak_gb
    result.steps[name] = {"status": status, "seconds": seconds, "gb": gb}
    if status == "ok":
        return value
    result.status, result.failed_step = status, name
    if status == "error":
        result.error = value
        warnings.warn(f"step {name} failed: {value}", stacklevel=3)
    return None


# ==========================================================================================
# Learning (S2-S7) and solving (S8)
# ==========================================================================================
def learning_steps(data: Dataset, method: Method, true_parents: dict | None = None) -> list:
    """The steps of method's learner as (name, function of the previous output)."""
    m = method
    if m.learner == "causal":
        if m.discovery == "oracle":
            if true_parents is None:
                raise ValueError("discovery 'oracle' needs true_parents (e.g. truth.true_parents)")
            discover = discovery_function("oracle")
            first = ("discovery", lambda d: discover(d, true_parents))
        else:
            discover = discovery_function(m.discovery)
            first = ("discovery", lambda d: discover(d, **m.discovery_options))
        select = SELECTION[m.selection]
        return [
            first,
            ("selection", lambda candidates: select(candidates, data)),
            ("learning", lambda graph: nbn_intervals.learn(data, graph, **m.learning_options)),
        ]
    if m.learner == "flat_rmdp":
        return [("learning", lambda d: flat_rmdp.learn(d, **m.learning_options))]
    if m.learner == "flat_mdp":
        return [("learning", lambda d: flat_mdp.learn(d, **m.learning_options))]
    if m.learner in OFFLINE_Q:
        return []  # model-free: the solver trains on the data
    if m.learner == "spibb":
        return [("learning", lambda d: spibb.flat_model(d, **m.learning_options))]
    if m.learner == "sl_spibb":
        return [("learning", lambda d: spibb.sl_model(d, **m.learning_options))]
    if m.learner == "focus":
        return [
            ("discovery", focus.discover),
            ("learning", lambda graph: focus.penalised_mdp(data, graph, **m.learning_options)),
        ]
    return [("discovery", lambda d: ActionGroups(difarl.groups(d, **m.discovery_options)))]


def solver_steps(data: Dataset, method: Method, gamma: float) -> list:
    """The steps of method's solver as (name, function of the learned model)."""
    opts = method.solver_options
    if method.solver == "box_robust_vi":
        return [("relax", product_bounds.compile), ("solve", lambda mdp: robust_vi.solve(mdp, gamma, **opts))]
    if method.solver == "vertex_robust_vi":
        return [("solve", lambda bounds: vertex_robust_vi.solve(bounds, gamma, **opts))]
    if method.solver == "robust_vi":
        return [("solve", lambda mdp: robust_vi.solve(mdp, gamma, **opts))]
    if method.solver == "vi":

        def vi(mdp):
            if not np.array_equal(mdp.L, mdp.U):
                raise ValueError("solver 'vi' needs an exact MDP (L == U)")
            return value_iteration.solve(mdp.L, mdp.reward, gamma, **opts)

        return [("solve", vi)]
    if method.solver == "spibb":
        return [("solve", lambda model: spibb.solve(model, gamma, **opts))]
    if method.solver == "greedy_q":
        alpha = {"cql_alpha": CQL_ALPHA} if method.learner == "offline_cql" else {}
        return [
            ("training", partial(_train_q, data, gamma, alpha | opts)),  # picklable: it runs in a new process
            ("solve", offline_q.greedy),
        ]
    return [("solve", lambda groups: difarl.fqi(data, groups.groups, gamma, **opts))]


def _train_q(data: Dataset, gamma: float, options: dict, _model=None):
    return offline_q.learn(data, gamma, **options)


def learn(
    data: Dataset, method: Method, limits: Limits | None = None, true_parents: dict | None = None
) -> Result:
    """Learn a model from the data. true_parents is used only by discovery "oracle"."""
    result, x = Result(method), data
    steps = learning_steps(data, method, true_parents)
    if not steps:
        return result  # model-free: nothing before the solver
    for name, step in steps:
        x = _run_step(result, name, step, x, limits)
        if x is None:
            return result
        if isinstance(x, Graph):
            result.graph = x
            result.info.update(x.report)
        if isinstance(x, ActionGroups):
            result.info["groups"] = len(x.groups)
    result.model = x
    return result


def solve(result: Result, data: Dataset, gamma: float, limits: Limits | None = None) -> Result:
    """Solve the learned model: result.solution is the policy and its promise (in place)."""
    if not result.ok:
        return result
    x = result.model
    for name, step in solver_steps(data, result.method, gamma):
        x = _run_step(result, name, step, x, limits)
        if x is None:
            return result
        if isinstance(x, offline_q.QTable):
            result.model = x  # what was learned: the Q-table (kept and saved)
    result.solution = x
    return result


def standard_methods(
    max_parents: int | None = 3,
    delta: float = 0.05,
    exact: bool = True,
    deep: bool = False,
    state_parents: int = 3,
) -> dict[str, Method]:
    """The methods of experiment 1 (used by experiments/exp1/*_complete.py):
    ours (BIC / PCMCI+ / GES -> RMDP), the oracle (Schnitzer et al.: IA, and exact if `exact`, which
    needs 0/1 state variables) and the baselines (flat RMDP, flat MDP, DiFaRL, flat SPIBB, SL-SPIBB
    with candidate parent sets of `state_parents` state variables, FOCUS; with `deep`, also offline
    DQN and offline CQL). Every baseline uses the defaults of its module in baselines/."""
    causal = {"selection": "bic_best", "learning_options": {"delta": delta}}
    oracle = {"discovery": "oracle", "selection": "given", "learning_options": {"delta": delta}}
    methods = {
        "BIC -> RMDP": Method(
            "causal",
            "box_robust_vi",
            discovery="bic",
            discovery_options={"max_parents": max_parents},
            **causal,
        ),
        "PCMCI+ -> RMDP": Method(
            "causal", "box_robust_vi", discovery="pcmci+", discovery_options={"alpha": 0.01}, **causal
        ),
        "GES -> RMDP": Method("causal", "box_robust_vi", discovery="ges", **causal),
        "Schnitzer IA": Method("causal", "box_robust_vi", **oracle),
        "Schnitzer exact": Method("causal", "vertex_robust_vi", **oracle),
        "flat RMDP": Method("flat_rmdp", "robust_vi", learning_options={"delta": delta}),
        "flat MDP": Method("flat_mdp", "vi"),
        "DiFaRL": Method(
            "difarl", "fqi", discovery_options={"p_threshold": 0.005}, solver_options={"max_iter": 30}
        ),
    }
    methods["flat SPIBB"] = Method("spibb", "spibb")
    methods["SL-SPIBB"] = Method("sl_spibb", "spibb", learning_options={"d": state_parents})
    methods["FOCUS"] = Method("focus", "vi")
    if not exact:
        del methods["Schnitzer exact"]
    if deep:
        methods["offline DQN"] = Method("offline_dqn", "greedy_q")
        methods["offline CQL"] = Method("offline_cql", "greedy_q")
    return methods
