"""Part 1 of the tool: learn a model from data (S2-S7), then solve it (S8).

A Method names one choice per step. The learner is one of:

    "causal"     S2 discovery -> S3 selection -> S4 parameter learning -> a factored interval MDP
                 discovery: "bic" | "pcmci+" | "ges" | "dynotears" | "oracle" (the true graph, given)
                            | "full" (the full factored graph: every slice-t input a parent of every
                            x_i', no same-step edge; discovery/full_factored.py)
                            | a control graph (discovery/controls.py: "logger_markov", "random_matched",
                            "plus_one"; built from the truth by the sweep and given like the oracle's)
                 selection: "bic_best" | "given" (for the oracle and the controls)
    "flat_rmdp"  baseline: one interval per (s, a, s'), no structure      -> an interval MDP
    "flat_mdp"   baseline: the plug-in MDP from counts, no guarantee      -> an exact MDP (L == U)
    "difarl"     baseline: DiFaRL's action groups, model-free, no guarantee -> action groups
    "offline_dqn", "offline_cql"
                 baselines: offline deep Q-learning on the logged data (baselines/offline_q.py),
                 model-free, no guarantee; nothing to learn before the solver, which trains
    "spibb", "sl_spibb"
                 baselines: safe policy improvement over the (estimated) behaviour policy, flat
                 (Laroche et al.) or with structure learning (Simao & Spaan; baselines/spibb.py)
    "factored_spibb"
                 reference: Factored SPIBB (Simao & Spaan, AAAI 2019), the factored model on a GIVEN
                 graph (discovery "oracle", selection "given": the true graph), solved like SPIBB
    "focus"      baseline: FOCUS's CI-test structure + a MOPO-penalised factored model (discrete
                 version, baselines/focus.py) -> an exact MDP with a penalised reward
    "oracle"     not a learner: the true observed MDP (baselines/true_model.py), exact (L == U, for
                 "vi") or within +-tol (for "robust_vi"); the sweep gives it the true kernel

and the solver one of (which ones fit which learner: SOLVERS_FOR):

    "box_robust_vi"     relax the factored set to its box (Schnitzer et al.'s IA), then robust VI
    "vertex_robust_vi"  exact robust VI over the factored set (0/1 state variables only)
    "robust_vi"         robust VI on an interval MDP (solver_options nature="cooperative": the optimistic
                        mirror, an upper bound and no guarantee; the default "adversarial" is the promise)
    "vi"                value iteration on an exact MDP
    "fqi"               DiFaRL's decomposed fitted Q iteration (uses the data again)
    "greedy_q"          train the Q-network on the data (step "training"), then its greedy policy
    "spibb"             Pi_b-SPIBB policy iteration (a stochastic policy)

Every step runs either in this process (limits=None) or in its own process with a memory and time
limit (Limits). A step that fails stops the method: Result.status says how ("memout", "timeout",
"error") and Result.failed_step where.
"""

import dataclasses
import inspect
import math
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from scm_to_mdp.baselines import difarl, flat_mdp, flat_rmdp, focus, offline_q, spibb, true_model
from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.core import ActionGroups, Dataset, FactorBounds, Graph, IntervalMDP, Reward, Solution
from scm_to_mdp.scm_learning import nbn_intervals, nbn_point
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
    "factored_spibb",
    "oracle",
)
BASELINES = LEARNERS[1:-1]  # every learner but ours ("causal") and the oracle
OFFLINE_Q = ("offline_dqn", "offline_cql")
POINT_ALPHA = 1.0  # X -> MDP: Laplace (Dirichlet(1) posterior mean; scm_learning/nbn_point.py)
WIDEN_K = 2  # BIC -> IMDP (hull / padded): inputs added to every family (scm_learning/nbn_intervals.py)
CQL_ALPHA = 0.1  # offline CQL's penalty weight (baselines/offline_q.py says how it was chosen)
GIVEN_GRAPHS = ("oracle", "logger_markov", "random_matched", "plus_one")  # graphs given, not discovered
DISCOVERY = ("bic", "pcmci+", "ges", "dynotears", "full") + GIVEN_GRAPHS
SELECTION = {"bic_best": bic_best.select, "given": given.select}
SOLVERS_FOR = {
    "causal": ("box_robust_vi", "vertex_robust_vi", "vi"),
    "flat_rmdp": ("robust_vi",),
    "flat_mdp": ("vi", "robust_vi"),
    "difarl": ("fqi",),
    "offline_dqn": ("greedy_q",),
    "offline_cql": ("greedy_q",),
    "spibb": ("spibb",),
    "sl_spibb": ("spibb",),
    "focus": ("vi",),
    "factored_spibb": ("spibb",),
    "oracle": ("vi", "robust_vi"),
}
ROBUST_SOLVERS = ("box_robust_vi", "vertex_robust_vi", "robust_vi")
STEPS = ("discovery", "selection", "learning", "relax", "training", "solve")  # every step a method can have
STOPPED = ("timeout", "memout")
# Method names before 2026-10-26: the robust arms are interval MDPs, so "-> RMDP" became "-> IMDP" (and the
# per-(s,a) product reference a factored IMDP). sweep.read_results renames older rows on reading.
RENAMED = {
    "flat RMDP": "flat IMDP",
    "true graph -> RMDP (per-(s,a) product)": "true graph -> factored IMDP (per-(s,a) product)",
    **{
        f"{x} -> RMDP{tail}": f"{x} -> IMDP{tail}"
        for x in (
            "BIC",
            "PCMCI+",
            "GES",
            "DYNOTEARS",
            "true graph",
            "oracle",
            "logger-Markov graph",
            "random matched graph",
            "true graph + 1",
        )  # fmt: skip
        for tail in ("", " (hull)", " (padded)")
    },
}


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
    if name == "dynotears":
        from scm_to_mdp.discovery import dynotears

        return dynotears.fit
    if name == "full":
        from scm_to_mdp.discovery import full_factored

        return full_factored.fit
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
            if (self.discovery in GIVEN_GRAPHS + ("full",)) != (self.selection == "given"):
                raise ValueError(
                    f"selection 'given' goes with the given graphs {GIVEN_GRAPHS} and 'full', and only"
                    " with them"
                )
            if self.discovery not in GIVEN_GRAPHS:
                _check_options(discovery_function(self.discovery), self.discovery_options, 1, "discovery")
            _check_options(causal_learner(self).learn, self.learning_options, 2, "learning")
        elif self.learner == "factored_spibb":  # its graph is given (Simao & Spaan, AAAI 2019)
            if (self.discovery, self.selection) != ("oracle", "given") or self.discovery_options:
                raise ValueError("factored_spibb takes the true graph: discovery 'oracle', selection 'given'")
            _check_options(spibb.factored_model, self.learning_options, 2, "learning (Factored SPIBB)")
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
                "oracle": true_model.model,
            }.get(self.learner)
            if self.learner == "focus":
                learn_fn = focus.penalised_mdp
                _check_options(learn_fn, self.learning_options, 2, "learning (FOCUS)")
                learn_fn = None
            if learn_fn is not None:
                n_inputs = 2 if self.learner == "oracle" else 1
                _check_options(learn_fn, self.learning_options, n_inputs, "learning")
            elif self.learning_options and self.learner != "focus":
                raise ValueError(f"{self.learner} takes no learning options (its options are solver options)")
            exact = self.learning_options.get("tol", 0) == 0
            if self.learner == "oracle" and (self.solver == "vi") != exact:
                raise ValueError(
                    "the oracle is solved with 'vi' when tol = 0 and with 'robust_vi' when tol > 0"
                )
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
        """ "baseline" (no causal structure learned), "reference" (given the true graph: true graph ->
        IMDP, true graph -> MDP), "control" (given a wrong graph of a known kind: discovery/controls.py)
        or "ours"; "oracle": the true observed MDP given to the solver. The full factored graph (no
        discovery: full-fact -> IMDP / MDP) is a baseline."""
        if self.learner == "oracle":
            return "oracle"
        if self.learner == "factored_spibb":  # a baseline method, but given the true graph
            return "reference"
        if self.learner in BASELINES or self.discovery == "full":
            return "baseline"
        if self.discovery == "oracle":
            return "reference"
        return "control" if self.discovery in GIVEN_GRAPHS else "ours"

    @property
    def guarantee(self) -> bool:
        """Whether the method's promise is meant as a lower bound on the true value (for "causal":
        when the selected graph contains the true parents)."""
        adversarial = self.solver_options.get("nature", "adversarial") == "adversarial"  # robust_vi
        return (
            self.solver in ROBUST_SOLVERS
            and self.learner in ("causal", "flat_rmdp", "oracle")
            and adversarial
        )

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
    timeout_s: dict = field(
        default_factory=lambda: {"discovery": 60.0, "selection": 60.0, "learning": 60.0, "solve": 60.0}
    )


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
def causal_learner(method: Method):
    """S4 of the causal learner: interval sets (nbn_intervals) for the robust solvers, point estimates
    (nbn_point, no guarantee) for value iteration ("X -> MDP")."""
    return nbn_point if method.solver == "vi" else nbn_intervals


def learning_steps(
    data: Dataset, method: Method, true_parents: dict | None = None, seed: int = 0, kernel=None
) -> list:
    """The steps of method's learner as (name, function of the previous output). seed: the run's seed,
    for the learners that draw random numbers (FOCUS's bootstrap ensemble) unless an option sets it.
    kernel: the oracle's true observed kernel on the data's action vectors [S, K, S] (tool/sweep.py)."""
    m = method
    if m.learner == "oracle":
        if kernel is None:
            raise ValueError("the oracle needs the true observed kernel (tool/sweep.py gives it)")
        return [("learning", lambda d: true_model.model(kernel, d, **m.learning_options))]
    if m.learner == "causal":
        if m.discovery in GIVEN_GRAPHS:
            if true_parents is None:
                raise ValueError(f"discovery {m.discovery!r} needs its graph as true_parents (tool/sweep.py)")
            discover = discovery_function("oracle")
            first = ("discovery", lambda d: discover(d, true_parents))
        else:
            discover = discovery_function(m.discovery)
            first = ("discovery", lambda d: discover(d, **m.discovery_options))
        select = SELECTION[m.selection]
        return [
            first,
            ("selection", lambda candidates: select(candidates, data)),
            ("learning", lambda graph: causal_learner(m).learn(data, graph, **m.learning_options)),
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
    if m.learner == "factored_spibb":
        if true_parents is None:
            raise ValueError("factored_spibb needs the true graph as true_parents (tool/sweep.py)")
        discover = discovery_function("oracle")
        return [
            ("discovery", lambda d: discover(d, true_parents)),
            ("selection", lambda candidates: given.select(candidates, data)),
            ("learning", lambda graph: spibb.factored_model(data, graph.parents, **m.learning_options)),
        ]
    if m.learner == "focus":
        options = {"seed": seed, **m.learning_options}
        return [
            ("discovery", focus.discover),
            ("learning", lambda graph: focus.penalised_mdp(data, graph, **options)),
        ]
    return [("discovery", lambda d: ActionGroups(difarl.groups(d, **m.discovery_options)))]


def solver_steps(data: Dataset, method: Method, gamma: float, seed: int = 0) -> list:
    """The steps of method's solver as (name, function of the learned model). seed: the run's seed,
    for the solvers that draw random numbers (offline DQN / CQL) unless an option sets it."""
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

        compile_first = [("relax", product_bounds.exact)] if method.learner == "causal" else []
        return compile_first + [("solve", vi)]  # causal: the factor estimates' product, an exact MDP
    if method.solver == "spibb":
        return [("solve", lambda model: spibb.solve(model, gamma, **opts))]
    if method.solver == "greedy_q":
        alpha = {"cql_alpha": CQL_ALPHA} if method.learner == "offline_cql" else {}
        return [
            # partial: picklable, the training runs in a new process
            ("training", partial(_train_q, data, gamma, {"seed": seed} | alpha | opts)),
            ("solve", offline_q.greedy),
        ]
    return [("solve", lambda groups: difarl.fqi(data, groups.groups, gamma, **opts))]


def _train_q(data: Dataset, gamma: float, options: dict, _model=None):
    return offline_q.learn(data, gamma, **options)


SHARED_STEPS = ("discovery", "selection")


def _shared_key(method: Method):
    """Methods with the same discovery and selection (on the same data) share their graph."""
    if method.learner != "causal":
        return None
    return (method.discovery, tuple(sorted(method.discovery_options.items())), method.selection)


def learn(
    data: Dataset,
    method: Method,
    limits: Limits | None = None,
    true_parents: dict | None = None,
    seed: int = 0,
    shared: dict | None = None,
    kernel=None,
) -> Result:
    """Learn a model from the data. true_parents: the graph of a given-graph discovery (the oracle's true
    parents, or a control graph); seed (the run's
    seed) only by learners that draw random numbers. shared: a dict kept across the methods of one
    dataset (tool/sweep.py): the first causal method with a given discovery and selection runs them and
    stores the graph (or how they stopped) and their step records; the others reuse it, so e.g.
    BIC -> IMDP, BIC -> IMDP (hull) and BIC -> MDP run BIC once and report the same discovery time.
    kernel: the oracle's true observed kernel (learning_steps)."""
    result, x = Result(method), data
    steps = learning_steps(data, method, true_parents, seed, kernel)
    if not steps:
        return result  # model-free: nothing before the solver
    key = _shared_key(method) if shared is not None else None
    if key is not None and key in shared:
        entry = shared[key]
        result.steps.update({name: dict(record) for name, record in entry["steps"].items()})
        result.info.update(entry["info"])
        if entry["graph"] is None:  # discovery or selection stopped: so does this method
            result.status, result.failed_step, result.error = (
                entry["status"],
                entry["failed_step"],
                entry["error"],
            )
            return result
        result.graph = x = entry["graph"]
        steps = [(name, step) for name, step in steps if name not in SHARED_STEPS]
    for name, step in steps:
        x = _run_step(result, name, step, x, limits)
        if (
            key is not None
            and key not in shared
            and name in SHARED_STEPS
            and (x is None or name == "selection")
        ):
            graph = x if isinstance(x, Graph) else None
            shared[key] = {
                "steps": {s: dict(result.steps[s]) for s in SHARED_STEPS if s in result.steps},
                "info": dict(result.info) | (dict(graph.report) if graph is not None else {}),
                "graph": graph,
                "status": result.status,
                "failed_step": result.failed_step,
                "error": result.error,
            }
        if x is None:
            return result
        if isinstance(x, Graph):
            result.graph = x
            result.info.update(x.report)
        if isinstance(x, ActionGroups):
            result.info["groups"] = len(x.groups)
        if isinstance(x, FactorBounds):
            result.info.update(x.info)
    result.model = x
    return result


def solve(result: Result, data: Dataset, gamma: float, limits: Limits | None = None, seed: int = 0) -> Result:
    """Solve the learned model: result.solution is the policy and its promise (in place). seed (the
    run's seed) is used only by solvers that draw random numbers."""
    if not result.ok:
        return result
    x = result.model
    for name, step in solver_steps(data, result.method, gamma, seed):
        x = _run_step(result, name, step, x, limits)
        if x is None:
            return result
        if isinstance(x, offline_q.QTable):
            result.model = x  # what was learned: the Q-table (kept and saved)
    result.solution = x
    if getattr(x, "iterations", None) is not None:  # value iteration: its sweeps and their mean time
        result.info |= {"solve_iterations": x.iterations, "solve_iteration_s": x.iteration_s}
    return result


# ---- another reward on the same model (reward transfer, S9) -----------------------------------------
MODEL_FREE = ("offline_dqn", "offline_cql", "difarl")  # no model to re-plan: their policy is kept


def with_reward(model, reward: Reward):
    """The learned model with another reward (the dynamics, sets or estimates untouched). FOCUS's
    uncertainty penalty is re-applied on the new reward's scale."""
    if isinstance(model, IntervalMDP) and "penalty" in model.info:
        step = reward.R[:, :, None] + reward.r_next[None, None, :]
        scale = float(step.max() - step.min()) or 1.0
        reward = Reward(reward.R - scale * model.info["penalty"], reward.r_next, reward.terminal)
    return dataclasses.replace(model, reward=reward)


def replan(
    result: Result, data: Dataset, gamma: float, limits: Limits | None = None, seed: int = 0
) -> Result:
    """result under the reward carried by data (the same transitions, another reward), without
    re-learning: a model-based method re-solves its learned model; a model-free one (MODEL_FREE)
    keeps its policy and has no promise for this reward (V = NaN)."""
    if not result.ok or result.solution is None:
        return result
    if result.method.learner in MODEL_FREE:
        out = Result(result.method, graph=result.graph, info=dict(result.info), model=result.model)
        old = result.solution
        out.solution = Solution(np.full_like(old.V, np.nan, dtype=float), old.pi, old.probs)
        return out
    out = Result(
        result.method,
        graph=result.graph,
        info=dict(result.info),
        model=with_reward(result.model, data.reward),
    )
    return solve(out, data, gamma, limits, seed)


def oracle_methods(tol: float = true_model.TOL) -> dict[str, Method]:
    """The oracle of every grid: the true observed MDP, exact (-> MDP, value iteration) and within
    +-tol (-> RMDP, robust value iteration; baselines/true_model.py)."""
    return {
        "oracle -> IMDP": Method("oracle", "robust_vi", learning_options={"tol": tol}),
        "oracle -> MDP": Method("oracle", "vi"),
    }


def standard_methods(
    max_parents: int | None = 3,
    delta: float = 0.05,
    exact: bool = True,
    deep: bool = False,
    state_parents: int = 3,
    spibb_options: dict | None = None,
    sl_spibb_options: dict | None = None,
    factored_spibb_options: dict | None = None,
) -> dict[str, Method]:
    """Every named method (experiments/exp1 and exp2 pick theirs in their protocol.py):
    ours (BIC / PCMCI+ / GES / DYNOTEARS -> IMDP; BIC -> IMDP with hull or padded families; the same
    four discoveries -> MDP, point estimates and value iteration, with true graph -> MDP), the references
    (true graph -> IMDP with the box relaxation of our robust arms; with `exact`, for 0/1 state
    variables, also true graph -> factored IMDP (per-(s,a) product): robust VI over the products of the factor
    intervals, Schnitzer et al.'s exact solver, which is exact per (s, a) but still lets nature choose
    every (s, a) independently; the pair measures what the box costs), the controls (given wrong
    graphs of a known kind, discovery/controls.py: logger-Markov graph, random matched graph, true
    graph + 1, each -> IMDP with the box), the full factored graph (full-fact -> IMDP / MDP: every
    slice-t input a parent of every next-state variable, no discovery) and the baselines (flat IMDP,
    flat MDP, DiFaRL, flat SPIBB, SL-SPIBB with candidate parent sets of `state_parents` state
    variables, FOCUS; with `deep`, also offline DQN and offline CQL) and Factored SPIBB on the true
    graph (a reference: its structure is given, Simao & Spaan, AAAI 2019). Every baseline uses the
    defaults of its module in baselines/."""
    causal = {"selection": "bic_best", "learning_options": {"delta": delta}}
    oracle = {"discovery": "oracle", "selection": "given", "learning_options": {"delta": delta}}
    methods = {
        "BIC -> IMDP": Method(
            "causal",
            "box_robust_vi",
            discovery="bic",
            discovery_options={"max_parents": max_parents},
            **causal,
        ),
        "PCMCI+ -> IMDP": Method(
            "causal", "box_robust_vi", discovery="pcmci+", discovery_options={"alpha": 0.01}, **causal
        ),
        "GES -> IMDP": Method("causal", "box_robust_vi", discovery="ges", **causal),
        "DYNOTEARS -> IMDP": Method("causal", "box_robust_vi", discovery="dynotears", **causal),
        # BIC's graph, with each family widened by the WIDEN_K inputs that BIC ranks next
        # (scm_learning/nbn_intervals.py): the hull over the families, or one padded family
        **{
            f"BIC -> IMDP ({label})": Method(
                "causal",
                "box_robust_vi",
                discovery="bic",
                discovery_options={"max_parents": max_parents},
                selection="bic_best",
                learning_options={"delta": delta, "widen": widen, "k": WIDEN_K},
            )
            for label, widen in (("hull", "hull"), ("padded", "pad"))
        },
        # the references: the true graph given, with the exact product set (Schnitzer et al.'s exact
        # solver, 0/1 variables; else the box relaxation, as our arms) or with point estimates and VI
        "true graph -> IMDP": Method("causal", "box_robust_vi", **oracle),
        # no discovery: the full factored graph, given (discovery/full_factored.py)
        "full-fact -> IMDP": Method("causal", "box_robust_vi", **(oracle | {"discovery": "full"})),
        **(
            {
                "true graph -> factored IMDP (per-(s,a) product)": Method(
                    "causal", "vertex_robust_vi", **oracle
                )
            }
            if exact
            else {}
        ),
        # the controls: wrong graphs of a known kind, learned and solved as the true graph
        **{
            label: Method("causal", "box_robust_vi", **(oracle | {"discovery": control}))
            for label, control in (
                ("logger-Markov graph -> IMDP", "logger_markov"),
                ("random matched graph -> IMDP", "random_matched"),
                ("true graph + 1 -> IMDP", "plus_one"),
            )
        },
        # certainty equivalence: the same discovery and selection, point estimates (nbn_point,
        # Laplace), value iteration; no guarantee. "true graph -> MDP" is their reference.
        "BIC -> MDP": Method(
            "causal",
            "vi",
            discovery="bic",
            discovery_options={"max_parents": max_parents},
            selection="bic_best",
            learning_options={"alpha": POINT_ALPHA},
        ),
        "PCMCI+ -> MDP": Method(
            "causal",
            "vi",
            discovery="pcmci+",
            discovery_options={"alpha": 0.01},
            selection="bic_best",
            learning_options={"alpha": POINT_ALPHA},
        ),
        "GES -> MDP": Method(
            "causal", "vi", discovery="ges", selection="bic_best", learning_options={"alpha": POINT_ALPHA}
        ),
        "DYNOTEARS -> MDP": Method(
            "causal",
            "vi",
            discovery="dynotears",
            selection="bic_best",
            learning_options={"alpha": POINT_ALPHA},
        ),
        "true graph -> MDP": Method(
            "causal", "vi", discovery="oracle", selection="given", learning_options={"alpha": POINT_ALPHA}
        ),  # fmt: skip
        "full-fact -> MDP": Method(
            "causal", "vi", discovery="full", selection="given", learning_options={"alpha": POINT_ALPHA}
        ),  # fmt: skip
        "flat IMDP": Method("flat_rmdp", "robust_vi", learning_options={"delta": delta}),
        "flat MDP": Method("flat_mdp", "vi"),
        "DiFaRL": Method(
            "difarl", "fqi", discovery_options={"p_threshold": 0.005}, solver_options={"max_iter": 30}
        ),
    }
    # SPIBB's count threshold m, SL-SPIBB's eps and the baseline policy (baselines/spibb.py)
    methods["flat SPIBB"] = Method("spibb", "spibb", learning_options=dict(spibb_options or {}))
    methods["Factored SPIBB (true graph)"] = Method(
        "factored_spibb", "spibb", discovery="oracle", selection="given",
        learning_options=dict(factored_spibb_options or {}),
    )  # fmt: skip
    methods["SL-SPIBB"] = Method(
        "sl_spibb", "spibb", learning_options={"d": state_parents, **(sl_spibb_options or {})}
    )
    methods["FOCUS"] = Method("focus", "vi")
    if deep:
        methods["offline DQN"] = Method("offline_dqn", "greedy_q")
        methods["offline CQL"] = Method("offline_cql", "greedy_q")
    return methods
