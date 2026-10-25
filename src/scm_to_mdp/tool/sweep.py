"""Sweeps: every method on every (problem size, number of rollouts, seed), into results.csv.

An Experiment says how to build the problem and its data, and what to sweep:

    Experiment(
        name="sysadmin",
        make_env=make_env,      # (size, seed) -> an object with .ground_truth() -> MDPGroundTruth
        collect=collect,        # (env, n_rollouts) -> DataFrame of transitions, index (episode, t)
        sizes=[3, 4, 5], n_rollouts=[50, 100, 200], seeds=[0, 1, 2],
        methods={"BIC -> RMDP": Method(...), ...},
    )

For one (size, seed) the data are collected once, with the largest number of rollouts, and every
smaller budget uses the first rollouts of that table: the datasets are nested, so the curves over
the number of rollouts differ only by the amount of data.

Rules:
- A method stopped by a limit (time or memory) at (size, n) is not run at any larger size with at
  least n rollouts, nor at larger n of the same size (row status "skipped").
- Rows are written to results.csv as soon as a (size, seed) finishes; run again and the finished
  (size, seed) are kept (resume). meta.json holds the configuration of the sweep; a resume with a
  different configuration (methods, gamma, limits, the largest number of rollouts, `settings`) is
  refused. Adding sizes, seeds or numbers of rollouts below the largest one is allowed.
- Nothing stops the sweep: a (size, seed) whose setup fails gets rows with status "error" (or
  "too_large" when the true model does not fit in memory), a failed evaluation gets status "error"
  for that method, and a (size, seed) whose worker process died twice gets status "crashed". These
  rows count as finished: delete them from results.csv to run them again.
- Settings (optional): knob and policy label the rows (e.g. a topology and a behaviour policy; see
  tool/grid.py, which makes one Experiment per knob and policy). behaviour(env) -> [S, A], the
  logger as an exact table, adds its coverage and confounding bias to every row (behaviour.py),
  computed for rollouts of `horizon` steps.
- For the first seed, every method's model is saved in out_dir/models/ (tool/store.py: the model,
  the policy and promise, the learned causal Bayesian network, pictures of small problems).
- Every row also holds the selected graph (parents, as JSON) and the true parents over the observed
  columns, so that every edge-level number can be recomputed from results.csv alone.
- The (size, seed) run in worker processes (in parallel when workers > 1; also with one worker, so
  that every step sees the run's fixed hash seed, utils/compute.py); make_env and collect must be
  functions defined at the top level of a module (not lambdas), because they are sent to the
  workers. Every worker may use up to Limits.mem_gb per step: the sweep refuses to start when
  workers x mem_gb is more than 80% of the RAM. workers="auto" takes as many as the machine allows
  (utils/compute.py); SCM_TO_MDP_WORKERS or the script's --workers N override it.
- run_sweeps runs several Experiments (the settings of a grid) in one pool: a free worker takes the
  next job of any setting, while each setting keeps its waves (a size starts when the smaller size
  of that setting is finished, so the limits' stops above still apply).
"""

import dataclasses
import io
import json
import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from scm_to_mdp.data import first_rollouts
from scm_to_mdp.evaluation import COVERAGE_COUNTS, actions_absent, coverage, random_starts
from scm_to_mdp.ground_truth import ModelTooLarge
from scm_to_mdp.tool import store
from scm_to_mdp.tool.evaluate import evaluate, reference_values
from scm_to_mdp.tool.methods import STEPS, STOPPED, Limits, Method, learn, replan, solve

# the columns of results.csv, in order (one row per size, seed, number of rollouts and method)
KEYS = ["experiment", "knob", "policy", "size", "size_label", "n_states", "seed", "n_rollouts", "transitions"]
KEYS += ["method"]
METHOD = ["role", "guarantee", "learner", "discovery", "selection", "solver"]
STATUS = ["status", "failed_step", "error"]
STEP_COLUMNS = [f"{step}_{x}" for step in STEPS for x in ("status", "s", "gb")]
VALUES = ["V_star_obs", "V_star", "V_star_K", "V_random", "V_random_K", "V_oracle_K"]
VALUES += ["promise", "true", "regret", "norm_regret"]
VALUES += ["regret_oracle", "optimal_share", "optimal_share_oracle", "gap"]
VALUES += ["regret_K", "gap_K", "safe", "worst_gap", "violation_rate", "worst_violation"]
BEHAVIOUR = ["V_logger", "C_star", "C_all_bound", "unvisited_pairs", "bias_max", "bias_mean"]
TRUE_MODEL = ["n_observed_states", "true_rows", "true_params", "n_true_edges", "true_parents"]
STRUCTURE = ["shd", "missing", "extra", "sid", "candidates", "dags", "tied", "intra_next", "groups"]
STRUCTURE += ["parents", "action_edges", "action_missing", "action_recall"]
CERTIFICATE = ["V_opt", "cert_gap", "cert_gap_K", "cert_holds"]  # robust methods (evaluation.certificate)
SETS = CERTIFICATE + ["s4_width", "s4_coverage", "s4_unseen", "s4_rows", "widen", "added_parents"]  # S4
# the data's coverage of the best policy's rows (true parents; evaluation.coverage) and of the actions
COVERAGE = [f"cov_{kind}_{k}" for k in COVERAGE_COUNTS for kind in ("joint", "factor")]
COVERAGE += ["cov_alt", "actions_absent", "lcov_factor_1", "lcov_alt"]
# the learned model against the true kernel (evaluation.model_errors)
ERRORS = [f"{m}_{w}" for m in ("tv", "kl", "contain") for w in ("data", "task", "uniform")]
ERRORS += ["contain_all", "width_seen", "width_unseen"]
# Every value a row may hold must be listed here: _write keeps exactly these columns.
COLUMNS = KEYS + METHOD + STATUS + STEP_COLUMNS + VALUES + BEHAVIOUR + TRUE_MODEL + STRUCTURE + SETS
COLUMNS += COVERAGE + ERRORS
# results_transfer.csv (reward transfer): the same, one row per task too
TRANSFER_COLUMNS = KEYS + ["task"] + METHOD + STATUS + STEP_COLUMNS + VALUES + STRUCTURE + SETS
TRANSFER_COLUMNS += COVERAGE + ERRORS
SHIFT = "same reward"  # the reward of a dynamics-shift task (run_point): no re-planning
NO_PROMISE = ["promise", "gap", "gap_K", "safe", "worst_gap", "violation_rate", "worst_violation"]


@dataclass(frozen=True)
class Experiment:
    name: str
    make_env: Callable[[Any, int], Any]
    collect: Callable[[Any, int], pd.DataFrame]
    sizes: list
    n_rollouts: list[int]
    seeds: list[int]
    methods: dict[str, Method]
    gamma: float = 0.9
    n_starts: int = 20  # K start states for regret_K / gap_K, drawn per seed among non-terminal states
    limits: Limits | None = field(default_factory=Limits)
    workers: int | str = 1  # parallel (size, seed) jobs, or "auto" (utils/compute.py); --workers overrides
    out_dir: Path = Path("results")
    size_name: str = "size"  # the name of the size on the figures, e.g. "N"
    save_models: str = "largest"  # first seed: save every model at the largest n ("all": every n; "none")
    pictures_max_states: int = 16  # and draw the MDPs (chain pictures) of problems with at most this many
    # observed states
    # also evaluate every method under the environment's alternative rewards (Env.alternative_rewards):
    # re-planned without re-learning, rows in results_transfer.csv
    reward_transfer: bool = False
    # dynamics shift: (size, seed) -> {name: Env}, the same observed states and reward with other
    # dynamics (e.g. a hidden disturbance more frequent at deployment); every learned policy is
    # evaluated there as it is (no re-learning, no re-planning), rows in results_shift.csv
    shifts: Callable[[Any, int], dict] | None = None
    knob: str = ""  # label of the setting (e.g. "topology=ring"), written in every row
    policy: str = ""  # label of the behaviour policy, written in every row
    behaviour: Callable[[Any], np.ndarray] | None = None  # env -> logger table [S, A] (coverage, bias)
    horizon: int | None = None  # rollout length, for the coverage and bias of `behaviour`
    settings: dict = field(default_factory=dict)  # constants of make_env / collect, checked on resume


def true_model_size(truth) -> dict:
    """Size of the true factored model over the observed columns: parent rows that can occur (the
    number of conditioning contexts) and free parameters, plus the true parents as JSON."""
    from scm_to_mdp.discovery.bic_search import feasible_rows

    spec = truth.learner_spec
    parents = {v: tuple(p for p in truth.true_parents[v] if p in spec.cards) for v in spec.state_vars}
    rows = {v: feasible_rows(ps, spec.cards, spec.action_vars, truth.actions) for v, ps in parents.items()}
    return {
        "n_observed_states": spec.n_states,
        "true_rows": sum(rows.values()),
        "true_params": sum(r * (spec.cards[v] - 1) for v, r in rows.items()),
        "n_true_edges": sum(len(ps) for ps in parents.values()),
        "true_parents": json.dumps(parents),
    }


# ==========================================================================================
# One (size, seed): every number of rollouts and every method
# ==========================================================================================
def _base(exp: Experiment, size, seed: int) -> dict:
    return {
        "experiment": exp.name,
        "knob": exp.knob,
        "policy": exp.policy,
        "size": str(size),
        "size_label": f"{exp.size_name}={size}",
        "seed": seed,
    }


def _rows_without_runs(exp: Experiment, size, seed: int, status: str, step: str, error: str) -> list[dict]:
    """One row per (number of rollouts, method) when nothing could be run for this (size, seed)."""
    base = _base(exp, size, seed)
    return [
        {
            **base,
            "n_rollouts": n,
            "method": name,
            **m.describe(),
            "status": status,
            "failed_step": step,
            "error": error,
        }  # fmt: skip
        for n in sorted(exp.n_rollouts)
        for name, m in exp.methods.items()
    ]


def run_point(exp: Experiment, size, seed: int, stopped_from: dict) -> list[dict]:
    """The rows of one (size, seed). stopped_from: {method: n} methods not to run from n rollouts on.
    Never raises: a failure becomes rows with status "too_large" or "error" (see the module docstring)."""
    import torch

    torch.set_num_threads(1)  # every step runs in its own process with one thread
    base = _base(exp, size, seed)
    try:
        env = exp.make_env(size, seed)
        truth = env.ground_truth()
        table = exp.collect(env, max(exp.n_rollouts))
        truth.check_data(table)
        starts = random_starts(truth, exp.n_starts, seed)
        reference = {"n_states": truth.n_states, **reference_values(truth, exp.gamma, starts)}
        if truth.true_parents is not None:
            reference |= true_model_size(truth)
        if exp.behaviour is not None:
            from scm_to_mdp.behaviour import behaviour_metrics

            reference |= behaviour_metrics(truth, exp.behaviour(env), exp.horizon, exp.gamma)
        tasks = {}  # name -> (ground truth, learner's reward function, reference values)
        if exp.reward_transfer:
            for task in env.alternative_rewards():
                task_truth, task_reward = env.task(task)
                task_ref = {"n_states": truth.n_states, **reference_values(task_truth, exp.gamma, starts)}
                tasks[task] = (task_truth, task_reward, task_ref)
        for shift, shifted_env in (exp.shifts(size, seed) if exp.shifts is not None else {}).items():
            shift_truth = shifted_env.ground_truth()
            same = shift_truth.learner_spec.state_vars == truth.learner_spec.state_vars
            if not same or shift_truth.learner_spec.n_states != truth.learner_spec.n_states:
                raise ValueError(f"shift {shift}: the observed states differ from the original's")
            shift_ref = {"n_states": truth.n_states, **reference_values(shift_truth, exp.gamma, starts)}
            tasks[shift] = (shift_truth, SHIFT, shift_ref)
    except (ModelTooLarge, MemoryError) as err:  # the true model does not fit: nothing can be evaluated
        return _rows_without_runs(exp, size, seed, "too_large", "setup", str(err) or "MemoryError")
    except Exception as err:  # noqa: BLE001 - reported in the rows, the sweep goes on
        return _rows_without_runs(exp, size, seed, "error", "setup", repr(err))
    stopped, rows, graphs = dict(stopped_from), [], {}
    for n in sorted(exp.n_rollouts):
        df = first_rollouts(table, n)
        data = truth.learner_spec.read(df)
        point = {**base, **reference, "n_rollouts": n, "transitions": len(df)}
        if truth.true_parents is not None:  # what the data cover of the best policy's rows
            point |= coverage(truth, data, exp.gamma, truth.parents_in(data))
        point["actions_absent"] = actions_absent(truth, data)
        shared: dict = {}  # discovery + selection, shared by the methods of this dataset (methods.learn)
        for name, method in exp.methods.items():
            if n >= stopped.get(name, math.inf):
                rows.append({**point, "method": name, **method.describe(), "status": "skipped"})
                for task, (_, task_reward, task_ref) in tasks.items():
                    task_point = {**base, **task_ref, "n_rollouts": n, "transitions": len(df), "task": task}
                    task_point |= {"_file": "shift"} if task_reward is SHIFT else {}
                    rows.append({**task_point, "method": name, **method.describe(), "status": "skipped"})
                continue
            parents = given_graph(method.discovery, truth, env, exp, seed, graphs)
            # shared: one discovery and selection per dataset for the methods that have the same ones
            learned = learn(data, method, exp.limits, parents, seed=seed, shared=shared)
            result = solve(learned, data, exp.gamma, exp.limits, seed=seed)
            graph = {"parents": json.dumps(result.graph.parents)} if result.graph is not None else {}
            try:
                metrics = evaluate(result, data, truth, exp.gamma, starts)
            except Exception as err:  # noqa: BLE001 - reported in the row, the sweep goes on
                metrics = {"status": "error", "failed_step": "evaluate", "error": repr(err)}
            rows.append({**point, "method": name, **result.row(), **metrics, **graph})
            for task, (task_truth, task_reward, task_ref) in tasks.items():
                context = (exp, base, task_ref, n, len(df), task, name, result, data)
                rows.append(_transfer_row(*context, task_truth, task_reward, graph, starts, seed))
            keep = exp.save_models == "all" or (exp.save_models == "largest" and n == max(exp.n_rollouts))
            if seed == exp.seeds[0] and keep:
                _save_model(exp, base, n, name, result, data, truth, rows[-1])
            if result.status in STOPPED:
                stopped[name] = n
            if result.status != "ok":
                detail = f": {result.error}" if result.error else ""
                tqdm.write(
                    f"   [{exp.name}] {base['size_label']} seed={seed} n={n} {name}: "
                    f"{result.failed_step} -> {result.status}{detail}"
                )
        first = seed == exp.seeds[0] and n == max(exp.n_rollouts) and exp.save_models != "none"
        folder = exp.out_dir / "models" / base["size_label"]
        title = f"{exp.name} {base['size_label']}: true MDP"
        if first and len(data.B) <= exp.pictures_max_states:
            _guarded(folder, store.draw_ground_truth, truth, data, title, folder / "ground_truth")
    return rows


def given_graph(discovery, truth, env, exp: Experiment, seed: int, cache: dict) -> dict | None:
    """The graph of a given-graph method: the true parents (oracle) or a control graph built from the
    truth (discovery/controls.py), once per (size, seed); None for a discovered graph."""
    from scm_to_mdp.discovery import controls

    if discovery == "oracle":
        return truth.true_parents
    if discovery not in controls.CONTROLS:
        return None
    if discovery not in cache:
        if discovery == "logger_markov":
            if exp.behaviour is None:
                raise ValueError("the logger_markov control needs the experiment's behaviour policy")
            cache[discovery] = controls.logger_markov(truth, exp.behaviour(env), exp.horizon or 50)
        else:
            cache[discovery] = getattr(controls, discovery)(truth, seed)
    return cache[discovery]


def _transfer_row(exp, base, task_ref, n, transitions, task, name, result, data, truth, reward, graph,
                  starts, seed) -> dict:  # fmt: skip
    """One row of results_transfer.csv: result re-planned under the task's reward (methods.replan:
    model-based methods re-solve their learned model, model-free ones keep their policy and have no
    promise), evaluated exactly on the same true transitions with that reward."""
    point = {**base, **task_ref, "n_rollouts": n, "transitions": transitions, "task": task, "method": name}
    task_data = data if reward is SHIFT else dataclasses.replace(data, reward=reward(data.B, data.actions))
    if truth.true_parents is not None:  # the task's best policy and what the data cover of its rows
        point |= coverage(truth, task_data, exp.gamma, truth.parents_in(task_data))
    point["actions_absent"] = actions_absent(truth, task_data)
    if reward is SHIFT:  # other dynamics, same reward: the policy as it is, evaluated on the shifted truth
        try:
            metrics = evaluate(result, data, truth, exp.gamma, starts)
        except Exception as err:  # noqa: BLE001 - reported in the row, the sweep goes on
            metrics = {"status": "error", "failed_step": "evaluate", "error": repr(err)}
        return {**point, "_file": "shift", **result.row(), **metrics, **graph}
    replanned = replan(result, task_data, exp.gamma, exp.limits, seed=seed)
    try:
        metrics = evaluate(replanned, task_data, truth, exp.gamma, starts)
    except Exception as err:  # noqa: BLE001 - reported in the row, the sweep goes on
        metrics = {"status": "error", "failed_step": "evaluate", "error": repr(err)}
    if replanned.solution is not None and np.isnan(replanned.solution.V).all():  # model-free: no promise
        metrics = {**metrics, **dict.fromkeys(NO_PROMISE, np.nan)}
    return {**point, **replanned.row(), **metrics, **graph}


def _guarded(where, fn, *args):
    """Saving a model or a picture must never stop a sweep."""
    try:
        fn(*args)
    except Exception as err:  # noqa: BLE001 - reported, the sweep goes on
        tqdm.write(f"   could not save {where}: {err!r}")


def _save_model(exp: Experiment, base: dict, n: int, name: str, result, data, truth, row: dict):
    folder = exp.out_dir / "models" / base["size_label"] / f"n={n}" / store.method_folder(name)
    info = {**base, "n_rollouts": n, "method": name, "status": row.get("status"),
            "promise": row.get("promise"), "true": row.get("true"), "gamma": exp.gamma}  # fmt: skip
    _guarded(folder, store.save, folder, result, data, truth, info, exp.pictures_max_states)


# ==========================================================================================
# The whole sweep
# ==========================================================================================
def _config(exp: Experiment) -> dict:
    """What must not change between a run and its resume, plus the grid."""
    return (
        {
            "name": exp.name,
            "gamma": exp.gamma,
            "n_starts": exp.n_starts,
            "limits": None if exp.limits is None else asdict(exp.limits),
            "max_rollouts": max(exp.n_rollouts),  # the data table: smaller budgets are its first rollouts
            "horizon": exp.horizon,
            "settings": exp.settings,
            "methods": {
                name: asdict(m) | {"role": m.role, "guarantee": m.guarantee}
                for name, m in exp.methods.items()
            },
        }
        | ({"reward_transfer": True} if exp.reward_transfer else {})
        | (  # only when on: older runs resume
            {"shifts": True} if exp.shifts is not None else {}
        )
    )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _check_memory(exp: Experiment, workers: int):
    if exp.limits is None or workers == 1:
        return
    ram = psutil.virtual_memory().total / 1e9
    need = workers * exp.limits.mem_gb
    if need > 0.8 * ram:
        raise ValueError(
            f"{workers} workers x {exp.limits.mem_gb:g} GB per step = {need:g} GB, more than 80% of the "
            f"RAM ({ram:.0f} GB): use fewer workers (--workers) or a lower Limits.mem_gb"
        )


LABELS = ("size", "knob", "policy")  # read back as text, exactly as written ("0.0" stays "0.0")


def read_results(out_dir: Path, current: bool = False, name: str = "results.csv") -> pd.DataFrame:
    """The rows of results.csv (or results_transfer.csv: name). current: only the sizes, seeds and
    numbers of rollouts of the last run's configuration (meta.json), e.g. after a seed was removed."""
    path = Path(out_dir) / name
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS if name == "results.csv" else TRANSFER_COLUMNS)
    text = path.read_text()
    if text and not text.endswith("\n"):  # a run killed while appending: drop the cut-off last line
        text = text[: text.rfind("\n") + 1]
    header = pd.read_csv(io.StringIO(text), nrows=0).columns
    na = {c: ([] if c in LABELS else ["", "nan", "NaN"]) for c in header}
    res = pd.read_csv(io.StringIO(text), dtype={c: str for c in LABELS}, keep_default_na=False, na_values=na)
    for column in ("knob", "policy"):  # a plain Experiment has empty labels
        if column not in res:
            res[column] = ""
    meta = Path(out_dir) / "meta.json"
    if current and meta.exists() and len(res):
        m = json.loads(meta.read_text())
        keep = (
            res["size"].isin(m["sizes"])
            & res["seed"].isin(m["seeds"])
            & res["n_rollouts"].isin(m["n_rollouts"])
        )
        res = res[keep].reset_index(drop=True)
    return res


def _write(path: Path, rows: list[dict], append: bool, columns: list[str] = COLUMNS):
    """Append rows in one write; a full rewrite goes through a temporary file (never half-written)."""
    text = pd.DataFrame(rows).reindex(columns=columns).to_csv(header=not append, index=False)
    if append:
        with open(path, "a") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
    else:
        _write_text(path, text)


def _write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class _Sweep:
    """The state of one Experiment while it runs: its folder, what is finished, the limits' stops,
    and the waves of (size, seed) jobs still to run (a wave = one size; see the module docstring)."""

    def __init__(self, exp: Experiment, resume: bool, machine: dict | None = None):
        self.exp = exp
        out = Path(exp.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.csv, meta_path = out / "results.csv", out / "meta.json"
        config = _config(exp)
        old = read_results(out) if resume else pd.DataFrame(columns=COLUMNS)
        if resume and meta_path.exists() and len(old):
            before = json.loads(meta_path.read_text())["config"]
            if before != json.loads(json.dumps(config)):
                raise ValueError(
                    f"{out} holds results of a different configuration: use another out_dir or resume=False"
                )
        # a (size, seed) is finished when it has a row for every number of rollouts and method
        need = {(n, m) for n in exp.n_rollouts for m in exp.methods}
        self.finished = {
            key
            for key, g in old.groupby(["size", "seed"])
            if need <= set(zip(g["n_rollouts"], g["method"], strict=True))
        }
        old = old[[(s, d) in self.finished for s, d in zip(old["size"], old["seed"], strict=True)]]
        _write(self.csv, old.to_dict("records"), append=False)
        self.transfer_csv = out / "results_transfer.csv"
        if exp.reward_transfer:
            old_t = read_results(out, name="results_transfer.csv") if resume else pd.DataFrame()
            if len(old_t):
                old_t = old_t[
                    [(s, d) in self.finished for s, d in zip(old_t["size"], old_t["seed"], strict=True)]
                ]
            _write(self.transfer_csv, old_t.to_dict("records"), append=False, columns=TRANSFER_COLUMNS)
        self.shift_csv = out / "results_shift.csv"
        if exp.shifts is not None:
            old_s = read_results(out, name="results_shift.csv") if resume else pd.DataFrame()
            if len(old_s):
                old_s = old_s[
                    [(s, d) in self.finished for s, d in zip(old_s["size"], old_s["seed"], strict=True)]
                ]
            _write(self.shift_csv, old_s.to_dict("records"), append=False, columns=TRANSFER_COLUMNS)
        meta = {
            "config": config,
            "sizes": [str(s) for s in exp.sizes],
            "size_name": exp.size_name,
            "knob": exp.knob,
            "policy": exp.policy,
            "size_labels": [f"{exp.size_name}={s}" for s in exp.sizes],
            "n_rollouts": sorted(exp.n_rollouts),
            "seeds": list(exp.seeds),
            "git_commit": _git_commit(),
            "started": datetime.now().isoformat(timespec="seconds"),
            "machine": machine,  # how the run used the machine (utils/compute.py); not checked on resume
        }
        _write_text(meta_path, json.dumps(meta, indent=2, default=str))
        # method -> smallest n at which a limit stopped it, per size. A (size, seed) gets the stops of
        # the smaller sizes only (also on a resume, where rows of the same size may already exist).
        self.stops: dict[str, dict[str, float]] = {}
        self.note(old.to_dict("records"))
        self.sizes = list(exp.sizes)  # sizes whose wave has not started
        self.running = 0  # jobs of the current wave not finished

    @property
    def todo(self) -> int:
        return sum(
            (str(size), seed) not in self.finished for size in self.exp.sizes for seed in self.exp.seeds
        )

    def note(self, rows):
        for r in rows:
            if r.get("status") in STOPPED:
                at = self.stops.setdefault(str(r["size"]), {})
                at[r["method"]] = min(at.get(r["method"], math.inf), r["n_rollouts"])

    def stops_before(self, size) -> dict:
        out: dict[str, float] = {}
        for smaller in self.exp.sizes[: [str(x) for x in self.exp.sizes].index(str(size))]:
            for method, n in self.stops.get(str(smaller), {}).items():
                out[method] = min(out.get(method, math.inf), n)
        return out

    def next_wave(self) -> list[tuple]:
        """The jobs (exp, size, seed, stops) of the next size with unfinished seeds; [] when done."""
        while self.sizes:
            size = self.sizes.pop(0)
            before = self.stops_before(size)
            jobs = [
                (self.exp, size, seed, dict(before))
                for seed in self.exp.seeds
                if (str(size), seed) not in self.finished
            ]
            if jobs:
                self.running = len(jobs)
                return jobs
        return []

    def record(self, job: tuple, rows) -> list[tuple]:
        """Write the rows of a finished job; returns the next wave when this one is complete."""
        if isinstance(rows, dict):  # the worker died twice (e.g. killed by the OS) or raised
            error = rows.get("error", "the worker process died twice (killed by the operating system?)")
            rows = _rows_without_runs(self.exp, job[1], job[2], "crashed", "worker", error)
        main = [r for r in rows if "task" not in r]
        _write(self.csv, main, append=True)
        transfer = [r for r in rows if "task" in r and r.get("_file") != "shift"]
        if transfer:
            _write(self.transfer_csv, transfer, append=True, columns=TRANSFER_COLUMNS)
        shifted = [r for r in rows if r.get("_file") == "shift"]
        if shifted:
            _write(self.shift_csv, shifted, append=True, columns=TRANSFER_COLUMNS)
        self.note(main)
        self.finished.add((str(job[1]), job[2]))
        self.running -= 1
        return self.next_wave() if self.running == 0 else []


def plan_for(exps: list[Experiment], n_jobs: int):
    """The machine plan of a run (utils/compute.py): the script's workers ("auto" or a number),
    overridden by SCM_TO_MDP_WORKERS / --workers; refuses workers x mem_gb above 80% of the RAM."""
    from scm_to_mdp.utils import compute

    limits = [e.limits for e in exps if e.limits is not None]
    mem_gb = max((lim.mem_gb for lim in limits), default=None)
    spec = os.environ.get(compute.ENV_WORKERS) or exps[0].workers
    p = compute.plan(mem_gb, workers=spec, n_jobs=n_jobs)
    for e in exps:
        _check_memory(e, p.workers)
    compute.apply(p)
    return p


def run_sweeps(exps: list[Experiment], resume: bool = True, title: str | None = None) -> None:
    """Run several Experiments in one pool of workers: a worker that frees up takes a job of any
    setting, so a large machine stays busy (each setting still runs its sizes in waves)."""
    from scm_to_mdp.utils import compute

    todo = sum(len(e.sizes) * len(e.seeds) for e in exps)  # an upper bound before the folders are read
    p = plan_for(exps, todo)
    print(p.describe())
    sweeps = [_Sweep(e, resume, compute.as_record(p)) for e in exps]
    total = sum(s.todo for s in sweeps)
    name = title or (exps[0].name if len(exps) == 1 else f"{len(exps)} settings")
    bar = tqdm(total=total, desc=f"{name} ({p.workers} workers)", unit="run", dynamic_ncols=True)
    first = [job for s in sweeps for job in s.next_wave()]
    owner = {id(s.exp): s for s in sweeps}
    # also with one worker: the steps then run in a fresh process with the run's hash seed
    # (utils/compute.py, HASH_SEED), not in this one, whose string hashing was fixed when it started
    from scm_to_mdp.utils.parallel import run_dynamic

    def then(job, rows):
        return owner[id(job[0])].record(job, rows)

    for _ in run_dynamic(run_point, first, p.workers, then=then, bar=bar):
        pass
    bar.close()


def run_experiment(exp: Experiment, resume: bool = True) -> pd.DataFrame:
    """Run the sweep (resuming from results.csv when resume) and return every row."""
    run_sweeps([exp], resume=resume)
    return read_results(Path(exp.out_dir))
