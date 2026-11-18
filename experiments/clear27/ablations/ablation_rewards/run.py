"""Ablation: how robust are the methods to the reward? A ladder of rewards on SysAdmin, Sepsis and Drone
(S8, S9).

Each method learns ONE model per dataset, as in clear27; that model is then re-planned (not re-learned)
under every reward of the ladder (rewards.py: from the environment's own reward to goals in rare states
and traps), and each policy is evaluated on the true model under that reward.

    methods   with a promise: the box interval MDP of PCMCI+ and DYNOTEARS (their lagged arms: no edge
              inside a slice), of the full graph, of the true graph (the reference: what a reward costs
              when the structure is known) and the flat interval MDP;
              without a promise: the plug-in MDP (point estimates, value iteration) of the same five
              structures ("<structure> MDP"), and the two structure learners of factored RL as their
              papers run them (KWIK_PAPER): SLF-Rmax (Strehl, Diuk, Littman, AAAI 2007) and Met-Rmax (Diuk,
              Li, Leffler, ICML 2009), i.e. the learner's own structure and predictions with the papers'
              thresholds, planned by value iteration. Offline, their optimism for "don't know" (it drives
              exploration) is replaced by pessimism: such a pair halts with the worst value
              (scm_learning/kwik_intervals.py). The value a point model gives its policy is recorded in
              `promise`, but it is no bound (CERTIFIED lists the methods whose promise is one);
              also without a promise on the value, the other model-based methods of the paper's
              comparison: SPIBB on a graph ("<structure> SPIBB": Factored SPIBB on the graph of
              DYNOTEARS, of PCMCI+ and on the true one), SL-SPIBB, flat SPIBB and FOCUS. The sparse
              graph is one more structure, in both forms. SLF-Rmax and Met-Rmax also run with their
              theorems' thresholds ("(theorem)"), and both thresholds also in KWIK form ("KWIK": the
              learner's predictions with its accuracy as intervals, the box interval MDP). The
              model-free methods of the paper (DiFaRL, offline DQN, offline CQL) learn no model: they
              are trained again on every reward but the environment's own (MODEL_FREE), and their rows
              record the time of that training (retrain_s) and of computing its policy (reinference_s)
    data      clear27's (protocol.py: the uniform logger, the same seeds): a dataset is the paper's
    limits    clear27's, per step (memory and time)
    rewards   rewards.py: 10 to 19 per size on SysAdmin, 16 or 17 on Sepsis, 15 on Drone

Run from the repository root:
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/run.py                  # SysAdmin
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/run.py --domain sepsis  # 2, 3, 4 vitals
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/run.py --domain drone --sizes 3 4 5
    --sizes: machines (SysAdmin), vitals (Sepsis), the side of the grid (Drone)
    --transitions 10 100 ...   --workers N|auto   --core-share P   --resume (continue the latest run)
    --seeds K: the first K seeds (0 .. K-1; default: the paper's 10)
    --methods A B ...: only these methods (default: all), into a new run folder; with --extends the
    new run is recorded as an extension of the latest one, and plot.py draws the two together
    (the datasets depend on the seed only, so the problems of the two runs are the same).
    A resumed run keeps the methods and the extended run of its config.json: --resume with other
    --methods, or with --extends on a run that is not an extension, is refused
Then the figures: .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py --domain <domain>

One launch script per domain (the paper's sizes and data sizes, then the figures; run_domain.sh):
    bash experiments/clear27/ablations/ablation_rewards/run_sysadmin.sh    # also run_sepsis.sh, run_drone.sh

Output: results/clear27/ablations/ablation_rewards/<domain>/<domain>_<date>_<time>/
    config.json                    the options, the commit, the methods and the rewards of every size
    timing.csv                     one row per (size, seed): seconds in all, and per step (summed over its
                                   methods)
    <size>=<s>/runs/seed=<s>.csv   one row per (transitions, method, reward); <size> is N, vitals or side

Columns of a run: value and promise are normalised with the reward's own references,
(V - V_random) / (V*_obs - V_random); `held`: promise <= true value in every observed live state;
Times, in seconds, per step of the method:
    discovery_s   causal discovery / structure learning   } once per (dataset, method): the same on
    selection_s   choosing the graph among the candidates  } every reward's row of that model
    learning_s    the parameters: the interval sets        }   (learn_s: their sum)
    relax_s       compiling the model to the box   } once per reward
    solve_s       robust value iteration           }   (replan_s: the wall time of both, with the
                                                       cost of starting the limited steps)
    retrain_s     model-free methods only: training on the row's reward (offline DQN and CQL: the
                  network; DiFaRL: its fitted Q-iteration, which also gives the policy); on the
                  environment's reward, the original training
    reinference_s model-free methods only: the greedy policy of the trained network over the states
                  (offline DQN and CQL; empty for DiFaRL, whose policy comes with the fit)
A step a method does not have is empty (the flat model has no discovery and no box to compile).
"""

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLEAR27 = HERE.parents[1]
sys.path.insert(0, str(CLEAR27))
sys.path.insert(0, str(HERE))

import protocol  # noqa: E402  (also puts experiments/ on the path: common.py, domains.py)

import common  # noqa: E402
import rewards  # noqa: E402

from scm_to_mdp.tool.runs import STAMP, add_core_share, runs_of, set_core_share  # noqa: E402

NAME = "ablation_rewards"
RESULTS = common.ROOT / "results" / protocol.EXPERIMENT / "ablations" / NAME
# domain -> (title, default sizes, the name of a size on an axis, in a folder)
DOMAINS = {
    "sysadmin": ("SysAdmin", [3, 4, 5, 6, 7, 8, 9, 10], "machines N", "N"),
    "sepsis": ("Sepsis", [2, 3, 4], "vitals", "vitals"),
    "drone": ("Drone", [3, 4, 5, 6, 7, 8, 9], "side of the grid", "side"),
}
TRANSITIONS = [10, 100, 1000, 10_000]
METHODS = {  # label -> (clear27 source, its arm)
    "PCMCI+": ("pcmci_lagged", "PCMCI+ (lagged) -> IMDP (IA)"),
    "DYNOTEARS": ("dynotears_lagged", "DYNOTEARS (lagged) -> IMDP (IA)"),
    "full graph": ("fullfact", "full-fact -> IMDP (IA)"),
    "flat": ("flat", "flat IMDP"),
    "true graph": ("truegraph", "true graph -> IMDP (IA)"),  # the reference (REFERENCE)
    "PCMCI+ MDP": ("pcmci_lagged", "PCMCI+ (lagged) -> MDP (plug-in)"),
    "DYNOTEARS MDP": ("dynotears_lagged", "DYNOTEARS (lagged) -> MDP (plug-in)"),
    "full graph MDP": ("fullfact", "full-fact -> MDP (plug-in)"),
    "flat MDP": ("flat", "flat -> MDP (plug-in)"),
    "true graph MDP": ("truegraph", "true graph -> MDP (plug-in)"),
    "SLF-Rmax": ("strehl", "Strehl-KWIK -> MDP (plug-in)"),  # with KWIK_PAPER's thresholds
    "Met-Rmax": ("met", "Met-Rmax-KWIK -> MDP (plug-in)"),
    "sparse": ("sparse", "sparse -> IMDP (IA)"),
    "sparse MDP": ("sparse", "sparse -> MDP (plug-in)"),
    "DYNOTEARS SPIBB": ("dynotears_lagged", "DYNOTEARS (lagged) -> Factored SPIBB"),  # the paper's CD-SPIBB
    "PCMCI+ SPIBB": ("pcmci_lagged", "PCMCI+ (lagged) -> Factored SPIBB"),
    "true graph SPIBB": ("truegraph", "true graph -> Factored SPIBB"),
    "SL-SPIBB": ("sota", "SL-SPIBB"),
    "flat SPIBB": ("sota", "flat SPIBB"),
    "FOCUS": ("sota", "FOCUS"),
    # the two KWIK learners in their four forms (2026-11-18): the papers' thresholds (KWIK_PAPER) or their
    # theorems' (protocol.kwik_options level "theorem": m calibrated at delta), each as the plug-in MDP
    # above (point predictions, HALT for "don't know") or in KWIK form: the learner's predictions with its
    # stated accuracy as intervals, solved as the box interval MDP
    "SLF-Rmax KWIK": ("strehl", "Strehl-KWIK -> IMDP (IA)"),
    "Met-Rmax KWIK": ("met", "Met-Rmax-KWIK -> IMDP (IA)"),
    "SLF-Rmax (theorem)": ("strehl_theorem", "Strehl-KWIK -> MDP (plug-in)"),
    "Met-Rmax (theorem)": ("met_theorem", "Met-Rmax-KWIK -> MDP (plug-in)"),
    "SLF-Rmax KWIK (theorem)": ("strehl_theorem", "Strehl-KWIK -> IMDP (IA)"),
    "Met-Rmax KWIK (theorem)": ("met_theorem", "Met-Rmax-KWIK -> IMDP (IA)"),
    # the model-free methods of the paper (2026-11-18): no model to re-plan, so they are trained again on
    # every reward of the ladder but the first (MODEL_FREE; retrain_s, reinference_s)
    "DiFaRL": ("sota", "DiFaRL"),
    "offline DQN": ("sota", "offline DQN"),
    "offline CQL": ("sota", "offline CQL"),
}
REFERENCE = "true graph"
CERTIFIED = ["PCMCI+", "DYNOTEARS", "full graph", "flat", "true graph", "sparse"]  # their promise is a bound
# The thresholds of the papers' experiments, as in the main runs (protocol.KWIK_PAPER, with its sources).
KWIK_PAPER = {"SLF-Rmax": protocol.KWIK_PAPER["strehl"], "Met-Rmax": protocol.KWIK_PAPER["met"]}
COLUMNS = [
    "size", "seed", "n_transitions", "method", "level", "reward", "status", "failed_step", "value",
    "promise", "held", "true", "promise_raw", "V_star_obs", "V_random", "learn_s", "replan_s",
    "discovery_s", "selection_s", "learning_s", "relax_s", "solve_s", "retrain_s", "reinference_s",
]  # fmt: skip
MODEL_STEPS = ("discovery", "selection", "learning")  # once per model
REWARD_STEPS = ("relax", "solve")  # once per reward

_SETUP: dict = {}


def fresh(result):
    """A copy of a learned Result that a solve can change without touching the original (tool.methods.solve
    works in place): its own steps and info."""
    out = copy.copy(result)
    out.steps, out.info = dict(result.steps), dict(result.info)
    return out


def retrain_times(result, model_free: bool) -> list:
    """[retrain_s, reinference_s] of a model-free method's result (README of the columns); [None, None]
    for a model-based one. DiFaRL's fit gives its policy: no separate inference time."""
    if not model_free:
        return [None, None]
    steps = result.steps
    if "training" in steps:  # offline DQN and CQL: the network, then its greedy policy ("solve")
        solve_s = steps["solve"]["seconds"] if "solve" in steps else None
        return [steps["training"]["seconds"], solve_s]
    return [steps["solve"]["seconds"] if "solve" in steps else None, None]  # DiFaRL: FQI


def env_size(domain: str, size: int):
    """The size as clear27's grid names it (Drone: "<side>-<drones>", one drone)."""
    return f"{size}-1" if domain == "drone" else size


def size_folder(domain: str, size: int) -> str:
    return f"{DOMAINS[domain][3]}={size}"


def setup(domain: str, labels: tuple[str, ...] | None = None) -> dict:
    """clear27's setting of the domain and the methods `labels` (all of them by default), once per
    process."""
    labels = tuple(labels or METHODS)
    if (domain, labels) not in _SETUP:
        from scm_to_mdp.tool.grid import experiments

        methods, exp = {}, None
        for label in labels:
            source, arm = METHODS[label]
            exp = experiments(protocol.experiment_grid(domain, source))[0]
            methods[label] = exp.methods[arm]  # SLF-Rmax and Met-Rmax: the papers' thresholds (KWIK_PAPER)
        _SETUP[(domain, labels)] = {"exp": exp, "methods": methods}
    return _SETUP[(domain, labels)]


def run_seed(folder: str, domain: str, size: int, seed: int, transitions: tuple[int, ...],
             labels: tuple[str, ...] | None = None) -> dict:  # fmt: skip
    """Every dataset of one (size, seed): writes <size>=<s>/runs/seed=<s>.csv and returns its timing."""
    warnings.filterwarnings("ignore")
    import numpy as np

    from scm_to_mdp import evaluation as exact
    from scm_to_mdp import reward_family as rf
    from scm_to_mdp.core import Reward
    from scm_to_mdp.data import first_transitions
    from scm_to_mdp.evaluation import random_starts
    from scm_to_mdp.tool.evaluate import reference_values
    from scm_to_mdp.tool.methods import MODEL_FREE, learn, replan, solve

    start, s = time.monotonic(), setup(domain, labels)
    exp = s["exp"]
    env = exp.make_env(env_size(domain, size), seed)
    truth = env.ground_truth()
    table = exp.collect(env, max(*transitions, max(protocol.N_TRANSITIONS)))  # the paper's table
    truth.check_data(table)
    starts = random_starts(truth, exp.n_starts, seed)
    full = truth.learner_spec.read(table)  # the whole table: the original terminals and the visits
    visits = np.bincount(full.state_index(full.df[full.state_vars].to_numpy()), minlength=len(full.B))
    terminal = np.asarray(full.reward.terminal, dtype=bool)
    observed = truth.observed_index()
    tasks = []  # (rung, the truth under its reward, V_target, V*_obs, V_random)
    # Sepsis and Drone: "rare" counts the arrivals (a state seen only as a start cannot be reached)
    arrivals = np.bincount(full.state_index(full.df[full.next_vars].to_numpy()), minlength=len(full.B))
    world = rewards.World(full.B, list(full.state_vars), np.asarray(truth.actions), terminal,
                          np.asarray(full.reward.r_next, dtype=float),
                          visits if domain == "sysadmin" else arrivals)  # fmt: skip
    for rung in rewards.ladder_of(domain, world):
        on_truth = rung.task.for_truth(observed)
        on_observed = Reward(rung.task.R, rung.task.r_next, rung.task.terminal)
        task_truth = truth.with_reward(lambda B, a, r=on_truth: r, lambda B, a, r=on_observed: r)
        ref = reference_values(task_truth, exp.gamma, starts)
        _, V_target = task_truth.observed_optimal(exp.gamma)
        tasks.append((rung, task_truth, V_target, ref["V_star_obs"], ref["V_random"]))
    setup_s, learn_s, replan_s = time.monotonic() - start, 0.0, 0.0
    spent_in = dict.fromkeys(MODEL_STEPS + REWARD_STEPS + ("retrain", "reinference"), 0.0)  # seconds, summed

    def seconds_of(result, steps) -> list:
        return [result.steps[name]["seconds"] if name in result.steps else None for name in steps]

    out = Path(folder) / size_folder(domain, size) / "runs"
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"seed={seed}.csv"
    partial = final.with_suffix(".csv.partial")
    with open(partial, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for n in sorted(transitions):
            data = truth.learner_spec.read(first_transitions(table, n))
            true_actions = truth.actions_of(data)
            original = tasks[0][0].task.for_learner(true_actions)  # level 1 must be the environment's reward
            if not np.allclose(original.R, data.reward.R) or not np.allclose(
                original.r_next, data.reward.r_next
            ):
                raise ValueError(f"level 1 of the ladder is not the reward of {domain}")
            shared: dict = {}
            for label, method in s["methods"].items():
                given = truth.true_parents if method.discovery == "oracle" else None  # the reference's graph
                model = learn(data, method, exp.limits, given, seed=seed, shared=shared)
                model_free = method.learner in MODEL_FREE
                learned = fresh(model) if model_free else None  # solve works in place: keep the learned one
                base = solve(model, data, exp.gamma, exp.limits, seed=seed, shared=shared)
                model_s = seconds_of(base, MODEL_STEPS)
                seconds = sum(x for x in model_s if x is not None)
                learn_s += seconds
                for name, x in zip(MODEL_STEPS, model_s, strict=True):
                    spent_in[name] += x or 0.0
                for k, (rung, task_truth, V_target, V_best, V_rand) in enumerate(tasks):
                    row = [size, seed, n, label, rung.level, rung.name]
                    if not base.ok or base.solution is None:  # the model was not learned: no reward runs
                        failed = [base.status, base.failed_step, *[None] * 7, seconds, None]
                        writer.writerow(
                            [*row, *failed, *model_s, None, None, *retrain_times(base, model_free)]
                        )
                        continue
                    task_data = rf.with_task(data, rung.task, true_actions)
                    again = time.monotonic()
                    if model_free and k > 0:  # no model to re-plan: train again on this reward
                        planned = solve(fresh(learned), task_data, exp.gamma, exp.limits, seed=seed)
                    elif model_free:  # the environment's own reward: the original training
                        planned = base
                    else:
                        planned = replan(base, task_data, exp.gamma, exp.limits, seed=seed)
                    spent = time.monotonic() - again
                    replan_s += spent
                    reward_s = seconds_of(planned, REWARD_STEPS)
                    for name, x in zip(REWARD_STEPS, reward_s, strict=True):
                        spent_in[name] += x or 0.0
                    again_s = retrain_times(planned, model_free)
                    for name, x in zip(("retrain", "reinference"), again_s, strict=True):
                        spent_in[name] += x or 0.0
                    times = [seconds, spent, *model_s, *reward_s, *again_s]
                    if not planned.ok or planned.solution is None:
                        stopped = [planned.status, planned.failed_step, *[None] * 5]
                        writer.writerow([*row, *stopped, V_best, V_rand, *times])
                        continue
                    m = exact.evaluate(task_truth, task_data, planned.solution, exp.gamma, V_target, starts)
                    span = V_best - V_rand
                    scaled = [
                        (m[key] - V_rand) / span if span > 1e-9 else None for key in ("true", "promise")
                    ]
                    writer.writerow([*row, "ok", None, *scaled, m["safe"], m["true"], m["promise"], V_best,
                                     V_rand, *times])  # fmt: skip
    os.replace(partial, final)  # a file with this name is a finished (size, seed) (--resume)
    timing = {
        "size": size, "seed": seed, "datasets": len(transitions), "rewards": len(tasks),
        "seconds": round(time.monotonic() - start, 2), "setup_s": round(setup_s, 2),
        "learn_s": round(learn_s, 2), "replan_s": round(replan_s, 2),
        **{f"{name}_s": round(x, 2) for name, x in spent_in.items()},
    }  # fmt: skip
    final.with_suffix(".json").write_text(json.dumps(timing))
    return timing


# ---- the run --------------------------------------------------------------------------------------
def _commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=common.ROOT, check=False)  # fmt: skip
        return out.stdout.strip() or "?"
    except OSError:
        return "?"


def run_folder(base: Path, resume: bool) -> Path:
    """The latest <base>_<date>_<time> folder on a resume (when there is one), else a new one."""
    found = runs_of(base) if resume else []
    if found:
        print(f"resuming {found[-1]}")
        return found[-1]
    return Path(f"{base}_{datetime.now().strftime(STAMP)}")


def done(folder: Path, domain: str, size: int, seed: int) -> bool:
    return (folder / size_folder(domain, size) / "runs" / f"seed={seed}.json").exists()


def write_timing(folder: Path):
    """timing.csv from the finished (size, seed) of the run; returns the table (or None)."""
    import pandas as pd

    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("*=*/runs/*.json"))]
    if not rows:
        return None
    table = pd.DataFrame(rows).sort_values(["size", "seed"])
    table.to_csv(folder / "timing.csv", index=False)
    return table


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    return f"{seconds / 60:.1f} minutes" if seconds < 5400 else f"{seconds / 3600:.1f} hours"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27 ablation: the methods along a ladder of rewards.")
    parser.add_argument("--domain", choices=list(DOMAINS), default="sysadmin", help="default: sysadmin")
    parser.add_argument(
        "--sizes",
        "--machines",
        dest="sizes",
        nargs="+",
        type=int,
        metavar="N",
        help="machines (SysAdmin, 2..11), vitals (Sepsis, 2..4), side (Drone, 3..9)",
    )
    parser.add_argument("--transitions", nargs="+", type=int, default=TRANSITIONS, metavar="n",
                        help=f"data sizes (default {' '.join(map(str, TRANSITIONS))})")  # fmt: skip
    parser.add_argument("--seeds", type=int, default=len(common.SEEDS), metavar="K",
                        help=f"how many seeds: 0 .. K-1 (default {len(common.SEEDS)})")  # fmt: skip
    parser.add_argument("--resume", action="store_true", help="continue the latest run")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), metavar="METHOD",
                        help="only these methods (default: all): " + ", ".join(METHODS))  # fmt: skip
    parser.add_argument(
        "--extends",
        action="store_true",
        help="a new run that extends the latest one (plot.py draws them together)",
    )
    parser.add_argument("--workers", metavar="N", help="parallel (size, seed) jobs: a number or 'auto'")
    add_core_share(parser)
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    set_core_share(parser, args.core_share)
    labels = tuple(m for m in METHODS if m in (args.methods or METHODS))
    domain = args.domain
    title, default_sizes, size_name, prefix = DOMAINS[domain]
    sizes = sorted(set(args.sizes or default_sizes))
    known = list(
        protocol.SIZES[domain]
    )  # N = 2 of SysAdmin allowed since 2026-11-15 (levels 1-4, duplicates dropped)
    unknown = [s for s in sizes if env_size(domain, s) not in known]
    if unknown:
        parser.error(
            f"{title} has no size {unknown}: choose among {' '.join(str(s).split('-')[0] for s in known)}"
        )
    if min(args.transitions) < 1 or args.seeds < 1:
        parser.error("--transitions must be >= 1 and --seeds >= 1")
    if args.workers is not None:
        if args.workers != "auto" and not (args.workers.isdigit() and int(args.workers) >= 1):
            parser.error(f"--workers must be a positive number or 'auto', not {args.workers!r}")
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    transitions = tuple(sorted(set(args.transitions)))
    seeds = list(range(args.seeds))

    from tqdm import tqdm

    from scm_to_mdp.utils import compute
    from scm_to_mdp.utils.parallel import run_dynamic

    earlier = runs_of(args.results / domain / domain)
    folder = run_folder(args.results / domain / domain, args.resume)
    extends = None  # the run this one adds methods to
    if (folder / "config.json").exists():  # a resumed run keeps its methods and the run it extends
        recorded = json.loads((folder / "config.json").read_text())
        extends = recorded.get("extends")
        kept = tuple(m for m in METHODS if m in recorded.get("methods", METHODS))
        if args.methods and labels != kept:
            parser.error(
                f"--resume: {folder.name} has the methods {', '.join(kept)}; --methods asks for "
                f"{', '.join(labels)}. A run has one set of methods: drop --resume for a new run"
            )
        if args.extends and not extends:
            parser.error(f"--resume --extends: the latest run, {folder.name}, extends no run; drop --resume "
                         f"to start its extension")  # fmt: skip
        labels = kept
    elif args.extends:
        if not earlier:
            parser.error(f"--extends: no earlier run of {domain} in {args.results / domain}")
        extends = earlier[-1].name
    jobs = [
        (str(folder), domain, size, seed, transitions, labels)
        for size in sizes
        for seed in seeds
        if not done(folder, domain, size, seed)
    ]
    limits = setup(domain, labels)["exp"].limits
    workers = os.environ.get(compute.ENV_WORKERS) or "auto"
    plan = compute.plan(limits.mem_gb if limits else None, workers=workers, n_jobs=max(len(jobs), 1))
    compute.apply(plan)
    certified = [m for m in labels if m in CERTIFIED]
    extension = f" (extends {extends})" if extends else ""
    print(f"\n===== {NAME}: {title}, {len(labels)} methods ({len(certified)} with a promise) along the "
          f"ladder of rewards -> {folder}{extension} =====")  # fmt: skip
    problems = 0
    for size in sizes:
        levels = sorted({level for level, _ in rewards.names_of(domain, size)})
        n_rewards = len(rewards.names_of(domain, size))
        problems += n_rewards * len(transitions) * len(seeds)
        print(f"  {prefix}={size}: {n_rewards} rewards (levels {levels[0]}-{levels[-1]}), "
              f"{len(transitions)} data sizes, {len(seeds)} seeds")  # fmt: skip
    print(f"  {problems:,} problem instances, {problems * len(labels):,} policies; "
          f"{len(jobs)} of {len(sizes) * len(seeds)} (size, seed) jobs to run")  # fmt: skip
    print(plan.describe())
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        "ablation": NAME, "domain": domain, "sizes": sizes, "size_name": size_name, "size_prefix": prefix,
        "transitions": list(transitions), "seeds": seeds, "commit": _commit(),
        "machine": compute.as_record(plan), "methods": {label: METHODS[label][1] for label in labels},
        "extends": extends, "reference": REFERENCE, "certified": certified,
        "kwik_paper": {label: options(protocol.DOMAINS[domain].sl_spibb_parents)
                       for label, options in KWIK_PAPER.items()},
        "levels": {str(level): name for level, name in rewards.LEVEL_NAMES[domain].items()},
        "rewards": {str(size): [{"level": lv, "name": name} for lv, name in rewards.names_of(domain, size)]
                    for size in sizes},
    }  # fmt: skip
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    bar = tqdm(total=len(jobs), desc=f"{NAME} ({plan.workers} workers)", unit="job", dynamic_ncols=True)
    started, failed = time.monotonic(), []
    try:
        # the small sizes first: their figures can be drawn while a large size still runs
        for job, row in run_dynamic(run_seed, jobs, plan.workers, bar=bar, priority=lambda job: job[2]):
            if row.get("crashed"):  # no file for this (size, seed): --resume runs it again
                failed.append(job)
                bar.write(f"   {prefix}={job[2]} seed={job[3]} failed: {row.get('error', 'the worker died')}")
    finally:
        bar.close()
        timing = write_timing(folder)
    wall = time.monotonic() - started
    if timing is not None:
        steps = [f"{name}_s" for name in MODEL_STEPS + REWARD_STEPS]
        by_size = timing.groupby("size")[["seconds", *steps]].sum()
        print("time per size (summed over its seeds and methods):")
        for m, t in by_size.iterrows():
            parts = ", ".join(f"{name[:-2]} {duration(t[name])}" for name in steps)
            print(f"  {prefix}={m}: {duration(t['seconds'])} ({parts})")
        print(f"  all: {duration(timing['seconds'].sum())} of computation, {duration(wall)} of wall clock "
              f"for this launch")  # fmt: skip
    note = f" ({len(failed)} job(s) failed: run again with --resume)" if failed else ""
    print(f"results: {folder}{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
