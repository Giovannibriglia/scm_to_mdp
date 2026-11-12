"""Ablation: how robust are the methods to the reward? A ladder of rewards, SysAdmin (S8, S9).

Each method learns ONE model per dataset, as in clear27; that model is then re-planned (not re-learned)
under every reward of the ladder (rewards.py: from the original, additive reward to goals in rare states
and traps), and each policy is evaluated on the true model under that reward.

    methods   the box interval MDP of: PCMCI+ and DYNOTEARS (their lagged arms: no edge inside a slice),
              Strehl and Met-Rmax (fixed hull), the full graph, and the flat model (METHODS)
    data      clear27's (protocol.py: ring, uniform logger, the same seeds): a dataset is the paper's
    limits    clear27's, per step (memory and time)
    rewards   10 at N=3, 11 at N=4, 19 from N=5 (rewards.py)

Run from the repository root:
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/run.py                  # N = 3..10
    .venv/bin/python experiments/clear27/ablations/ablation_rewards/run.py --machines 3 4 --seeds 3
    --transitions 10 100 ...   --workers N|auto   --core-share P   --resume (continue the latest run)
    --seeds K: the first K seeds (0 .. K-1; default: the paper's 10)
Then the figures: .venv/bin/python experiments/clear27/ablations/ablation_rewards/plot.py

Output: results/clear27/ablations/ablation_rewards/sysadmin_<date>_<time>/
    config.json               the options, the commit, the methods and the rewards of every size
    timing.csv                one row per (N, seed): seconds in all, to learn the models, to re-plan
    N=<N>/runs/seed=<s>.csv   one row per (transitions, method, reward)

Columns of a run: value and promise are normalised with the reward's own references,
(V - V_random) / (V*_obs - V_random); `held`: promise <= true value in every observed live state;
learn_s: the seconds to learn the method's model on that dataset (the same on every reward's row);
replan_s: the seconds to compile and solve that model under the reward.
"""

import argparse
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
DOMAIN = "sysadmin"
RESULTS = common.ROOT / "results" / protocol.EXPERIMENT / "ablations" / NAME
MACHINES = [3, 4, 5, 6, 7, 8, 9, 10]
TRANSITIONS = [10, 100, 1000, 10_000]
METHODS = {  # label -> (clear27 source, its arm)
    "PCMCI+": ("pcmci_lagged", "PCMCI+ (lagged) -> IMDP (IA)"),
    "DYNOTEARS": ("dynotears_lagged", "DYNOTEARS (lagged) -> IMDP (IA)"),
    "Strehl": ("strehl", "Strehl (fixed hull) -> IMDP (IA)"),
    "Met-Rmax": ("met", "Met-Rmax (fixed hull) -> IMDP (IA)"),
    "full graph": ("fullfact", "full-fact -> IMDP (IA)"),
    "flat": ("flat", "flat IMDP"),
}
COLUMNS = [
    "machines", "seed", "n_transitions", "method", "level", "reward", "status", "failed_step", "value",
    "promise", "held", "true", "promise_raw", "V_star_obs", "V_random", "learn_s", "replan_s",
]  # fmt: skip

_SETUP: dict = {}


def setup() -> dict:
    """clear27's SysAdmin setting and the methods (once per process)."""
    if not _SETUP:
        from scm_to_mdp.tool.grid import experiments

        methods, exp = {}, None
        for label, (source, arm) in METHODS.items():
            exp = experiments(protocol.experiment_grid(DOMAIN, source))[0]
            methods[label] = exp.methods[arm]
        _SETUP.update(exp=exp, methods=methods)
    return _SETUP


def run_seed(folder: str, machines: int, seed: int, transitions: tuple[int, ...]) -> dict:
    """Every dataset of one (size, seed): writes N=<N>/runs/seed=<s>.csv and returns its timing."""
    warnings.filterwarnings("ignore")
    import numpy as np

    from scm_to_mdp import evaluation as exact
    from scm_to_mdp import reward_family as rf
    from scm_to_mdp.core import Reward
    from scm_to_mdp.data import first_transitions
    from scm_to_mdp.evaluation import random_starts
    from scm_to_mdp.tool.evaluate import reference_values
    from scm_to_mdp.tool.methods import learn, replan, solve

    start, s = time.monotonic(), setup()
    exp = s["exp"]
    env = exp.make_env(machines, seed)
    truth = env.ground_truth()
    table = exp.collect(env, max(*transitions, max(protocol.N_TRANSITIONS)))  # the paper's table
    truth.check_data(table)
    starts = random_starts(truth, exp.n_starts, seed)
    full = truth.learner_spec.read(table)  # the whole table: the original terminals and the visits
    visits = np.bincount(full.state_index(full.df[full.state_vars].to_numpy()), minlength=len(full.B))
    terminal = np.asarray(full.reward.terminal, dtype=bool)
    observed = truth.observed_index()
    tasks = []  # (rung, the truth under its reward, V_target, V*_obs, V_random)
    for rung in rewards.ladder(full.B, np.asarray(truth.actions), terminal, visits):
        on_truth = rung.task.for_truth(observed)
        on_observed = Reward(rung.task.R, rung.task.r_next, rung.task.terminal)
        task_truth = truth.with_reward(lambda B, a, r=on_truth: r, lambda B, a, r=on_observed: r)
        ref = reference_values(task_truth, exp.gamma, starts)
        _, V_target = task_truth.observed_optimal(exp.gamma)
        tasks.append((rung, task_truth, V_target, ref["V_star_obs"], ref["V_random"]))
    setup_s, learn_s, replan_s = time.monotonic() - start, 0.0, 0.0

    out = Path(folder) / f"N={machines}" / "runs"
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
                raise ValueError("the additive reward of the ladder is not SysAdmin's reward")
            shared: dict = {}
            for label, method in s["methods"].items():
                model = learn(data, method, exp.limits, None, seed=seed, shared=shared)
                base = solve(model, data, exp.gamma, exp.limits, seed=seed, shared=shared)
                seconds = sum(step["seconds"] for name, step in base.steps.items() if name != "solve")
                learn_s += seconds
                for rung, task_truth, V_target, V_best, V_rand in tasks:
                    row = [machines, seed, n, label, rung.level, rung.name]
                    if not base.ok or base.solution is None:  # the model was not learned: no reward runs
                        writer.writerow([*row, base.status, base.failed_step, *[None] * 7, seconds, None])
                        continue
                    task_data = rf.with_task(data, rung.task, true_actions)
                    again = time.monotonic()
                    planned = replan(base, task_data, exp.gamma, exp.limits, seed=seed)
                    spent = time.monotonic() - again
                    replan_s += spent
                    if not planned.ok or planned.solution is None:
                        stopped = [planned.status, planned.failed_step, *[None] * 5]
                        writer.writerow([*row, *stopped, V_best, V_rand, seconds, spent])
                        continue
                    m = exact.evaluate(task_truth, task_data, planned.solution, exp.gamma, V_target, starts)
                    span = V_best - V_rand
                    scaled = [
                        (m[key] - V_rand) / span if span > 1e-9 else None for key in ("true", "promise")
                    ]
                    writer.writerow([*row, "ok", None, *scaled, m["safe"], m["true"], m["promise"], V_best,
                                     V_rand, seconds, spent])  # fmt: skip
    os.replace(partial, final)  # a file with this name is a finished (size, seed) (--resume)
    timing = {
        "machines": machines, "seed": seed, "datasets": len(transitions), "rewards": len(tasks),
        "seconds": round(time.monotonic() - start, 2), "setup_s": round(setup_s, 2),
        "learn_s": round(learn_s, 2), "replan_s": round(replan_s, 2),
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


def done(folder: Path, machines: int, seed: int) -> bool:
    return (folder / f"N={machines}" / "runs" / f"seed={seed}.json").exists()


def write_timing(folder: Path):
    """timing.csv from the finished (size, seed) of the run; returns the table (or None)."""
    import pandas as pd

    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("N=*/runs/*.json"))]
    if not rows:
        return None
    table = pd.DataFrame(rows).sort_values(["machines", "seed"])
    table.to_csv(folder / "timing.csv", index=False)
    return table


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    return f"{seconds / 60:.1f} minutes" if seconds < 5400 else f"{seconds / 3600:.1f} hours"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27 ablation: the methods along a ladder of rewards.")
    parser.add_argument("--machines", nargs="+", type=int, default=MACHINES, metavar="N",
                        help=f"SysAdmin sizes (default {MACHINES[0]}..{MACHINES[-1]})")  # fmt: skip
    parser.add_argument("--transitions", nargs="+", type=int, default=TRANSITIONS, metavar="n",
                        help=f"data sizes (default {' '.join(map(str, TRANSITIONS))})")  # fmt: skip
    parser.add_argument("--seeds", type=int, default=len(common.SEEDS), metavar="K",
                        help=f"how many seeds: 0 .. K-1 (default {len(common.SEEDS)})")  # fmt: skip
    parser.add_argument("--resume", action="store_true", help="continue the latest run")
    parser.add_argument("--workers", metavar="N", help="parallel (size, seed) jobs: a number or 'auto'")
    add_core_share(parser)
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    set_core_share(parser, args.core_share)
    if min(args.machines) < 3 or min(args.transitions) < 1 or args.seeds < 1:
        parser.error("--machines must be >= 3, --transitions >= 1 and --seeds >= 1")
    if args.workers is not None:
        if args.workers != "auto" and not (args.workers.isdigit() and int(args.workers) >= 1):
            parser.error(f"--workers must be a positive number or 'auto', not {args.workers!r}")
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    machines, transitions = sorted(set(args.machines)), tuple(sorted(set(args.transitions)))
    seeds = list(range(args.seeds))

    from tqdm import tqdm

    from scm_to_mdp.utils import compute
    from scm_to_mdp.utils.parallel import run_dynamic

    folder = run_folder(args.results / DOMAIN, args.resume)
    jobs = [(str(folder), m, s, transitions) for m in machines for s in seeds if not done(folder, m, s)]
    limits = setup()["exp"].limits
    workers = os.environ.get(compute.ENV_WORKERS) or "auto"
    plan = compute.plan(limits.mem_gb if limits else None, workers=workers, n_jobs=max(len(jobs), 1))
    compute.apply(plan)
    print(f"\n===== {NAME}: SysAdmin, {len(METHODS)} methods along the ladder of rewards -> {folder} =====")
    problems = 0
    for m in machines:
        n_rewards = len(rewards.names(m))
        problems += n_rewards * len(transitions) * len(seeds)
        print(f"  N={m}: {n_rewards} rewards (levels {rewards.levels_of(m)[0]}-{rewards.levels_of(m)[-1]}), "
              f"{len(transitions)} data sizes, {len(seeds)} seeds")  # fmt: skip
    print(f"  {problems:,} problem instances, {problems * len(METHODS):,} policies; "
          f"{len(jobs)} of {len(machines) * len(seeds)} (size, seed) jobs to run")  # fmt: skip
    print(plan.describe())
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        "ablation": NAME, "domain": DOMAIN, "machines": machines, "transitions": list(transitions),
        "seeds": seeds, "commit": _commit(), "machine": compute.as_record(plan),
        "methods": {label: arm for label, (_, arm) in METHODS.items()},
        "rewards": {str(m): [{"level": lv, "name": name} for lv, name in rewards.names(m)] for m in machines},
    }  # fmt: skip
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    bar = tqdm(total=len(jobs), desc=f"{NAME} ({plan.workers} workers)", unit="job", dynamic_ncols=True)
    started, failed = time.monotonic(), []
    try:
        # the small sizes first: their figures can be drawn while a large size still runs
        for job, row in run_dynamic(run_seed, jobs, plan.workers, bar=bar, priority=lambda job: job[1]):
            if row.get("crashed"):  # no file for this (size, seed): --resume runs it again
                failed.append(job)
                bar.write(f"   N={job[1]} seed={job[2]} failed: {row.get('error', 'the worker died')}")
    finally:
        bar.close()
        timing = write_timing(folder)
    wall = time.monotonic() - started
    if timing is not None:
        by_size = timing.groupby("machines")[["seconds", "learn_s", "replan_s"]].sum()
        print("time per size (summed over its seeds):")
        for m, t in by_size.iterrows():
            print(f"  N={m}: {duration(t['seconds'])} (learning {duration(t['learn_s'])}, "
                  f"re-planning {duration(t['replan_s'])})")  # fmt: skip
        print(f"  all: {duration(timing['seconds'].sum())} of computation, {duration(wall)} of wall clock "
              f"for this launch")  # fmt: skip
    note = f" ({len(failed)} job(s) failed: run again with --resume)" if failed else ""
    print(f"results: {folder}{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
