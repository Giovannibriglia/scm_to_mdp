"""Ablation: does a better BIC mean a better policy? Every graph, SysAdmin (S3 -> S8).

For SysAdmin with N machines, EVERY graph our constraints allow is scored and run through the pipeline:

    constraints   the time order, and no edge inside a slice: an edge goes from an input at t (the N
                  machines and the action `reboot`) to a machine at t+1. N (N+1) possible edges, so
                  2^(N (N+1)) graphs: 64 at N=2, 4,096 at N=3, 1,048,576 at N=4.
    per graph     three scores on the data: BIC (the score of the selection step, selection/bic_best.py),
                  AIC (the log-likelihood minus the number of parameters) and BDeu (the Bayesian score
                  with a uniform Dirichlet prior of equivalent sample size BDEU_ESS); then the
                  steps of "PCMCI+ (lagged) -> IMDP (IA)" on that graph: the interval sets (delta and the
                  post-selection family of the lagged arms), the box, robust value iteration, and the
                  evaluation on the true model.
    marked        the true graph, the empty and the full graph, the best graph of each score, and the
                  graphs PCMCI+ and DYNOTEARS find under the same constraints (their "lagged" arms).

The data are clear27's (protocol.py: ring, uniform logger, the same seeds), so a dataset is the paper's.

Run from the repository root:
    .venv/bin/python experiments/clear27/ablations/ablation_graph_score/run.py            # N = 2 and 3
    .venv/bin/python experiments/clear27/ablations/ablation_graph_score/run.py --machines 2 3 4 \
        --transitions 100 1000
    --seeds 0 1 ...   --workers N|auto   --core-share P   --resume (continue the latest run)
Then the figures: .venv/bin/python experiments/clear27/ablations/ablation_graph_score/plot.py

Output: results/clear27/ablations/ablation_graph_score/sysadmin_<date>_<time>/
    config.json                      the options, the commit, and the edge of every bit of a graph's number
    N=<N>/datasets.csv               one row per (seed, transitions): reference values, marked graphs
    N=<N>/graphs/seed=<s>_n=<n>.csv  one row per graph of that dataset

A graph is a number: bit k is the edge EDGES[k] = (machine at t+1, input at t), the machines in order and,
for each, the inputs in the order of the data (config.json, "edges"); parents_of() decodes it.
normalised value / promise = (V - V_random) / (V*_obs - V_random), as in plot.py.

The confidence statement of the interval sets holds for a graph that contains the true parents; a graph
that misses a true edge has no guarantee, and its promise may fail (column `held`).
"""

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

CLEAR27 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CLEAR27))

import protocol  # noqa: E402  (also puts experiments/ on the path: common.py, domains.py)

import common  # noqa: E402

from scm_to_mdp.core import NEXT_MARK  # noqa: E402
from scm_to_mdp.tool.runs import STAMP, add_core_share, runs_of, set_core_share  # noqa: E402

NAME = "ablation_graph_score"
DOMAIN = "sysadmin"
RESULTS = common.ROOT / "results" / protocol.EXPERIMENT / "ablations" / NAME
MACHINES = [2, 3]
TRANSITIONS = [10, 100, 1000, 10_000]
ARM = "PCMCI+ (lagged) -> IMDP (IA)"  # the arm whose learning and solver every graph gets
DISCOVERIES = {  # mark -> (clear27 source, its arm): the discoveries under our constraints
    "pcmci": ("pcmci_lagged", "PCMCI+ (lagged) -> IMDP (IA)"),
    "dynotears": ("dynotears_lagged", "DYNOTEARS (lagged) -> IMDP (IA)"),
}
SCORES = ("bic", "aic", "bdeu")  # higher is better for each
BDEU_ESS = 1.0  # BDeu's equivalent sample size (Heckerman, Geiger, Chickering 1995; 1 is the usual choice)
SECONDS_PER_GRAPH = 0.006  # for the estimate printed at the start (measured at N=2 and 3: about 5 ms)
GRAPH_COLUMNS = [
    "seed", "n_transitions", "graph", "n_edges", "missing", "extra", "bic", "aic", "bdeu", "loglik",
    "n_params", "status",
    "value", "promise", "held", "covered", "width", "rows", "unseen", "true", "promise_raw",
]  # fmt: skip


# ---- graphs as numbers ----------------------------------------------------------------------------
def n_graphs(machines: int) -> int:
    return 2 ** (machines * (machines + 1))


def edge_list(state_vars: list[str], inputs: list[str]) -> list[tuple[str, str]]:
    """EDGES: bit k of a graph's number is the edge inputs -> state var at t+1, as (state var, input)."""
    return [(v, p) for v in state_vars for p in inputs]


def parents_of(graph: int, state_vars: list[str], inputs: list[str]) -> dict[str, tuple[str, ...]]:
    """The parents (inputs at t) of every machine at t+1, in the graph with this number."""
    width = len(inputs)
    return {
        v: tuple(p for j, p in enumerate(inputs) if (graph >> (i * width + j)) & 1)
        for i, v in enumerate(state_vars)
    }


def graph_of(parents: dict, state_vars: list[str], inputs: list[str]) -> int:
    """The number of a graph given by its parents (same-step parents, if any, are left out)."""
    width = len(inputs)
    bits = [(i, j) for i, v in enumerate(state_vars) for j, p in enumerate(inputs) if p in parents[v]]
    return sum(1 << (i * width + j) for i, j in bits)


# ---- one dataset ----------------------------------------------------------------------------------
_SETUP: dict = {}


def setup() -> dict:
    """clear27's SysAdmin setting and the methods (once per process)."""
    if not _SETUP:
        from scm_to_mdp.tool.grid import experiments

        arms = {}
        for mark, (source, arm) in DISCOVERIES.items():
            exp = experiments(protocol.experiment_grid(DOMAIN, source))[0]
            arms[mark] = exp.methods[arm]
        exp = experiments(protocol.experiment_grid(DOMAIN, DISCOVERIES["pcmci"][0]))[0]
        graph_given = {"discovery": "oracle", "selection": "given", "discovery_options": {}}
        given = dataclasses.replace(exp.methods[ARM], **graph_given)
        _SETUP.update(exp=exp, given=given, discoveries=arms)
    return _SETUP


def family_scores(data, cards: dict) -> dict:
    """(state var index, parent mask) -> (BIC, log-likelihood, parameters, BDeu) of that family: N 2^(N+1)
    families give the scores of every graph (each score is a sum over the machines). Parameters and
    BDeu's prior count the parent rows that can occur (bic_search.feasible_rows), as the BIC does."""
    import math

    import numpy as np
    from scipy.special import gammaln

    from scm_to_mdp.core import encode
    from scm_to_mdp.discovery.bic_search import bic, feasible_rows

    scores, half_log_n = {}, 0.5 * math.log(len(data.df))
    for i, v in enumerate(data.state_vars):
        child, values = data.df[v + NEXT_MARK].to_numpy(), cards[v]
        for mask in range(2 ** len(data.inputs)):
            parents = [p for j, p in enumerate(data.inputs) if (mask >> j) & 1]
            parent_cards = [cards[p] for p in parents]
            rows = encode(data.df[parents].to_numpy(), parent_cards) if parents else np.zeros(len(child), int)
            counts = np.zeros((math.prod(parent_cards), values))
            np.add.at(counts, (rows, child), 1)
            feasible = feasible_rows(tuple(parents), cards, data.action_vars, data.actions)
            params = feasible * (values - 1)
            score = bic(counts, feasible)
            row_prior, cell_prior = BDEU_ESS / feasible, BDEU_ESS / (feasible * values)
            seen = counts[counts.sum(1) > 0]  # a row without data adds nothing
            bdeu = (gammaln(row_prior) - gammaln(row_prior + seen.sum(1))).sum()
            bdeu += (gammaln(cell_prior + seen) - gammaln(cell_prior)).sum()
            scores[i, mask] = (score, score + half_log_n * params, params, float(bdeu))
    return scores


def run_dataset(folder: str, machines: int, seed: int, n: int) -> dict:
    """Every graph of one dataset: writes N=<N>/graphs/seed=<s>_n=<n>.csv and returns the dataset's row."""
    warnings.filterwarnings("ignore")  # a graph that misses parents gives the library's own warnings
    from scm_to_mdp.data import first_transitions
    from scm_to_mdp.evaluation import random_starts
    from scm_to_mdp.tool.evaluate import evaluate, reference_values
    from scm_to_mdp.tool.methods import learn, solve

    start, s = time.monotonic(), setup()
    exp = s["exp"]
    env = exp.make_env(machines, seed)
    truth = env.ground_truth()
    table = exp.collect(env, max(n, max(protocol.N_TRANSITIONS)))  # the paper's table, then its first n
    truth.check_data(table)
    starts = random_starts(truth, exp.n_starts, seed)
    reference = reference_values(truth, exp.gamma, starts)
    data = truth.learner_spec.read(first_transitions(table, n))
    state_vars, inputs = data.state_vars, data.inputs
    width, total = len(inputs), n_graphs(len(state_vars))
    cards = {c: data.cards[c.removesuffix(NEXT_MARK)] for c in data.df.columns}
    scores = family_scores(data, cards)
    true_graph = graph_of(truth.parents_in(data), state_vars, inputs)
    span = reference["V_star_obs"] - reference["V_random"]

    def normalised(v):
        return (v - reference["V_random"]) / span if v is not None else None

    out = Path(folder) / f"N={machines}" / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"seed={seed}_n={n}.csv"
    partial = final.with_suffix(".csv.partial")
    best = dict.fromkeys(SCORES, (-float("inf"), None))  # score -> (its best value, the graph)
    with open(partial, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(GRAPH_COLUMNS)
        for graph in range(total):
            family = [scores[i, (graph >> (i * width)) & ((1 << width) - 1)] for i in range(len(state_vars))]
            bic, loglik, params, bdeu = (sum(x[k] for x in family) for k in range(4))
            aic = loglik - params
            for name, score in zip(SCORES, (bic, aic, bdeu), strict=True):
                if score > best[name][0]:
                    best[name] = (score, graph)  # a tie: the lowest number
            result = learn(data, s["given"], None, parents_of(graph, state_vars, inputs), seed=seed)
            result = solve(result, data, exp.gamma, None, seed=seed)
            m = evaluate(result, data, truth, exp.gamma, starts, exp.failure_horizon) if result.ok else {}
            writer.writerow([
                seed, n, graph, graph.bit_count(), (true_graph & ~graph).bit_count(),
                (graph & ~true_graph).bit_count(), bic, aic, bdeu, loglik, params, result.status,
                normalised(m.get("true")), normalised(m.get("promise")), m.get("safe"), m.get("contain_all"),
                m.get("s4_width"), m.get("s4_rows"), m.get("s4_unseen"), m.get("true"), m.get("promise"),
            ])  # fmt: skip
    os.replace(partial, final)  # a file with this name is a finished dataset (--resume)
    marks = {}
    for mark, method in s["discoveries"].items():  # what the discoveries pick on this dataset
        found = learn(data, method, None, seed=seed)
        graph = found.graph if getattr(found, "graph", None) is not None else None
        marks[f"{mark}_graph"] = graph_of(graph.parents, state_vars, inputs) if graph is not None else None
        marks[f"{mark}_status"] = found.status
    row = {
        "machines": machines, "seed": seed, "n_transitions": n, "transitions": len(data.df), "graphs": total,
        "V_star_obs": reference["V_star_obs"], "V_random": reference["V_random"],
        "true_graph": true_graph, "empty_graph": 0, "full_graph": total - 1,
        **{f"best_{name}_graph": best[name][1] for name in SCORES},
        **marks, "seconds": round(time.monotonic() - start, 2),
    }  # fmt: skip
    final.with_suffix(".json").write_text(json.dumps(row))
    return row


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


def done(folder: Path, machines: int, seed: int, n: int) -> bool:
    return (folder / f"N={machines}" / "graphs" / f"seed={seed}_n={n}.json").exists()


def write_datasets(folder: Path) -> None:
    """N=<N>/datasets.csv from the finished datasets of the run."""
    import pandas as pd

    for sub in sorted(folder.glob("N=*")):
        rows = [json.loads(p.read_text()) for p in sorted((sub / "graphs").glob("*.json"))]
        if rows:
            table = pd.DataFrame(rows).sort_values(["seed", "n_transitions"])
            table.to_csv(sub / "datasets.csv", index=False)


def duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    return f"{seconds / 60:.0f} minutes" if seconds < 5400 else f"{seconds / 3600:.1f} hours"


def describe_edges(machines: int) -> list:
    """The edge of every bit at N machines (the data's column order: m0.., reboot)."""
    state_vars = [f"m{i}" for i in range(machines)]
    return [list(e) for e in edge_list(state_vars, state_vars + ["reboot"])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="clear27 ablation: BIC against performance, every graph.")
    parser.add_argument("--machines", nargs="+", type=int, default=MACHINES, metavar="N",
                        help=f"SysAdmin sizes (default {' '.join(map(str, MACHINES))})")  # fmt: skip
    parser.add_argument("--transitions", nargs="+", type=int, default=TRANSITIONS, metavar="n",
                        help=f"data sizes (default {' '.join(map(str, TRANSITIONS))})")  # fmt: skip
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.SEEDS), metavar="SEED",
                        help="seeds (default: the paper's)")  # fmt: skip
    parser.add_argument("--resume", action="store_true", help="continue the latest run")
    parser.add_argument("--workers", metavar="N", help="parallel datasets: a number or 'auto'")
    add_core_share(parser)
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    set_core_share(parser, args.core_share)
    if min(args.machines) < 2 or min(args.transitions) < 1:
        parser.error("--machines must be >= 2 and --transitions >= 1")
    if args.workers is not None:
        if args.workers != "auto" and not (args.workers.isdigit() and int(args.workers) >= 1):
            parser.error(f"--workers must be a positive number or 'auto', not {args.workers!r}")
        os.environ["SCM_TO_MDP_WORKERS"] = args.workers
    machines, transitions = sorted(set(args.machines)), sorted(set(args.transitions))
    seeds = sorted(set(args.seeds))

    from tqdm import tqdm

    from scm_to_mdp.utils import compute
    from scm_to_mdp.utils.parallel import run_dynamic

    folder = run_folder(args.results / DOMAIN, args.resume)
    every = [(m, s, n) for m in machines for n in transitions for s in seeds]
    jobs = [(str(folder), m, s, n) for m, s, n in every if not done(folder, m, s, n)]
    workers = os.environ.get(compute.ENV_WORKERS) or "auto"
    plan = compute.plan(None, workers=workers, n_jobs=max(len(jobs), 1))
    compute.apply(plan)
    print(f"\n===== {NAME}: SysAdmin, {ARM} on every graph -> {folder} =====")
    cpu = 0.0
    for m in machines:
        todo = sum(job[1] == m for job in jobs)
        seconds = todo * n_graphs(m) * SECONDS_PER_GRAPH
        cpu += seconds
        print(f"  N={m}: {n_graphs(m):,} graphs per dataset, {todo} of {len(seeds) * len(transitions)} "
              f"datasets to run (about {duration(seconds)} of CPU)")  # fmt: skip
    print(f"  about {duration(cpu / plan.workers)} with {plan.workers} workers (a rough estimate)")
    print(plan.describe())
    folder.mkdir(parents=True, exist_ok=True)
    config = {
        "ablation": NAME, "domain": DOMAIN, "arm": ARM, "machines": machines, "transitions": transitions,
        "seeds": seeds, "commit": _commit(), "machine": compute.as_record(plan),
        "edges": {str(m): describe_edges(m) for m in machines},
    }  # fmt: skip
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    bar = tqdm(total=len(jobs), desc=f"{NAME} ({plan.workers} workers)", unit="dataset", dynamic_ncols=True)
    failed = []
    try:
        # the small sizes first: their figures can be drawn while a large size still runs
        for job, row in run_dynamic(run_dataset, jobs, plan.workers, bar=bar, priority=lambda job: job[1]):
            if row.get("crashed"):  # the dataset has no file: --resume runs it again
                failed.append(job)
                bar.write(
                    f"   N={job[1]} seed={job[2]} n={job[3]} failed: {row.get('error', 'the worker died')}"
                )
    finally:
        bar.close()
        write_datasets(folder)
    print(
        f"results: {folder}"
        + (f" ({len(failed)} dataset(s) failed: run again with --resume)" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
