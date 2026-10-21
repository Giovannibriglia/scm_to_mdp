"""Experiment 1, SysAdmin: causal discovery algorithms and published baselines, over N.

Question: from offline data of ONE behaviour policy, does learning the causal structure give a
better policy, and a valid and tighter guarantee, than methods without structure?

Setting: SysAdmin with N machines (TOPOLOGY, REPAIR_COST; hidden repair mapping), EPISODES x HORIZON
steps of the POLICY behaviour policy. The learner is told which columns are actions, that x' is the
next-step value of x, how many values each variable has, and the reward.

Arms (each a recipe of pipeline steps; see make_arms):
    BIC / PCMCI+ / GES -> RMDP   discovery -> NBN Clopper-Pearson factor intervals -> product
                                  bounds -> robust VI. PCMCI+ and GES return a class of graphs:
                                  every DAG of the class is scored by BIC and the best is kept
                                  (src/scm_to_mdp/discovery/cpdag_to_dag.py).
    Schnitzer IA / exact         Schnitzer, Abate, Parker (AAAI 2026): the TRUE graph, the same
                                  factor intervals, then the interval relaxation (IA) or exact
                                  vertex enumeration.
    flat RMDP                    no structure: one interval per (s, a, s'), robust VI.
    flat MDP                     no structure, no guarantee: the plug-in MDP, VI.
    DiFaRL                       action groups by CI tests + decomposed FQI (CLeaR 2026).

Limits: every step runs in its own process with a memory limit (MEM_LIMIT_GB); discovery and solve
steps also have a time limit (TIMEOUT_S). A step that hits a limit stops its arm for that run.

Run (from the repository root):
    .venv/bin/python experiments/exp1/sysadmin.py               run the grid, save CSV, tables, figure
    .venv/bin/python experiments/exp1/sysadmin.py --plot-only   tables and figure from the saved CSV
Output: results/exp1/sysadmin/: runs.csv, figures/{performance,validity,structure,time,memory}.{pdf,png},
        mdp/N=<n>/<method>.{pdf,png} (seed 0, problems with at most MDP_PLOT_MAX_STATES states).
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from common import draw_models, make_figures, print_tables, run_arms, run_waves

from scm_to_mdp.baselines import difarl, flat_mdp, flat_rmdp
from scm_to_mdp.compilation import product_bounds
from scm_to_mdp.data import read_dataset
from scm_to_mdp.discovery import bic_search, ges, oracle, pcmci_plus
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.evaluation import random_starts
from scm_to_mdp.plotting.figures import Axis
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.solving import robust_vi, value_iteration, vertex_robust_vi

warnings.filterwarnings("ignore")
torch.set_num_threads(1)  # steps run in their own processes, one thread each

# ==========================================================================================
# 1. Config
# ==========================================================================================
N_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # the true model is a dense [2^N, N+1, 2^N] array: ~3.5 GB at peak for N=12, 275 GB for N=15
SEEDS = [0, 1, 2, 3, 4]
N_STARTS = 20  # K start states drawn uniformly among the non-terminal states, per seed
MDP_PLOT_MAX_STATES = 20  # pictures of the true and learned models (seed 0) for problems this small
TOPOLOGY, REPAIR_COST = "ring", 0.4  # SysAdmin settings: topology in ring / bidirectional / star
EPISODES, HORIZON, POLICY = 300, 20, "random"
GAMMA, DELTA = 0.9, 0.05
MAX_PARENTS = None  # BIC search: at most MAX_PARENTS parents per variable (true in-degree: 3 on the ring)
PCMCI_ALPHA = 0.01
DIFARL_P, DIFARL_ITERS = 0.005, 30  # CI-test threshold, FQI iterations (their Sec. 5.1, 6)
MEM_LIMIT_GB = 8.0  # every step
TIMEOUT_S = {"discovery": 60.0, "solve": 60.0}  # seconds; other steps have no time limit
WORKERS = 3  # runs (N, seed) in parallel; WORKERS x MEM_LIMIT_GB must stay below the RAM (32 GB)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "sysadmin"
ARMS = [
    "BIC -> RMDP",
    "PCMCI+ -> RMDP",
    "GES -> RMDP",
    "Schnitzer IA",
    "Schnitzer exact",
    "flat RMDP",
    "flat MDP",
    "DiFaRL",
]


# ==========================================================================================
# 2. Arms: each arm is a list of (step kind, function); see common.run_arms
# ==========================================================================================
def make_arms(data, true_parents):
    learn = lambda graph: nbn_intervals.learn(data, graph, DELTA)  # noqa: E731
    robust = lambda mdp: robust_vi.solve(mdp, GAMMA)  # noqa: E731
    return {
        "BIC -> RMDP": [
            ("discovery", lambda d: bic_search.fit(d, MAX_PARENTS)),
            ("learning", learn),
            ("compile", product_bounds.compile),
            ("solve", robust),
        ],
        "PCMCI+ -> RMDP": [
            ("discovery", lambda d: pcmci_plus.fit(d, PCMCI_ALPHA)),
            ("learning", learn),
            ("compile", product_bounds.compile),
            ("solve", robust),
        ],
        "GES -> RMDP": [
            ("discovery", ges.fit),
            ("learning", learn),
            ("compile", product_bounds.compile),
            ("solve", robust),
        ],
        "Schnitzer IA": [
            ("discovery", lambda d: oracle.fit(d, true_parents)),
            ("learning", learn),
            ("compile", product_bounds.compile),
            ("solve", robust),
        ],
        "Schnitzer exact": [
            ("discovery", lambda d: oracle.fit(d, true_parents)),
            ("learning", learn),
            ("solve", lambda bounds: vertex_robust_vi.solve(bounds, GAMMA)),
        ],
        "flat RMDP": [
            ("learning", lambda d: flat_rmdp.learn(d, DELTA)),
            ("solve", robust),
        ],
        "flat MDP": [
            ("learning", flat_mdp.learn),
            ("solve", lambda mdp: value_iteration.solve(mdp.L, mdp.reward, GAMMA)),
        ],
        "DiFaRL": [
            ("discovery", lambda d: difarl.groups(d, DIFARL_P)),
            ("solve", lambda groups: difarl.fqi(data, groups, GAMMA, DIFARL_ITERS)),
        ],
    }


# ==========================================================================================
# 3. Pipeline: one run (N, seed) -> one row of results
# ==========================================================================================
def run(n: int, seed: int, skip: tuple = ()) -> dict:
    env = SysAdmin(n, seed, topology=TOPOLOGY, repair_cost=REPAIR_COST)
    data = read_dataset(env.sample(EPISODES, HORIZON, POLICY), env.action_vars, env.cards, env.reward)
    arms = make_arms(data, env.true_parents(data))
    pictures = seed == 0 and len(env.B) <= MDP_PLOT_MAX_STATES
    row, models = run_arms(
        env,
        data,
        arms,
        GAMMA,
        MEM_LIMIT_GB,
        TIMEOUT_S,
        random_starts(env, N_STARTS, seed),
        skip=skip,
        keep_models=pictures,
        label=f"N={n} seed={seed}",
    )
    if pictures:
        draw_models(env, data, models, OUT_DIR / "mdp" / f"N={n}", f"SysAdmin N={n}")
    return {"N": n, "seed": seed, "rows": len(data.df), **row}


def main():
    rows = run_waves(run, N_VALUES, SEEDS, WORKERS, desc="exp1 sysadmin")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["N", "seed"]).to_csv(OUT_DIR / "runs.csv", index=False)
    report()


# ==========================================================================================
# 4. Tables and figures (from the CSV only)
# ==========================================================================================
def report():
    res = pd.read_csv(OUT_DIR / "runs.csv")
    axis = Axis(res, "N", N_VALUES, None, "N (machines)")
    print_tables(res, axis, ARMS)
    limits = {"time": TIMEOUT_S, "memory": MEM_LIMIT_GB}
    make_figures(res, axis, ARMS, N_STARTS, limits, DELTA, OUT_DIR / "figures")


if __name__ == "__main__":
    report() if "--plot-only" in sys.argv else main()
