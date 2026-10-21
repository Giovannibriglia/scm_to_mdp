"""Experiment 1, Sepsis: causal discovery algorithms and published baselines, over problem sizes.

Question: the same as experiment 1 on SysAdmin, on a medical simulator with non-binary variables,
terminal states (death, discharge), factored actions and context-specific effects (a treatment that
just stopped matters only in some states).

Setting: the sepsis simulator of Oberst & Sontag (ICML 2019) at sizes small (72 states), medium
(144) and full (1440; src/scm_to_mdp/envs/sepsis.py). EPISODES episodes of at most HORIZON steps of
the RANDOM behaviour policy (an episode stops at death or discharge). Values are averaged over the
start distribution of the simulator. The learner is told which columns are actions, that x' is the
next-step value of x, how many values each variable has, and the reward (with terminal states).

Arms: as in sysadmin.py, without "Schnitzer exact" (vertex enumeration is for 0/1 variables only).
    BIC / PCMCI+ / GES -> RMDP, Schnitzer IA (true graph), flat RMDP, flat MDP, DiFaRL.

Limits: every step runs in its own process with a memory limit (MEM_LIMIT_GB); discovery and solve
steps also have a time limit (TIMEOUT_S). A step that hits a limit stops its arm for that run.

Run (from the repository root):
    .venv/bin/python experiments/exp1/sepsis.py               run the grid, save CSV, tables, figure
    .venv/bin/python experiments/exp1/sepsis.py --plot-only   tables and figure from the saved CSV
Output: results/exp1/sepsis/: runs.csv, figures/{performance,validity,structure,time,memory}.{pdf,png},
        mdp/<size>/<method>.{pdf,png} (seed 0, problems with at most MDP_PLOT_MAX_STATES states).
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
from scm_to_mdp.envs import Sepsis
from scm_to_mdp.evaluation import random_starts
from scm_to_mdp.plotting.figures import Axis
from scm_to_mdp.scm_learning import nbn_intervals
from scm_to_mdp.solving import robust_vi, value_iteration

warnings.filterwarnings("ignore")
torch.set_num_threads(1)  # steps run in their own processes, one thread each

# ==========================================================================================
# 1. Config
# ==========================================================================================
SIZES = ["small", "medium", "full"]  # see envs/sepsis.py; Sepsis("custom", vitals=..., diabetes=...)
SEEDS = [0, 1, 2, 3, 4]
N_STARTS = 20  # K start states drawn uniformly among the non-terminal states, per seed
MDP_PLOT_MAX_STATES = 20  # pictures of the true and learned models (seed 0) for problems this small
EPISODES, HORIZON, POLICY = 2000, 20, "random"
GAMMA, DELTA = 0.9, 0.05
MAX_PARENTS = 6  # BIC search: at most 6 parents per variable (true in-degree: up to 6, sysbp')
PCMCI_ALPHA = 0.01
DIFARL_P, DIFARL_ITERS = 0.005, 30  # CI-test threshold, FQI iterations (their Sec. 5.1, 6)
MEM_LIMIT_GB = 8.0  # every step
TIMEOUT_S = {"discovery": 60.0, "solve": 60.0}  # seconds; other steps have no time limit
WORKERS = 4  # runs (size, seed) in parallel
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "sepsis"
ARMS = ["BIC -> RMDP", "PCMCI+ -> RMDP", "GES -> RMDP", "Schnitzer IA", "flat RMDP", "flat MDP", "DiFaRL"]


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
# 3. Pipeline: one run (size, seed) -> one row of results
# ==========================================================================================
def run(size: str, seed: int, skip: tuple = ()) -> dict:
    env = Sepsis(size, seed=seed)
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
        label=f"{size} seed={seed}",
    )
    if pictures:
        draw_models(env, data, models, OUT_DIR / "mdp" / size, f"Sepsis {size}")
    return {"size": size, "seed": seed, "states": len(env.B), "rows": len(data.df), **row}


def main():
    rows = run_waves(run, SIZES, SEEDS, WORKERS, desc="exp1 sepsis")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res["order"] = res["size"].map(SIZES.index)
    res.sort_values(["order", "seed"]).drop(columns="order").to_csv(OUT_DIR / "runs.csv", index=False)
    report()


# ==========================================================================================
# 4. Tables and figures (from the CSV only)
# ==========================================================================================
def report():
    res = pd.read_csv(OUT_DIR / "runs.csv")
    states = res.groupby("size")["states"].first()
    axis = Axis(
        res, "size", SIZES, [f"{s}\n{states.get(s, '?')}" for s in SIZES], "problem size (number of states)"
    )
    print_tables(res, axis, ARMS)
    limits = {"time": TIMEOUT_S, "memory": MEM_LIMIT_GB}
    make_figures(res, axis, ARMS, N_STARTS, limits, DELTA, OUT_DIR / "figures")


if __name__ == "__main__":
    report() if "--plot-only" in sys.argv else main()
