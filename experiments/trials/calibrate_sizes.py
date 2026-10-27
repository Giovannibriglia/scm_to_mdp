"""Which sizes fit the common budget (protocol.LIMITS)? The rule of the trials: every size of a grid must
be solved by the oracle (the true observed MDP within +-tol, robust value iteration; tool.oracle_methods)
within the limits of one step. For every candidate size of a domain, in increasing order:

    build     the environment and its dense true model (in this process, as a sweep worker does: not
              under the step limits), and the observed kernel: seconds and memory added
    oracle    oracle -> IMDP: robust value iteration on [P - tol, P + tol] (the step "solve", under the
              memory limit of protocol.LIMITS and no time limit, as in a sweep), and oracle -> MDP:
              value iteration (seconds); the other methods will then have SOLVE_FACTOR x the oracle's
              time per sweep x the sweeps the stopping rule may need at the rate gamma (tool/sweep.py)

The candidates are the sizes of the trial scripts (chosen with this script on 2026-10-26). A domain stops
at its first size that does not fit in memory, or whose oracle takes more than MAX_ORACLE_S
(a larger one would not do better). Prints one table
per domain and writes results/trials/calibration/<domain>.csv.

    .venv/bin/python experiments/trials/calibrate_sizes.py                    # every domain
    .venv/bin/python experiments/trials/calibrate_sizes.py sysadmin drone     # only these
"""

import gc
import importlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from protocol import LIMITS, SOLVE_FACTOR, SOLVE_FLOOR_S  # noqa: E402

from scm_to_mdp.baselines.true_model import TOL  # noqa: E402
from scm_to_mdp.core import IntervalMDP, Reward, all_states  # noqa: E402
from scm_to_mdp.ground_truth import ModelTooLarge  # noqa: E402
from scm_to_mdp.solving import robust_vi, value_iteration  # noqa: E402
from scm_to_mdp.solving.stopping import iterations_bound, reward_scale  # noqa: E402
from scm_to_mdp.utils.limits import run_limited  # noqa: E402

warnings.filterwarnings("ignore")
MAX_ORACLE_S = 1800.0  # stop a domain once its oracle -> IMDP takes longer (a wall-clock budget, not a limit)

CANDIDATES = {  # in the size format of each <domain>_uniform_rmdp.py
    "sysadmin": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "chemistry": [2, 3, 4, 5, 6, 7],
    "aircraft": ["4x3", "5x4", "6x4", "8x5", "8x6", "10x7", "12x8", "14x10", "16x12", "18x13"],
    "drone": ["3-1", "4-1", "5-1", "6-1", "7-1", "8-1", "9-1", "10-1", "11-1", "12-1"],
    # (map, agents): the knob of frozenlake_uniform_*.py is the number of agents
    "frozenlake": [
        ("4x4", "1"),
        ("8x8", "1"),
        ("16x16", "1"),
        ("32x32", "1"),
        ("4x4", "2"),
        ("5x5", "2"),
        ("6x6", "2"),
        ("7x7", "2"),
    ],  # fmt: skip
    "stock": ["1x2", "1x3", "1x4", "2x2", "1x6", "2x3", "3x2", "2x4", "4x2"],
    "sepsis": [2, 3, 4],
    "coffee": ["coffee", "coffee+snack"],
    "taxi": ["deterministic", "noisy"],
}


def observed_reward(truth) -> Reward:
    if not truth.hidden:
        return truth.reward
    obs = truth.learner_spec
    return obs.reward(all_states([obs.cards[v] for v in obs.state_vars]), truth.actions)


def oracle_solve(L, U, reward, gamma):
    solution = robust_vi.solve(IntervalMDP(L, U, reward), gamma)
    return solution.iterations, solution.iteration_s


def calibrate(domain: str) -> pd.DataFrame:
    module = importlib.import_module(f"{domain}_uniform_rmdp")
    grid = module.GRID
    gamma = grid.gamma
    rows = []
    process = psutil.Process()
    for candidate in CANDIDATES[domain]:
        size, knob = candidate if isinstance(candidate, tuple) else (candidate, grid.knobs[0])
        row = {"domain": domain, "size": str(size), "knob": str(knob), "gamma": gamma}
        gc.collect()
        before = process.memory_info().rss / 1e9
        start = time.monotonic()
        try:
            env = grid.make_env(size, knob, 0)
            truth = env.ground_truth()
            kernel = truth.observed_kernel()
        except (ModelTooLarge, MemoryError) as err:
            rows.append(row | {"fits": False, "why": f"true model: {err}"})
            print(f"  {domain} {size} ({knob}): the true model does not fit ({err})")
            break
        row |= {
            "states": truth.n_states,
            "observed_states": truth.learner_spec.n_states,
            "actions": len(truth.actions),
            "build_s": time.monotonic() - start,
            "build_gb": process.memory_info().rss / 1e9 - before,
        }
        reward = observed_reward(truth)
        L, U = np.clip(kernel - TOL, 0, 1), np.clip(kernel + TOL, 0, 1)
        out = run_limited(oracle_solve, L, U, reward, gamma, mem_gb=LIMITS.mem_gb, timeout_s=None)
        row |= {"oracle_rmdp": out.status, "oracle_rmdp_s": out.seconds, "oracle_rmdp_gb": out.peak_gb}
        if out.status == "ok":
            sweeps, sweep_s = out.value
            bound = iterations_bound(reward_scale(reward), 1e-7, gamma)
            row |= {"oracle_sweeps": sweeps, "sweep_s": sweep_s, "sweeps_bound": bound}
            row["solve_limit_s"] = max(SOLVE_FLOOR_S, SOLVE_FACTOR * sweep_s * bound)
        start = time.monotonic()
        value_iteration.solve(kernel, reward, gamma)
        row["oracle_mdp_s"] = time.monotonic() - start
        row["fits"] = out.status == "ok"
        rows.append(row)
        print(
            f"  {domain} {size} ({knob}): {row['observed_states']} observed states, "
            f"{row['actions']} actions; "
            f"build {row['build_s']:.1f} s +{row['build_gb']:.2f} GB; oracle -> IMDP {out.status} "
            f"{out.seconds:.1f} s ({row.get('oracle_sweeps')} sweeps) {out.peak_gb:.2f} GB; oracle -> MDP "
            f"{row['oracle_mdp_s']:.2f} s; solve limit of the others "
            f"{row.get('solve_limit_s', float('nan')):.0f} s "
            f"({row.get('sweeps_bound')} sweeps at the rate gamma)",
            flush=True,
        )
        del env, truth, kernel, L, U
        if not row["fits"] or out.seconds > MAX_ORACLE_S:
            break
    return pd.DataFrame(rows)


if __name__ == "__main__":
    domains = sys.argv[1:] or list(CANDIDATES)
    out = HERE.parents[1] / "results" / "trials" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    print(f"limits: {LIMITS}")
    for domain in domains:
        print(f"== {domain}")
        table = calibrate(domain)
        table.to_csv(out / f"{domain}.csv", index=False)
