"""Stock trading, uniform logger: every causal discovery -> MDP method against the baselines without a
guarantee (S2-S9).

    size        sectors x stocks: 2x2, 2x3, 3x2 (64, 256, 512 observed states; experiment 1's 3x3 needs a
                server)
    knob        scale 0.8: P(rise) = 0.1 + 0.8 k / M (Schnitzer, Abate, Parker's; experiment 1 also runs 0.4)
    logger      uniform (every action tried in every visited state: no positivity problem)
    rollouts    50, 250, 750 (horizon 20, gamma 0.9), 5 seeds (protocol.py)
    methods     BIC / PCMCI+ / GES / DYNOTEARS -> MDP (point estimates, value iteration; no guarantee),
                flat MDP, DiFaRL, FOCUS, offline DQN, offline CQL, and true graph -> MDP as the reference
                (protocol.py, MDP_METHODS)
    evaluation  only in the MDP the data come from (no reward transfer, no dynamics shift)
    hidden      as in experiment 1: the news lowers the rise probability of stock s0_0 (x 0.4) w.p. 0.1 at
                every step; never in the data

Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/trials/stock_uniform_mdp.py                # a new run
    .venv/bin/python experiments/trials/stock_uniform_mdp.py --resume       # continue the latest
    .venv/bin/python experiments/trials/stock_uniform_mdp.py --report-only  # tables and figures
    .venv/bin/python experiments/trials/stock_uniform_mdp.py --workers 6    # own plan
Output: results/trials/stock_uniform_mdp_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DELTA, MDP_METHODS, N_ROLLOUTS, POLICIES, SEEDS, TRAIN_HIDDEN, methods

from scm_to_mdp.envs import StockTrading
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["2x2", "2x3", "3x2"]  # sectors x stocks per sector
SCALE = 0.8
HORIZON, GAMMA = 20, 0.9
NEWS_EFFECT = 0.6
MODEL_GB = 12.0
LIMITS = Limits(
    mem_gb=2.5,  # peak without BIC 0.5 GB (offline CQL, 3x2, 750 rollouts)
    timeout_s={"discovery": 120.0, "selection": 120.0, "learning": 120.0, "training": 1800.0, "solve": 600.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "trials" / "stock_uniform_mdp"
METHODS = methods(MDP_METHODS)


def make_env(size: str, scale, seed: int) -> StockTrading:
    sectors, stocks = (int(v) for v in str(size).split("x"))
    return StockTrading(sectors, stocks, seed, scale=float(scale), news=TRAIN_HIDDEN, news_effect=NEWS_EFFECT,
                        max_model_gb=MODEL_GB)  # fmt: skip


GRID = Grid(
    name="Stock trading",
    make_env=make_env,
    sizes=SIZES,
    size_name="sectors x stocks",
    knobs=[SCALE],
    knob_name="scale",
    policies=POLICIES,
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=METHODS,
    limits=LIMITS,
    workers="auto",
    out_dir=OUT_DIR,
    reward_transfer=False,
    shifts=None,
    constants={"news": TRAIN_HIDDEN, "news_effect": NEWS_EFFECT},
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)  # a new <OUT_DIR>_<date>_<time>, or --resume / --report-only
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
