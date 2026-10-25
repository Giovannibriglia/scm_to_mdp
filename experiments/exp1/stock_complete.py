"""Stock trading, complete grid: every size x coupling x behaviour policy x number of rollouts
(protocol.py: the claims, the loggers, the seeds, the shifts and the expectations common to every domain).

    size        sectors x stocks: 2x2, 2x3, 3x2, 3x3 (64, 256, 512, 4096 observed states), the instances of
                Schnitzer, Abate, Parker (AAAI 2026, App. E.4); 3x3 needs a server
    knob        scale: how much a sector's rising stocks lift each stock, P(rise) = 0.1 + scale * k / M;
                0.8 is theirs (and Strehl et al.'s), 0.4 a weaker coupling (harder to discover)
    policies    uniform, expert eps 0.05, expert eps 0, confounded (sells sector 0 during news w.p. 0.95)
    hidden      the news (src/scm_to_mdp/envs/stock_trading.py) lowers the rise probability of ONE stock,
                s0_0 (x 0.4), w.p. 0.1 in the data, 0.3 and 0.6 at deployment
    rollouts    10 ... 1600 (horizon 20, gamma 0.9), 5 seeds, the standard methods with the per-(s,a)
                product set (every variable is 0/1)

No trap (no end state, no sparse goal): the control domain of claim A, where the robust arms should
only pay for their guarantee.

Expectations for this domain, written 2026-10-24 before the runs:
    H1  structure pays off: small families per sector against 2^(N(M+1)) rows of the flat model (A);
    H2  no trap: the point models do as well as or better than the robust arms in true value; among
        the arms whose promise holds, the causal ones beat flat RMDP (A1);
    H3  scale 0.4: weaker dependence between stocks, so discovery misses edges at small budgets;
    H4  sector0_only and loss_aversion change which sectors to hold: rows the expert rarely visits
        (B1); confounded: the promise fails for s0_0 (control).

Run from the repository root (the 3x3 instances need about 8 GB per step):
    .venv/bin/python experiments/exp1/stock_complete.py                 # a new run
    .venv/bin/python experiments/exp1/stock_complete.py --resume        # continue the latest run
    .venv/bin/python experiments/exp1/stock_complete.py --report-only   # tables and figures only
    .venv/bin/python experiments/exp1/stock_complete.py --workers 4 --device cpu  # own plan
Output: results/exp1/stock_complete_<date>_<time>/.
"""

import dataclasses
import warnings
from pathlib import Path

from protocol import DEEP_BASELINES, DELTA, N_ROLLOUTS, SEEDS, SHIFT_HIDDEN, TRAIN_HIDDEN, policies

from scm_to_mdp.envs import StockTrading
from scm_to_mdp.tool import Grid, Limits, report_grid, run_folder, run_grid, standard_methods

warnings.filterwarnings("ignore", category=FutureWarning)

SIZES = ["2x2", "2x3", "3x2", "3x3"]  # sectors x stocks per sector
SCALES = [0.8, 0.4]
HORIZON, GAMMA = 20, 0.9
NEWS_EFFECT = 0.6
MODEL_GB = 12.0  # the true model of 3x3 with news (8192 states) needs about 7.5 GB to build
LIMITS = Limits(
    mem_gb=16.0,
    timeout_s={"discovery": 120.0, "selection": 120.0, "learning": 120.0, "solve": 600.0, "training": 1800.0},
)
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1" / "stock_complete"


def stock(size: str, scale, seed: int, news: float) -> StockTrading:
    sectors, stocks = (int(v) for v in str(size).split("x"))
    return StockTrading(sectors, stocks, seed, scale=float(scale), news=news, news_effect=NEWS_EFFECT,
                        max_model_gb=MODEL_GB)  # fmt: skip


def make_env(size: str, scale, seed: int) -> StockTrading:
    return stock(size, scale, seed, TRAIN_HIDDEN)


def shifts(size: str, scale, seed: int) -> dict:
    """The same market with more news (the learner never sees it)."""
    return {f"news={p:g}": stock(size, scale, seed, p) for p in SHIFT_HIDDEN}


GRID = Grid(
    name="Stock trading",
    make_env=make_env,
    sizes=SIZES,
    size_name="sectors x stocks",
    knobs=SCALES,
    knob_name="scale",
    policies=policies(),
    n_rollouts=N_ROLLOUTS,
    seeds=SEEDS,
    horizon=HORIZON,
    gamma=GAMMA,
    methods=standard_methods(max_parents=3, delta=DELTA, exact=True, deep=DEEP_BASELINES, state_parents=3),
    limits=LIMITS,
    workers="auto",
    out_dir=OUT_DIR,
    reward_transfer=True,
    shifts=shifts,
    constants={
        "news": TRAIN_HIDDEN,
        "shift_news": SHIFT_HIDDEN,
        "news_effect": NEWS_EFFECT,
        "model_gb": MODEL_GB,
    },
)

if __name__ == "__main__":
    out_dir, report_only = run_folder(OUT_DIR)
    grid = dataclasses.replace(GRID, out_dir=out_dir)
    if not report_only:
        run_grid(grid)
    report_grid(grid.out_dir, target=0.95, delta=DELTA)
