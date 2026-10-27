"""Experiment 1, Stock trading: causal discovery vs the full factored graph, the true graph and flat;
our arms with interval sets and robust value iteration (-> IMDP, a promise).

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1/stock_imdp.py                # a new run
    .venv/bin/python experiments/exp1/stock_imdp.py --resume       # continue the latest
    .venv/bin/python experiments/exp1/stock_imdp.py --report-only  # tables and figures
    .venv/bin/python experiments/exp1/stock_imdp.py --workers 6    # own plan
Output: results/exp1/stock_imdp_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("stock", "imdp")

if __name__ == "__main__":
    main(GRID)
