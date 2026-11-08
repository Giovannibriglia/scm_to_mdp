"""Smoke run of exp1/stock_mdp.py: the same methods on a tiny grid (experiments/smoke.py), in minutes.

    .venv/bin/python experiments/run.py exp1_smoke stock mdp
Output: results/exp1_smoke/stock_mdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("stock", "mdp")

if __name__ == "__main__":
    main(GRID)
