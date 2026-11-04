"""Smoke run of exp1_sparse/taxi_mdp.py: the same methods on a tiny grid (experiments/smoke.py).

    .venv/bin/python experiments/run.py exp1_sparse_smoke taxi mdp
Output: results/exp1_sparse_smoke/taxi_mdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("taxi", "mdp")

if __name__ == "__main__":
    main(GRID)
