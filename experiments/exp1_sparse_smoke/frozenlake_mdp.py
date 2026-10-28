"""Smoke run of exp1_sparse/frozenlake_mdp.py: the same methods on a tiny grid (experiments/smoke.py).

    .venv/bin/python experiments/run.py exp1_sparse_smoke frozenlake mdp
Output: results/exp1_sparse_smoke/frozenlake_mdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("frozenlake", "mdp")

if __name__ == "__main__":
    main(GRID)
