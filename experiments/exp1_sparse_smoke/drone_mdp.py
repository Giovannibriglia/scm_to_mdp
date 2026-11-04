"""Smoke run of exp1_sparse/drone_mdp.py: the same methods on a tiny grid (experiments/smoke.py).

    .venv/bin/python experiments/run.py exp1_sparse_smoke drone mdp
Output: results/exp1_sparse_smoke/drone_mdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("drone", "mdp")

if __name__ == "__main__":
    main(GRID)
