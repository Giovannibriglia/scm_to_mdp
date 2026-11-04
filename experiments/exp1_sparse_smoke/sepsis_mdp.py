"""Smoke run of exp1_sparse/sepsis_mdp.py: the same methods on a tiny grid (experiments/smoke.py).

    .venv/bin/python experiments/run.py exp1_sparse_smoke sepsis mdp
Output: results/exp1_sparse_smoke/sepsis_mdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("sepsis", "mdp")

if __name__ == "__main__":
    main(GRID)
