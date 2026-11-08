"""Smoke run of exp2/frozenlake_imdp.py: the same methods on a tiny grid (experiments/smoke.py), in minutes.

    .venv/bin/python experiments/run.py exp2_smoke frozenlake imdp
Output: results/exp2_smoke/frozenlake_imdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("frozenlake", "imdp")

if __name__ == "__main__":
    main(GRID)
