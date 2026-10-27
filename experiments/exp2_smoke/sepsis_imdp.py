"""Smoke run of exp2/sepsis_imdp.py: the same methods on a tiny grid (experiments/smoke.py), in minutes.

    .venv/bin/python experiments/run.py exp2_smoke sepsis imdp
Output: results/exp2_smoke/sepsis_imdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("sepsis", "imdp")

if __name__ == "__main__":
    main(GRID)
