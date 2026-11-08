"""Smoke run of exp2/coffee_mdp.py: the same methods on a tiny grid (experiments/smoke.py), in minutes.

    .venv/bin/python experiments/run.py exp2_smoke coffee mdp
Output: results/exp2_smoke/coffee_mdp_<date>_<time>/ (summary/: tables and figures).
"""

from protocol import experiment_grid, main

GRID = experiment_grid("coffee", "mdp")

if __name__ == "__main__":
    main(GRID)
