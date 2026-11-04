"""Experiment 1 (sparse graph), Coffee robot: sparse -> MDP (and the oracle) on experiment 1's data;
experiments/merge_runs.py joins it to experiment 1's run.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp1_sparse/coffee_mdp.py                # a new run
    .venv/bin/python experiments/exp1_sparse/coffee_mdp.py --resume       # continue the latest
    .venv/bin/python experiments/exp1_sparse/coffee_mdp.py --report-only  # tables and figures
    .venv/bin/python experiments/exp1_sparse/coffee_mdp.py --workers 6    # own plan
Output: results/exp1_sparse/coffee_mdp_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("coffee", "mdp")

if __name__ == "__main__":
    main(GRID)
