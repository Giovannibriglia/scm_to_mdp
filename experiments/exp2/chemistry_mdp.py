"""Experiment 2, Chemistry: ours (PCMCI+ / DYNOTEARS) vs DiFaRL, FOCUS, SL-SPIBB and offline DQN;
our arms with point estimates and value iteration (-> MDP, no guarantee).

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/exp2/chemistry_mdp.py                # a new run
    .venv/bin/python experiments/exp2/chemistry_mdp.py --resume       # continue the latest
    .venv/bin/python experiments/exp2/chemistry_mdp.py --report-only  # tables and figures
    .venv/bin/python experiments/exp2/chemistry_mdp.py --workers 6    # own plan
Output: results/exp2/chemistry_mdp_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("chemistry", "mdp")

if __name__ == "__main__":
    main(GRID)
