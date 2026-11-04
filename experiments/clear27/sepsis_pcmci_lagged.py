"""clear27, Sepsis: pool 3, PCMCI+ without same-step links -> point 2; the oracle as yardstick.

Methods, data and evaluation: protocol.py and experiments/common.py; the domain: experiments/domains.py.
Run from the repository root (a stopped run resumes setting by setting):
    .venv/bin/python experiments/clear27/sepsis_pcmci_lagged.py                # a new run
    .venv/bin/python experiments/clear27/sepsis_pcmci_lagged.py --resume       # continue the latest
    .venv/bin/python experiments/clear27/sepsis_pcmci_lagged.py --report-only  # tables and figures
Output: results/clear27/sepsis_pcmci_lagged_<date>_<time>/.
"""

from protocol import experiment_grid, main

GRID = experiment_grid("sepsis", "pcmci_lagged")

if __name__ == "__main__":
    main(GRID)
