# scm_to_mdp

Code and experiments for the SCM → MDP-family pipeline: from offline data collected by several
behaviour policies to a policy with a guarantee. Causal structure learning and probability sets
over factored transition models come from the [NBN](https://github.com/Giovannibriglia/NeuralBayesianNetworks)
library (`pip install "nbn>=0.24"`); this repository holds the pipeline glue, the benchmark
environments and the experiments of the paper.

**Status:** under construction.

| Path | Content |
|---|---|
| `src/scm_to_mdp/` | pipeline glue (steps S0–S9) on top of NBN |
| `envs/` | benchmark environments (stage 1: SysAdmin, stock trading) |
| `experiments/<name>/` | one folder per experiment: `config.yaml`, `run.py`, `README.md` (claim → table / figure) |
| `tests/` | tests |
| `results/` | outputs of runs (git-ignored) |

Environments and dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).
