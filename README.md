# scm_to_mdp

Code and experiments for the SCM → MDP-family pipeline: from offline data to a policy with a
guarantee on its value. The pipeline learns the causal structure of the transitions from data,
learns probability intervals for each causal mechanism, compiles them into a robust (interval) MDP
and solves it. Probability sets come from the [NBN](https://github.com/Giovannibriglia/NeuralBayesianNetworks)
library (`nbn>=0.26`).

## Pipeline

```
env.sample ──► read_dataset ──► discovery ──► scm_learning ──► compilation ──► solving ──► evaluation
 (S0, S1)        Dataset          Graph        FactorBounds      IntervalMDP      Solution     (S9)
                                 (S2–S3)          (S4)              (S7)            (S8)
```

The data classes passed between steps are in `src/scm_to_mdp/core.py`.

## Layout

| Path | Content |
|---|---|
| `src/scm_to_mdp/core.py` | `Dataset`, `Graph`, `FactorBounds`, `IntervalMDP`, `Solution`, `Reward` (step reward + reward on arrival + terminal states) |
| `src/scm_to_mdp/data.py` | table of transitions → `Dataset` (the learner is told which columns are actions, that `x'` is the next value of `x`, the number of values of each variable, and the reward) |
| `src/scm_to_mdp/envs/` | factored environments with their exact model and behaviour policies, `env.sample(episodes, horizon, policy="random")`: `SysAdmin(n, seed, topology, repair_cost)` and `Sepsis(size)` (Oberst & Sontag 2019; `small` 72, `medium` 144, `full` 1440 states, or `custom`) |
| `src/scm_to_mdp/discovery/` | `bic_search`, `pcmci_plus`, `ges`, `oracle`; `cpdag_to_dag` = how one DAG is chosen from a class |
| `src/scm_to_mdp/scm_learning/` | `nbn_intervals`: Clopper–Pearson interval per factor row, union bound |
| `src/scm_to_mdp/compilation/` | `product_bounds`: factor intervals → interval MDP |
| `src/scm_to_mdp/solving/` | `value_iteration`, `robust_vi`, `vertex_robust_vi` (exact over factor boxes, 0/1 variables) |
| `src/scm_to_mdp/baselines/` | `flat_mdp`, `flat_rmdp`, `difarl` |
| `src/scm_to_mdp/evaluation.py` | V*, true value of a policy, promise, safety, structure score |
| `src/scm_to_mdp/utils/` | `limits` (memory and time limit per step), `parallel` (pool of runs) |
| `experiments/exp1/` | experiment 1, discovery algorithms and baselines: `sysadmin.py` (over N), `sepsis.py` (over sizes), `common.py` (step runner, tables, figure) |
| `tests/` | one test file per step, plus a smoke test of exp1 |
| `results/` | outputs (git-ignored) |

## Usage

```bash
uv sync --extra discovery                              # environment (uv)
.venv/bin/python -m pytest -q                          # tests
.venv/bin/python experiments/exp1/sysadmin.py              # -> results/exp1/sysadmin/
.venv/bin/python experiments/exp1/sepsis.py                # -> results/exp1/sepsis/
.venv/bin/python experiments/exp1/sepsis.py --plot-only    # tables and figure from the saved CSV
```

## Adding things

- **An environment:** a file in `envs/` that subclasses `Env` and brings its behaviour policies.
- **A method:** a file in the folder of its step, with `fit` / `learn` / `compile` / `solve`.
- **An experiment:** a folder `experiments/expK/` with one script per environment, each with the same
  sections as `exp1/sysadmin.py` (config, arms, pipeline, report); results go to `results/expK/<env>/`.
  An arm is a list of steps.
