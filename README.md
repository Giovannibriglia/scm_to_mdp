# Learning Robust Markov Decision Processes from Data through Structural Causal Models

Code and experiments for the SCM → MDP-family pipeline: from offline data to a policy with a
guarantee on its value. The pipeline learns the causal structure of the transitions from data,
learns probability intervals for each causal mechanism, turns them into a robust (interval) MDP
and solves it. Probability sets come from the [NBN](https://github.com/Giovannibriglia/NeuralBayesianNetworks)
library (`nbn>=0.26`).

## Quick start

```bash
uv sync --extra discovery                                   # environment (uv; Python 3.12, from uv.lock)
.venv/bin/python experiments/exp1/exp_smoke.py              # is everything installed? (~15 minutes)
.venv/bin/python -m pytest -q                               # the tests (6 minutes on a recent laptop, ~20 on an older server)
.venv/bin/python experiments/exp1/sysadmin_complete.py      # experiment 1, SysAdmin grid (hours)
```

Tested on macOS (arm64) and Linux (Ubuntu 20.04, x86_64). Without `uv`: `pip install --user uv`.

### How a run uses the machine

The scripts run as they are on a laptop or a server: at start they look at the machine, choose, print
the choice and save it in each setting's `meta.json` (`"machine"`; `src/scm_to_mdp/utils/compute.py`).

| | default (`auto`) | override |
|---|---|---|
| parallel jobs | one per physical core, but at most 80% of the RAM / `LIMITS.mem_gb` (8 GB per step) | `--workers N` or `SCM_TO_MDP_WORKERS=N` |
| offline DQN / CQL training | CUDA if a test step runs on it, else the CPU; never Apple's MPS (slower here) | `--device cpu\|cuda\|cuda:N\|mps` or `SCM_TO_MDP_DEVICE` |
| threads | one per process (torch and BLAS) | `SCM_TO_MDP_THREADS` |

- All the settings of a grid share one pool of workers: a free worker takes the next (size, seed) of
  any setting, so a machine with more cores finishes sooner. The rows do not depend on the number
  of workers (tested: `tests/tool/test_compute.py`).
- The RAM rule guarantees that a step is never killed because other steps used the memory, so a
  `memout` means the same on every machine. On a 34 GB laptop it gives 3 workers; lower
  `LIMITS.mem_gb` for more (the largest step seen so far used 0.5 GB).
- Only offline DQN / CQL use a GPU: they take most of the run time, everything else is counting work
  on the CPU. On CUDA the training step is recorded once as a CUDA graph and replayed (5x faster
  alone on a TITAN X), and a training waits for the GPU by sleeping, so ten trainings sharing one
  GPU use about 1 s of CPU each instead of a full core. Measured on 2026-10-22 (4,000 steps, two Sepsis datasets): Apple M5, CPU
  4.7 s vs MPS 6.1 s; Xeon E5-2640 v4 + TITAN X, ten trainings at once: CPU 60 s, CUDA 38-40 s.
- On Linux x86_64, `uv.lock` installs PyTorch built for CUDA 12.6 (`pyproject.toml`): it works with
  NVIDIA drivers from CUDA 12.x on and with GPUs from Pascal to Hopper. For Blackwell GPUs (RTX 50xx,
  B200) change `cu126` to `cu128` there and run `uv lock`.
- A run's numbers are reproducible on the same machine and device. A GPU and a CPU training agree up
  to float rounding, which a DQN amplifies (the greedy action differs in about 10-20% of the states
  in our check), so do not mix devices within one run; `meta.json` records the device.
- Stopping a run (IDE stop button, Ctrl+C, kill) stops its workers and their steps too; continue it
  with `--resume`.

## Experiment 1: complete grids

`experiments/exp1/` holds one script per domain (`*_complete.py`) and the protocol they share
(`protocol.py`). The protocol states what the grids test and how:

- **A, sample efficiency:** an RMDP learned through causality (discovery + interval learning) reaches
  a near-optimal policy with a promise that holds at fewer rollouts than flat RMDP and the baselines.
  Measured by `rollouts_needed`, the validity-gated comparison and the certificate V_opt − V_rob.
  Against the point models it is reported apart, in true value and in guaranteed value.
- **B, transfer:**
  - B1, new rewards on the same dynamics (`results_transfer.csv`), read against the data's coverage
    of each task's rows: `cov_joint_k − cov_factor_k` are the rows only a factored model can estimate;
  - B2, a dynamics shift (`results_shift.csv`): the hidden variable is more frequent at deployment.
    This is robustness, not causal transfer; a single-mechanism change with new data is experiment 2.
- **C, understanding:** the learned model against the true kernel under the logged visits, the best
  policy's occupancy and uniform weights (`tv_*`, `kl_*`, `contain_*`, `width_*`), and the structure
  (SHD, SID, action recall).
- **Control:** the confounded logger. Every model-based promise is expected to fail there, so it is
  not evidence for A–C.

Every grid runs every size × knob × logger × number of rollouts × 5 seeds with the standard methods
(`standard_methods(..., deep=True)`):

- **ours:** BIC / PCMCI+ / GES / DYNOTEARS (Pamfil et al., AISTATS 2020) → a robust MDP over the box
  of the factor intervals. Also BIC → RMDP with widened families (S4,
  `scm_learning/nbn_intervals.py`): each parent set plus the 2 inputs BIC ranks next, either as the
  hull of the families' intervals ("hull") or as one padded family ("padded"). Finally, the same
  discoveries → MDP: point estimates and value iteration.
- **the references:**
  - true graph → RMDP, the true graph with the box (what discovery costs);
  - on 0/1 domains, also true graph → RMDP (per-(s,a) product), robust VI over the products of the
    factor intervals (Schnitzer, Abate & Parker, AAAI 2026). It is exact per (s, a), but nature still
    picks every (s, a) independently. The pair measures the cost of the box (`box_looseness.csv`);
  - true graph → MDP.
- **the controls** (`discovery/controls.py`), wrong graphs of a known kind, given like the true one:
  - the logger-Markov graph: the smallest parent sets that fit the logger's distribution. It drops
    the parents a near-deterministic logger makes redundant;
  - a random graph matched in free parameters;
  - the true graph + 1 parent (fragmentation only).
- **baselines:**
  - flat RMDP (one interval per transition) and flat MDP (the plug-in estimate);
  - DiFaRL (Ehrlichman et al., CLeaR 2026);
  - flat SPIBB (Laroche et al., ICML 2019) and SL-SPIBB (Simão & Spaan, IJCAI 2019);
  - FOCUS (Zhu et al., 2025), in a discrete version;
  - offline DQN and offline CQL (Kumar et al., NeurIPS 2020).

Every baseline's choices and fixed parameters (and how a pilot chose CQL's α and FOCUS's λ) are in the
docstring of its module in `src/scm_to_mdp/baselines/`. `protocol.py` states the expectations common to
every domain, and each script states its own (H1–H4). Both were written before any run, with their
reasons from the literature.

| Script | Size | Knob | Hidden variable (never in the data; one child) |
|---|---|---|---|
| `sysadmin_complete.py` | N = 2–10 machines | ring / bidirectional / star / ring+crash | a storm on machine 0 |
| `sepsis_complete.py` | vitals: 2 / 3 / 4 | diabetes: absent / observed / hidden | diabetes (persistent: the limit case) |
| `chemistry_complete.py` | objects: 4–7 | graph density: 0 / 0.4 / 1 | a catalyst of one reaction |
| `aircraft_complete.py` | 8x6, 12x8, 15x15 | proximity on / off | turbulence on the own altitude |
| `drone_complete.py` | side 3–6, 1 or 2 drones | gust: calm / gust | a gust on drone 0's y |
| `stock_complete.py` | 2x2 … 3x3 sectors x stocks | coupling 0.8 / 0.4 | news on stock s0_0 |
| `frozenlake_complete.py` | 4x4 / 8x8, 1 or 2 agents | wind: calm / wind | wind on agent 0's row |
| `coffee_complete.py` | coffee, coffee+snack | – | none (the readable graph; a 0/1 domain) |
| `taxi_complete.py` | deterministic / noisy | – | none (the limit case: context-specific structure) |

The loggers of every grid (`Env.behaviour`, exact tables over the full state):
- **uniform:** every action equally likely;
- **expert ε = 0.05:** the best policy over the observed state, with a uniform action w.p. 0.05;
- **expert ε = 0:** the thinnest coverage;
- **confounded:** the expert, but when the hidden variable is on it takes one action w.p. 0.95.

A hidden variable fresh at every step is on w.p. 0.1 in the data and w.p. 0.3 / 0.6 at deployment
(the shift). Every row also holds the logger's coverage (`C_star`, `C_all_bound`, `unvisited_pairs`)
and confounding bias (`bias_max`, `bias_mean`), computed exactly in `scm_to_mdp/behaviour.py`, and
the true value of the behaviour policy (`V_logger`, the reference for SPIBB).

```bash
.venv/bin/python experiments/exp1/sepsis_complete.py                 # a new run
.venv/bin/python experiments/exp1/sepsis_complete.py --resume        # continue the latest run
.venv/bin/python experiments/exp1/sepsis_complete.py --report-only   # tables and figures of the latest run
.venv/bin/python experiments/exp1/calibration.py                     # 100 seeds on small settings: does 1 - delta hold?
.venv/bin/python experiments/exp1/claims_report.py results/exp1/<run>  # claim-by-claim figures and tables (summary/claims/)
```

Every run gets its own folder, `results/exp1/<name>_<date>_<time>/`, so no run overwrites another;
`--resume FOLDER` and `--report-only FOLDER` pick an older one (`tool/runs.py`). A run writes
`results.csv` row by row, so a stopped run resumes where it stopped (with `--resume`). The grid report
(`summary/`) adds three tables:
- `validity.csv`: the share of seeds whose promise held, with a Clopper–Pearson interval and a verdict
  against 1 − δ;
- `coverage.csv`: the coverage of the best policy's rows;
- `box_looseness.csv`.

For the first seed, every method's learned model is saved at every size and at the largest number of
rollouts, in `<setting>/models/<size>/n=<n>/<method>/` (`tool/store.py`):
- `model.npz`: the model the solver used, the policy and its promise (`load_model(folder)` reads it
  back);
- `cbn.json`: the learned causal Bayesian network (graph, counts, estimates and intervals of every
  conditional distribution, with the true parents);
- `cbn.png`: the learned graph against the true one;
- `mdp.png`, for problems with at most 16 states: the chain picture of the MDP the model induces.
  Nodes are states, edges are actions with one colour per action, solid for the likeliest next
  states and dashed for the less likely ones. A one-agent grid world is laid out on its grid, and
  the true MDP is drawn next to it.

The data determine a causal Bayesian network, not a structural causal model: many SCMs induce the same
network, and the network is what the pipeline uses. To change a grid, edit the constants at the top of
its script (`SIZES`, `N_ROLLOUTS`, `LIMITS`) or of `protocol.py`. If you change anything other than
adding sizes, seeds or budgets, use a new output folder: otherwise the old rows are kept and mixed with
the new ones.

## Pipeline steps

The code names the steps of the pipeline S0–S9: S0 the unknown system (an environment and its true
model), S1 data collection, S2 causal discovery, S3 hypotheses (here one graph, chosen by a fixed
rule), S4 parameter learning (probability intervals per factor), S5–S6 merging hypotheses and the
decision criterion over several of them (not used here), S7 compilation to an MDP model, S8 solving
(exact, or on a relaxation), S9 checking the policy and its promise against the true model.

## The tool in three parts

```
            ┌───────────── 1. learn + solve (S1–S8) ─────────────┐
DataFrame ─▶│ spec.read(df) → learn(data, method) → solve(...)    │─▶ policy + promise
            └─────────────────────────────────────────────────────┘
            ┌──────────────── 2. evaluate (S9) ───────────────────┐
            │ evaluate(result, data, truth)   truth: MDPGroundTruth│─▶ true value, regret, validity, SHD/SID
            └─────────────────────────────────────────────────────┘
            ┌──────────────── 3. report ──────────────────────────┐
            │ run_experiment(Experiment(...)) → report(out_dir)    │─▶ results.csv, tables.md, figures/
            └─────────────────────────────────────────────────────┘
```

**1. Learn and solve.** Learning sees only the data and a `DataSpec`, i.e. what the learner is
told: the state variables, the action columns, the number of values of each variable and the
reward. Every column is one dimension of the state or of the action (categorical, with a "do
nothing" value where the domain has one); no done or terminated variable (the end of an episode is a
set of states of the reward). A `Method` names one choice per step:

```python
from scm_to_mdp.tool import Method, learn, solve

method = Method("causal", "box_robust_vi",           # learner, solver
                discovery="pcmci+",                  # S2: bic | pcmci+ | ges | oracle
                selection="bic_best",                # S3: how ONE graph is chosen (oracle: "given")
                discovery_options={"alpha": 0.01},
                learning_options={"delta": 0.05})    # S4: confidence of the intervals
data = spec.read(df)                                 # df: columns x, a, x'; index (episode, t)
result = solve(learn(data, method), data, gamma=0.9)
result.solution.pi, result.solution.V                # the policy and its promise (a lower bound)
```

| learner | what it learns | solvers | role |
|---|---|---|---|
| `causal` | S2 discovery → S3 selection → S4 intervals per factor: a factored interval MDP | `box_robust_vi` (relax to the box, Schnitzer et al.'s IA), `vertex_robust_vi` (exact, 0/1 variables) | ours; with `discovery="oracle"` (the true graph given) a reference: true graph → RMDP / MDP |
| `flat_rmdp` | one interval per (s, a, s′) | `robust_vi` | baseline |
| `flat_mdp` | the plug-in MDP from counts (no guarantee) | `vi`, `robust_vi` | baseline |
| `difarl` | DiFaRL's action groups (model-free, no guarantee) | `fqi` | baseline |
| `offline_dqn`, `offline_cql` | a Q-network trained on the logged data (model-free, no guarantee) | `greedy_q` | baseline |
| `spibb`, `sl_spibb` | the estimated MDP, flat or with SL structure learning, and the bootstrapped pairs | `spibb` (a stochastic policy) | baseline |
| `focus` | FOCUS's CI-test graph, the factored model with MOPO's penalised reward | `vi` | baseline |

**2. Evaluate.** The ground truth is used only here. `MDPGroundTruth` holds the `DataSpec`
and the true `P [S, A, S']`, the action columns of every true action, the start distribution and,
optionally, the true parents:

```python
from scm_to_mdp.tool import DataSpec, MDPGroundTruth, evaluate

truth = MDPGroundTruth(spec=DataSpec(state_vars, action_vars, cards, reward),
                       P=P, actions=actions, start=start, true_parents=None)
metrics = evaluate(result, data, truth, gamma=0.9)   # promise, true, regret, optimal_share, gap, safe, shd, ...
```

States are every combination of the state variables in C order (`scm_to_mdp.core.all_states`).
For a factored model use `MDPGroundTruth.from_factored(spec, next_distributions, actions, start)`.
Our environments give `env.spec()` and `env.ground_truth()`.
**Hidden variables:** pass `observed=DataSpec(...)` with a subset of the state variables (e.g. a
confounder kept out of the data). A policy over the observed state is evaluated exactly in the full
model, and V* is that of an agent who also sees the hidden variables. A dense `P` grows as $S^2$: building one
that needs more than 4 GB is refused (`ModelTooLarge`). Without a truth, `evaluate(result, data)`
returns only the promise.

**3. Report.** An `Experiment` sweeps problem sizes × numbers of rollouts × seeds (with nested
datasets: smaller budgets use the first rollouts of one table). Each step runs in its own process
with a memory limit and, for discovery, selection, parameter learning, solving and training, a time limit. Rows are written to `results.csv` as each (size, seed) finishes, and
a second run resumes. `report(out_dir)` writes:

- `figures/rollouts=<n>/{performance,validity,structure,time,memory}` over the problem size, one
  folder per number of rollouts;
- `figures/sample_efficiency`: x = number of states, y = rollouts needed for a policy that is at
  least 95% optimal ($V^\pi \ge 0.95\,V^\star_{obs}$ at the start, at that number of rollouts and every
  larger one; median over seeds, quartile bars); $V^\star_{obs}$ is the best policy over the observed
  state (= $V^\star$ when nothing is hidden);
- `tables.md` with every table.

A grid (`Grid`, `run_grid`, `report_grid` in `tool/grid.py`, `tool/report_grid.py`) runs one
Experiment per (knob, behaviour policy) and writes `summary/`, built level by level from the most
atomic records, every level saved as CSV: `runs.csv` (one row per run), `edges.csv` (one row per
run and edge: in the true graph? in the learned one?), `behaviour.csv` (one row per logger), then
`rollouts_needed_by_seed.csv`, `cells.csv` (mean / s.e. / min / max over seeds), `rollouts_needed.csv`,
`promise_held.csv`, `best_method.csv` and `method_rank.csv`; every figure has the exact numbers it
draws in `figure_data/`. `summary/README.md` lists every file.

Baselines are dashed, ours solid, the references (true graph) dotted; methods without a guarantee (X → MDP, true graph → MDP) dash-dot.

## Layout

| Path | Content |
|---|---|
| `src/scm_to_mdp/tool/` | the tool: `methods.py` (Method, learn, solve), `evaluate.py`, `sweep.py` (Experiment, run_experiment), `report.py`; grids: `grid.py` (Grid, Policy, run_grid), `report_grid.py`; `store.py` (saved models, `load_model`), `runs.py` (one folder per run) |
| `src/scm_to_mdp/behaviour.py` | behaviour policies as exact tables; data occupancy, coverage ($C^\star$) and confounding bias of a logger |
| `src/scm_to_mdp/core.py` | data passed between steps: `Dataset`, `Candidates`, `Graph`, `FactorBounds`, `IntervalMDP`, `ActionGroups`, `Solution`, `Reward` |
| `src/scm_to_mdp/data.py` | `DataSpec`: table of transitions → `Dataset`, with checks |
| `src/scm_to_mdp/ground_truth.py` | `MDPGroundTruth`, `from_factored`, the size check |
| `src/scm_to_mdp/envs/` | `SysAdmin(n, seed, topology, repair_cost)`, `Sepsis(size, hide_diabetes=...)`, `Coffee(size)`, `Chemistry(density, seed)`, `Taxi(size)`, with `spec()`, `ground_truth()`, `sample(rollouts, horizon, policy)`, `hide(df)` |
| `src/scm_to_mdp/discovery/` | S2: `bic_search`, `pcmci_plus`, `ges`, `dynotears`, `oracle` → `Candidates`; `controls` (the control graphs of the evaluation) |
| `src/scm_to_mdp/selection/` | S3: `bic_best` (best BIC in the class; ties: transition BIC, then a fixed order), `given` |
| `src/scm_to_mdp/scm_learning/` | S4: `nbn_intervals`, a Clopper–Pearson interval per factor row with a union bound |
| `src/scm_to_mdp/compilation/` | S7: `product_bounds`, factor intervals → their box, an interval MDP (the relaxation the box solvers use in S8) |
| `src/scm_to_mdp/solving/` | S8: `value_iteration`, `robust_vi`, `vertex_robust_vi` (`tol` = distance to the fixed point, `stopping.py`) |
| `src/scm_to_mdp/baselines/` | `flat_mdp`, `flat_rmdp`, `difarl`, `offline_q` (offline DQN / CQL), `spibb` (flat and SL-SPIBB), `focus` |
| `src/scm_to_mdp/evaluation.py` | S9: V*, the true value of a policy, promise, safety, SHD and SID, the data's coverage of a task's rows, the learned model against the true kernel |
| `src/scm_to_mdp/plotting/` | result figures (from `results.csv`) and pictures of MDPs |
| `experiments/exp1/` | experiment 1: the protocol (`protocol.py`), one complete grid per domain (table above), `calibration.py` (100 seeds), `claims_report.py` (the figures and tables per claim: A, A3, discovery, B1, B2, C) and `exp_smoke.py` (a check of the installation) |
| `tests/` | one folder per part (`envs/`, `data/`, `discovery/`, `learning/`, `solving/`, `baselines/`, `evaluation/`, `plotting/`, `tool/`); `tests/tool/test_experiment_scripts.py` checks every grid against the protocol |
| `results/` | outputs (git-ignored) |

## Adding things

- **Your own problem:** write `make_env(size, seed)`, returning an object with `ground_truth()`, and
  `collect(env, n_rollouts)`, returning the table (`tool/sweep.py`, `Experiment`). Or subclass `Env`
  (`envs/base.py`) and implement `next_distributions`, `action_columns`, `reward`, `true_parents`
  and `start_distribution`.
- **A method for a step:** a module in the folder of its step with the same signature as its
  neighbours, plus its name in `tool/methods.py`.
- **An experiment:** a folder `experiments/expK/` with one config script per environment, like
  `exp1/sysadmin_complete.py`; results go to `results/expK/<name>/`.
