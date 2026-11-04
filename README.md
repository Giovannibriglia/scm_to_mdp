# Learning Robust Markov Decision Processes from Data through Structural Causal Models

Code and experiments for the SCM → MDP-family pipeline: from offline data to a policy with a
guarantee on its value. The pipeline learns the causal structure of the transitions from data,
learns probability intervals for each causal mechanism, turns them into a robust (interval) MDP
and solves it. Probability sets come from the [NBN](https://github.com/Giovannibriglia/NeuralBayesianNetworks)
library (`nbn>=0.26`).

## Quick start

```bash
uv sync --extra discovery          # environment (uv; Python 3.12, from uv.lock)
.venv/bin/python -m pytest -q      # the tests (about 2 minutes)
```

Tested on macOS (arm64) and Linux (Ubuntu 20.04, x86_64). Without `uv`: `pip install --user uv`.

## Running the experiments

From the repository root. A script is `<experiment> <domain> <kind>`: experiment `exp1` or `exp2`,
domain one of `sysadmin chemistry aircraft drone frozenlake stock sepsis coffee taxi`, kind `imdp`
(with a promise) or `mdp` (without).

**One domain** (e.g. SysAdmin, IMDP arms of experiment 1):

```bash
.venv/bin/python experiments/run.py exp1 sysadmin imdp            # run
.venv/bin/python experiments/run.py exp1 sysadmin imdp --resume   # continue a stopped run
.venv/bin/python experiments/run.py exp1 sysadmin imdp --show     # print what it would run, without running
```

**A whole experiment** (every domain, both kinds, one after the other; logs in `results/<experiment>/logs/`):

```bash
bash experiments/run_chain.sh exp1                     # all of experiment 1
bash experiments/run_chain.sh exp2                     # all of experiment 2
bash experiments/run_chain.sh exp1 sysadmin taxi       # only these domains
RESUME=1 bash experiments/run_chain.sh exp1            # continue a stopped chain
SHOW=0 bash experiments/run_chain.sh exp1              # log only (default: progress also shown here)
```

**A quick check first** (minutes): `exp1_smoke` and `exp2_smoke` run the same scripts on a tiny grid
(2 smallest sizes, 100 / 500 / 2,000 transitions, 3 seeds; `experiments/smoke.py`) to check that everything
runs and to see the final tables and figures. Their numbers are not results.

```bash
.venv/bin/python experiments/run.py exp1_smoke sysadmin imdp      # one domain
bash experiments/run_chain.sh exp1_smoke                          # every domain
```

Results go to `results/<experiment>/<domain>_<kind>_<date>_<time>/` (tables and figures in `summary/`).
`.venv/bin/python experiments/run.py --list` lists every script; `--workers N` sets the parallel jobs.

**Arms added after a run started** go in their own run on the same grids and are joined afterwards.
Experiment 1's sparse graph (`sparse -> IMDP / MDP`: each next-state variable gets itself and its own
action variables; `discovery/sparse.py`) is `exp1_sparse` (smoke: `exp1_sparse_smoke`):

```bash
RESUME=1 bash experiments/run_chain.sh exp1_sparse                                       # the extra arms
.venv/bin/python experiments/merge_runs.py results/exp1 results/exp1_sparse --out results/exp1_merged
```

`merge_runs.py` joins a pair only if the grids, the protocol and the data are the same (the oracle, run
in both, must have the same reference values), keeps the first run's rows for shared methods, and
rebuilds every table and figure in the new folder; the two runs are left as they are.

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

## Experiments 1 and 2

Both experiments share the data and the evaluation (`experiments/common.py`) and the nine domains
(`experiments/domains.py`: SysAdmin, Chemistry, Aircraft, Drone, Frozen Lake, Stock trading, Sepsis,
Coffee robot, Taxi; sizes calibrated so that the oracle solves each within 8 GB):

- **data:** a uniform-random logger; budgets of 10, 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k,
  50k and 100k transitions per dataset (the first n logged transitions of one table, so the datasets
  are nested; the episode where the budget ends is cut); in domains with terminal states every episode
  runs until the task ends, in the others it resets after 64 steps (256 in Coffee, whose logger
  forgets the start more slowly; `experiments/domains.py`); 10 seeds;
- **evaluation:** in the MDP the data come from, with the same reward, at the nominal start and from 20
  random starts; the value of a policy is reported normalised between the random policy (0%) and the
  optimal one over the observed state (100%);
- **summary over seeds:** the interquartile mean (IQM) with a 95% bootstrap interval (Agarwal et al.,
  NeurIPS 2021; `plotting/stats.py`); rates (how often a promise held) as plain shares; the
  transitions needed for a 95%-optimal policy as the median with its exact 95% interval from order
  statistics (some seeds may never get there; with 10 seeds the 2nd and 9th smallest values);
- **budget:** 8 GB per step; solving limited to 3x the oracle's work (the oracle is the true observed
  MDP, solved first on every dataset);
- **robustness of a promise** (`src/scm_to_mdp/robustness.py`): every check rests on one event, "the
  learned set contains the true kernel", which depends on the transitions only. So on that event the
  promise is a lower bound for every reward and every policy at once, and each check below must hold:
  the value bracket V_lo ≤ V^π ≤ V_up (the policy evaluated against an adversarial and a cooperative
  nature), the optimistic bound V*_obs ≤ V_opt, a one-sweep check that the promise is below its
  policy's robust value (`cert_residual` ≥ 0), and, in domains with failure states (Frozen Lake's holes,
  Sepsis's death), an upper bound on the probability of failing within 256 steps. `summary/robustness.csv`
  counts as `bugs` the runs whose set contains the truth while a check fails (0 expected);
- **many rewards, experiment 1 on Sepsis only** (`src/scm_to_mdp/reward_family.py`): every learned
  model is re-planned, not re-learned, under 20 random rewards (dense, sparse goals, white noise, and
  "stress" rewards on the least-visited states) at every data budget, and under Sepsis's 3
  alternative rewards (`results_transfer.csv`); `rf_safe_all` says whether the promise held for every
  random reward. Sepsis is the domain where this test is cleanest: nothing is hidden (so the promise
  for every reward is a theorem, not a hope), the vitals have treatment-specific parents (a missing
  edge can matter for one objective and not another), and death is a failure state.

Every setting above is a named constant in `experiments/common.py` (data, budget, random rewards),
`experiments/domains.py` (the domains) or the experiment's `protocol.py` (the methods).

Each experiment has two scripts per domain: `<domain>_mdp.py` (point estimates and value iteration, no
guarantee) and `<domain>_imdp.py` (interval sets and robust value iteration: a promise that is a lower
bound on the true value with probability at least 1 - δ).

- **Experiment 1** (`experiments/exp1/`): is causal discovery better than assuming the full factored
  graph? PCMCI+ and DYNOTEARS against full-fact (every state and action variable a parent of every
  next-state variable, no same-step edge), the true graph and the flat model. The discovered graphs
  are also scored against the true one: recall (true edges found), precision (found edges that are
  true), SHD and SID.
- **Experiment 2** (`experiments/exp2/`): ours (PCMCI+ and DYNOTEARS) against DiFaRL, FOCUS, flat
  Π_b-SPIBB (Laroche et al., ICML 2019), SL-SPIBB (Simão & Spaan, IJCAI 2019) and offline DQN, with Factored SPIBB (Simão & Spaan, AAAI 2019) on the
  true graph as a reference.

Every script runs on its own, by name, from any folder (`experiments/run.py`), or directly:

```bash
.venv/bin/python experiments/run.py --list                         # every script
.venv/bin/python experiments/run.py exp1 sepsis imdp --show        # what it would run, without running it
.venv/bin/python experiments/run.py exp1 sepsis imdp               # a new run
.venv/bin/python experiments/run.py exp1 sepsis imdp --resume      # continue the latest run
.venv/bin/python experiments/run.py exp2 taxi mdp --report-only    # tables and figures of the latest run
.venv/bin/python experiments/exp1/sysadmin_imdp.py --workers 6     # the same, calling the script itself
bash experiments/run_chain.sh exp1                                 # every script of experiment 1, in sequence
.venv/bin/python experiments/calibration.py                        # does 1 - δ hold? 60 seeds, smallest sizes
```

Ten seeds cannot confirm δ = 0.05 (with no failure, the 95% upper bound on the failure rate is still
0.26), so `experiments/calibration.py` runs the → IMDP arms of experiment 1 with 60 seeds on the two
smallest sizes of every domain; `summary/validity.csv` gives the verdict against 1 − δ per cell.

Every run gets its own folder, `results/<experiment>/<domain>_<kind>_<date>_<time>/`, so no run
overwrites another; `--resume FOLDER` and `--report-only FOLDER` pick an older one (`tool/runs.py`). A
run writes `results.csv` row by row, so a stopped run resumes where it stopped (with `--resume`).
Figures are written as PDF (vector) and PNG (500 dpi), without titles, for the paper.

The earlier designs (`experiments/exp1_wrong/`, `experiments/trials_wrong/`) are kept on disk only
(git-ignored).

For the first seed, every method's learned model is saved at every size and every data budget, in `<setting>/models/<size>/n=<n>/<method>/` (`tool/store.py`):
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
network, and the network is what the pipeline uses. To change a grid, edit `experiments/domains.py` (a domain),
`experiments/common.py` (data, budget) or the experiment's `protocol.py` (methods). If you change anything other than
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

- `figures/transitions=<n>/{performance,validity,structure,time,memory}` over the problem size, one
  folder per transition budget;
- `figures/sample_efficiency`: x = number of states, y = transitions needed for a policy that is at
  least 95% optimal ($V^\pi \ge 0.95\,V^\star_{obs}$ at the start, at that budget and every larger
  one; median over seeds, band = its 95% CI); $V^\star_{obs}$ is the best policy over the observed
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
| `src/scm_to_mdp/discovery/` | S2: `bic_search`, `pcmci_plus`, `ges`, `dynotears`, `oracle`, `full_factored` → `Candidates`; `controls` (the control graphs of the evaluation) |
| `src/scm_to_mdp/selection/` | S3: `bic_best` (best BIC in the class; ties: transition BIC, then a fixed order), `given` |
| `src/scm_to_mdp/scm_learning/` | S4: `nbn_intervals`, a Clopper–Pearson interval per factor row with a union bound |
| `src/scm_to_mdp/compilation/` | S7: `product_bounds`, factor intervals → their box, an interval MDP (the relaxation the box solvers use in S8) |
| `src/scm_to_mdp/solving/` | S8: `value_iteration`, `robust_vi`, `vertex_robust_vi` (`tol` = distance to the fixed point, `stopping.py`) |
| `src/scm_to_mdp/baselines/` | `flat_mdp`, `flat_rmdp`, `difarl`, `offline_q` (offline DQN / CQL), `spibb` (flat, SL-SPIBB and Factored SPIBB), `focus` |
| `src/scm_to_mdp/evaluation.py` | S9: V*, the true value of a policy, promise, safety, SHD, SID, recall and precision, the data's coverage of a task's rows, the learned model against the true kernel |
| `src/scm_to_mdp/robustness.py`, `src/scm_to_mdp/reward_family.py` | S9: the robustness checks of a promise; random rewards on the same dynamics |
| `src/scm_to_mdp/plotting/` | result figures (from `results.csv`) and pictures of MDPs |
| `experiments/` | `domains.py` (the nine domains), `common.py` (data, budget, evaluation), `exp1/` and `exp2/` (a `protocol.py` with the methods and two scripts per domain), `calibration.py` (60 seeds: does 1 − δ hold?), `run_chain.sh` (run an experiment's scripts in sequence) |
| `tests/` | one folder per part (`envs/`, `data/`, `discovery/`, `learning/`, `solving/`, `baselines/`, `evaluation/`, `plotting/`, `tool/`); `tests/tool/test_experiment_scripts.py` checks every script against the protocol |
| `results/` | outputs (git-ignored) |

## Adding things

- **Your own problem:** write `make_env(size, seed)`, returning an object with `ground_truth()`, and
  `collect(env, n_rollouts)`, returning the table (`tool/sweep.py`, `Experiment`). Or subclass `Env`
  (`envs/base.py`) and implement `next_distributions`, `action_columns`, `reward`, `true_parents`
  and `start_distribution`.
- **A method for a step:** a module in the folder of its step with the same signature as its
  neighbours, plus its name in `tool/methods.py`.
- **An experiment:** a folder `experiments/expK/` with a `protocol.py` (its methods) and one script
  per domain and kind, like `exp1/sysadmin_imdp.py`; results go to `results/expK/<name>/`.
