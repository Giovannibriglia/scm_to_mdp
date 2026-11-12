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

One script per domain (`experiments/clear27/`): its 11 graph sources share one pool of workers, sized to
the machine by default (`SCM_TO_MDP_WORKERS=N` to choose).

Smoke tests first (a tiny grid of the same scripts, minutes):

```bash
SMOKE=1 bash experiments/clear27/run_sysadmin.sh
SMOKE=1 bash experiments/clear27/run_drone.sh
SMOKE=1 bash experiments/clear27/run_sepsis.sh
SMOKE=1 bash experiments/clear27/run_aircraft.sh
```

Then the experiments:

```bash
bash experiments/clear27/run_sysadmin.sh
bash experiments/clear27/run_drone.sh
bash experiments/clear27/run_sepsis.sh
bash experiments/clear27/run_aircraft.sh
```

Or the four in one pool, SysAdmin first (a free worker takes a job of the first domain that has one):
`bash experiments/clear27/run_all.sh`. Results in `results/clear27/<domain>/` (smoke: `results/clear27_smoke/`); `RESUME=1` continues a stopped run.

One method alone, with the oracle: `bash experiments/clear27/run_single.sh sysadmin --list`, then
`bash experiments/clear27/run_single.sh sysadmin "PCMCI+ -> IMDP (IA)"`. `CORE_SHARE=50` in front of any
launcher uses at most 50% of the physical cores.

## Figures

After (or during) the runs of a domain, one command draws every figure from its results:

```bash
.venv/bin/python experiments/clear27/plot.py sysadmin        # or: sepsis drone
.venv/bin/python experiments/clear27/plot.py --list          # the figure names
```

Figures go to `outputs/clear27/<domain>/paper/` and `explore/` (PDF and PNG), with the numbers drawn in
`outputs/clear27/<domain>/data/`. The sizes and panels of a domain are read from its results.

## Ablations

`experiments/clear27/ablations/<name>/` holds one ablation: `run.py` (results in
`results/clear27/ablations/<name>/`) and `plot.py` (figures in `outputs/clear27/ablations/<name>/`).

```bash
.venv/bin/python experiments/clear27/ablations/ablation_bic_score/run.py --machines 2    # default: 2 3
.venv/bin/python experiments/clear27/ablations/ablation_bic_score/plot.py
```

`ablation_bic_score`: on SysAdmin, every graph the constraints allow (2^(N(N+1)): 64 at N=2, 4,096 at N=3)
is scored with BIC, AIC and BDeu and run through the box IMDP, to see whether a better score gives a
better policy.

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

method = Method("causal", "exact_robust_vi",         # learner, solver
                discovery="pcmci+",                  # S2: pcmci+ | dynotears | bic | ges | full | sparse | oracle
                selection="bic_best",                # S3: how ONE graph is chosen (given graphs: "given")
                discovery_options={"alpha": 0.01, "same_step": False},
                learning_options={"delta": 0.05, "n_candidates": "inputs"})  # S4: level, post-selection family
data = spec.read(df)                                 # df: columns x, a, x'; index (episode, t)
result = solve(learn(data, method), data, gamma=0.9)
result.solution.pi, result.solution.V                # the policy and its promise (a lower bound)
```

| learner | what it learns | solvers | role |
|---|---|---|---|
| `causal` | S2 discovery → S3 selection → S4 Clopper-Pearson intervals per factor (a factored interval MDP; with the post-selection union bound for a learned graph); `vi`: point estimates | IMDP: `box_robust_vi` (interval arithmetic), `mccormick_robust_vi`, `exact_robust_vi` (vertex products; Schnitzer et al., AAAI 2026); MDP: `vi` (plug-in), `worst_mdp_vi` / `best_mdp_vi` (nature's worst / best MDP of a relaxation) | ours; the true graph (`oracle`) a reference; `full`, `sparse` baselines |
| `kwik` | the structure of Strehl et al. (2007) or Met-Rmax (2009) (`discovery="strehl"` / `"met_rmax"`); then our S4 hull, their own predictions ± accuracy, a point model with HALT, or Factored SPIBB (`mode`) | as `causal`, and `spibb` | ours |
| `factored_spibb` | Factored Π_b-SPIBB (Simão & Spaan, AAAI 2019) on a given or learned graph | `spibb` | reference on the true graph, else baseline / ours |
| `flat_rmdp` | one interval per (s, a, s′) | `robust_vi`, `worst_mdp_vi`, `best_mdp_vi` | baseline |
| `flat_mdp` | the plug-in MDP from counts (no guarantee) | `vi`, `robust_vi` | baseline |
| `spibb`, `sl_spibb` | the estimated MDP, flat or with SL structure learning, and the bootstrapped pairs | `spibb` (a stochastic policy) | baseline |
| `focus` | FOCUS's CI-test graph, the factored model with MOPO's penalised reward | `vi` | baseline |
| `difarl` | DiFaRL's action groups (model-free, no guarantee) | `fqi` | baseline |
| `offline_dqn`, `offline_cql` | a Q-network trained on the logged data (model-free, no guarantee) | `greedy_q` | baseline |
| `oracle` | the true observed MDP (the yardstick: V*_obs and the solve budget) | `vi`, `robust_vi` | oracle |

McCormick and exact need factors without same-step parents (else the run is `not_applicable`); a worst
MDP reuses its IMDP arm's robust solve on the same data.

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
with a memory limit (PSS on Linux, RSS on macOS) and a time limit per step (discovery, selection,
learning, compilation, training, solving); a step over its limit is a `timeout` / `memout` and the
larger budgets of that size are skipped. Rows are written to `results.csv` as each (size, seed)
finishes, and a second run resumes. `report(out_dir)` writes:

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
