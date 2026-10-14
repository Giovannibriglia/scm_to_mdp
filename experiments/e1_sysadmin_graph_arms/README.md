# E1 — Does graph uncertainty change decisions?

## The question
An agent must learn to act from **offline data** logged by **several behaviour policies** (loggers) on a
small, fully observed, stochastic system. To plan safely it builds a model of the dynamics, and a
**causal graph** says which variables each next-state variable depends on. With finite data the graph
is uncertain.

1. Does it matter whether we use **one learned graph**, keep **several plausible graphs**, use **no
   graph at all**, or know the **true graph**?
2. When the plausible graphs disagree, does the disagreement **change the decision**?
3. Do **more, and more diverse, loggers** reduce that uncertainty, and not only add data?

We measure the answer as **regret** (how much worse than optimal the chosen policy is) and
**soundness** (whether the value we promise is at most the value we really get).

## The process (pipeline steps S0–S9)
```
S0 environment ──▶ S1 loggers + data ──▶ S2/S3 graphs (the arms) ──▶ S4 probability sets per factor
      ──▶ S5 merge graphs that agree ──▶ S6 declare the criterion (worst case)
      ──▶ S7 compile the model ──▶ S8 robust planning ──▶ S9 exact check on the true system
```
- **S0, the system:** the SysAdmin ring, with N machines that are up or down. A failed left neighbour
  makes a machine fail more often, and the agent repairs one machine per step. Its exact model is
  known to the evaluator only.
- **S1, the data:** seven logger families: uniform random, region specialist, aliasing rule, softmax
  over the optimal values, lazy, history-dependent, and confounded (it reads a hidden load). Each
  covers different parts of the state-action space.
- **S2/S3, the arms:**
  - *oracle* graph;
  - *one learned graph* (best score);
  - *plausible graphs* (all candidates the data do not refute: the agreement rule of NBN);
  - *no graph* (one table row per state-action);
  - *plug-in* versions without uncertainty sets.
- **S4–S8:** every arm gets per-factor confidence sets (NBN), is compiled with the same rules, and is
  solved for the worst case.
- **S5** is where question 2 is answered: graphs whose sets agree everywhere they matter are merged.
  If the survivors would all act the same, the disagreement does not matter.
- **S9:** the returned policy is evaluated exactly on the true system: promise, true value, regret.

The full design (arms, measurements, knobs and the expected results, written before any run) is in
[`DESIGN.md`](DESIGN.md).

## How we answer it
- Every arm sees **the same data** (paired seeds); results are reported with bootstrap confidence
  intervals, and failures are counted separately.
- Knobs: episodes per logger, exploration, edge strength (a weak edge is hard to detect), and which
  loggers are combined.
- First a **pilot**: one dataset per logger family, to check that each logger produces the data it is
  meant to (coverage, aliasing, confounding) before any learning.

## How to run
From the repository root, with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
uv run pytest -q
uv run python experiments/e1_sysadmin_graph_arms/pilot_loggers.py --seeds 10
uv run python experiments/e1_sysadmin_graph_arms/analyze.py
```
In order: install the environment and dependencies; run the tests of the environment and loggers;
simulate one dataset per logger family and seed (seconds); draw the figure and write the tables.
(No inline `#` comments: zsh passes them on as arguments unless `interactivecomments` is set.)
The pilot prints one line per logger family and writes `results/e1/pilot_loggers.csv`. `analyze.py`
reads it (it never simulates) and writes:

- `results/e1/figures/pilot_loggers.pdf` / `.png`. **Caption:** *What each logger family's data
  cover and can tell (SysAdmin ring, N = 4; 100 episodes of 30 steps per dataset; 10 seeds). (a)
  Share of the 80 state-action rows visited jointly (dark) and covered only factor by factor
  (light), mean over seeds. (b) Aliasing index: how strongly a repair flag determines the state of
  the machine on its left (normalised mutual information, maximum over machines). (c) Confounding
  excess: the largest gap between observed and interventional transition probabilities beyond four
  standard errors. In (b) and (c), each dot is one seed, and the black bar is the mean with a 95%
  percentile-bootstrap interval.*
- `results/e1/tables/pilot_loggers.csv` and `.tex`: every metric, mean [95% bootstrap interval]
  over seeds (a LaTeX `tabular` body, ready for `\input`).

### Stage 2: assumption checks and graph arms (S1–S3)
```bash
uv sync --extra discovery
uv run --extra discovery python experiments/e1_sysadmin_graph_arms/structure_arms.py --seeds 10
uv run python experiments/e1_sysadmin_graph_arms/analyze.py
```
The first command adds Tigramite for J-PCMCI+ (GPL-3.0, an optional extra; without it the runner skips
that arm). The runner (progress bar with time left; about 2 minutes) simulates every logger suite at
40, 100 and 400 episodes, with a strong and a weak edge, and for each run:

1. **S1 checks from data only**: the confounding gate (is the kernel the same under every logger?), a
   first-order Markov test and a stationarity test. All use a permutation test within each input
   configuration. Their level is exact for independent rows at any sample size, and approximate (close
   to nominal in simulations) on trajectories.
2. **Episode split**: discovery half for the graph arms, estimation half kept for S4.
3. **Graph arms**: one graph by BIC, one graph by J-PCMCI+ (logger as context), the Bayesian credible
   family, and the ICP family (the accepted parent sets: contains the true set w.p. ≥ 0.9).

`analyze.py` then writes:

- `results/e1/figures/structure_arms.pdf` / `.png`. **Caption:** *Graph arms on the SysAdmin ring
  (10 seeds per point; bands are 95% bootstrap intervals). (a) Share of runs in which a one-graph
  method misses a true parent, for a weak and a strong edge. (b) Size of the ICP family per target
  (weak edge): with one logger (S-A) nothing can be rejected, with two diverse loggers the family
  shrinks as data grow. (c) Share of runs whose family contains the true parents of every target
  (weak edge), against the nominal 0.9 of the ICP family.*
- `results/e1/tables/structure_arms.csv` (every suite, edge, size and arm) and `.tex` (weak edge:
  valid share and family size per arm), and `results/e1/tables/s1_checks.csv` (how often each S1
  check rejects; nominal 0.10 family-wise, since setting 1 holds in these suites).

Columns of the pilot table:

| Column | Meaning |
|---|---|
| `joint` | share of state-action rows visited |
| `factor` | share of rows whose per-machine parent values were all visited |
| `f-only` | rows covered only factor by factor (where structure can help) |
| `min n` | fewest visits on a visited row |
| `alias` | how strongly a repair flag is tied to a neighbour's state (0–1) |
| `empty cells` (in the tables) | parent configurations of a repair flag and its neighbour never seen (not estimable from the data) |
| `conc.` | single-policy concentrability: largest ratio of the optimal policy's discounted visits to the data's row frequency (∞ = the optimal policy uses a row the data never visit). A descriptive number: the theory's coefficient takes the worst case over all policies (Agarwal et al., *RL: Theory and Algorithms*, Assumption 4.1, p. 54) |
| `confound` | gap between observed and interventional kernels, beyond sampling noise |

Options: `--n` (machines), `--episodes`, `--horizon`, `--seeds`, `--load-effect`, `--no-progress`.
A progress bar (on stderr) shows the datasets done, the elapsed time and an estimate of the time left.

### Stage 3: the whole pipeline, S1–S9, for every arm
```bash
uv run python experiments/e1_sysadmin_graph_arms/pipeline_arms.py --seeds 10
uv run python experiments/e1_sysadmin_graph_arms/analyze_pipeline.py
```
The runner (progress bar with time left; a few minutes) uses the same suites, sizes and edges as
stage 2. For each run it splits the data by episode (discovery half: S2/S3; estimation half: S4) and
runs nine arms:

| Arm | S2/S3: parent sets per machine | S4 (NBN `agreement_set`) |
|---|---|---|
| `oracle` | the true parents | sets on them |
| `flat` | all 8 inputs (no structure) | one set per visited row |
| `bic` | the BIC-best set (one learned graph) | sets on it |
| `posterior` | the Bayesian credible family | union over the family (`refute=False`) |
| `icp` | the ICP accepted family | union over the family (`refute=False`) |
| `agreement` | all size-3 sets, refuted by their extensions | the agreement rule (`refute=True`) |
| `icp_agreement` | the ICP family, then refuted | the agreement rule on it |
| `plugin_oracle`, `plugin_bic` | the true / BIC parents | maximum-likelihood tables, no sets |

Then: **S7** exports the per-machine sets with NBN (`nbn.factored_sets/1`) and reads each joint
row's box back (`joint_row_bounds`, an outer relaxation); **S8** runs robust value iteration on the
boxes (worst case, dynamic nature, γ = 0.95); **S9** evaluates the returned policy exactly. The
budget is δ = 0.1 for the sets (split over the machines), plus α = 0.1 for the ICP family.
**S5** (multi-candidate arms): up to 32 hypotheses, one candidate per machine, are drawn; each one's
plug-in policy is evaluated, giving the largest regret among them and the share of states where they
choose different actions.

`analyze_pipeline.py` reads `results/e1/pipeline_arms.csv` and writes:

- `results/e1/figures/pipeline_regret.pdf` / `.png`. **Caption:** *Regret of the returned policy,
  $V^*-V^{\hat\pi}$ at "all machines up", against total episodes, per logger suite (columns) and
  edge strength (rows); 10 seeds per point, bands are 95% bootstrap intervals. Solid: arms with
  uncertainty sets (robust planning); dashed: plug-in controls.*
- `results/e1/figures/pipeline_promise.pdf` / `.png`. **Caption:** *(a) Share of runs where the
  promise held (bars: promise ≤ true value; hatched: plug-in controls, which carry no guarantee) and
  where the true kernel lay inside every box (diamonds: the event the guarantee rests on).
  (b) Feasible state-action rows (of 80) whose box is narrower than the simplex. (c) Promise
  against true value per run, coloured as in (a): points above the diagonal kept their promise.*
- `results/e1/figures/pipeline_s5.pdf` / `.png`. **Caption:** *S5, does graph disagreement matter?
  For arms with several candidates per machine: the largest regret among the plug-in policies of
  the sampled hypotheses, and the share of states where they disagree.*
- `results/e1/tables/pipeline_arms.csv` and `.tex`: per arm and size (suites and edges pooled):
  regret, promise, sound share, truth inside every box, rows with a set, valid structure, mean
  [95% bootstrap interval].

**First full run (2026-10-14; 4 suites × 40/100/400 episodes × 2 edges × 10 seeds, 9 arms;
`results/e1/tables/pipeline_arms.csv`):**
- *Sets keep their promise; plug-in does not.* Every set-based arm kept promise ≤ true value in all
  240 runs; the plug-in controls broke it in 75–90% of runs, with near-zero regret.
- *The one-graph arm breaks the guarantee's event.* The true kernel lay outside some BIC box in 21%
  of runs (43% when BIC missed a parent), and outside some credible-family box in 5%. Its promises
  still held only because pessimism is wide here, so "promise ≤ true value" alone is a weak test on
  this domain; "truth inside every box" is the sharper one. Oracle, flat, agreement: 100%; ICP and
  ICP + agreement: 99% (as their level allows).
- *Pessimism costs decisions.* Robust arms have regret 4–35 where the plug-in policies are near
  optimal; the gap shrinks with data (oracle: 14.0 → 5.6 from 40 to 400 episodes).
- *Structure buys coverage.* Feasible rows with a set at 400 episodes: oracle and BIC 79, flat 29.
- *Valid families are wide at small n.* The agreement rule cannot refute much with 40–100 episodes
  (about 20 survivors per machine), so its rows stay open and its regret is the largest; ICP +
  agreement is the best valid-family arm at 400 episodes (regret 8.9 vs agreement 11.9, ICP 10.0).
- *S5:* the sampled hypotheses of the valid families disagree on most states (ICP 0.64, agreement
  0.69 at 400 episodes): on this domain graph ambiguity does change the plug-in decision.

These are pre-registered expectations 2–3 of `DESIGN.md` §7 confirmed in part: (3) yes; (2) the
one-graph arm misses parents and leaves its boxes, but its promise does not fail, because of the
pessimism margin. Expectation (1), every graph arm ≈ oracle with many episodes, does not hold at
400 episodes for the valid-family arms.

**Status:** S0–S9 implemented and tested (environment, loggers, S1 checks, graph arms, NBN sets and
export, robust value iteration, exact check, S5 sampling).
