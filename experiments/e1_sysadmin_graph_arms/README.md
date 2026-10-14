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
uv run python experiments/e1_sysadmin_graph_arms/pipeline_arms.py
uv run python experiments/e1_sysadmin_graph_arms/pipeline_arms.py --d 2 --suites "S-B two aliasing" --per-logger 50 200 --eps 0.2 --no-alone --out results/e1/ablation_d2.csv
uv run python experiments/e1_sysadmin_graph_arms/pipeline_arms.py --suites "S-X distractor" --per-logger 50 200 1000 --eps 0.2 --no-alone --out results/e1/ablation_distractor.csv
uv run python experiments/e1_sysadmin_graph_arms/pipeline_arms.py --n 6 --suites "S-B two aliasing" "S-D two specialists" --per-logger 50 200 --eps 0.2 --seeds 10 --no-alone --out results/e1/ablation_n6.csv
uv run python experiments/e1_sysadmin_graph_arms/analyze_pipeline.py
```
The first command is the core grid: suites S-A to S-E, 20, 50, 200 and 1000 episodes **per logger**,
strong and weak edge, exploration ε ∈ {0.02, 0.2} for every logger, 20 paired seeds, ten arms, and
every multi-logger suite also run with each of its loggers **alone** at the same total data (4160
runs; about 12 minutes on 8 worker processes, progress bar with time left). The next three are the
ablations (a false in-degree bound, a distractor bit, six machines); the last draws every figure and
table from whichever result files exist.

**Suites.** S-A aliasing (offset −1) alone; S-B two aliasing loggers (offsets −1 and +1); S-C softmax
over $Q^*$ + lazy; S-D two region specialists; **S-E** a region specialist + a logger that reads a hidden
load which also moves the machines (load effect 0.4): setting 1 fails, and the S1 gate should reject;
**S-X** (ablation) a specialist that acts only when an observed, irrelevant bit is 1 + the other
specialist, with the bit offered to S2/S3 as a candidate parent.

For each run the data are split by episode (discovery half: the S1 gate and S2/S3; estimation half: S4),
and ten arms run on the same data:

| Arm | S2/S3: parent sets per machine | S4 (NBN) |
|---|---|---|
| `oracle` | the true parents | sets on them (`agreement_set`, one candidate) |
| `flat` | all inputs (no structure) | one set per visited row |
| `bic` | the BIC-best set (one learned graph) | sets on it |
| `posterior` | the Bayesian credible family | union over the family (`refute=False`) |
| `icp` | the ICP accepted family | union over the family (`refute=False`) |
| `agreement` | all size-$d$ sets, refuted by their extensions | the agreement rule (`refute=True`) |
| `icp_agreement` | the ICP family, then refuted | the agreement rule on it |
| `oracle_gated` | the true parents | **follows the S1 gate**: if it rejects, confounded rows (`confounded_interval_set`, declared randomised loggers from their metadata) combined by Fréchet bounds; else as `oracle` |
| `plugin_oracle`, `plugin_bic` | the true / BIC parents | maximum-likelihood tables, no sets |

Then: **S7** exports the per-machine sets (`nbn.factored_sets/1`) and turns each joint row into a box:
the product of the factor bounds (machines independent given $(s,a)$), or the Fréchet bounds for the
gated route (a hidden load can move all machines together); an extra input such as the distractor bit is
hulled over its values (nature picks it). **S8** runs robust value iteration on the boxes (worst case,
dynamic nature, γ = 0.95); the promise is the returned policy's own robust value. **S9** evaluates it
exactly. Budgets: δ = 0.1 for the sets (split over the machines), α = 0.1 for the ICP family and for
the S1 gate (both split over the machines). **S5** (multi-candidate arms): per machine, candidates whose
sets agree within τ = 0.05 on every row of a state the data reached are merged ($K\to K'$); up to 32
hypotheses are drawn, and we report the largest regret of their plug-in policies and the share of
reached states where those policies differ. Each logger's own value (Monte Carlo, 500 episodes of 150
steps) is recorded as a reference.

`analyze_pipeline.py` writes (each figure as PDF and 300-dpi PNG; tables as CSV and LaTeX):

- `figures/pipeline_regret_eps02` and `_eps20` (ε = 0.02 and 0.2). **Caption:** *Regret $V^*-V^{\hat\pi}$ at "all
  machines up" against episodes per logger, per suite (columns) and edge (rows), exploration ε as in the
  file name; 20 seeds per point, bands are 95% bootstrap intervals. Solid: arms with sets (robust
  planning); dashed: plug-in controls; dotted grey: the best logger's own regret.*
- `figures/pipeline_promise`. **Caption:** *Setting-1 suites (S-A to S-D), all sizes, edges and ε. (a)
  Bars: share of runs whose promise held (promise ≤ true value; hatched: plug-in controls, no
  guarantee); diamonds: share where the true kernel lay inside every box. (b) Feasible rows with a set
  that the data never visited jointly (covered factor by factor only). (c) Promise against true value
  per run, coloured as in (a).*
- `figures/pipeline_s5`. **Caption:** *S5 for the multi-candidate arms (setting-1 suites). (a) Number of
  hypotheses before (dotted) and after (solid) merging candidates whose sets agree within τ = 0.05 on
  reached rows ($\log_{10}$). (b) Largest regret among the plug-in policies of up to 32 sampled
  hypotheses. (c) Share of reached states where those policies choose different actions.*
- `figures/pipeline_loggers`. **Caption:** *Each multi-logger suite against each of its loggers alone, at
  the same total number of episodes: ICP family size per machine, regret of ICP + agreement, and rows
  with a set under the oracle graph.*
- `figures/pipeline_gate`. **Caption:** *(a) Rejection rate of the S1 gate per suite (family-wise level
  0.1; dotted): false alarms in S-A to S-D, power in S-E. (b, c) S-E: share of runs with the true
  interventional kernel inside every box, and regret, for the oracle graph, ICP + agreement, the plug-in
  oracle, and the oracle that follows the gate.*
- `figures/pipeline_ablations`. **Caption:** *Ablations against the matching core runs (same suite,
  sizes and ε): regret (top) and share of runs with the truth inside every box (bottom), per arm.*
- `tables/pipeline_arms` (per arm and size: regret, promise, sound, truth in boxes, rows with a set
  visited jointly / factor only, valid structure, SHD), `tables/model_cards` (candidates per machine,
  factor rows, joint rows per factor row, hypotheses before and after S5), `tables/pipeline_loggers`,
  `tables/ablations`.

#### Results (2026-10-14; core grid 4160 runs × 10 arms, 20 seeds; ablations as above)
Setting-1 suites S-A to S-D unless stated (`tables/pipeline_arms`, `tables/model_cards`,
`tables/pipeline_loggers`, `tables/ablations`).

- **Sets keep their promise; one learned graph breaks the guarantee's event.** Every set arm kept
  promise ≤ true value, except BIC in 3 of 1280 runs (expectation 2 of DESIGN §7: a missed parent can
  over-promise). The true kernel lay outside some box in 244 of 1280 BIC runs (19%), 57 credible-family
  runs (4%), 21 ICP and 16 ICP + agreement runs (≤ 2%, within their level), and never for oracle, flat
  and agreement. The plug-in controls kept their promise in only 12–25% of runs.
- **Structure buys coverage.** Feasible rows with a set that the data never visited jointly, at 20
  episodes per logger: BIC 28, oracle and credible family 14–15, ICP ≈ 1, agreement 0, flat 0 (by
  construction); the gap closes with data (≈ 3.5 at 1000).
- **Pessimism costs decisions; exploration helps.** Robust arms' regret falls with data (oracle 13.5 →
  1.9 from 20 to 1000 episodes per logger) while plug-in policies are near optimal throughout; with
  ε = 0.2 instead of 0.02 the oracle's regret drops from 9.4 to 5.8 and BIC's from 9.0 to 4.0.
- **S5: graph ambiguity does not collapse and does change decisions.** Merging candidates whose sets
  agree within τ = 0.05 on reached rows barely reduces the hypotheses (e.g. ICP + agreement at 1000
  episodes per logger: $10^{2.06}\to10^{2.01}$); the sampled hypotheses' plug-in policies still differ
  on 45–62% of reached states for the valid families at 1000 episodes per logger (credible family 12%).
- **More loggers, less graph uncertainty (not just more data).** At 400 total episodes, any logger
  alone leaves ICP with all 93 candidate sets (no environments to compare), while two loggers shrink
  it to 3–6 per machine with validity 0.95–0.96; ICP + agreement's regret drops from 17–22 (S-B and
  S-D loggers alone) to 0.3–9.6 with both. (S-C's softmax logger alone is already near optimal.)
- **The S1 gate.** False alarms 6–10% in S-B to S-D (nominal 10%); in S-E it rejects 24% of runs at 20
  episodes per logger and 100% from 200. In S-E, ignoring the gate leaves the true interventional
  kernel outside some box in half of the oracle runs (all runs from 200 episodes per logger);
  `oracle_gated` keeps it inside in every run, at a regret of 12.1 (oracle 4.3; plug-in 0.9 but its
  promise held in only 55%). In setting 1, its false alarms cost little (regret 8.7 vs 7.6).
- **Ablations.** *False bound d = 2:* every learned arm loses the truth (inside every box in 25–50% of
  runs); the agreement rule's refutation flag fires in only 25% of runs (detection is not guaranteed)
  and the ICP family is empty in 49%; BIC's regret happens to be low while its boxes miss the truth in
  75%. *Distractor bit:* the learned graphs mostly ignore it (BIC valid 84%), ICP's family grows (5.2
  sets per machine) and its regret rises (3.4 → 8.6), flat suffers most (14 → 30). *N = 6:* soundness
  holds (truth inside every box ≥ 97%), but the valid families are wide (15–30 sets per machine) and
  their regret is 30–39, against 10 for the oracle.

Against DESIGN §7: (3) plug-in good on average but unsound — confirmed; (2) one learned graph misses
parents, leaves its boxes and occasionally over-promises — confirmed (rarely: 3 of 1280); (1) every
graph arm ≈ oracle with many episodes — holds for BIC and the credible family at 1000 episodes per
logger, not for the ICP family in S-A (one logger) or for the valid families in general.

**Status:** S0–S9 implemented and tested, with the second-pass items (S5 merge, coverage split, SHD,
logger values, model cards, ε, 1000 episodes per logger, 20 seeds, S-E with the gate, loggers alone,
ablations d = 2, distractor, N = 6). Open: the Storm cross-check of S8.
