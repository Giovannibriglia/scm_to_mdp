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

**Status:** environment, loggers, data and diagnostics are implemented and tested. The learning arms
(S2–S9, on NBN ≥ 0.24) are next.
