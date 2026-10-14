# E1 design — does graph uncertainty change decisions? (SysAdmin ring)

**Status:** design (2026-10-14), nothing implemented. Expected results are written before any run (§7).

**Question.** Offline data come from several exploratory behaviour policies on a small, fully observed,
stochastic MDP. We compare a pipeline that uses **one learned graph**, one that keeps **several
plausible graphs**, one with **no structure**, and the **oracle graph**. We measure regret and soundness
(promise ≤ true value), and check whether **graph disagreement changes the decision**.

## 1. The pipeline, step by step

| Step | What it does in E1 | Choice |
|---|---|---|
| **S0** Unknown system | SysAdmin one-way ring, $N=4$ (then $N=6$); exact model known to the evaluator only | §2 |
| **S1** Data and profiling | two half-ring loggers; trajectories; regime label kept; the policy-as-context test is run as a gate (it should not reject: no confounding) | §3 |
| **S2** Causal discovery | depends on the arm (§4): none, exact score search, or none (candidates enumerated) | §4 |
| **S3** Hypotheses | depends on the arm: one graph, a set of candidate parent sets refuted by the agreement rule, or the true graph | §4 |
| **S4** Parameter learning | per-factor L1 sets from counts (NBN v0.24: `HypothesisSet`, `agreement_set`), `sampling="trajectory"`, total δ = 0.1 | §4 |
| **S5** Collapse | merge hypotheses whose sets agree on every reachable $(s,a)$ within a tolerance; log $K\to K'$. **This is the measurement of "does disagreement matter"** (§5) | §5 |
| **S6** Decision criterion | declared, not learned: $J$ = worst case; nature **dynamic** (re-picks after seeing $(s,a)$), so the exact target is an $(s,a)$-rectangular robust MDP; no weights; Markov stationary policies; discounted, $\gamma=0.95$, $\mu$ = all machines up | §6 |
| **S7** Compilation | exact factored model with its sharing: NBN `export_factored_sets` (schema `nbn.factored_sets/1`); one model card per arm (size, rows with a set, coupling) | §6 |
| **S8** Solving | relaxation chosen under the budget: per-row interval boxes from `joint_row_bounds` (outer, so still sound), robust value iteration; the same relaxation for every arm | §6 |
| **S9** Check | exact: the true value of the returned policy on the true MDP, regret, and whether promise ≤ true value | §5 |

## 2. Environment (S0)
- State: machines $m_0..m_3\in\{\text{up},\text{down}\}$ (16 states). Actions: no-op or repair $i$ (5 actions; 80 feasible rows).
  The action enters each machine as a derived flag $\mathrm{rep}_i=[A=i]$ (declared in S0).
- Dynamics: a repaired machine is up w.p. 0.95; an up machine stays up w.p. $p_{\text{ok}}=0.95$ if its left
  neighbour $m_{i-1}$ is up and $p_{\text{bad}}$ if it is down; a down machine recovers w.p. 0.05.
  True parents: $\mathrm{Pa}(m_i')=\{m_i,m_{i-1},\mathrm{rep}_i\}$ (in-degree 3).
- Reward: number of machines up, with $m_0$ (the server) weighted 2. The edge $m_3\to m_0'$ then decides
  whether repairing $m_3$ protects the server: a graph that misses it can choose the wrong machine.
- Everything is exact: the true kernel, $V^*$, and the true value of any policy.
- Why this domain: stochastic, factored, a known sparse graph whose strength we control, an exact
  model, and natural loggers with partly disjoint coverage. It is also a standard benchmark for
  structure learning (Simão & Spaan, IJCAI 2019) and for robust factored MDPs (Schnitzer, Abate,
  Parker, AAAI 2026, where the structure is known: our oracle arm).

## 3. Loggers (S1)
- **L** repairs a failed machine among $\{m_0,m_1\}$; **R** among $\{m_2,m_3\}$; each takes a random action
  with probability ε. Horizon 30 per episode.
- Joint rows such as "repair $m_0$ while $m_3$ is down" are rare, but each factor's parent values are seen
  by at least one logger: the situation where structure can predict rows never visited jointly.

## 3b. Logger families and suites (updated 2026-10-14)
All seven families are implemented (`src/scm_to_mdp/loggers.py`): uniform random, region specialist,
aliasing rule (repair $m_i$ iff $m_{i+\text{offset}}$ is down), softmax over $Q^*$, lazy,
recent failure (history-dependent), and confounded (reads the hidden load; outside setting 1).

- **Pilot:** one dataset per family (`pilot_loggers.py`, 10 seeds): coverage, aliasing, confounding,
  single-policy concentrability. It checks that each logger produces the data it is meant to.
- **Suites for the arms:**

| Suite | Loggers | What it tests |
|---|---|---|
| S-A | aliasing (offset −1) alone | a single learned graph can be wrong; the plausible set stays sound with a small gain |
| S-B | aliasing (offset −1) + aliasing (offset +1) | diverse loggers separate candidate graphs: **more loggers, less graph uncertainty** |
| S-C | softmax over $Q^*$ + lazy | complementary coverage (good states vs failures) |
| S-D | the two region specialists | joint vs factor coverage (the original design) |
| S-E | specialist + confounded | the S1 gate must reject; not a setting-1 result |

Each suite is compared with its loggers alone (same total data), so the effect of adding a logger is
measured, not assumed.

## 4. Arms (S2–S4)

| Arm | S2 | S3 output | S4 |
|---|---|---|---|
| **Oracle** | — | the true parent sets | sets on the true parents |
| **Flat** (no structure) | — | none: one row per $(s,a)$ | tabular sets; unvisited rows = the whole simplex |
| **One learned graph** | exact BIC search over parent sets of size ≤ 3 from the 8 inputs (93 per node) | one parent set per node (may miss a parent) | sets on that parent set |
| **Bayesian credible set** | exact posterior over the 93 parent sets (BDeu, uniform structure prior; K&F Prop. 18.7) | smallest set of candidates with posterior mass $\ge 1-\delta/N$ | sets on every candidate (factor cache) |
| **ICP set** | invariance test of $P(m_i'\mid C)$ across loggers for every candidate $C$ (loggers = environments) | the accepted sets of size $\le d$: contains the true set w.p. $\ge 1-\alpha/N$ (ECI eq. 7.8) | sets on every candidate |
| **Agreement rule** | — (candidates enumerated) | all size-3 candidates; refuted against their one-variable extensions; declared bound $d=3$ | one set per row from the survivors (`agreement_set`) |
| **ICP + agreement** | ICP as above | the agreement rule run on the ICP-accepted candidates only | one set per row from the survivors |
| **Plug-in controls** | oracle; one learned graph | as above | maximum-likelihood tables, **no** sets (ordinary value iteration) |

The plug-in controls separate the role of the sets (pessimism) from the role of the graph.
Every arm fits on the same pooled data (all loggers of a suite), with the same seeds.

**Graph-level levels (S3).** ICP gives a frequentist level for the hypothesis set: the true parent
set is accepted with probability $\ge 1-\alpha$ per target (Peters, Janzing & Schölkopf, *ECI*,
§7.2.5, eq. 7.8, pp. 153–154), so the family of accepted sets is valid at $1-\alpha$ after a Bonferroni split over
the $N$ targets. ICP's intersection $\hat S$ is *not* used: it controls false parents only and can
miss true ones. The Bayesian credible set has a posterior level, not a frequentist one.

**ICP test (S2).** For candidate $C$, compare $P(m_i'\mid C=u)$ across loggers within each
configuration $u$ (G statistic summed over configurations with data from at least two loggers). The
reference distribution is a **permutation** of the logger labels within each configuration (each
configuration's table redrawn with its margins fixed): given the true parents (or a superset), next
states are conditionally independent of the past and of the logger, so the labels are exchangeable
there and the level holds on sparse data and on trajectories. The chi-square approximation is not
used: on sparse cells it rejected 23% of null datasets at a nominal 5% (permutation: 4%). With one
logger there are no environments: every set is accepted.

**A prediction for the one-graph arm.** BIC keeps a parent roughly when
$M/\log M \gtrsim \Delta\mathrm{Dim}/(2\Delta)$, with $\Delta$ the conditional mutual information the
parent adds (nats) and $\Delta\mathrm{Dim}$ its extra parameters (our derivation from Koller &
Friedman, p. 803). It predicts at which data size the one-graph arm stops missing the weak edge;
E1 checks it.

## 5. Measurements (S5, S9)
Per arm and run:
- **Promise** (robust value at $\mu$), **true value** of the returned policy, **regret** $V^*-V^{\hat\pi}$.
- **Soundness:** whether promise ≤ true value held (the guarantee; it can fail for an arm whose graph
  misses a parent, and for the plug-in controls).
- **Coverage:** feasible rows with a set narrower than the simplex, split into rows visited jointly and
  rows covered only factor by factor.
- **Graph quality:** SHD of the learned graph; whether the plausible set contains a superset of the true
  parents for every node; number of survivors.
- **S5, does disagreement matter?** For the surviving hypotheses:
  - $K\to K'$ after collapse (sets agree on every reachable row within tolerance $\tau$);
  - the plug-in policy of each survivor, its regret on the true MDP, and the **largest** such regret;
  - the share of reachable states where survivors choose different actions.
  If all three are ≈ 0, graph ambiguity does not matter for this task: a valid answer.
- References: each logger's own value, and $V^*$.

## 6. Declared semantics and solving (S6–S8)
- $J$ = worst case over everything that is uncertain (sets and surviving hypotheses).
- Nature is dynamic: it re-chooses the transition after seeing $(s,a)$. Under this semantics the
  $(s,a)$-rectangular robust MDP is the exact target (not a relaxation). The static version (one world
  per episode, coupled rows) is the exact target of a later experiment (E2, coupled solving).
- S7 keeps the factored model exact (per-factor sets, row maps that record which rows share a factor row).
- S8 replaces each row's product of per-factor sets by its interval box (outer, so sound), then runs
  robust value iteration. All arms use the same relaxation, so the comparison is fair.

## 7. Knobs, seeds and expected results
| Knob | Values |
|---|---|
| Episodes per logger | 20, 50, 200, 1000 |
| Exploration ε | 0.02, 0.2 |
| Strength of the $m_{i-1}$ edge | strong ($p_{\text{bad}}=0.60$), weak ($p_{\text{bad}}=0.85$) |

16 configurations × 20 paired seeds. Report means with bootstrap confidence intervals; count failures
separately; compare arms on the same seeds.

**Ablations (after the core runs):** declared $d=2$ (the bound is false: the refutation flag should fire and
the arm fall back to flat); a distractor bit read by logger L that no machine depends on; $N=6$.

**Expected (pre-registered):**
1. Strong edge, many episodes: every graph arm ≈ oracle; S5 collapses the survivors; disagreement changes
   nothing (negative control).
2. Weak edge, few episodes: "one learned graph" sometimes drops $m_{i-1}$, over-promises (promise > true
   value) and repairs the wrong machine; "plausible graphs" stays sound and promises more than "flat"
   on rows covered factor by factor only.
3. Plug-in controls: good on average, unsound on some seeds.

If (2) does not appear, the claim that structure buys coverage fails on this domain; we then lower the
loggers' overlap before scaling up.

## 8. Planned layout
```
envs/sysadmin.py                         ring, exact kernel, reward, loggers
experiments/e1_sysadmin_graph_arms/
  config.yaml                            knobs, seeds, budgets
  run.py                                 one run = one (config, seed): all arms on the same data
  analyze.py                             tables and figures with bootstrap intervals
results/e1/                              (git-ignored)
```
Dependencies: `nbn>=0.24` (sets, hypotheses, agreement rule, export), NumPy (robust value iteration,
exact evaluation). Storm (interval models) as an optional cross-check of S8.
