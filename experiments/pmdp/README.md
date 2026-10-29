# Tied CPD rows: from a declared SCM to a parametric MDP (exploratory)

**Status: exploratory, not part of the paper's main pipeline.** This folder measures *how much a correct
declaration of the mechanism is worth* on top of the CBN -> IMDP pipeline. It is **not** a method for
domains whose mechanisms are unknown: there, the default stays CBN -> IMDP (structure assumed or
learned, nothing assumed about the mechanisms). See "How to read the results" below.

Steps of the pipeline: S2-S3 (the graph), S4 (the declared mechanism and the shared parameters), S7 (the
coupled pMDP), S8 (option a: box relaxation; option b: the static worst case by parameter lifting), S9.

## 1. The idea in five steps (general case)

1. **Declare the form of f.** For every state variable $x_i$: $x_i' = f_i(s, a, U_i)$, $U_i \sim P_\theta(U_i)$,
   where $f_i$ reads $(s, a)$ only through a **class map** $c_i(s, a)$ (which situations behave alike) and
   the noise law has unknown parameters $\theta_c$ **shared by every row of the same class**, across
   variables and states. The class map and the sharing are *declared*; a few unknowns inside $f$
   (parents, which action does what) are *learned* from the data and the discovered graph.
2. **Learn P(U).** Put every logged transition in its class, pool the outcomes per class, estimate
   $\theta_c$; a confidence set per class with a union bound gives $\Theta$, which contains the true $\theta$
   with probability $\ge 1-\delta$ **if the declaration is right**.
3. **Sum out the noise.** $P_\theta(s' \mid s, a) = \prod_i P_\theta(x_i' \mid c_i(s, a))$, $\theta \in \Theta$: a
   parametric MDP whose rows are coupled through the shared $\theta$ (the set is not rectangular; an entry
   can contain $\theta_c^2$).
4. **Solve.** (a) relax every row to its box and run robust VI: a policy $\pi$ and a sound promise;
   (b) for that $\pi$, bound the worst case with $\theta$ fixed for the whole run, $\min_{\theta \in \Theta} V^\pi_\theta$,
   by parameter lifting with region refinement: tighter, still sound.
5. **What you get.** With probability $\ge 1-\delta$, promise $\le$ true value, **provided the declared form of
   f holds**. The gain comes from pooling (few parameters, and unseen rows inherit their class's interval);
   the risk is the same pooling: the data test the declaration only on the rows they visited.

Only one input is new compared with CBN -> IMDP: the declared form of $f$ (a side input of S4).
Where the line between "$f$" and "$P(U)$" falls is a convention: SCMs with the same kernel give the same
MDP and differ only in counterfactuals, which are not used here.

## 2. What is assumed

Common to every arm (as experiments 1 and 2): the uniform logger (no confounding and positivity on
actions by design), a Markov and stationary observed state, no same-step edges, a known reward.

The **pMDP arms add** a declared template (`templates.py`):

| Domain | Declared | Learned |
|---|---|---|
| **SysAdmin** (`SysAdminTemplate`, 4 classes) | T1 every machine has the same probabilities (one $\theta$ for all machines); T2 a machine's chance to be up next depends on its parents only through 4 situations (*repaired*, *up and every neighbour up*, *up and some neighbour down*, *down*); T3 its own state and the action are parents, each machine has one repair action, no two the same | the neighbours (the other machines among the graph's parents) and the repair actions (one-to-one assignment from the recovery rates) |
| **Stock trading** (`StockTemplate`) | owning a sector after buying, selling or neither (owned or not): 4 classes, the same for every sector; a stock's chance to rise depends only on how many of its stock parents rise, the same for every stock | the buy and sell action of every sector, and the stock parents |

Both simulators are written **exactly** in this tied form (SysAdmin uses one `p_up` for all machines,
Stock one `base + scale * rising / M` for all stocks): they are the best case for the pMDP.

**Impact of the assumption** (SysAdmin, N = 6): the IMDP on the true graph estimates 168 separate
probabilities (6 machines x 4 parent configurations x 7 actions), the pMDP 4. So about 40x more data per
estimated number, a union bound over 4 instead of 168 intervals, and unseen rows get their class's
interval instead of [0, 1]. The promise becomes "PAC under the model" (NBN's `"pac_under_model"`): if
T1-T3 are wrong it can fail while looking confident, and the tying test (`tied.tying_test`) can only
check the rows that have data, with the least power at the small budgets where the pMDP gains most.

## 3. Option (b): the static worst case, and why the chain is sequentialised

A row of the flat chain multiplies the factors, so it contains $\theta_c^2$ when two variables share a class:
parameter lifting needs entries affine in each parameter of a state (Storm stopped with "states with
multiple parameters"). `tied.prism_pdtmc` decides one variable per intermediate state:

    (s, j=0)       --gamma-->    (s, j=1, b=0)  and  --(1 - gamma)--> sink   (reward R[s, pi(s)])
    (s, j=i+1, b)  --theta_c-->  x_i = 1 in b   and  --(1 - theta_c)--> x_i = 0
    (s, j=n+1, b)  --1-->        (s'=b, j=0)

For every fixed $\theta$ this chain has the same discounted value as the flat one, so region refinement
converges to $\min_\theta V^\pi_\theta$. Storm 2.0 (`stormpy.pars.compute_extremal_value`, robust parameter
lifting, sound solution methods, absolute precision 0.01) returns a value within the precision above the
minimum: Storm's value - 0.01 is a sound lower bound. Every Storm call runs in a fresh process (a second
parametric model in one process failed inside Storm) and stops after 60 s; the promise of (b) is the
larger of Storm's bound and a Python region refinement (`tied.region_refinement`, 81 sub-boxes, each
relaxed to its box), both sound (`b_source` in the results says which one). Checks in every (b) row: the
exact value at the theta Storm returns equals Storm's value (`storm_value_check`), and
a <= b_py <= Storm's value, Storm's bound <= `sampled_upper` (exact values at sampled theta).

Storm is slow when the intervals are wide and touch 0 (graph-changing regions, many parameters): on
Stock at 100 transitions it does not finish within 120 s even at precision 0.1, at 2k transitions it
takes about 1 s. There the Python bound is used.

## 4. Expert checks (agents of the knowledge base, 2026-10-29)

- **koller_claude** (Koller & Friedman 2009): pooled counts are the MLE for shared parameters (§17.5,
  pp. 754-761); within a transition the tied draws are independent given the parents; the anytime
  correction for trajectory data is needed (NBN's `sampling="trajectory"`); choosing the repair actions
  and the neighbours from the same data is a selection step the intervals ignore (Box 16.A,
  pp. 705-708); the chi-square tying test is rough on sparse rows (a parametric bootstrap is better).
- **badings_claude** (Badings thesis 2025; Suilen et al. 2024; Haddad & Monmege 2018): (a) and (b) are
  sound on the event $\theta^* \in \Theta$; the rows are not multilinear (hence the sequentialised chain); the
  sequentialised chain has the same value for every fixed $\theta$; Storm's bound needs sound numerics; call
  (b) "the static worst case over $\Theta$ to within the precision", not "exact"; "static" here means $\theta$
  fixed for the whole run, not the robust-MDP sense of fixed per state-action pair.

## 5. Results (results/pmdp/2026-10-29_123708/{sysadmin,sysadmin_storm,stock}/, git-ignored; 5 seeds, uniform logger, delta 0.05)

Each setting folder has `summary.md`, `tables.md`, the figures of `tool/report.py` (as experiments 1 and 2:
`figures/transitions=<n>/`, `figures/sample_efficiency`) and the pMDP pictures (`figures/pmdp_graph/`).

Normalised values: 0 = random policy, 1 = $V^*_{obs}$; means over seeds.

**SysAdmin, the declaration holds** (promise):

| N, transitions | true graph -> IMDP | DYNOTEARS -> IMDP | true graph -> pMDP (a) / (b) | DYNOTEARS -> pMDP (a) / (b) |
|---|---|---|---|---|
| 3, 2k | -1.20 | -1.20 | 0.41 / 0.46 | 0.41 / 0.46 |
| 6, 100 | -2.36 | -2.37 | -1.66 / -1.19 | -1.68 / -1.24 |
| 6, 2k | -1.87 | -1.87 | 0.12 / 0.41 | 0.12 / 0.41 |

- the pMDP's policy is near-optimal at every budget (true value 0.90-1.00), the IMDP's is below random at
  100 transitions; true graph and DYNOTEARS give almost the same pMDP: the gain is the tying, not the graph;
- (b) adds a gain that grows with N (+0.29 at N = 6); Storm gave the (b) bound in 80% of runs;
- the promise held in 100% of runs for every method.

**SysAdmin with the storm (control: the tying is wrong for machine 0).** The promise still held in 100%
of runs: in the observed model the storm lowers machine 0's chance to be up by about 8%, small next to
the promise's slack. The tying test rejected in 40% (true graph) and 55% (DYNOTEARS) of runs, against 7%
and 18% without the storm. The control shows the test sees the error; it does **not** show that the
promise survives a wrong declaration in general.

**Stock trading, the declaration holds** (promise; layouts sectors x stocks):

| layout, transitions | true graph -> IMDP | DYNOTEARS -> IMDP | true graph -> pMDP (a) / (b) | DYNOTEARS -> pMDP (a) / (b) |
|---|---|---|---|---|
| 1x2, 2k | 0.34 | 0.34 | 0.52 / 0.56 | 0.52 / 0.56 |
| 2x2, 500 | -1.45 | -1.70 | 0.17 / 0.34 | 0.12 / 0.21 |
| 2x3, 500 | -2.21 | -2.47 | 0.08 / 0.20 | -0.02 / 0.03 |
| 2x3, 2k | -0.44 | -0.49 | 0.41 / 0.49 | 0.41 / 0.49 |

- the same pattern as SysAdmin, smaller: the pMDP's promise is above the IMDP's everywhere, and its policy
  is better or equal (2x3 at 500 transitions: true value 0.84 against 0.40); here DYNOTEARS costs
  something at small budgets on the larger layouts (2x3, 500: 0.03 against 0.20 with the true graph);
- the promise held in 100% of runs for every method; the tying test rejected in 2% (true graph) and 12%
  (DYNOTEARS) of runs;
- Storm finished in 61% of the (b) runs: in 10% at 100 transitions (wide intervals that touch 0, up to
  ~10 parameters), in 74-80% from 500 on; elsewhere the Python bound was used. When Storm finished, its
  value matched the exact value at its theta to 3e-6.

## 6. How to read the results

- The pMDP arms assume structure **and** mechanism; the IMDP arms assume structure only. The comparison
  measures **the value of a correct mechanism declaration**, not a method that beats the others. Do not put
  these numbers next to the baselines of experiment 2.
- In a domain whose mechanisms are unknown, declaring $f$ is not defensible: the default is CBN -> IMDP.
  Without an assumption linking rows, an unseen row is [0, 1]; even "learning which rows to merge" carries
  the assumption that rows the data cannot tell apart are equal.
- Assumptions an engineer can state without knowing the probabilities, as a weaker alternative to T1-T3:
  symmetry by design ("these machines are the same model") and the direction of an effect ("a failed
  neighbour never makes recovery more likely"), each with a tolerance.

## 7. Next steps (not done)

- **A ladder of weaker assumptions**, from weakest to strongest: sparse factorisation (the CBN);
  context-specific independence within one table; monotonicity (inequalities between rows); bounded
  heterogeneity, "$\varepsilon$-tying": rows of the same kind differ by at most $\varepsilon$, i.e. the interval
  $[\theta_{lo} - \varepsilon, \theta_{hi} + \varepsilon]$ intersected with the row's own interval ($\varepsilon = 0$: this pMDP;
  $\varepsilon = 1$: the IMDP); smoothness; exact tying with a declared form (this folder). Report the
  promise as a curve over $\varepsilon$, with the lower bound on $\varepsilon$ that the data give.
- **Controls that break the assumption**: heterogeneous machines (each machine's probabilities shifted by
  a random +-0.05-0.15); a wrong form (P(up) falling with the *number* of neighbours down); a stronger storm.
- The tying test by parametric bootstrap on whole trajectories; selection on one half of the data and
  intervals on the other.
- Storm (stormpy 2.0) was installed in the venv with `uv pip install stormpy`; it is not in
  `pyproject.toml` / `uv.lock`. Add it as an optional extra if this work continues.

## Files

| File | Content |
|---|---|
| `tied.py` | domain-independent: pooled counts, theta's box (NBN Clopper-Pearson), the tying test, the factor bounds, option (b) (Python region refinement, sampled upper estimate, Storm on the sequentialised chain) |
| `templates.py` | the declared tyings: `SysAdminTemplate`, `StockTemplate` |
| `draw.py` | pictures of the learned pMDP: every edge labelled with its expression in theta and its interval (uses `plotting/mdp_graph.draw(..., edge_labels=...)`) |
| `run.py` | the experiment: settings `sysadmin`, `sysadmin_storm`, `stock`; results in the sweep's format, then `tool/report.py`'s tables and figures (as experiments 1 and 2) and `summary.md`; pictures in `figures/pmdp_graph/` |
| `../../tests/pmdp/test_tied.py` | the templates reproduce the true kernels; counts add up; the bounds come in the required order |

    uv pip install --python .venv/bin/python stormpy                 # once (option b)
    .venv/bin/python experiments/pmdp/run.py sysadmin sysadmin_storm stock
    .venv/bin/python experiments/pmdp/run.py stock --smoke           # 2 sizes, 100 / 500, 2 seeds
    .venv/bin/python experiments/pmdp/run.py --report results/pmdp/<run>/<setting>
