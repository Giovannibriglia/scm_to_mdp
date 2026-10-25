"""The protocol shared by every complete grid of experiment 1 (one script per domain, *_complete.py).

What the grids test (the claims), and how:
    A  sample efficiency   an RMDP learned through causality (discovery + interval learning) reaches a
                           near-optimal policy with a promise that holds at fewer rollouts than flat
                           RMDP and the baselines: rollouts_needed, the validity-gated comparison, the
                           certificate V_opt - V_rob; against the point models (X -> MDP, flat MDP) in
                           true value and in guaranteed value, reported apart. Its controls separate
                           "the right structure" from "a small model" (discovery/controls.py): the
                           logger-Markov graph (fits the logged data, not the mechanisms), a random
                           graph matched in free parameters, the true graph + 1 parent; and, on 0/1
                           domains, the cost of the box (true graph -> RMDP with the per-(s,a) product).
    B  transfer            B1 new rewards on the same dynamics (results_transfer.csv), read against the
                           data's coverage of each task's rows (cov_joint_k - cov_factor_k: rows that only
                           a factored model can estimate); B2 dynamics shift (results_shift.csv: the
                           hidden variable more frequent at deployment) is robustness, NOT causal
                           transfer (a single-mechanism change with new data is experiment 2).
    C  understanding       the learned model against the true kernel under the logged visits, the best
                           policy's occupancy and uniform weights (tv_*, kl_*, contain_*, width_*), and
                           the structure (SHD, SID, action recall).
    control                the confounded logger: every model-based promise is expected to fail once the
                           intervals are narrower than the confounding bias (not evidence for A-C).

The same for every domain:
    loggers     uniform; expert with eps = 0.05; expert with eps = 0 (the thinnest coverage); confounded
                (the expert, plus the confounded action w.p. 0.95 while the hidden variable is on) where
                the domain has a hidden variable
    hidden      a hidden variable drawn afresh at every step is on w.p. TRAIN_HIDDEN in the data, and w.p.
                SHIFT_HIDDEN at deployment (the shift); it acts on ONE next-state variable, so the true
                graph's factored model of the observed variables stays exact
    seeds       5 (descriptive: a promise held in 5/5 seeds has a 95% Clopper-Pearson lower end of 0.48;
                experiments/exp1/calibration.py runs 100 seeds on small settings to test 1 - delta)
    transfer    every domain's alternative rewards (re-planned, not re-learned)
    methods     tool.standard_methods (ours, the references, the controls, the baselines, deep included)

Expectations common to every domain, written 2026-10-24 before the runs (after a one-seed pilot on Frozen
Lake), with the reasons from the references (AJKS: Agarwal et al., RL Theory and Algorithms, 2026 draft;
ECI: Peters, Janzing, Scholkopf, Elements of Causal Inference, 2017; K&F: Koller & Friedman, PGM, 2009):
    A1  uniform logger: the point models match or beat the robust arms in true value (plug-in is
        minimax-optimal with good coverage, AJKS Thm 2.6-2.7, pp. 26-27); among the arms whose promise
        holds, causal -> RMDP needs fewer rollouts than flat RMDP where the domain has structure
        (fewer parameters: a finite-sample effect, ECI p. 118; K&F Thm 17.4, p. 775).
    A2  expert eps = 0: Laplace's guess for the rows the expert never tries can be optimistic and lead
        the point models off the expert's path (no method can identify unseen actions, AJKS Prop 4.6,
        p. 57); the robust arms keep to the expert. The difference shows in true value where there is
        a trap.
    A3  controls: the logger-Markov graph is the true graph under the uniform and eps = 0.05 loggers and
        differs under eps = 0 (ECI p. 206), where its promise can fail; the random matched graph's
        promise fails (a missing parent: K&F Thm 17.3, p. 774); true graph + 1 keeps its promise and
        needs more rollouts (fragmentation, K&F pp. 784-785).
    B1  the transfer gain of causal -> RMDP over flat RMDP grows with cov_joint - cov_factor.
    C1  tv_uniform (and contain_uniform at a given width) separates the models more than tv_data: the
        flat models know nothing off the data, the factored ones extrapolate where every factor's row
        was seen.
"""

from scm_to_mdp.tool import Policy

SEEDS = [0, 1, 2, 3, 4]
DELTA = 0.05
TRAIN_HIDDEN = 0.1  # P(hidden variable on) at every step, in the data
SHIFT_HIDDEN = [0.3, 0.6]  # ... at deployment (the dynamics shift)
N_ROLLOUTS = [10, 25, 50, 100, 200, 400, 1600]
DEEP_BASELINES = True  # offline DQN and CQL (a run: M5 CPU ~12 s; Xeon E5 CPU ~100 s, CUDA ~10 s)


def policies(confounded: bool = True) -> list[Policy]:
    """The loggers of every grid (the confounded one only where the domain has a hidden variable)."""
    out = [Policy("uniform"), Policy("expert", epsilon=0.05), Policy("expert", epsilon=0.0)]
    return out + ([Policy("confounded", epsilon=0.05, strength=0.95)] if confounded else [])
