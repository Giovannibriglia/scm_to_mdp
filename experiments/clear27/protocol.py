"""clear27: from the graph to the policy, every graph source through the same compilations (S2-S9).

Data and evaluation as experiments 1 and 2 (experiments/common.py: uniform logger, 10 seeds, the same MDP,
the nominal start and 20 random starts), with budgets of 10 ... 500k transitions (N_TRANSITIONS) and the
domains SysAdmin (N = 2..11), Sepsis (2..4 vitals), Drone (side 3..12) and Aircraft (4x3 .. 16x12).

Limits per step (LIMITS, the same for every domain and size; the oracle has the memory limit only):
discovery / selection / learning / training 600 s, compilation (relax) 300 s, solve 400 s, memory 6 GB
(PSS on Linux, physical footprint on macOS: utils/limits.py). Chosen with the timing pilot and the worst-problem
simulation of 2026-11-04 (laptop and server): McCormick times out on the larger problems and exact runs
out of memory on Sepsis 4, Drone and Aircraft (its vertex products), by design; everything else fits
except SL-SPIBB at SysAdmin N >= 10 with 500k and flat at N = 11 with the large budgets.

SMOKE=1 (environment): a tiny grid of the same script (its smallest size, 100 and 1000 transitions,
seed 0) into results/clear27_smoke, to check every script runs before the real launch.

"Point 2", the compilation block, applied to every graph source (COMPILATIONS):

    IMDP    Clopper-Pearson intervals (delta = DELTA; for a learned graph with the post-selection union
            bound over its candidate family, Agarwal et al., as in Lemma A.20, with Lemma 6.2 for the random
            counts: n_candidates), then one of
            Schnitzer, Abate, Parker's relaxations (AAAI 2026, arXiv:2508.00707), then robust VI:
                IA         interval arithmetic, the box of the products (Eq. 11)
                McCormick  the recursive McCormick relaxation (Eqs. 12a-d, App. B.2)
                exact      the vertex products (Thm 1)
    MDP     (a) plug-in   point estimates (Laplace), value iteration; one MDP for the three relaxations
            (b) worst     nature's choice at the robust fixed point, per relaxation, then VI (Badings,
                          RVSS Sec. 3.3.2, p. 42: fixing nature gives an MDP; Prop. 3.33-3.35, pp. 44-45:
                          the same value as the IMDP arm, and the robust policy is optimal in it up to ties)
            (c) best      nature's choice at the optimistic fixed point, per relaxation, then VI: an upper
                          bound when the truth is in the set (Suilen et al. 2024, p. 12; Jackson et al.
                          2021, Thm 2; Badings, RVSS Lemma 3.32); its policy lacks the max-min guarantee
                          of the IMDP arm
            Factored SPIBB  Simao & Spaan (AAAI 2019) on the same graph, solved by Pi_b-SPIBB policy
                          iteration, not VI (baselines/spibb.py); its guarantee is an improvement over the
                          baseline w.h.p., not a value bound, and its Thm 1 needs the graph given in advance:
                          on a learned graph it is an empirical arm

The pools, one script per (domain, source) so that each discovery runs once per dataset and every
compilation of it reuses its graph (tool/methods.py: _shared_key). Every script also runs the oracle
(pool 1: the true observed MDP by VI, V*_obs and the solve budget's yardstick):

    truegraph       pool 2: the true graph -> point 2
    dynotears, dynotears_lagged, pcmci, pcmci_lagged
                    pool 3: DYNOTEARS / PCMCI+ with and without same-step links -> point 2
    strehl, met     pool 3: Strehl, Diuk, Littman (AAAI 2007) / Met-Rmax (Diuk, Li, Leffler, ICML 2009):
                    without KWIK (our S4 on their structure, hull "fixed" or "accepted") -> IMDP, worst,
                    best; with KWIK (their predictions +- accuracy) -> IMDP, worst, best; their point
                    predictions with HALT for "don't know" -> MDP; Factored SPIBB with the "don't know"
                    pairs bootstrapped. m at the papers' theorem level (KWIK_OPTIONS)
    fullfact, sparse, flat
                    pool 4: the full factored graph, the sparse graph -> point 2; flat (one factor: the
                    three relaxations coincide) -> IMDP, plug-in, worst, best (flat -> Factored SPIBB is
                    flat SPIBB, in pool 5)
    sota            pool 5: DiFaRL, FOCUS, SL-SPIBB, flat SPIBB, offline DQN, offline CQL

A script: experiments/clear27/<domain>_<source>.py; results: results/clear27/<domain>_<source>_<date>_<time>.
A domain's 11 sources in one pool of workers: bash experiments/clear27/run_<domain>.sh (run_domain.sh ->
run_pool.py: a free worker takes the next job of any source).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/: common.py, domains.py

import common  # noqa: E402
from common import main  # noqa: E402, F401
from domains import DOMAINS  # noqa: E402

from scm_to_mdp.tool import Grid, Limits, Method, oracle_methods, standard_methods  # noqa: E402
from scm_to_mdp.tool.methods import FIXED_GRAPHS, POINT_ALPHA  # noqa: E402

EXPERIMENT = "clear27"
N_TRANSITIONS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000,
                 500_000]  # fmt: skip
LIMITS = Limits(mem_gb=6.0, timeout_s={"discovery": 600.0, "selection": 600.0, "learning": 600.0,
                                       "relax": 300.0, "training": 600.0, "solve": 400.0})  # fmt: skip
SOLVE_FLOOR_S = 400.0  # the relative solve limit (3 x the oracle's sweeps) is far below it in every domain
# the models (and their pictures) of the first seed at every budget: SysAdmin only ("all": about 25-35 GB,
# mostly full-fact's, which stores every parent row); the other domains keep the results only, from which
# every figure of the report is made (the report reads results.csv, never the models)
SAVE_MODELS = {"sysadmin": "all"}
SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TRANSITIONS = [100, 1000]
DELTA = common.DELTA
SIZES = {  # the domains and their sizes (domains.py's, without Aircraft 18x13)
    "sysadmin": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "sepsis": DOMAINS["sepsis"].sizes,
    "drone": DOMAINS["drone"].sizes,
    "aircraft": [s for s in DOMAINS["aircraft"].sizes if s != "18x13"],
}
RELAXATIONS = {"IA": ("box_robust_vi", "ia"), "McCormick": ("mccormick_robust_vi", "mccormick"),
               "exact": ("exact_robust_vi", "exact")}  # fmt: skip
PCMCI_ALPHA = 0.01  # as experiments 1 and 2
MET_M = 50  # Met-Rmax's m (Alg. 1, line 18): the paper's SysAdmin setting; Thm 1 gives no constant
KWIK_EPS = 0.2  # Strehl's eps1 and Met-Rmax's eps (the papers' settings)


def set_arms(source: str, base: dict, learning: dict) -> dict:
    """The IMDP arms and nature's MDPs (worst, best) of one set-valued source: base = Method's learner,
    discovery, selection, options; learning = its learning options."""
    out = {}
    for name, (robust, _) in RELAXATIONS.items():
        out[f"{source} -> IMDP ({name})"] = Method(solver=robust, learning_options=learning, **base)
    for choice in ("worst", "best"):
        for name, (_, relaxation) in RELAXATIONS.items():
            solver_options = {"relaxation": relaxation}
            out[f"{source} -> MDP ({choice}, {name})"] = Method(
                solver=f"{choice}_mdp_vi", learning_options=learning, solver_options=solver_options, **base
            )
    return out


def graph_arms(source: str, discovery: str, domain, options: dict | None = None) -> dict:
    """Point 2 on one graph source (a discovery, the true graph or a fixed graph)."""
    options = options or {}
    learned = discovery not in ("oracle",) + FIXED_GRAPHS
    selection = "bic_best" if learned else "given"
    base = {"learner": "causal", "discovery": discovery, "selection": selection, "discovery_options": options}
    # a learned graph: the union bound over every parent set it could have chosen (with same-step links,
    # the other next-state variables are candidates too)
    family = ("inputs+next" if options.get("same_step", True) else "inputs") if learned else 1
    out = set_arms(source, base, {"delta": DELTA, "n_candidates": family})
    out[f"{source} -> MDP (plug-in)"] = Method(solver="vi", learning_options={"alpha": POINT_ALPHA}, **base)
    out[f"{source} -> Factored SPIBB"] = Method(
        "factored_spibb", "spibb", discovery=discovery, selection=selection, discovery_options=options,
        learning_options={"m": domain.sl_spibb_m, "behaviour": "uniform"},
    )  # fmt: skip
    return out


def kwik_options(learner: str, domain) -> dict:
    """The KWIK learners at their theorems' level: k (D) at least the true in-degree (SL-SPIBB's d),
    Strehl's m and Met-Rmax's m_sub calibrated at delta (kwik_structure.strehl_calibrated_m, met_sub_m)."""
    if learner == "strehl":
        return {"k": domain.sl_spibb_parents, "m": "calibrated", "eps1": KWIK_EPS, "delta": DELTA}
    return {"D": domain.sl_spibb_parents, "m": MET_M, "m_sub": "calibrated", "eps": KWIK_EPS, "delta": DELTA}


def kwik_arms(learner: str, domain) -> dict:
    title = {"strehl": "Strehl", "met": "Met-Rmax"}[learner]
    base = {"learner": "kwik", "discovery": "strehl" if learner == "strehl" else "met_rmax",
            "discovery_options": kwik_options(learner, domain)}  # fmt: skip
    out = {}
    for hull in ("fixed", "accepted"):  # without KWIK: our S4 on the structure
        out |= set_arms(f"{title} ({hull} hull)", base, {"mode": "s4", "hull": hull, "delta": DELTA})
    out |= set_arms(f"{title}-KWIK", base, {"mode": "kwik"})  # with KWIK: the learner's own inference
    out[f"{title}-KWIK -> MDP (plug-in)"] = Method(solver="vi", learning_options={"mode": "point"}, **base)
    out[f"{title}-KWIK -> Factored SPIBB"] = Method(
        solver="spibb", learning_options={"mode": "spibb", "behaviour": "uniform"}, **base
    )
    return out


def flat_arms() -> dict:
    flat = {"flat IMDP": Method("flat_rmdp", "robust_vi"), "flat -> MDP (plug-in)": Method("flat_mdp", "vi")}
    for choice in ("worst", "best"):
        flat[f"flat -> MDP ({choice})"] = Method("flat_rmdp", f"{choice}_mdp_vi")
    return flat


SOTA = ("DiFaRL", "FOCUS", "SL-SPIBB", "flat SPIBB", "offline DQN", "offline CQL")


def sota_arms(domain) -> dict:
    every = standard_methods(
        max_parents=None, delta=DELTA, exact=False, deep=True, state_parents=domain.sl_spibb_parents,
        spibb_options={"m": domain.spibb_m, "behaviour": "uniform"},
        sl_spibb_options={"m": domain.sl_spibb_m, "eps": domain.sl_spibb_eps, "behaviour": "uniform"},
    )  # fmt: skip
    return {name: every[name] for name in SOTA}


SOURCES = ("truegraph", "dynotears", "dynotears_lagged", "pcmci", "pcmci_lagged", "strehl", "met",
           "fullfact", "sparse", "flat", "sota")  # fmt: skip


def source_methods(source: str, domain) -> dict:
    """The methods of one script (module docstring), the oracle first."""
    pc, lagged = {"alpha": PCMCI_ALPHA}, {"same_step": False}
    arms = {
        "truegraph": lambda: graph_arms("true graph", "oracle", domain),
        "dynotears": lambda: graph_arms("DYNOTEARS", "dynotears", domain),
        "dynotears_lagged": lambda: graph_arms("DYNOTEARS (lagged)", "dynotears", domain, lagged),
        "pcmci": lambda: graph_arms("PCMCI+", "pcmci+", domain, pc),
        "pcmci_lagged": lambda: graph_arms("PCMCI+ (lagged)", "pcmci+", domain, pc | lagged),
        "strehl": lambda: kwik_arms("strehl", domain),
        "met": lambda: kwik_arms("met", domain),
        "fullfact": lambda: graph_arms("full-fact", "full", domain),
        "sparse": lambda: graph_arms("sparse", "sparse", domain, {"actions": domain.sparse_actions}),
        "flat": flat_arms,
        "sota": lambda: sota_arms(domain),
    }
    if source not in arms:
        raise ValueError(f"source must be one of {SOURCES}, not {source!r}")
    return {"oracle -> MDP": oracle_methods()["oracle -> MDP"]} | arms[source]()


def experiment_grid(domain: str, source: str) -> Grid:
    d = DOMAINS[domain]
    smoke = SMOKE
    return Grid(
        name=d.title,
        make_env=d.make_env,
        sizes=SIZES[domain][:1] if smoke else SIZES[domain],
        size_name=d.size_name,
        knobs=d.knobs,
        knob_name=d.knob_name,
        constants=d.constants,
        policies=common.POLICIES,
        n_transitions=SMOKE_TRANSITIONS if smoke else N_TRANSITIONS,
        seeds=[0] if smoke else common.SEEDS,
        horizon=d.horizon,
        failure_horizon=common.FAILURE_HORIZON,
        gamma=d.gamma,
        n_starts=common.N_STARTS,
        methods=source_methods(source, d),
        limits=LIMITS,
        solve_factor=common.SOLVE_FACTOR,
        solve_floor_s=SOLVE_FLOOR_S,
        save_models="none" if smoke else SAVE_MODELS.get(domain, "none"),
        workers="auto",
        out_dir=common.ROOT / "results" / (EXPERIMENT + ("_smoke" if smoke else "")) / f"{domain}_{source}",
        reward_transfer=False,
        reward_family=None,
        shifts=None,
    )
