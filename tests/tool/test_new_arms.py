"""The arms added for clear27, end to end on the shared SysAdmin(3) data: Schnitzer, Abate, Parker's three
relaxations (S7/S8) as robust VI and as nature's MDP (worst / best), Factored SPIBB on any graph, the KWIK
structure learners (Strehl et al. 2007, Met-Rmax) in every S4 mode, and discovery without same-step links."""

import numpy as np
import pytest

from scm_to_mdp.tool import Method, evaluate, learn, solve

GAMMA = 0.9
TRUE = {"discovery": "oracle", "selection": "given"}
STREHL = {"discovery": "strehl", "discovery_options": {"k": 2, "m": 20}}
MET = {"discovery": "met_rmax", "discovery_options": {"D": 2, "m": 7}}

ARMS = {
    # the true graph -> the three relaxations (robust VI), nature's MDP of each, Factored SPIBB
    "ia": Method("causal", "box_robust_vi", **TRUE),
    "mccormick": Method("causal", "mccormick_robust_vi", **TRUE),
    "exact": Method("causal", "exact_robust_vi", **TRUE),
    **{
        f"{c} {r}": Method("causal", f"{c}_mdp_vi", solver_options={"relaxation": r}, **TRUE)
        for c in ("worst", "best")
        for r in ("ia", "mccormick", "exact")
    },
    "plug-in": Method("causal", "vi", **TRUE),
    "factored spibb": Method("factored_spibb", "spibb", **TRUE),
    # discovery without same-step links, then a compilation
    "pcmci+ lagged": Method(
        "causal",
        "box_robust_vi",
        discovery="pcmci+",
        selection="bic_best",
        discovery_options={"same_step": False},
    ),
    "pcmci+ lagged -> factored spibb": Method(
        "factored_spibb",
        "spibb",
        discovery="pcmci+",
        selection="bic_best",
        discovery_options={"same_step": False},
    ),
    # KWIK structures, every S4 mode
    "strehl fixed": Method("kwik", "exact_robust_vi", learning_options={"hull": "fixed"}, **STREHL),
    "strehl accepted": Method("kwik", "exact_robust_vi", **STREHL),
    "strehl kwik": Method("kwik", "box_robust_vi", learning_options={"mode": "kwik"}, **STREHL),
    "strehl point": Method("kwik", "vi", learning_options={"mode": "point"}, **STREHL),
    "strehl spibb": Method("kwik", "spibb", learning_options={"mode": "spibb"}, **STREHL),
    "met accepted": Method("kwik", "mccormick_robust_vi", **MET),
    "met worst": Method("kwik", "worst_mdp_vi", **MET),
}


@pytest.fixture(scope="module")
def runs(truth, dataset):
    shared = {}
    out = {}
    for name, method in ARMS.items():
        result = solve(learn(dataset, method, true_parents=truth.true_parents, shared=shared), dataset, GAMMA)
        assert result.ok, (name, result.error)
        out[name] = (result, evaluate(result, dataset, truth, GAMMA))
    return out


def test_every_new_arm_runs_and_no_policy_beats_the_optimum(runs):
    for name, (_, ev) in runs.items():
        assert ev["regret"] >= -1e-6, name


def test_true_graph_robust_promises_are_lower_bounds(runs):
    for name in ("ia", "mccormick", "exact", "worst ia", "worst mccormick", "worst exact"):
        _, ev = runs[name]
        assert ev["promise"] <= ev["true"] + 1e-6, name


def test_relaxation_ladder_and_nature_mdps(runs):
    """exact >= McCormick >= IA (Schnitzer et al.: tighter relaxations, a larger robust value); nature's
    worst MDP has the robust value of its relaxation; its best MDP is at least that."""
    V = {name: r.solution.V for name, (r, _) in runs.items()}
    assert np.all(V["exact"] >= V["mccormick"] - 1e-6) and np.all(V["mccormick"] >= V["ia"] - 1e-6)
    for r in ("ia", "mccormick", "exact"):
        assert np.allclose(V[f"worst {r}"], V[r], atol=1e-5), r
        assert np.all(V[f"best {r}"] >= V[r] - 1e-6), r


def test_one_discovery_per_structure(runs):
    """Every KWIK mode on the same learner reuses one structure (the same discovery record)."""
    times = {runs[n][0].steps["discovery"]["seconds"] for n in runs if n.startswith("strehl")}
    assert len(times) == 1
    assert runs["strehl accepted"][0].info["kwik_dont_know"] == runs["strehl point"][0].info["kwik_dont_know"]


def test_flat_nature_mdps_match_flat_robust_vi(truth, dataset):
    """The flat IMDP is one factor: its worst MDP has the robust value, its best MDP is above it."""
    V = {}
    for solver in ("robust_vi", "worst_mdp_vi", "best_mdp_vi"):
        r = solve(learn(dataset, Method("flat_rmdp", solver)), dataset, GAMMA)
        assert r.ok, (solver, r.error)
        V[solver] = r.solution.V
    assert np.allclose(V["worst_mdp_vi"], V["robust_vi"], atol=1e-5)
    assert np.all(V["best_mdp_vi"] >= V["robust_vi"] - 1e-6)


def test_factored_solvers_are_not_applicable_with_same_step_links(truth, dataset):
    """Schnitzer et al.'s factors depend on (s, a) only (Eq. 1): with a same-step link the run's status is
    "not_applicable" (not an error), in process and under limits."""
    from scm_to_mdp.tool import Limits

    parents = dict(truth.true_parents)
    parents["m1"] = parents["m1"] + ("m0'",)  # m1' gets m0' as a same-step parent
    method = Method("causal", "exact_robust_vi", discovery="oracle", selection="given")
    for limits in (None, Limits(mem_gb=4.0, timeout_s={})):
        r = solve(learn(dataset, method, limits, true_parents=parents), dataset, GAMMA, limits)
        assert r.status == "not_applicable" and r.failed_step == "solve", (limits, r.status, r.error)
