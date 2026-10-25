"""Behaviour policies, coverage and confounding bias (behaviour.py), the observed optimum and the
environments' hidden variables."""

import numpy as np
import pytest

from scm_to_mdp.behaviour import behaviour_metrics, data_occupancy
from scm_to_mdp.envs import Chemistry, Sepsis, SysAdmin

# ---- behaviour policies -----------------------------------------------------------------------


def test_expert_puts_one_minus_epsilon_on_the_observed_optimum():
    env = SysAdmin(3, 0, storm=0.3)
    truth = env.ground_truth()
    probs = env.behaviour("expert", 0.9, epsilon=0.1)
    pi_obs, _ = truth.observed_optimal(0.9)
    A = len(truth.actions)
    assert np.allclose(probs.sum(1), 1)
    chosen = probs[np.arange(truth.n_states), pi_obs[truth.observed_index()]]
    assert np.allclose(chosen, 0.9 + 0.1 / A)
    # the expert reads the observed state only: equal rows for the two storm values
    assert np.allclose(probs[0::2], probs[1::2])


def test_only_the_confounded_logger_biases_the_data():
    env = SysAdmin(3, 0, storm=0.3)
    truth = env.ground_truth()
    for name in ("uniform", "expert"):
        assert behaviour_metrics(truth, env.behaviour(name, 0.9), 20, 0.9)["bias_max"] < 1e-12
    assert behaviour_metrics(truth, env.behaviour("confounded", 0.9), 20, 0.9)["bias_mean"] > 0.01


def test_chemistry_without_reactions_is_a_negative_control():
    env = Chemistry(0.0, 0, n_objects=3, catalyst=0.3)
    assert env.exposed is None  # no object has parents: the catalyst acts on nothing
    metrics = behaviour_metrics(env.ground_truth(), env.behaviour("confounded", 0.9), 20, 0.9)
    assert metrics["bias_max"] < 1e-12


def test_coverage_numbers():
    env = SysAdmin(3, 0)
    truth = env.ground_truth()
    uniform = behaviour_metrics(truth, env.behaviour("uniform", 0.9), 20, 0.9)
    assert uniform["unvisited_pairs"] == 0 and 1 <= uniform["C_star"] < np.inf
    nu = data_occupancy(truth, env.behaviour("uniform", 0.9), 20)
    assert np.isclose(nu.sum(), 1) and (nu >= 0).all()


def test_the_sampler_follows_the_table():
    env = SysAdmin(2, 0, storm=0.3)
    probs = env.behaviour("confounded", 0.9, epsilon=0.05, strength=1.0)
    df = env.sample(400, 5, probs)  # full table, with the storm column
    storm = df[df["storm"] == 1]
    repairs_m0 = storm["reboot"] == env.repairs.index(0) + 1  # the value of reboot that repairs machine 0
    assert repairs_m0.all()  # strength 1: every storm step repairs machine 0


# ---- the observed optimum ---------------------------------------------------------------------


def test_observed_optimum():
    plain = SysAdmin(3, 0).ground_truth()
    pi, V = plain.observed_optimal(0.9)
    assert np.allclose(V, plain.optimal_values(0.9))
    hidden = SysAdmin(3, 0, storm=0.3, storm_effect=0.8).ground_truth()
    _, V_obs = hidden.observed_optimal(0.9)
    assert hidden.start @ V_obs <= hidden.start @ hidden.optimal_values(0.9) + 1e-9


# ---- environments -----------------------------------------------------------------------------


def test_hidden_variables_stay_out_of_the_data():
    for env in (SysAdmin(3, 0, storm=0.3), Chemistry(0.4, 0, n_objects=3, catalyst=0.3)):
        df = env.hide(env.sample(5, 3, env.behaviour("uniform", 0.9)))
        assert env.hidden_vars[0] not in df.columns
        env.spec().check(df)


def test_storm_and_catalyst_off_change_nothing():
    a, b = SysAdmin(3, 0), SysAdmin(3, 0, storm=None)
    assert np.array_equal(a.P, b.P)
    c = Chemistry(0.4, 0, n_objects=3)
    assert c.ground_truth().n_states == 27


def test_sepsis_diabetes_effect():
    original = Sepsis("custom", 0, vitals=("hr", "sysbp"), diabetes=True)
    assert np.array_equal(original.P, Sepsis("custom", 0, vitals=("hr", "sysbp"), diabetes=True,
                                             diabetes_effect=1.0).P)  # fmt: skip
    none = Sepsis("custom", 0, vitals=("hr", "sysbp"), diabetes=True, diabetes_effect=0.0)
    d = none.state_vars.index("diabetic")
    diabetic, healthy = none.B[:, d] == 1, none.B[:, d] == 0
    # with k = 0 a diabetic's vitals move as a non-diabetic's (diabetes itself stays)
    other = [i for i in range(len(none.state_vars)) if i != d]
    marginal = lambda rows: none.next_distributions(none.B[rows], np.zeros(rows.sum(), int))[:, other]  # noqa: E731
    assert np.allclose(marginal(diabetic), marginal(healthy))
    with pytest.raises(ValueError):
        Sepsis("full", 0, diabetes_effect=2.0)


# ---- the grid ---------------------------------------------------------------------------------
