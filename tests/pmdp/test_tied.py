"""The pMDP of experiments/pmdp (tied.py, templates.py): on the true graph the templates reproduce the
true kernel, the pooled counts add up, the learned template is the true one with enough data, and the
bounds of options (a) and (b) come in the order soundness requires."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "pmdp"))
import templates  # noqa: E402
import tied  # noqa: E402

from scm_to_mdp.compilation import product_bounds  # noqa: E402
from scm_to_mdp.data import first_transitions  # noqa: E402
from scm_to_mdp.envs import StockTrading, SysAdmin  # noqa: E402
from scm_to_mdp.solving import robust_vi  # noqa: E402

GAMMA = 0.9


def _data(env, n_transitions: int):
    truth = env.ground_truth()
    table = env.sample(n_transitions // 64 + 2, 64, "random")
    return truth, truth.learner_spec.read(first_transitions(table, n_transitions))


def _sysadmin_theta_star(template) -> np.ndarray:
    return np.array([0.95, 0.90, 0.60, 0.05])  # SysAdmin's p_up per class (envs/sysadmin.p_up)


def _stock_theta_star(template, M: int) -> np.ndarray:
    theta = np.zeros(template.n_classes)
    theta[:4] = [1.0, 0.0, 1.0, 0.0]  # buy, sell, hold_owned, hold_not
    for r in range(M + 1):
        theta[template.offsets[M] + r] = 0.1 + 0.8 * r / M  # base + scale * rising / M
    return theta


@pytest.mark.parametrize("n_machines", [2, 3, 4])
def test_sysadmin_template_reproduces_the_true_kernel(n_machines):
    env = SysAdmin(n_machines, 0, topology="ring")
    truth, data = _data(env, 2000)
    template = templates.SysAdminTemplate.learn(data, truth.parents_in(data))
    assert template.repair == {f"m{env.repairs[j]}": j + 1 for j in range(n_machines)}
    theta = _sysadmin_theta_star(template)
    mdp = product_bounds.compile(tied.factor_bounds(data, template, theta, theta))
    kernel = truth.observed_kernel()[:, truth.actions_of(data), :]
    np.testing.assert_allclose(mdp.L, kernel, atol=1e-12)
    np.testing.assert_allclose(mdp.U, kernel, atol=1e-12)


@pytest.mark.parametrize("layout", [(1, 2), (2, 2)])
def test_stock_template_reproduces_the_true_kernel(layout):
    env = StockTrading(*layout, 0, scale=0.8)
    truth, data = _data(env, 2000)
    template = templates.StockTemplate.learn(data, truth.parents_in(data))
    theta = _stock_theta_star(template, layout[1])
    mdp = product_bounds.compile(tied.factor_bounds(data, template, theta, theta))
    kernel = truth.observed_kernel()[:, truth.actions_of(data), :]
    np.testing.assert_allclose(mdp.L, kernel, atol=1e-12)
    np.testing.assert_allclose(mdp.U, kernel, atol=1e-12)


def test_pooled_counts_add_up():
    env = SysAdmin(3, 0, topology="ring")
    truth, data = _data(env, 500)
    template = templates.SysAdminTemplate.learn(data, truth.parents_in(data))
    counts = tied.pooled_counts(data, template)
    assert counts.sum() == 500 * 3
    assert counts[:, 1].sum() == sum(int(data.df[m + "'"].sum()) for m in data.state_vars)


def test_bounds_are_ordered():
    """a <= b_py <= Storm's value; Storm's lower bound <= the exact value at theta* (theta* in the box),
    and Storm's value is the exact value at the theta it returns."""
    pytest.importorskip("stormpy")
    env = SysAdmin(3, 0, topology="ring")
    truth, data = _data(env, 500)
    template = templates.SysAdminTemplate.learn(data, truth.parents_in(data))
    lo, hi = tied.theta_box(tied.pooled_counts(data, template))
    theta_star = _sysadmin_theta_star(template)
    assert np.all((lo <= theta_star) & (theta_star <= hi))
    solution = robust_vi.solve(product_bounds.compile(tied.factor_bounds(data, template, lo, hi)), GAMMA)
    start = truth.start
    a = float(start @ solution.V)
    b_py = tied.region_refinement(data, template, lo, hi, solution.pi, GAMMA, start)
    storm = tied.storm_lower_bound(data, template, lo, hi, solution.pi, GAMMA, int(np.argmax(start)), 1e-3, 120)
    assert storm["storm_status"] == "ok"
    exact = tied.exact_value(data, template, theta_star, solution.pi, GAMMA, start)
    assert a <= b_py + 1e-9
    assert b_py <= storm["storm_value"] + 1e-6
    assert storm["storm_lower"] <= exact + 1e-6
    assert abs(storm["storm_value"] - storm["storm_value_check"]) < 1e-3
    assert storm["storm_lower"] <= tied.sampled_upper(data, template, lo, hi, solution.pi, GAMMA, start) + 1e-6
