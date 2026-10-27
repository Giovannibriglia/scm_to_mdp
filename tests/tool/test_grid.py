"""The grid of experiments (tool/grid.py); the experiment scripts run in test_experiment_scripts.py."""

import pickle

from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Grid, Policy, experiments, standard_methods

# ---- behaviour policies -----------------------------------------------------------------------


def make_env(n, topology, seed):
    return SysAdmin(int(n), seed, topology=topology, storm=0.3)


def test_grid_expands_to_one_experiment_per_setting(tmp_path):
    grid = Grid(
        name="g", make_env=make_env, sizes=[2], knobs=["ring", "star"],
        policies=[Policy("uniform"), Policy("expert"), Policy("confounded")],
        n_transitions=[10], seeds=[0], methods=standard_methods(), horizon=5, out_dir=tmp_path,
        skip=lambda knob, policy: knob == "star" and policy.name == "confounded",
    )  # fmt: skip
    exps = experiments(grid)
    assert len(exps) == 5
    assert len({e.out_dir for e in exps}) == 5
    assert {e.policy for e in exps} == {"uniform", "expert ε=0.05", "confounded ε=0.05 s=0.95"}
    pickle.dumps(exps[0].make_env), pickle.dumps(exps[0].collect), pickle.dumps(exps[0].behaviour)
    env = exps[0].make_env(2, 0)
    table = exps[0].collect(env, 12)  # at least 12 transitions: whole episodes of at most 5 steps
    assert env.topology == "ring" and len(table) >= 12
    assert table.index.get_level_values("episode").nunique() == 3 and table.index.is_unique
