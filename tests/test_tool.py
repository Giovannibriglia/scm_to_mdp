"""The tool: methods (learn + solve), evaluate, sweep and report."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scm_to_mdp.tool import Method, evaluate, learn, solve

GAMMA = 0.9
ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "BIC -> RMDP": Method(
        "causal", "box_robust_vi", discovery="bic", selection="bic_best", discovery_options={"max_parents": 3}
    ),
    "Schnitzer exact": Method("causal", "vertex_robust_vi", discovery="oracle", selection="given"),
    "flat RMDP": Method("flat_rmdp", "robust_vi"),
    "flat MDP": Method("flat_mdp", "vi"),
    "DiFaRL": Method("difarl", "fqi", solver_options={"max_iter": 30}),
}


def test_every_method_learns_solves_and_is_evaluated(truth, dataset):
    for name, method in METHODS.items():
        result = solve(learn(dataset, method, true_parents=truth.true_parents), dataset, GAMMA)
        assert result.ok, (name, result.error)
        out = evaluate(result, dataset, truth, GAMMA)
        assert out["regret"] >= -1e-6, name  # no policy beats the optimal one
        if method.guarantee:
            assert out["promise"] <= out["true"] + 1e-6, name  # the true graph, or no graph: valid
        if method.learner == "causal":
            assert out["shd"] == 0


def test_roles_and_guarantees():
    assert METHODS["BIC -> RMDP"].role == "ours" and METHODS["BIC -> RMDP"].guarantee
    assert METHODS["Schnitzer exact"].role == "oracle"
    assert {METHODS[m].role for m in ["flat RMDP", "flat MDP", "DiFaRL"]} == {"baseline"}
    assert (
        not METHODS["flat MDP"].guarantee
        and not METHODS["DiFaRL"].guarantee
        and METHODS["flat RMDP"].guarantee
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"learner": "magic", "solver": "vi"}, "learner must be"),
        ({"learner": "flat_mdp", "solver": "fqi"}, "is solved with"),
        ({"learner": "causal", "solver": "box_robust_vi", "discovery": "bic"}, "needs selection"),
        (
            {"learner": "causal", "solver": "box_robust_vi", "discovery": "bic", "selection": "given"},
            "goes with",
        ),
        (
            {
                "learner": "causal",
                "solver": "box_robust_vi",
                "discovery": "bic",
                "selection": "bic_best",
                "discovery_options": {"alpha": 0.1},
            },
            "unknown options",
        ),
        (
            {"learner": "flat_rmdp", "solver": "robust_vi", "learning_options": {"delt": 0.1}},
            "unknown options",
        ),
        ({"learner": "flat_rmdp", "solver": "robust_vi", "discovery": "bic"}, "no discovery"),
    ],
)
def test_bad_methods_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Method(**kwargs)


def test_oracle_needs_the_true_parents(dataset):
    with pytest.raises(ValueError, match="true_parents"):
        learn(dataset, METHODS["Schnitzer exact"])


def test_without_ground_truth_only_the_promise(dataset):
    result = solve(learn(dataset, METHODS["flat RMDP"]), dataset, GAMMA)
    assert set(evaluate(result, dataset)) == {"promise"}


def test_solving_the_truth_gives_regret_zero(truth, dataset):
    """The evaluation checked on the true model itself (every action vector seen in the data)."""
    from scm_to_mdp.solving import value_iteration
    from scm_to_mdp.tool.methods import Result

    assert len(dataset.actions) == len(truth.actions)
    order = truth.actions_of(dataset)  # learner action k = true action order[k]
    P = truth.P[:, order]
    result = Result(METHODS["flat MDP"], solution=value_iteration.solve(P, dataset.reward, GAMMA))
    out = evaluate(result, dataset, truth, GAMMA)
    assert abs(out["regret"]) < 1e-6 and abs(out["gap"]) < 1e-6 and np.isclose(out["optimal_share"], 1.0)


def test_limited_steps_report_a_timeout(dataset):
    from scm_to_mdp.tool import Limits

    slow = Method("causal", "box_robust_vi", discovery="ges", selection="bic_best")  # GES takes ~1 s here
    result = learn(dataset, slow, Limits(mem_gb=4, timeout_s={"discovery": 0.05}))
    assert result.status == "timeout" and result.failed_step == "discovery" and result.model is None


SWEEP = """
import sys
from pathlib import Path
from scm_to_mdp.envs import SysAdmin
from scm_to_mdp.tool import Experiment, Limits, Method, run_experiment, report

def make_env(size, seed):
    return SysAdmin(size, seed)

def collect(env, n_rollouts):
    return env.sample(n_rollouts, 10, "random")

exp = Experiment(
    name="tiny", make_env=make_env, collect=collect, sizes=[2, 3], n_rollouts=[20, 60], seeds=[0, 1],
    methods={
        "BIC -> RMDP": Method("causal", "box_robust_vi", discovery="bic", selection="bic_best"),
        "flat MDP": Method("flat_mdp", "vi"),
    },
    limits=Limits(mem_gb=2), workers=int(sys.argv[2]), out_dir=Path(sys.argv[1]), size_name="N",
    pictures_max_states=4,
)
if __name__ == "__main__":
    rows = run_experiment(exp)
    report(exp.out_dir, show=False)
    print(len(rows))
"""


def run_sweep(tmp_path, workers):
    script = tmp_path / "sweep.py"
    script.write_text(SWEEP)
    out = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "out"), str(workers)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return int(out.stdout.strip().splitlines()[-1])


def test_sweep_resume_and_report(tmp_path):
    """2 sizes x 2 seeds x 2 rollouts x 2 methods = 16 rows; a second run resumes (no new rows)."""
    import pandas as pd

    assert run_sweep(tmp_path, workers=2) == 16
    out = tmp_path / "out"
    res = pd.read_csv(out / "results.csv")
    assert set(res["status"]) == {"ok"} and set(res["role"]) == {"ours", "baseline"}
    assert (res.groupby(["size", "seed", "n_rollouts"])["transitions"].nunique() == 1).all()
    assert (res[res["n_rollouts"] == 20]["transitions"] == 200).all()  # 20 rollouts of 10 steps
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_rollouts"] == [20, 60] and meta["size_labels"] == ["N=2", "N=3"]
    for n in [20, 60]:
        for name in ["performance", "validity", "structure", "time", "memory"]:
            assert (out / "figures" / f"rollouts={n}" / f"{name}.png").exists()
    assert (out / "figures" / "sample_efficiency.pdf").exists() and (out / "tables.md").exists()
    assert (out / "models" / "N=2" / "ground_truth.png").exists()
    # the models of the first seed, at the largest number of rollouts, read back
    from scm_to_mdp.core import FactorBounds, IntervalMDP
    from scm_to_mdp.tool import load_model

    bic = load_model(out / "models" / "N=3" / "n=60" / "BIC_RMDP")
    assert isinstance(bic["model"], FactorBounds) and set(bic["cbn"]["cpds"]) == {"m0'", "m1'", "m2'"}
    assert bic["info"]["seed"] == 0 and bic["solution"].V.shape == (8,)
    flat = load_model(out / "models" / "N=3" / "n=60" / "flat_MDP")
    assert isinstance(flat["model"], IntervalMDP) and flat["cbn"] is None
    assert not (out / "models" / "N=3" / "n=20").exists()
    # resume: every (size, seed) is finished, nothing runs again
    before = res.drop(columns=[c for c in res.columns if c.endswith(("_s", "_gb"))])
    assert run_sweep(tmp_path, workers=1) == 16
    after = pd.read_csv(out / "results.csv").drop(
        columns=[c for c in res.columns if c.endswith(("_s", "_gb"))]
    )
    pd.testing.assert_frame_equal(before, after)
