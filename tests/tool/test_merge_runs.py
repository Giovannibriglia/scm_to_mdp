"""experiments/merge_runs.py: two runs of the same grids on other methods join into one; anything else is
refused."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import merge_runs  # noqa: E402

SETTING = "topology=ring/uniform"


def fake_run(root: Path, name: str, methods: list[str], v_star: float = 1.0, seeds=(0, 1)) -> Path:
    run = root / f"sysadmin_imdp_{name}"
    (run / SETTING / "models" / "N=2" / "n=10").mkdir(parents=True)
    (run / "grid.json").write_text(json.dumps({"seeds": list(seeds), "settings": [{"folder": SETTING}]}))
    meta = {"config": {"name": "x", "limits": {"mem_gb": 8}, "methods": {m: {"role": "r"} for m in methods}},
            "git_commit": name}  # fmt: skip
    (run / SETTING / "meta.json").write_text(json.dumps(meta))
    rows = [{"size": 2, "seed": s, "n_transitions": 10, "method": m, "V_star": v_star, "true": i}
            for s in seeds for i, m in enumerate(methods)]  # fmt: skip
    pd.DataFrame(rows).to_csv(run / SETTING / "results.csv", index=False)
    for m in methods:
        (run / SETTING / "models" / "N=2" / "n=10" / m.replace(" ", "_")).mkdir()
    return run


def test_the_extra_methods_join_and_shared_ones_keep_the_first_run(tmp_path):
    a = fake_run(tmp_path / "exp1", "2026-10-28_08-00-00", ["PCMCI+ -> IMDP", "oracle -> IMDP"])
    b = fake_run(tmp_path / "exp1_sparse", "2026-10-29_08-00-00", ["sparse -> IMDP", "oracle -> IMDP"])
    assert merge_runs.check_same_experiment(a, b) == []
    code = merge_runs.main([str(a.parent), str(b.parent), "--out", str(tmp_path / "merged"), "--no-report"])
    out = tmp_path / "merged" / a.name / SETTING
    rows = pd.read_csv(out / "results.csv")
    assert code == 0 and sorted(rows["method"].unique()) == [
        "PCMCI+ -> IMDP",
        "oracle -> IMDP",
        "sparse -> IMDP",
    ]
    assert len(rows) == 6 and (rows[rows["method"] == "oracle -> IMDP"]["true"] == 1).all()  # first run's
    meta = json.loads((out / "meta.json").read_text())
    assert set(meta["config"]["methods"]) == set(rows["method"]) and meta["merged"][
        "methods from the second run"
    ] == ["sparse -> IMDP"]
    assert (out / "models" / "N=2" / "n=10" / "sparse_->_IMDP").is_dir()


def test_other_grids_or_other_data_are_refused(tmp_path):
    a = fake_run(tmp_path / "a", "2026-10-28_08-00-00", ["PCMCI+ -> IMDP", "oracle -> IMDP"])
    other_seeds = fake_run(
        tmp_path / "b", "2026-10-28_08-00-00", ["sparse -> IMDP", "oracle -> IMDP"], seeds=(0, 2)
    )
    other_data = fake_run(
        tmp_path / "c", "2026-10-28_08-00-00", ["sparse -> IMDP", "oracle -> IMDP"], v_star=2.0
    )
    assert any("grid.json" in p for p in merge_runs.check_same_experiment(a, other_seeds))
    assert any("other data" in p for p in merge_runs.check_same_experiment(a, other_data))
    code = merge_runs.main(
        [str(a.parent), str(other_data.parent), "--out", str(tmp_path / "m"), "--no-report"]
    )
    assert code == 1 and not (tmp_path / "m").exists()


def test_latest_runs_reads_runs_started_at_20_hours(tmp_path):
    """A run started at 20:xx has "_20" in its time too (frozenlake_mdp_2026-10-29_20-29-11)."""
    for name in ("frozenlake_mdp_2026-10-29_20-29-11", "frozenlake_mdp_2026-10-28_09-00-00",
                 "sysadmin_imdp_2026-10-28_08-28-52"):  # fmt: skip
        (tmp_path / name).mkdir()
        (tmp_path / name / "grid.json").write_text("{}")
    runs = merge_runs.latest_runs(tmp_path)
    assert runs == {"frozenlake_mdp": tmp_path / "frozenlake_mdp_2026-10-29_20-29-11",
                    "sysadmin_imdp": tmp_path / "sysadmin_imdp_2026-10-28_08-28-52"}  # fmt: skip
