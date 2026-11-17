"""The thirty-four pooled figures of experiments/clear27/plot.py (its section 6): drawn from a table of
instances, whatever its origin (the main runs, or an ablation with its rewards)."""

import importlib.util
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("clear27_plot", ROOT / "experiments" / "clear27" / "plot.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)

MODEL_FREE = ["DiFaRL", "offline DQN", "offline CQL"]
TRANSITIONS = [100, 1000, 10_000, 100_000]


def fake_instances(methods: list, rewards: list | None = None) -> pd.DataFrame:
    """A table of instances: 2 domains x 2 sizes x 3 seeds (x the rewards), with more instances solved as
    the data grow; the plug-in MDPs and the baselines are rarely safe, and one in ten runs does not finish."""
    rng = np.random.default_rng(0)
    rows = []
    for domain in ("sysadmin", "drone"):
        for size in (3, 4):
            for seed in range(3):
                for reward in rewards or [None]:
                    for k, n in enumerate(TRANSITIONS):
                        for method in methods:
                            promise = method in plot.GRAPHS
                            row = {"domain": domain, "size": size, "seed": seed, "n_transitions": n,
                                   "method": method, "ok": rng.random() > 0.1,
                                   "optimality": 0.3 * k + rng.normal(0, 0.3), "seconds": rng.lognormal(),
                                   "safe": rng.random() < (0.95 if promise else 0.3)}  # fmt: skip
                            row["true_value"] = row["optimality"]
                            row["reported"] = row["true_value"] + rng.normal(-0.5 if promise else 0.3, 0.4)
                            rows.append(row if reward is None else row | {"reward": reward})
    return pd.DataFrame(rows)


METHODS = [*plot.METHODS, "true graph MDP", "DYNOTEARS MDP", "PCMCI+ MDP", "flat MDP", "full graph MDP"]


@pytest.mark.parametrize("rewards", [None, ["original", "goal"]])
@pytest.mark.parametrize("model_free", [False, True])
def test_the_pooled_figures_are_drawn_from_a_table_of_instances(tmp_path, rewards, model_free):
    """In the two sets: paper/ (DYNOTEARS as ours, no PCMCI+) and appendix/ (both as ours)."""
    methods = [m for m in METHODS if model_free or m not in MODEL_FREE]
    instances = fake_instances(methods, rewards)
    written = plot.pooled_figures(instances, tmp_path)
    # the instances stop at 100,000 transitions: the figures drawn at 500,000 (along the observed states,
    # and against the required normalised value) are skipped
    at_500k = ("solved_and_scalability", "tradeoff_and_optimality", "solved_reliably_vs_states",
               "solved_reliably_vs_states_everywhere", "tradeoff_and_scalability",
               "imdp_by_graph/solved_vs_optimality",
               "mdp_by_graph/solved_vs_optimality", "vs_baselines/solved_vs_optimality",
               "vs_baselines/solved_vs_optimality_to90", "vs_baselines/solved_vs_optimality_grey",
               "vs_baselines/solved_vs_regret", "vs_baselines/solved_vs_regret_grey",
               "vs_baselines/solved_vs_regret_fade",
               "vs_baselines/solved_vs_regret_fade90",
               "vs_baselines/solved_vs_regret_fade_ends",
               "vs_baselines/solved_vs_optimality_fade_ends")  # fmt: skip
    drawn_names = [n for n in plot.POOLED_FIGURES if n not in at_500k]
    appendix_only = ("reported_vs_true_all",)  # every method: in the appendix set only

    def drawn_in(group):
        return [n for n in drawn_names if group == "appendix" or n not in appendix_only]

    assert written == [f"{group}/{name}" for group in plot.SETS for name in drawn_in(group)]
    assert len(plot.POOLED_FIGURES) == 34
    for group in plot.SETS:
        for name, (folder, _, _) in plot.POOLED_FIGURES.items():
            if name not in drawn_in(group):
                assert not (tmp_path / group / f"{name}.pdf").exists()
                continue
            assert (tmp_path / group / f"{name}.pdf").stat().st_size > 0
            assert (tmp_path / group / f"{name}.png").stat().st_size > 0
            csv = "_".join(filter(None, [group, folder, name.rsplit("/", 1)[-1]]))
            assert (tmp_path / "data" / f"{csv}.csv").stat().st_size > 0
    for file in (tmp_path / "data").glob("paper_*.csv"):  # PCMCI+ is in no figure of the paper set
        assert "PCMCI" not in file.read_text(), file.name
    drawn = pd.read_csv(tmp_path / "data" / "paper_vs_baselines_solved_vs_transitions.csv")
    assert {"CD-IMDP (ours)", "CD-MDP (ours)", "Full-IMDP", "Flat-IMDP", "FOCUS"} <= set(drawn["label"])
    assert (
        set(MODEL_FREE) <= set(drawn["method"]) if model_free else not set(MODEL_FREE) & set(drawn["method"])
    )
    assert set(drawn["instances"]) == {12 * len(rewards or [1])}  # every instance, finished or not
    both = pd.read_csv(tmp_path / "data" / "appendix_tradeoff.csv")  # four paths of ours
    ours = {"DYNOTEARS-IMDP (ours)", "PCMCI+-IMDP (ours)", "DYNOTEARS-MDP (ours)", "PCMCI+-MDP (ours)"}
    assert ours <= set(both["label"]) and "True-IMDP (reference)" in set(both["label"])
    same = ["method", "n_transitions", "instances", "solved", "safe"]  # the numbers do not depend on the set
    paper = pd.read_csv(tmp_path / "data" / "paper_tradeoff.csv")
    pd.testing.assert_frame_equal(
        paper[same], both[~both["method"].str.startswith("PCMCI+")][same].reset_index(drop=True)
    )
    plane = pd.read_csv(tmp_path / "data" / "paper_tradeoff_pareto.csv")
    assert plane["n_transitions"].min() > 100 and plot.REFERENCE in set(plane["method"])
    for ring in ("safest", "most_solved"):  # one method per data size, never the reference
        best = plane[plane[ring]]
        assert len(best) == 3 and plot.REFERENCE not in set(best["method"])
    a, b, p = plane[["envelope_a", "envelope_b", "envelope_p"]].iloc[0]
    compared = plane[plane["method"] != plot.REFERENCE]
    assert ((compared["solved"] / a) ** p + (compared["safe"] / b) ** p <= 1 + 1e-6).all()  # none beyond it
    for group, ours in (("paper", "CD-IMDP (ours)"), ("appendix", "DYNOTEARS-IMDP (ours)")):
        scatter = pd.read_csv(tmp_path / "data" / f"{group}_reported_vs_true.csv")
        drawn_methods = list(scatter.drop_duplicates("method")["method"])  # the panels, in their order
        assert drawn_methods == [m for m in REPORTED if model_free or m not in MODEL_FREE]
        assert scatter["label"].iloc[0] == ours and not scatter["method"].str.contains("PCMCI").any()
        assert set(scatter["domain"]) == {"sysadmin", "drone", "all"}
        whole = scatter[scatter["domain"] == "all"].set_index("method")
        parts = scatter[scatter["domain"] != "all"].groupby("method")["runs"].sum()
        assert (whole["runs"] == parts[whole.index]).all()
        assert list(whole["panel"]) == list(range(1, len(whole) + 1))
    only = plot.pooled_figures(instances, tmp_path / "one", sets=["appendix"], only=["tradeoff"])
    assert only == ["appendix/tradeoff"] and not (tmp_path / "one" / "paper").exists()


def test_solved_and_safe_need_a_finished_run_and_the_threshold():
    """Solved: finished with a normalised value of at least THRESHOLD (0.9); safe: finished and the
    reported value held. A run that did not finish is neither, and stays in the denominator."""
    assert plot.THRESHOLD == 0.9 and plot.REFERENCE_N == 100_000
    rows = [
        (True, 0.95, True),
        (True, 0.9, False),
        (True, 0.89, True),
        (False, 0.99, True),
        (False, np.nan, None),
    ]
    table = pd.DataFrame(rows, columns=["ok", "optimality", "safe"]).assign(
        method="DYNOTEARS", n_transitions=1000
    )
    judged = plot.judged(table)
    assert list(judged["solved"]) == [1, 1, 0, 0, 0] and list(judged["safe"]) == [1, 0, 1, 0, 0]
    (share,) = plot.shares(judged, ["DYNOTEARS"]).to_dict("records")
    assert (
        share["instances"] == 5
        and share["solved"] == pytest.approx(40)
        and share["safe"] == pytest.approx(40)
    )


def test_the_data_a_method_needs_is_read_between_two_data_sizes():
    path = pd.DataFrame(
        {"n_transitions": [100, 1000, 10_000], "solved": [0.0, 20.0, 60.0], "safe": [100.0, 90, 70]}
    )
    assert plot.reach(path, 10) == pytest.approx((10**2.5, 95.0))  # half way, in the log of the transitions
    assert plot.reach(path, 60) == pytest.approx((10_000, 70.0)) and plot.reach(path, 70) is None
    assert plot.reach(path, 0) == pytest.approx((100, 100.0))  # reached at the first data size


def test_the_envelope_is_the_smallest_superellipse_with_no_point_beyond_it():
    solved, safe = np.array([10.0, 60.0, 95.0]), np.array([98.0, 90.0, 40.0])
    a, b, p = plot.envelope(solved, safe)
    assert a > 95 and b > 98 and ((solved / a) ** p + (safe / b) ** p).max() == pytest.approx(1, abs=1e-6)


REPORTED = ["DYNOTEARS", "DYNOTEARS MDP", "flat MDP", "full graph MDP", "FOCUS", "SL-SPIBB", "flat SPIBB",
            "SLF-Rmax", "Met-Rmax", "DiFaRL", "offline DQN", "offline CQL"]  # fmt: skip


def drawn_figure(monkeypatch, draw, instances, methods, group="paper"):
    """The figure and the table a pooled figure would save (not written)."""
    kept = {}
    monkeypatch.setattr(
        plot, "save_pooled", lambda to, folder, fig, name, data: kept.update(fig=fig, data=data)
    )
    to = plot.FigureSet(group, plot.SETS[group], Path("."))
    with mpl.rc_context(plot.PAPER_STYLE):
        assert draw(plot.judged(instances), to.methods(methods), "", to)
    return kept["fig"], kept["data"]


def texts_of(fig) -> list:
    """Every word a figure shows: titles, axis labels, the texts in the panels, the legends."""
    words = [text.get_text() for text in fig.texts]
    for ax in fig.axes:
        words += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel(), *(text.get_text() for text in ax.texts)]
    return words + [text.get_text() for legend in fig.legends for text in legend.get_texts()]


def reported_rows(shares: dict, domain: str = "sysadmin") -> list:
    """Four runs per method: `under` of them report at most the true value (0.5), the others more."""
    return [
        {"domain": domain, "size": 3, "seed": i, "n_transitions": 1000, "method": method, "ok": True,
         "optimality": 0.5, "safe": i < under, "seconds": 1.0, "true_value": 0.5,
         "reported": 0.2 if i < under else 0.9}
        for method, under in shares.items() for i in range(4)
    ]  # fmt: skip


def test_reported_against_true_has_a_panel_per_method_in_the_order_of_the_paper(monkeypatch):
    """reported_vs_true: twelve panels, four to a row, in the order of REPORTED; one dot per finished run
    in the colour of its domain; the share of a panel is a count of its runs, in one wording for every
    method ("reliability"); the grey region is named in the legend only, and no label says "promise"."""
    assert plot.REPORTED == [m.replace("DYNOTEARS", plot.OURS) for m in REPORTED]
    assert list(plot.DOMAIN_COLOURS) == ["sysadmin", "drone", "sepsis"] == plot.POOLED_DOMAINS
    assert len(set(plot.DOMAIN_COLOURS.values())) == 3
    shares = {method: i % 5 for i, method in enumerate(REPORTED)}  # 0 ... 4 of 4 runs under the true value
    rows = reported_rows(shares) + reported_rows(shares, "sepsis")
    rows.append(rows[0] | {"ok": False, "seed": 9})  # a run that did not finish is not drawn
    rows.append(rows[0] | {"reported": np.nan, "seed": 8})  # nor one that reported nothing
    fig, table = drawn_figure(monkeypatch, plot.reported_vs_true, pd.DataFrame(rows), plot.REPORTED)
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert fig.axes[0].get_gridspec().get_geometry() == (3, 4) and len(panels) == 12
    names = ["CD-IMDP (ours)", "CD-MDP (ours)", "Flat-MDP", "Full-MDP", "FOCUS", "SL-SPIBB", "flat SPIBB",
             "SLF-Rmax", "Met-Rmax", "DiFaRL", "offline DQN", "offline CQL"]  # fmt: skip
    # the name, then the share of the runs of the method that returned a value: 8 of the 10 of the first
    assert [ax.get_title() for ax in panels] == [
        f"{name} – {'80.0' if i == 0 else '100.0'}%" for i, name in enumerate(names)
    ]
    first = table[(table["method"] == "DYNOTEARS") & (table["domain"] == "all")].iloc[0]
    assert (first["possible_runs"], first["returned_runs"], first["returned_share"]) == (10, 8, 80.0)
    assert (table[table["method"] != "DYNOTEARS"]["returned_share"] == 100).all()
    whole = table[table["domain"] == "all"].set_index("method")
    for method, under in shares.items():
        assert whole.loc[method, "runs"] == 8 and whole.loc[method, "reported_le_true"] == 25 * under
    assert (table[table["domain"] == "sepsis"]["runs"] == 4).all() and (table["outside"] == 0).all()
    for ax, under in zip(panels, shares.values(), strict=True):
        assert [text.get_text() for text in ax.texts] == [f"reliability: {25 * under:.1f}%"]  # and no more
        assert ax.title.get_color() == plot.INK and ax.title.get_fontweight() in ("normal", "regular", 400)
        (dots,) = [c for c in ax.collections if c.get_offsets().shape[0] == 8]
        colours = {mpl.colors.to_hex(c) for c in dots.get_facecolors()}
        assert colours == {plot.DOMAIN_COLOURS["sysadmin"], plot.DOMAIN_COLOURS["sepsis"]}
        assert dots.get_rasterized()
    words = " ".join(texts_of(fig))
    assert "promise" not in words.lower() and "reported >" not in words and "≤ true" not in words
    assert [text.get_text() for text in fig.legends[0].get_texts()] == [
        "SysAdmin", "Sepsis", "not reliable",
    ]  # fmt: skip
    assert (table["mean_value"] == 0.5).all() and (table["median_value"] == 0.5).all()
    assert {text.get_text() for text in fig.texts} == {"normalised reported value", "normalised true value"}
    plot.plt.close(fig)


def test_reported_against_true_draws_the_methods_it_has_and_a_far_run_at_the_border(monkeypatch):
    """Nine methods (an ablation has no model-free one): three rows of three, in the same order; the
    appendix set has the same panels, of the first discovery method, under its name there; a run beyond
    REPORTED_X is drawn at the border and counted (`outside`)."""
    nine = [m for m in REPORTED if m not in MODEL_FREE]
    rows = reported_rows({method: 2 for method in [*nine, "PCMCI+", "PCMCI+ MDP"]}, "drone")
    rows[0] = rows[0] | {"reported": -50.0}
    table = pd.DataFrame(rows)
    fig, table = drawn_figure(monkeypatch, plot.reported_vs_true, table, plot.REPORTED, "appendix")
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert fig.axes[0].get_gridspec().get_geometry() == (3, 3) and len(panels) == 9
    titles = [ax.get_title().replace("\n", " – ") for ax in panels]  # the share: on a second line if long
    assert titles[:2] == ["DYNOTEARS-IMDP (ours) – 100.0%", "DYNOTEARS-MDP (ours) – 100.0%"]
    assert len({ax.get_title().count("\n") for ax in panels}) == 1  # on one line in every panel, or in none
    assert list(table["method"]) == nine and set(table["domain"]) == {"drone"}  # one domain: no row "all"
    assert list(table["outside"]) == [1] + [0] * 8 and table["reported_le_true"].eq(50).all()
    (dots,) = [c for c in panels[0].collections if c.get_offsets().shape[0] == 4]
    assert dots.get_offsets()[:, 0].min() == plot.REPORTED_X[0]
    assert [text.get_text() for text in panels[0].texts] == ["reliability: 50.0%"]
    assert "promise" not in " ".join(texts_of(fig)).lower()
    plot.plt.close(fig)


def test_the_paper_grid_has_the_eight_most_reliable_methods_in_order(monkeypatch):
    """reported_vs_true_top8: of the candidates (REPORTED_TOP: the interval MDPs of the full graph and of
    no graph too), the TOP_N with the highest share of runs that reported at most the true value, two rows
    of four in that order. Ties: ours first, then the higher share of runs that returned a value, then
    the order of the candidates. The csv has every candidate, with its rank and whether it is shown."""
    assert plot.TOP_N == 8 and plot.REPORTED_TOP == [*plot.REPORTED, "full graph", "flat"]
    shares = {"DYNOTEARS": 4, "full graph": 4, "flat": 4, "FOCUS": 3, "DiFaRL": 3, "SL-SPIBB": 2,
              "SLF-Rmax": 2, "Met-Rmax": 2, "DYNOTEARS MDP": 1, "flat MDP": 1, "full graph MDP": 0,
              "flat SPIBB": 0, "offline DQN": 0, "offline CQL": 0}  # fmt: skip
    rows = reported_rows(shares, "drone")
    rows.append(next(r for r in rows if r["method"] == "flat") | {"ok": False, "seed": 9})  # 4 of 5 returned
    fig, table = drawn_figure(monkeypatch, plot.reported_vs_true_top8, pd.DataFrame(rows), plot.REPORTED_TOP)
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert fig.axes[0].get_gridspec().get_geometry() == (2, 4) and len(panels) == 8
    assert [ax.get_title().split(" – ")[0] for ax in panels] == [
        "CD-IMDP (ours)", "Full-IMDP", "Flat-IMDP", "FOCUS", "DiFaRL", "SL-SPIBB", "SLF-Rmax", "Met-Rmax",
    ]  # fmt: skip
    assert panels[2].get_title() == "Flat-IMDP – 80.0%"
    assert [t.get_text() for t in panels[3].texts] == ["reliability: 75.0%"]
    assert list(table["rank"]) == list(range(1, 15)) and list(table["shown"]) == [True] * 8 + [False] * 6
    assert list(table["method"])[:3] == ["DYNOTEARS", "full graph", "flat"]
    assert list(table["reported_le_true"]) == [100, 100, 100, 75, 75, 50, 50, 50, 25, 25, 0, 0, 0, 0]
    assert "promise" not in " ".join(texts_of(fig)).lower()
    plot.plt.close(fig)


def test_reported_against_true_needs_the_reported_values(capsys):
    to = plot.FigureSet("paper", plot.SETS["paper"], Path("."))
    without = pd.DataFrame(reported_rows({"DYNOTEARS": 2})).drop(columns=["reported", "true_value"])
    assert not plot.reported_vs_true(plot.judged(without), to.methods(plot.REPORTED), "", to)
    assert "skipped paper/reported_vs_true: its instances have no reported value" in capsys.readouterr().out


def test_the_common_runs_are_those_every_method_finished_with_a_value(monkeypatch, capsys):
    """reported_vs_true_common: the intersection over the methods of the figure of their runs (domain,
    size, seed, transitions and, in an ablation, reward) that finished with a value; every panel then has
    the same runs, and their number is in the legend and in the csv."""
    methods = ["DYNOTEARS", "DYNOTEARS MDP", "FOCUS"]
    rows = [
        {"domain": domain, "size": 3, "seed": seed, "n_transitions": 1000, "reward": reward, "method": method,
         "ok": True, "optimality": 0.5, "safe": True, "seconds": 1.0, "true_value": 0.1 * seed,
         "reported": 0.0 if method == "DYNOTEARS" else 0.25}
        for domain in ("sysadmin", "drone") for seed in range(4) for reward in ("original", "goal")
        for method in methods
    ]  # fmt: skip
    table = pd.DataFrame(rows)
    seed, method, reward = table["seed"], table["method"], table["reward"]
    table.loc[(seed == 0) & (method == "FOCUS"), "ok"] = False  # did not finish: seed 0 is not common
    table.loc[(seed == 1) & (method == "DYNOTEARS MDP") & (reward == "goal"), "reported"] = np.nan  # no value
    table = table[~((seed == 2) & (method == "DYNOTEARS") & (table["domain"] == "drone"))]  # a missing run
    kept = plot.common_runs(table[table["ok"] & table["reported"].notna()], methods)
    runs = kept[["domain", "seed", "reward"]].drop_duplicates()
    # 2 domains x 4 seeds x 2 rewards = 16; without seed 0 (4), seed 1 with "goal" (2), seed 2 of drone (2)
    assert len(runs) == 8 and len(kept) == 24 and set(kept.groupby("method").size()) == {8}
    assert 0 not in set(runs["seed"])
    assert not ((runs["domain"] == "drone") & (runs["seed"] == 2)).any()
    fig, drawn = drawn_figure(monkeypatch, plot.reported_vs_true_common, table, plot.REPORTED)
    assert set(drawn["common_runs"]) == {8} and set(drawn[drawn["domain"] == "all"]["runs"]) == {8}
    assert set(drawn[drawn["domain"] == "sysadmin"]["runs"]) == {5} and len(drawn) == 9
    panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(panels) == 3
    values = 0.1 * runs["seed"]
    for ax in panels:  # the same runs in every panel
        (dots,) = [c for c in ax.collections if c.get_offsets().shape[0] == 8]
        assert sorted(dots.get_offsets()[:, 1].round(6)) == sorted(values.round(6))
    said = [text.get_text() for text in fig.legends[0].get_texts()]
    assert said == ["SysAdmin", "Drone", "not reliable", "N common runs: 8"]
    assert [text.get_text() for text in panels[0].texts] == ["reliability: 100.0%"]  # a lower bound of 0
    assert [ax.get_title() for ax in panels] == ["CD-IMDP (ours)", "CD-MDP (ours)", "FOCUS"]  # no share
    assert drawn["mean_value"].iloc[-1] == pytest.approx(values.mean())  # in the csv, not in the figure
    held = 100 * (values >= 0.25 - 1e-9).mean()  # FOCUS reports 0.25
    assert [text.get_text() for text in panels[2].texts] == [f"reliability: {held:.1f}%"]
    focus = drawn[(drawn["method"] == "FOCUS") & (drawn["domain"] == "all")].iloc[0]
    assert (focus["possible_runs"], focus["returned_runs"], focus["runs"]) == (16, 12, 8)  # seed 0 failed
    plot.plt.close(fig)
    apart = table[(table["seed"] == 0) == (table["method"] == "DYNOTEARS")]  # no run that the three share
    to = plot.FigureSet("paper", plot.SETS["paper"], Path("."))
    assert not plot.reported_vs_true_common(plot.judged(apart), to.methods(plot.REPORTED), "", to)
    printed = capsys.readouterr().out
    assert "skipped paper/reported_vs_true_common: no run that every one of its methods finished" in printed


def test_the_figures_say_reliable_and_never_safe_or_promise():
    """What a reader sees: "reliable" on the axis and in the legend of the plane (the column stays `safe`)."""
    assert plot.SAFE_AXIS == "instances reliable (reported value ≤ true value, %)"
    assert [handle.get_label() for handle in plot.ring_handles()] == [
        "most reliable of a data size", "most solved of a data size"]  # fmt: skip
    for text in (plot.SAFE_AXIS, plot.SOLVED_AXIS, *(h.get_label() for h in plot.ring_handles())):
        assert not any(word in text.lower() for word in ("safe", "promise"))


def test_the_two_panel_figure_lists_every_method_once_and_leaves_its_two_figures_as_they_are(tmp_path):
    """solved_and_tradeoff: (a) vs_baselines/solved_vs_transitions and (b) tradeoff_pareto with one legend,
    whose index (the csv) has every method of either panel once; the two figures' numbers do not change."""
    instances = fake_instances(METHODS, None)
    alone = ["vs_baselines/solved_vs_transitions", "tradeoff_pareto"]
    plot.pooled_figures(instances, tmp_path / "alone", only=alone)
    written = plot.pooled_figures(instances, tmp_path / "with", only=[*alone, "solved_and_tradeoff"])
    assert "paper/solved_and_tradeoff" in written and "appendix/solved_and_tradeoff" in written
    for group in plot.SETS:
        assert (tmp_path / "with" / group / "solved_and_tradeoff.pdf").stat().st_size > 0
        index = pd.read_csv(tmp_path / "with" / "data" / f"{group}_solved_and_tradeoff.csv")
        assert index["method"].is_unique and index["label"].is_unique
        assert set(index["panels"]) <= {"a", "b", "a+b"}
        left = pd.read_csv(tmp_path / "with" / "data" / f"{group}_vs_baselines_solved_vs_transitions.csv")
        right = pd.read_csv(tmp_path / "with" / "data" / f"{group}_tradeoff_pareto.csv")
        assert set(index["method"]) == set(left["method"]) | set(right["method"])
        assert set(index.loc[index["panels"].str.contains("a"), "method"]) == set(left["method"])
        assert set(index.loc[index["panels"].str.contains("b"), "method"]) == set(right["method"])
        assert index["method"].tolist()[:2] == plot.REFERENCES and set(index["panels"]) == {"a+b"}
        assert index["label"].tolist()[:2] == ["True-IMDP (reference)", "True-MDP (reference)"]
        assert not right.loc[right["method"].isin(plot.REFERENCES), ["safest", "most_solved"]].any().any()
        for name in ("vs_baselines_solved_vs_transitions", "tradeoff_pareto"):
            same = (tmp_path / "alone" / "data" / f"{group}_{name}.csv").read_bytes()
            assert same == (tmp_path / "with" / "data" / f"{group}_{name}.csv").read_bytes()


def test_the_reliable_share_of_the_two_checks_is_a_count_by_hand(tmp_path):
    """reliable_vs_transitions: reported at most the true value from the start, over the runs that returned
    a value; reliable_everywhere_vs_transitions: `safe`, over all the instances (unfinished: not reliable)."""
    rows = [  # one method, one data size: 5 instances, 4 returned a value, 3 of them reliable from the start
        dict(ok=True, reported=0.2, true_value=0.5, safe=True),
        dict(ok=True, reported=0.5, true_value=0.5, safe=False),  # equal: reliable; not in every state
        dict(ok=True, reported=0.9, true_value=0.5, safe=False),
        dict(ok=True, reported=-1.0, true_value=0.1, safe=True),
        dict(ok=False, reported=np.nan, true_value=np.nan, safe=True),  # did not finish: never reliable
    ]
    base = dict(domain="sysadmin", size=3, n_transitions=1000, method="FOCUS", seconds=1.0)
    rows = [base | row | {"seed": i, "optimality": row["true_value"]} for i, row in enumerate(rows)]
    instances = pd.DataFrame(rows)
    t = plot.reliable_shares(plot.judged(instances), ["FOCUS"]).iloc[0]
    assert (t["instances"], t["returned"]) == (5, 4)
    assert t["reliable"] == pytest.approx(75.0) and t["reliable_of_all"] == pytest.approx(60.0)
    assert t["reliable_everywhere"] == pytest.approx(40.0)  # two of five: the unfinished one does not count
    names = ["vs_baselines/reliable_vs_transitions", "vs_baselines/reliable_everywhere_vs_transitions"]
    written = plot.pooled_figures(instances, tmp_path, sets=["paper"], only=names)
    assert written == [f"paper/{n}" for n in names]
    start = pd.read_csv(tmp_path / "data" / "paper_vs_baselines_reliable_vs_transitions.csv")
    assert start["reliable"].tolist() == [75.0] and start["returned"].tolist() == [4]
    # without a reported value the check from the start has nothing to draw; the one in every state has
    bare = instances.drop(columns=["reported", "true_value"])
    assert plot.pooled_figures(bare, tmp_path / "bare", sets=["paper"], only=names) == [f"paper/{names[1]}"]
    assert plot.RELIABLE["reliable_vs_transitions"][1] == "runs reliable (%)"
    assert plot.RELIABLE["reliable_everywhere_vs_transitions"][1] == "instances reliable in every state (%)"


def test_the_two_panel_reliable_figures_and_the_compact_plane_leave_the_other_figures_as_they_are(
    tmp_path, monkeypatch
):
    """solved_and_reliable and solved_and_reliable_everywhere: (a) solved_vs_transitions, (b) the reliable
    figure of the check, with one legend that has every method once; tradeoff_pareto_compact: the plane of
    tradeoff_pareto with its numbers and no method in its legend."""
    instances = fake_instances(METHODS, None)
    alone = ["vs_baselines/solved_vs_transitions", "tradeoff_pareto", "solved_and_tradeoff"]
    new = ["vs_baselines/reliable_vs_transitions", "vs_baselines/reliable_everywhere_vs_transitions",
           "solved_and_reliable", "solved_and_reliable_everywhere", "tradeoff_pareto_compact"]  # fmt: skip
    plot.pooled_figures(instances, tmp_path / "alone", only=alone)
    written = plot.pooled_figures(instances, tmp_path / "with", only=[*alone, *new])
    data = tmp_path / "with" / "data"
    for group in plot.SETS:
        assert all(f"{group}/{name}" in written for name in new)
        for name in new:
            assert (tmp_path / "with" / group / f"{name}.pdf").stat().st_size > 0
        left = pd.read_csv(data / f"{group}_vs_baselines_solved_vs_transitions.csv")
        for pair, panel in plot.RELIABLE_PAIRS.items():
            index = pd.read_csv(data / f"{group}_{pair}.csv")
            assert index["method"].is_unique and index["label"].is_unique
            assert set(index["panels"]) == {"a+b"}
            right = pd.read_csv(data / f"{group}_vs_baselines_{panel}.csv")
            assert set(index["method"]) == set(left["method"]) == set(right["method"])
            assert index["method"].tolist()[:2] == plot.REFERENCES
            assert right[plot.RELIABLE[panel][0]].between(0, 100).all()
        compact = (data / f"{group}_tradeoff_pareto_compact.csv").read_bytes()
        assert compact == (data / f"{group}_tradeoff_pareto.csv").read_bytes()  # the same plane
        for name in ("vs_baselines_solved_vs_transitions", "tradeoff_pareto", "solved_and_tradeoff"):
            same = (tmp_path / "alone" / "data" / f"{group}_{name}.csv").read_bytes()
            assert same == (data / f"{group}_{name}.csv").read_bytes()
    with plot.references_named():
        fig, _ = drawn_figure(monkeypatch, plot.solved_and_reliable, instances, plot.VS_BASELINES)
        a, b = fig.axes
        assert (a.get_ylabel(), b.get_ylabel()) == ("instances solved (%)", "runs reliable (%)")
        assert a.get_xlabel() == b.get_xlabel() == "transitions" and a.get_ylim() == b.get_ylim()
        assert fig.get_size_inches().tolist() == list(plot.PAIR_SIZE)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert len(labels) == len(set(labels)) == len(a.get_lines()) == len(b.get_lines())
        ours = [line for line in b.get_lines() if line.get_label() == "CD-IMDP (ours)"][0]
        assert ours.get_zorder() == max(line.get_zorder() for line in b.get_lines())  # on top at 100%
        fig, _ = drawn_figure(monkeypatch, plot.tradeoff_pareto_compact, instances, plot.TRADEOFF)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert labels == ["envelope of all points", "most reliable of a data size",
                          "most solved of a data size", "True-IMDP (reference)", "True-MDP (reference)",
                          "ideal"]  # fmt: skip
        assert fig.get_size_inches().tolist() == list(plot.COMPACT_SIZE)


def states_instances() -> pd.DataFrame:
    """Two domains at 500,000 transitions, three seeds per size; ours on the learned graph solves and is
    reliable but for one seed of the largest SysAdmin size, which does not finish; FOCUS solves everything
    and reports too much on Drone."""
    rows = []
    for domain, sizes in (("sysadmin", (3, 5, 11)), ("drone", ("3-1", "8-1"))):
        for size in sizes:
            for seed in range(3):
                lost = domain == "sysadmin" and size == 11 and seed == 0
                shared = {"domain": domain, "size": size, "seed": seed, "n_transitions": plot.STATES_BUDGET,
                          "seconds": 1.0, "optimality": 0.95}  # fmt: skip
                rows.append(shared | {"method": "DYNOTEARS", "ok": not lost, "safe": True,
                                      "true_value": np.nan if lost else 0.95,
                                      "reported": np.nan if lost else 0.5})  # fmt: skip
                rows.append(shared | {"method": "FOCUS", "ok": True, "safe": False, "true_value": 0.95,
                                      "reported": 1.2 if domain == "drone" else 0.9})  # fmt: skip
    return pd.DataFrame(rows)


def test_the_observed_states_of_a_size_follow_the_rule_of_its_domain():
    assert plot.observed_states("sysadmin", "11") == 2048 and plot.observed_states("sysadmin", 3) == 8
    assert plot.observed_states("drone", "12-1") == 1728 and plot.observed_states("drone", 3) == 27
    assert [plot.observed_states("sepsis", v) for v in (2, 3, 4)] == [144, 288, 1440]
    for unknown in (("aircraft", 3), ("drone", "3-2"), ("sepsis", 7), ("sysadmin", "N")):
        assert np.isnan(plot.observed_states(*unknown))


def test_the_line_of_a_method_is_an_average_of_its_shares_weighted_by_the_instances(monkeypatch):
    """trend: a Gaussian-weighted average on the log2 of the states, weighted by the instances, between the
    smallest and the largest size and within 0-100; equal neighbours give back the raw share."""
    states, values = plot.trend([4, 8, 16, 32], [70.0] * 4, [10] * 4)
    assert values == pytest.approx(70.0) and len(states) == plot.SMOOTH_GRID
    assert (states.min(), states.max()) == pytest.approx((4.0, 32.0))  # no extrapolation
    _, values = plot.trend([4, 8, 16, 32], [100.0, 0.0, 100.0, 0.0], [10, 10, 10, 30])
    assert ((values >= 0) & (values <= 100)).all() and len(set(values.round(6))) > 1
    # two sizes an octave apart, 10 and 30 instances, a width of one octave: at the first size the weights
    # are 10 and 30 exp(-1/2); half-way the two Gaussians are equal, and the weights are the instances
    monkeypatch.setattr(plot, "SMOOTH_OCTAVES", 1.0)
    monkeypatch.setattr(plot, "SMOOTH_GRID", 3)
    states, values = plot.trend([4, 8], [100.0, 0.0], [10, 30])
    far = np.exp(-0.5)
    assert states == pytest.approx([4.0, 2**2.5, 8.0])
    assert values == pytest.approx([1000 / (10 + 30 * far), 25.0, 1000 * far / (10 * far + 30)])


def test_solved_reliably_against_the_observed_states_is_one_smoothed_panel(tmp_path, monkeypatch):
    """solved_reliably_vs_states, over several domains with 500,000 transitions: a share per distinct number
    of observed states, counted by hand (a run that did not finish is neither solved nor reliable and stays
    in the denominator; solved AND reliable, not one of the two); the line of a method is trend of its
    shares (SMOOTH_OCTAVES) and the shares are not drawn; no method in the legend and no number of
    transitions in the figure; one domain: its shares themselves, not smoothed."""
    instances = states_instances()
    name = "solved_reliably_vs_states"
    written = plot.pooled_figures(instances, tmp_path, sets=["paper"])
    assert f"paper/{name}" in written and f"paper/{name}_everywhere" in written
    assert all((tmp_path / "paper" / f"{name}.{kind}").stat().st_size > 0 for kind in ("pdf", "png"))
    table = pd.read_csv(tmp_path / "data" / f"paper_{name}.csv")
    assert set(table["octaves"]) == {plot.SMOOTH_OCTAVES} and set(table["n_transitions"]) == {
        plot.STATES_BUDGET
    }
    shares, lines = table[table["row"] == "share"], table[table["row"] == "line"]
    ours = shares[shares["method"] == "DYNOTEARS"].set_index("states")
    assert list(ours.index) == [8, 27, 32, 512, 2048] and set(ours["instances"]) == {3}
    assert ours["solved_reliably"].tolist() == pytest.approx([100, 100, 100, 100, 200 / 3])
    assert ours.loc[2048, "solved_n"] == 2 and ours.loc[2048, "reliable_n"] == 2
    focus = shares[shares["method"] == "FOCUS"].set_index("states")  # solved; Drone reports too much
    assert focus["solved"].tolist() == [100] * 5 and focus["solved_reliably"].tolist() == [
        100,
        0,
        100,
        0,
        100,
    ]
    for method, rows in shares.groupby("method"):
        states, values = plot.trend(rows["states"], rows["solved_reliably"], rows["instances"])
        line = lines[lines["method"] == method]
        assert line["states"].to_numpy() == pytest.approx(states)
        assert line["smoothed"].to_numpy() == pytest.approx(values) and line["smoothed"].between(0, 100).all()
        assert line["label"].nunique() == 1
    alone = plot.pooled_figures(instances.query("domain == 'drone'"), tmp_path / "one", sets=["paper"])
    assert f"paper/{name}" in alone
    one = pd.read_csv(tmp_path / "one" / "data" / f"paper_{name}.csv")
    one_shares, one_lines = one[one["row"] == "share"], one[one["row"] == "line"]
    for method, rows in one_shares.groupby("method"):  # one domain: the line is the shares themselves
        line = one_lines[one_lines["method"] == method]
        assert line["states"].tolist() == rows["states"].tolist()
        assert line["smoothed"].to_numpy() == pytest.approx(rows["solved_reliably"].to_numpy())
    draw = plot.POOLED_FIGURES[name][1]
    fig, data = drawn_figure(monkeypatch, draw, instances, plot.VS_BASELINES)
    assert fig.get_size_inches().tolist() == list(plot.COMPACT_SIZE) and not fig.legends  # no reference here
    (ax,) = fig.axes
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("observed states", plot.STATES_AXIS)
    assert plot.STATES_AXIS == "instances solved (≥ 0.9) and reliable (%)" and ax.get_xscale() == "log"
    words = texts_of(fig)
    for word in ("transitions", "initial", "start", "safe", "promise"):
        assert not any(word in t.lower() for t in words)
    drawn = {line.get_label(): line for line in ax.get_lines()}
    assert len(ax.get_lines()) == 2  # a line per method, and no raw point
    assert all(line.get_marker() in (None, "None", "") for line in ax.get_lines())
    ours, other = drawn["CD-IMDP (ours)"], drawn["FOCUS"]
    kept = data[(data["row"] == "line") & (data["method"] == "DYNOTEARS")]
    assert ours.get_ydata() == pytest.approx(kept["smoothed"].to_numpy())
    assert ours.get_linewidth() > 1.5 * other.get_linewidth()  # the other methods are drawn thin
    assert ours.get_zorder() == max(line.get_zorder() for line in ax.get_lines())
    with plot.references_named():  # with the references: they alone are in the legend, and the two compact
        truth = instances[instances["method"] == "DYNOTEARS"].assign(method="true graph")  # figures match
        both = pd.concat([instances, truth], ignore_index=True)
        fig, _ = drawn_figure(monkeypatch, draw, both, plot.VS_BASELINES)
        assert [text.get_text() for text in fig.legends[0].get_texts()] == ["True-IMDP (reference)"]
        assert list(fig.axes[0].get_position().bounds) == pytest.approx(plot.compact_frame(fig, 1))


def test_the_plane_and_the_states_are_two_panels_with_one_legend(tmp_path, monkeypatch):
    """tradeoff_and_scalability: (a) the plane of tradeoff_pareto and (b) solved_reliably_vs_states, with
    the legend of solved_and_tradeoff (every method once, then the envelope and the two rings); the csv
    files of its two figures do not depend on it; one domain: skipped."""
    instances = states_instances()
    name = "tradeoff_and_scalability"
    sources = ["tradeoff_pareto", "solved_reliably_vs_states"]
    plot.pooled_figures(instances, tmp_path / "alone", sets=["paper"], only=sources)
    written = plot.pooled_figures(instances, tmp_path, sets=["paper"], only=[*sources, name])
    assert f"paper/{name}" in written
    assert all((tmp_path / "paper" / f"{name}.{kind}").stat().st_size > 0 for kind in ("pdf", "png"))
    for source in sources:
        file = f"paper_{source}.csv"
        assert (tmp_path / "data" / file).read_bytes() == (tmp_path / "alone" / "data" / file).read_bytes()
    index = pd.read_csv(tmp_path / "data" / f"paper_{name}.csv")
    assert index["method"].is_unique and set(index["panels"]) <= {"a", "b", "a+b"}
    fig, _ = drawn_figure(monkeypatch, plot.POOLED_FIGURES[name][1], instances, plot.VS_BASELINES)
    (legend,) = fig.legends
    labels = [text.get_text() for text in legend.get_texts()]
    assert len(labels) == len(set(labels)) == len(index) + 3
    assert labels[-3:] == [
        "envelope of all points",
        "most reliable of a data size",
        "most solved of a data size",
    ]
    plane, states = fig.axes
    assert (plane.get_xlabel(), plane.get_ylabel()) == ("instances solved (%)", "instances reliable (%)")
    assert (
        states.get_xlabel() == "observed states"
        and states.get_ylabel().replace("\n", " ") == plot.STATES_AXIS
    )
    assert plane.get_position().height == pytest.approx(states.get_position().height)
    assert {"(a)", "(b)"} <= set(texts_of(fig))
    alone = plot.pooled_figures(instances.query("domain == 'drone'"), tmp_path / "one", sets=["paper"])
    assert not any(name in w for w in alone)


def test_the_figures_of_the_paper_have_three_forms_and_the_appendix_every_method(tmp_path):
    """Figure 3 (solved_and_scalability, at 500,000 transitions) and figure 5 (tradeoff_and_rewards: the main
    runs beside the runs with other rewards) as one figure with its legend, each panel alone (_a, _b) and
    the legend alone; figure 4 complete (reported_vs_true_all) in the appendix set only, PCMCI+ too."""
    methods = [m for m in METHODS if m not in MODEL_FREE]
    instances = fake_instances(methods)
    at_500k = instances[instances["n_transitions"] == 100_000].assign(n_transitions=500_000)
    instances = pd.concat([instances, at_500k], ignore_index=True)
    instances["n_observed_states"] = 2 ** instances["size"]
    only = ["solved_and_scalability", "reported_vs_true_all"]
    written = plot.pooled_figures(instances, tmp_path, only=only)
    assert sorted(written) == ["appendix/reported_vs_true_all", "appendix/solved_and_scalability",
                               "paper/solved_and_scalability"]  # fmt: skip
    for suffix in ("_a", "_b", "_legend", "_a_full", "_b_full"):
        assert (tmp_path / "paper" / f"solved_and_scalability{suffix}.pdf").stat().st_size > 0
    assert not (tmp_path / "paper" / "reported_vs_true_all.pdf").exists()
    every = pd.read_csv(tmp_path / "data" / "appendix_reported_vs_true_all.csv")
    assert {"PCMCI+", "DYNOTEARS", "SLF-Rmax", "flat MDP"} <= set(every["method"])
    rewards = fake_instances(methods, ["original", "goal"])
    to = plot.FigureSet("paper", plot.SETS["paper"], tmp_path / "fig5")
    with mpl.rc_context(plot.PAPER_STYLE), plot.references_named():
        assert plot.tradeoff_and_rewards(instances, rewards, to)
        assert not plot.tradeoff_and_rewards(instances[instances["domain"] == "drone"], rewards, to)
    for suffix in ("_a", "_b", "_legend", "_a_full", "_b_full"):
        assert (tmp_path / "fig5" / "paper" / f"tradeoff_and_rewards{suffix}.pdf").stat().st_size > 0
    index = pd.read_csv(tmp_path / "fig5" / "data" / "paper_tradeoff_and_rewards.csv")
    assert set(index["panels"]) == {"a+b"} and "CD-IMDP (ours)" in set(index["label"])
