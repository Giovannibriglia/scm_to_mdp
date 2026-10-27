"""Report of a grid (tool/grid.py), from grid.json and the results.csv of its settings only.

    report_grid(out_dir)

Every table is built from lower levels only, and every level is saved in <out_dir>/summary/:

  level 0  runs.csv        one row per run (knob, policy, size, seed, n_rollouts, method): every
                           settings' results.csv, stacked (only the sizes, seeds and numbers of
                           rollouts of the last run), plus missing_share and extra_per_variable;
                           safe as 0/1. Everything below comes from this file.
           edges.csv       one row per run of a method that selects a graph (the oracle included),
                           and per candidate edge (target x', parent) in the true or the learned
                           graph: in_true, in_learned
           behaviour.csv   one row per (knob, policy, size, seed): the logger's coverage (C_star,
                           C_all_bound, unvisited_pairs) and confounding bias (bias_max, bias_mean),
                           V_star_obs, V_star and the size of the true model
  level 1  rollouts_needed_by_seed.csv   n* per (knob, policy, size, seed, method): the fewest
                           rollouts (episodes) from which the policy is >= 95% of V_star_obs (inf:
                           never), and t*, the transitions (samples) of that dataset
  level 2  cells.csv       per (knob, policy, size, n_rollouts, method) and metric: mean, standard
                           error, min, max and number of seeds (long format)
           rollouts_needed.csv          median and quartiles of n* and of t* (median_t, q25_t,
                           q75_t) over seeds, share of seeds reached
           promise_held.csv per (knob, policy, size, method): attempted runs (every status but
                           "skipped"), finished runs ("runs", status ok), runs whose promise holds in
                           every state, and the share among the finished runs
           validity.csv    per (knob, policy, size, n_rollouts, method with a promise): runs, held
                           (the promise held in every state), its share with a two-sided 95%
                           Clopper-Pearson interval, and the verdict against 1 - delta: "confirmed"
                           (lower end >= 1 - delta), "refuted" (upper end < 1 - delta), else
                           "undecided" (5 seeds can only describe: 5/5 gives a lower end of 0.48);
                           contain_all_share (every row of the learned set contains the true one: the
                           event the promise relies on); conditional: the graph was chosen from the
                           same data, so the 1 - delta is conditional on that choice
           box_looseness.csv  per (knob, policy, size, n_rollouts), on 0/1 domains: the promise and
                           the true value of true graph -> IMDP with the per-(s,a) product set minus
                           the same with the box, and the share of the box's conservatism (true -
                           promise) due to the box; mean over the seeds
           coverage.csv    per (knob, policy, size, n_rollouts): the data's coverage of the best
                           policy's rows (cov_*, evaluation.coverage), mean over the seeds
  level 3  best_method.csv per (knob, policy, size): the method with the fewest rollouts among those
                           whose promise held in >= 95% of the cell's runs
           method_rank.csv mean rank of every method's n* over the cells, and how often it is best
figures/<name>.{pdf,png} with figure_data/<name>.csv (exactly the numbers drawn), tables.md, and a
README.md that lists all of the above. Each setting also gets its own report (tool/report.py) in its
folder: the figures over the size for every number of episodes. The figures put the sizes at their
number of observed states and count data in transitions (episodes x horizon, less where episodes
end early); the tables keep both.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scm_to_mdp.plotting import grid_figures
from scm_to_mdp.plotting.figures import rollouts_needed, sample_grid, samples_label, summarize_rollouts
from scm_to_mdp.plotting.style import method_styles
from scm_to_mdp.tool.report import ROLE_ORDER, report
from scm_to_mdp.tool.sweep import read_results

SETTING = ["knob", "policy"]
METRICS = ["optimal_share", "regret", "norm_regret", "gap", "safe", "violation_rate", "worst_violation"]
METRICS += ["missing_share", "extra_per_variable", "shd", "sid", "discovery_s", "learning_s", "solve_s"]
METRICS += [f"{m}_{w}" for m in ("tv", "kl", "contain") for w in ("data", "task", "uniform")]
METRICS += ["width_seen", "width_unseen", "lcov_factor_1", "lcov_alt", "cert_gap"]
BOX, PRODUCT = "true graph -> IMDP", "true graph -> factored IMDP (per-(s,a) product)"
VALID_SHARE = 0.95  # a method is "valid" in a cell when its promise held in at least this share of runs


# ==========================================================================================
# Level 0: the runs, their edges and the loggers
# ==========================================================================================
def load_runs(out_dir: Path) -> tuple[dict, pd.DataFrame]:
    grid = json.loads((out_dir / "grid.json").read_text())
    frames = [read_results(out_dir / s["folder"], current=True) for s in grid["settings"]]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError(f"{out_dir}: no results yet (run the grid first)")
    runs = pd.concat(frames, ignore_index=True)
    runs["safe"] = runs["safe"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    n_vars = runs["true_parents"].map(lambda j: len(json.loads(j)) if isinstance(j, str) else np.nan)
    runs["missing_share"] = runs["missing"] / runs["n_true_edges"]
    runs["extra_per_variable"] = runs["extra"] / n_vars
    return grid, runs


def edges_table(runs: pd.DataFrame) -> pd.DataFrame:
    keys = SETTING + ["size", "seed", "n_rollouts", "method"]
    rows = []
    for record in runs[runs["parents"].notna()].to_dict("records"):
        learned, true = json.loads(record["parents"]), json.loads(record["true_parents"])
        for target in true:
            t, g = set(true[target]), set(learned.get(target, ()))
            for parent in sorted(t | g):
                rows.append(
                    {
                        **{k: record[k] for k in keys},
                        "target": target,
                        "parent": parent,
                        "in_true": int(parent in t),
                        "in_learned": int(parent in g),
                    }
                )
    return pd.DataFrame(rows, columns=keys + ["target", "parent", "in_true", "in_learned"])


def behaviour_table(runs: pd.DataFrame) -> pd.DataFrame:
    columns = ["n_states", "n_observed_states", "true_rows", "true_params", "n_true_edges", "V_star_obs"]
    columns += [
        "V_star",
        "V_random",
        "V_logger",
        "C_star",
        "C_all_bound",
        "unvisited_pairs",
        "bias_max",
        "bias_mean",
    ]
    present = [c for c in columns if c in runs]
    return runs.drop_duplicates(SETTING + ["size", "seed"])[SETTING + ["size", "seed"] + present]


# ==========================================================================================
# Levels 1-3
# ==========================================================================================
def needed_by_seed(runs: pd.DataFrame, target: float) -> pd.DataFrame:
    parts = []
    for (knob, policy), g in runs.groupby(SETTING, sort=False):
        n = rollouts_needed(g, target)
        if len(n):  # a setting with nothing to evaluate (e.g. every size too large) has no n*
            parts.append(n.assign(knob=knob, policy=policy))
    columns = SETTING + ["size", "n_states", "n_observed_states", "seed", "method", "n_star", "t_star"]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return out[columns].astype({"n_star": float, "t_star": float})


def needed_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    parts = [
        summarize_rollouts(g).assign(knob=k, policy=p) for (k, p), g in by_seed.groupby(SETTING, sort=False)
    ]
    columns = SETTING + ["size", "n_states", "n_observed_states", "method", "median", "q25", "q75", "reached"]
    columns += ["median_t", "q25_t", "q75_t"]
    return pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)


def cells_table(runs: pd.DataFrame) -> pd.DataFrame:
    keys = SETTING + ["size", "n_rollouts", "method"]
    long = runs.melt(id_vars=keys, value_vars=[m for m in METRICS if m in runs], var_name="metric")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    g = long.groupby(keys + ["metric"], sort=False)["value"]
    return g.agg(mean="mean", sem="sem", min="min", max="max", n_seeds="count").reset_index()


def promise_held(runs: pd.DataFrame) -> pd.DataFrame:
    """Shares among the finished runs; `attempted` shows how many runs a method did not finish."""
    keys = SETTING + ["size", "method"]
    attempted = runs[runs["status"] != "skipped"].groupby(keys, sort=False).size().rename("attempted")
    done = runs[runs["status"] == "ok"]
    out = done.groupby(keys, sort=False)["safe"].agg(runs="count", held="sum")
    out = attempted.to_frame().join(out, how="left").fillna({"runs": 0, "held": 0}).reset_index()
    out["share"] = out["held"] / out["runs"].where(out["runs"] > 0)
    return out[keys + ["attempted", "runs", "held", "share"]]


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Two-sided exact binomial interval for a share k / n."""
    from scipy.stats import beta

    if n == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def validity_table(runs: pd.DataFrame, delta: float) -> pd.DataFrame:
    keys = SETTING + ["size", "n_rollouts", "method"]
    done = runs[(runs["status"] == "ok") & runs["guarantee"].astype(bool)].copy()
    done["contain_all"] = pd.to_numeric(done.get("contain_all"), errors="coerce")
    rows = []
    for key, g in done.groupby(keys, sort=False):
        n, k = len(g), int(g["safe"].sum())
        lo, hi = clopper_pearson(k, n)
        verdict = "confirmed" if lo >= 1 - delta else "refuted" if hi < 1 - delta else "undecided"
        rows.append(
            dict(zip(keys, key, strict=True))
            | {"runs": n, "held": k, "share": k / n, "cp_low": lo, "cp_high": hi, "verdict": verdict,
               "contain_all_share": g["contain_all"].mean(), "conditional": g["role"].iloc[0] == "ours"}
        )  # fmt: skip
    columns = keys + [
        "runs",
        "held",
        "share",
        "cp_low",
        "cp_high",
        "verdict",
        "contain_all_share",
        "conditional",
    ]
    return pd.DataFrame(rows, columns=columns)


def box_looseness(runs: pd.DataFrame) -> pd.DataFrame:
    keys = SETTING + ["size", "n_rollouts", "seed"]
    ok = runs[runs["status"] == "ok"]
    box, product = (ok[ok["method"] == m].set_index(keys) for m in (BOX, PRODUCT))
    if box.empty or product.empty:
        return pd.DataFrame(columns=keys[:-1] + ["promise_gain", "true_gain", "box_share", "seeds"])
    num = lambda f, c: pd.to_numeric(f[c], errors="coerce")  # noqa: E731
    both = pd.DataFrame(
        {
            "promise_gain": num(product, "promise") - num(box, "promise"),
            "true_gain": num(product, "true") - num(box, "true"),
            "conservatism": num(box, "true") - num(box, "promise"),
        }
    ).dropna(subset=["promise_gain"])
    both["box_share"] = both["promise_gain"] / both["conservatism"].where(both["conservatism"] > 1e-12)
    g = both.reset_index().groupby(keys[:-1], sort=False)
    out = g[["promise_gain", "true_gain", "box_share"]].mean()
    return out.assign(seeds=g.size()).reset_index()


def coverage_table(runs: pd.DataFrame) -> pd.DataFrame:
    keys = SETTING + ["size", "n_rollouts"]
    columns = [c for c in runs.columns if c.startswith("cov_")] + ["actions_absent"]
    one = runs.drop_duplicates(keys + ["seed"])  # method-independent: one row per dataset
    values = one[keys + columns].copy()
    values[columns] = values[columns].apply(pd.to_numeric, errors="coerce")
    return values.groupby(keys, sort=False)[columns].mean().reset_index()


def best_method(summary: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    cell = SETTING + ["size"]
    table = summary.merge(held[cell + ["method", "share"]], on=cell + ["method"], how="left")
    rows = []
    for key, g in table.groupby(cell, sort=False):
        valid = g[g["share"] >= VALID_SHARE]
        reached = valid[np.isfinite(valid["median"])]
        if reached.empty:
            best, n_star = "none valid and reached", np.inf
        else:
            n_star = reached["median"].min()
            best = " = ".join(reached.loc[reached["median"] == n_star, "method"])
        rows.append(
            dict(zip(cell, key, strict=True))
            | {"best_valid_method": best, "n_star": n_star, "valid_methods": ", ".join(valid["method"])}
        )
    return pd.DataFrame(rows, columns=cell + ["best_valid_method", "n_star", "valid_methods"])


def method_rank(summary: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    cell = SETTING + ["size"]
    ranked = summary.assign(rank=summary.groupby(cell)["median"].rank(method="average"))
    wins = {}
    for names in best["best_valid_method"]:
        for m in names.split(" = "):
            wins[m] = wins.get(m, 0) + 1
    out = ranked.groupby("method")["rank"].agg(mean_rank="mean", cells="count").reset_index()
    out["best_valid_in_cells"] = out["method"].map(wins).fillna(0).astype(int)
    return out.sort_values("mean_rank")


# ==========================================================================================
# The report
# ==========================================================================================
def _fmt_n(r) -> str:
    """The transitions needed: median [quartiles] over seeds."""
    if not np.isfinite(r["median_t"]):
        return ">"
    hi = "inf" if not np.isfinite(r["q75_t"]) else samples_label(r["q75_t"])
    return f"{samples_label(r['median_t'])} [{samples_label(r['q25_t'])}, {hi}]"


def report_grid(out_dir, target: float = 0.95, delta: float | None = 0.05, settings_too: bool = True) -> dict:
    """Write <out_dir>/summary/ (the CSV levels above, figures/, figure_data/, tables.md, README.md)
    from grid.json and the settings' results.csv. target: the share of V_star_obs (the best policy
    over the observed state) that counts as optimal; delta: the failure probability of the promises
    (the 1 - delta line of the validity panels); settings_too: also run report() in every setting's
    folder. Returns the tables by name."""
    out = Path(out_dir)
    grid, runs = load_runs(out)
    summary_dir, fig_dir, data_dir = (
        out / "summary",
        out / "summary" / "figures",
        out / "summary" / "figure_data",
    )
    for d in (summary_dir, fig_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)
    sizes, knobs, policies = grid["sizes"], grid["knobs"], grid["policies"]
    knobs = [k for k in knobs if k in set(runs["knob"])]
    policies = [p for p in policies if p in set(runs["policy"])]
    first = runs.drop_duplicates("method").set_index("method")
    methods = sorted(
        first.index, key=lambda m: (ROLE_ORDER.get(first.at[m, "role"], 3), list(first.index).index(m))
    )
    styles = method_styles({m: (first.at[m, "role"], bool(first.at[m, "guarantee"])) for m in methods})
    graph_methods = [m for m in methods if runs.loc[runs["method"] == m, "parents"].notna().any()]
    learned_graph = [
        m for m in graph_methods if first.at[m, "role"] not in ("oracle", "reference", "control")
    ]
    guaranteed = [m for m in methods if bool(first.at[m, "guarantee"])]
    n_grid = sample_grid(runs)  # the transitions of every number of episodes (median over the runs)
    name = grid["name"]

    # ---- the tables, level by level ------------------------------------------------------------
    edges, behaviour = edges_table(runs), behaviour_table(runs)
    by_seed = needed_by_seed(runs, target)
    summary = needed_summary(by_seed)
    cells, held = cells_table(runs), promise_held(runs)
    best = best_method(summary, held)
    rank = method_rank(summary, best)
    tables = {
        "runs": runs,
        "edges": edges,
        "behaviour": behaviour,
        "rollouts_needed_by_seed": by_seed,
        "cells": cells,
        "rollouts_needed": summary,
        "promise_held": held,
        "validity": validity_table(runs, delta if delta is not None else 0.05),
        "box_looseness": box_looseness(runs),
        "coverage": coverage_table(runs),
        "best_method": best,
        "method_rank": rank,
    }
    for file, frame in tables.items():
        frame.to_csv(summary_dir / f"{file}.csv", index=False)

    # ---- figures, each with the numbers it draws ------------------------------------------------
    def figure(stem, data, draw):
        data.to_csv(data_dir / f"{stem}.csv", index=False)
        draw(fig_dir / stem)

    figure(
        "sample_efficiency_grid",
        summary,
        lambda stem: grid_figures.sample_efficiency_grid(
            summary,
            sizes,
            grid["size_name"],
            knobs,
            grid["knob_name"],
            policies,
            methods,
            styles,
            n_grid,
            target,
            name,
            stem,
        ),
    )
    sizes_of = behaviour.groupby(SETTING + ["size"])["true_rows"].mean().reset_index()
    points = summary.merge(sizes_of, on=SETTING + ["size"], how="left")
    figure(
        "rollouts_vs_model_size",
        points,
        lambda stem: grid_figures.rollouts_vs_model_size(
            points, policies, methods, styles, n_grid, name, stem
        ),
    )
    overall = runs[runs["status"] == "ok"].groupby(["method"] + SETTING)["safe"].mean()
    heat = overall.unstack(SETTING)
    heat = heat.reindex(
        index=[m for m in methods if m in heat.index],
        columns=[(k, p) for p in policies for k in knobs if (k, p) in heat.columns],
    )
    heat.columns = [f"{p}\n{grid['knob_name']}={k}" for k, p in heat.columns]
    figure("promise_heatmap", heat.reset_index(), lambda stem: grid_figures.promise_heatmap(heat, name, stem))
    if learned_graph:
        done = runs[(runs["status"] == "ok") & runs["method"].isin(learned_graph)]
        curves = done.groupby(SETTING + ["n_rollouts", "method"])[["missing_share", "extra_per_variable"]]
        curves = curves.mean().reset_index()
        samples = pd.to_numeric(runs["transitions"], errors="coerce").groupby(runs["n_rollouts"]).mean()
        curves["transitions"] = curves["n_rollouts"].map(samples)
        for column, ylabel in [
            ("missing_share", "share of true edges missed"),
            ("extra_per_variable", "extra edges per variable"),
        ]:
            figure(
                f"discovery_{column}",
                curves[SETTING + ["n_rollouts", "transitions", "method", column]],
                lambda stem, c=column, y=ylabel: grid_figures.discovery_errors(
                    curves, knobs, grid["knob_name"], policies, learned_graph, styles, c, y, name, stem
                ),
            )
    if "C_star" in behaviour:
        cov = by_seed.merge(behaviour[SETTING + ["size", "seed", "C_star"]], on=SETTING + ["size", "seed"])
        figure(
            "rollouts_vs_coverage",
            cov,
            lambda stem: grid_figures.rollouts_vs_coverage(
                cov, policies, methods, styles, n_grid, name, stem
            ),
        )
    if guaranteed and "bias_mean" in runs:
        last = runs[
            (runs["n_rollouts"] == max(grid["n_rollouts"]))
            & (runs["status"] == "ok")
            & runs["method"].isin(guaranteed)
        ]
        conf = last[SETTING + ["size", "seed", "method", "bias_mean", "gap"]]
        figure(
            "confounding",
            conf,
            lambda stem: grid_figures.confounding(conf, policies, guaranteed, styles, name, stem),
        )

    # ---- tables.md ----------------------------------------------------------------------------
    lines = [
        f"# {name}: summary of the grid",
        "",
        f"{grid['size_name']} = {', '.join(sizes)}; {grid['knob_name']} = {', '.join(knobs)}; policies: "
        f"{'; '.join(policies)}; episodes {grid['n_rollouts']} (transitions "
        f"{', '.join(samples_label(t) for t in n_grid)}); seeds {grid['seeds']}; horizon {grid['horizon']}; "
        f"gamma {grid['gamma']}.",
        "",
        f"Target: V^pi >= {target:.0%} of V*_obs (the best policy over the observed state) at that number "
        "of episodes and every larger one.",
        "",
    ]
    lines += ["## Transitions needed (median [quartiles] over seeds; '>' = not reached)", ""]
    for policy in policies:
        for knob in knobs:
            cell = summary[(summary["policy"] == policy) & (summary["knob"] == knob)]
            if cell.empty:
                continue
            t = cell.assign(v=cell.apply(_fmt_n, axis=1)).pivot(index="size", columns="method", values="v")
            t = t.reindex(index=[s for s in sizes if s in t.index], columns=[m for m in methods if m in t])
            states = cell.groupby("size")["n_observed_states"].median()
            t.index = [
                f"{s} ({int(states[s]):,} states)" if np.isfinite(states.get(s, np.nan)) else s
                for s in t.index
            ]
            t.index.name = grid["size_name"]
            lines += [f"### {policy}, {grid['knob_name']} = {knob}", "```", t.to_string(), "```", ""]
    held_all = heat.copy()
    held_all.columns = [c.replace("\n", ", ") for c in held_all.columns]
    lines += [
        "## Share of finished runs whose promise holds in every state (all sizes and rollouts)",
        "```",
        held_all.to_string(float_format=lambda v: f"{v:.0%}"),
        "```",
        "",
    ]
    b = best.pivot_table(index=SETTING, columns="size", values="best_valid_method", aggfunc="first")
    b = b.reindex(columns=[s for s in sizes if s in b.columns])
    lines += [
        f"## Best valid method (promise held in >= {VALID_SHARE:.0%} of the cell's runs; fewest episodes)",
        "```",
        b.to_string(),
        "```",
        "",
        "## Mean rank of the episodes needed (1 = fewest)",
        "```",
        rank.to_string(index=False),
        "```",
        "",
    ]
    beh = behaviour.groupby(SETTING + ["size"])[
        [c for c in ["C_star", "unvisited_pairs", "bias_mean", "bias_max"] if c in behaviour]
    ].median()
    lines += [
        "## Loggers: coverage and confounding (median over seeds)",
        "```",
        beh.to_string(float_format=lambda v: f"{v:.3g}"),
        "```",
        "",
    ]
    (summary_dir / "tables.md").write_text("\n".join(lines))
    (summary_dir / "README.md").write_text(
        _readme(grid, tables, sorted(p.stem for p in fig_dir.glob("*.pdf")))
    )
    if settings_too:
        for s in grid["settings"]:
            report(out / s["folder"], target=target, delta=delta, show=False)
    print(f"summary: {summary_dir}/ (tables.md, README.md, figures/, figure_data/)")
    return tables


def _readme(grid: dict, tables: dict, figures: list[str]) -> str:
    doc = (__doc__ or "").split("\n\n", 2)[-1]
    lines = [
        f"# {grid['name']}: what is in this folder",
        "",
        "```",
        doc.strip(),
        "```",
        "",
        "Files and their rows:",
    ]
    lines += [
        f"- `{name}.csv`: {len(frame)} rows; columns: {', '.join(map(str, frame.columns))}"
        for name, frame in tables.items()
    ]
    lines += ["", "Figures (each with `figure_data/<name>.csv`): " + ", ".join(f"`{f}`" for f in figures), ""]
    lines += ["Settings:"] + [
        f"- {s['policy']}, {grid['knob_name']} = {s['knob']}: `{s['folder']}/`" for s in grid["settings"]
    ]
    return "\n".join(lines) + "\n"
