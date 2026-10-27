"""Compare every trial grid (experiments/trials/<domain>_uniform_{rmdp,mdp}.py) in one table (S2-S9).

Reads the latest run folder of each grid under results/trials/ (its summary/runs.csv and
summary/rollouts_needed.csv) and writes results/trials/comparison/:
    all_grids.csv   one row per (domain, kind, method): sizes, sizes where the target is reached
                    (median n* over seeds finite), median n* over those sizes, finished / attempted
                    runs, how they failed, share of finished runs whose promise holds in every state,
                    and at the largest number of rollouts: mean promise, true value, gap, norm_regret
    all_grids.md    the headline table (methods x domains) and the detail table

    python experiments/trials/compare_grids.py [--root results/trials]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["sysadmin", "aircraft", "coffee", "drone", "frozenlake", "sepsis", "stock", "chemistry", "taxi"]
NAMES = {
    "sysadmin": "SysAdmin",
    "aircraft": "Aircraft",
    "coffee": "Coffee",
    "drone": "Drone",
    "frozenlake": "FrozenLake",
    "sepsis": "Sepsis",
    "stock": "Stock",
    "chemistry": "Chemistry",
    "taxi": "Taxi",
}
METHODS = {
    "rmdp": [
        "BIC -> IMDP",
        "PCMCI+ -> IMDP",
        "GES -> IMDP",
        "DYNOTEARS -> IMDP",
        "true graph -> IMDP",
        "flat IMDP",
        "flat SPIBB",
        "SL-SPIBB",
    ],
    "mdp": [
        "BIC -> MDP",
        "PCMCI+ -> MDP",
        "GES -> MDP",
        "DYNOTEARS -> MDP",
        "true graph -> MDP",
        "flat MDP",
        "DiFaRL",
        "FOCUS",
        "offline DQN",
        "offline CQL",
    ],
}


def latest(root: Path, domain: str, kind: str) -> Path | None:
    runs = sorted(
        p
        for p in root.glob(f"{domain}_uniform_{kind}_*")
        if re.search(r"_\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d$", p.name)
    )
    return runs[-1] if runs else None


def grid_rows(folder: Path, domain: str, kind: str) -> list[dict]:
    runs = pd.read_csv(folder / "summary" / "runs.csv", low_memory=False)
    need = pd.read_csv(folder / "summary" / "rollouts_needed.csv")
    runs["safe"] = runs["safe"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    n_max = runs["n_rollouts"].max()
    sizes = sorted(runs["size"].unique())
    out = []
    for method in METHODS[kind]:
        r = runs[runs["method"] == method]
        ok = r[r["status"] == "ok"]
        last = ok[ok["n_rollouts"] == n_max]
        nd = need[need["method"] == method]
        reached = nd[np.isfinite(nd["median"])]
        fails = r[~r["status"].isin(["ok", "skipped"])]
        how = fails.groupby(["failed_step", "status"]).size()
        out.append(
            {
                "domain": NAMES[domain],
                "kind": {"rmdp": "IMDP", "mdp": "MDP"}[kind],  # the robust grids solve interval MDPs
                "method": method,
                "folder": folder.name,
                "role": r["role"].iloc[0] if len(r) else "",
                "sizes": len(sizes),
                "reached": len(reached),
                "largest_reached": reached["size"].max() if len(reached) else np.nan,
                "median_n_star": reached["median"].median() if len(reached) else np.nan,
                "runs": len(r),
                "finished": len(ok),
                "failed": len(fails),
                "skipped": int((r["status"] == "skipped").sum()),
                "failures": "; ".join(f"{s} {st} x{n}" for (s, st), n in how.items()),
                "held": ok["safe"].mean() if len(ok) else np.nan,
                "n_max": n_max,
                "finished_at_n_max": len(last),
                "promise": last["promise"].mean(),
                "true": last["true"].mean(),
                "V_star_obs": last["V_star_obs"].mean(),
                "gap": last["gap"].mean(),
                "norm_regret": last["norm_regret"].mean(),
            }
        )
    return out


def cell(row: pd.Series) -> str:
    """Headline cell: sizes reached / sizes, then the promise-held share (if < 100%), then '*' if runs
    failed."""
    if row["finished"] == 0:
        return "—"
    s = f"{row['reached']}/{row['sizes']}"
    if row["held"] < 0.995:
        s += f" ({100 * row['held']:.0f}%)"
    if row["failed"] + row["skipped"] > 0:
        s += f" *{100 * (row['failed'] + row['skipped']) / row['runs']:.0f}%"
    return s


def bold_best(head: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Bold, per kind and per column, the best method: most sizes reached, then fewest rollouts (median n*).

    Candidates: not the reference (true graph), at least one finished run, and for IMDP a promise that held in
    >= 95% of the finished runs (as best_method.csv). Ties are all bolded; nothing is bolded if no candidate
    reaches the target anywhere in the column.
    """
    head = head.copy()
    ok = (df["role"] != "reference") & (df["finished"] > 0) & ((df["kind"] == "MDP") | (df["held"] >= 0.95))
    total = df.groupby(["kind", "method"], sort=False).agg(
        reached=("reached", "sum"),
        median_n_star=("median_n_star", "median"),
        role=("role", "first"),
        finished=("finished", "sum"),
        all_ok=("method", lambda m: ok[m.index].all()),
    )
    for kind in df["kind"].unique():
        for column in head.columns:
            if column == "reached (all)":
                c = total.loc[kind]
                c = c[(c["role"] != "reference") & (c["finished"] > 0) & ((kind == "MDP") | c["all_ok"])]
            else:
                c = df[(df["kind"] == kind) & (df["domain"] == column) & ok].set_index("method")
            c = c[c["reached"] > 0]
            if c.empty:
                continue
            key = list(zip(-c["reached"], c["median_n_star"].fillna(np.inf), strict=True))
            best = [m for m, k in zip(c.index, key, strict=True) if k == min(key)]
            for m in best:
                head.loc[(kind, m), column] = f"**{head.loc[(kind, m), column]}**"
    return head


def md(df: pd.DataFrame, index: bool = True) -> str:
    """A pipe table (no tabulate dependency)."""
    d = df.reset_index() if index else df

    def fmt(v):
        return (
            "" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3g}" if isinstance(v, float) else str(v))
        )

    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/trials")
    root = Path(ap.parse_args().root)
    rows = []
    for domain in DOMAINS:
        for kind in ("rmdp", "mdp"):
            folder = latest(root, domain, kind)
            if folder is None:
                print(f"missing: {domain} {kind}")
                continue
            rows += grid_rows(folder, domain, kind)
    df = pd.DataFrame(rows)
    out = root / "comparison"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "all_grids.csv", index=False)

    df["cell"] = df.apply(cell, axis=1)
    head = df.pivot_table(
        index=["kind", "method"], columns="domain", values="cell", aggfunc="first", sort=False
    )
    head = head[[NAMES[d] for d in DOMAINS]]
    head["reached (all)"] = df.groupby(["kind", "method"], sort=False).apply(
        lambda g: f"{g['reached'].sum()}/{g['sizes'].sum()}"
    )
    head = bold_best(head, df)
    detail = df[
        [
            "domain",
            "kind",
            "method",
            "reached",
            "sizes",
            "median_n_star",
            "finished",
            "runs",
            "held",
            "promise",
            "true",
            "V_star_obs",
            "gap",
            "norm_regret",
            "failures",
        ]
    ].round(3)
    folders = "\n".join(
        f"- {d}: `{f}`" for d, f in df.groupby(["domain", "kind"], sort=False)["folder"].first().items()
    )
    text = f"""# Trials: all grids compared

Cell = sizes where the target is reached (median n* over 5 seeds finite; target V^pi >= 95% of V*_obs at
that number of rollouts and every larger one) / sizes in the grid; (x%) = share of finished runs whose
promise holds in every state, shown when below 100%; *y% = share of runs that did not finish (memout /
timeout, or skipped after a failure at fewer rollouts); — = no run finished. MDP methods give no
guarantee, so their "promise" is the model's own value.
Bold = best per kind and column: most sizes reached, then fewest rollouts (median n*); the true graph (a
reference) is not a candidate, and an IMDP method is one only if its promise held in >= 95% of finished runs
(in every domain, for the last column). Ties are all bold.

{md(head)}

## Detail (mean over finished runs at n = {int(df["n_max"].max())} rollouts)

promise, true, V*_obs, gap = promise - true, norm_regret

{md(detail, index=False)}

Run folders:
{folders}
"""
    (out / "all_grids.md").write_text(text)
    print(md(head))
    print(f"\nwritten: {out}/all_grids.csv, {out}/all_grids.md")


if __name__ == "__main__":
    main()
