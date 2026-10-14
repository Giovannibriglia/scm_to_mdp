"""E1 stage 2: S1 checks and S2/S3 graph arms on the SysAdmin ring, scored against the true graph.

For every (logger suite, total episodes, edge strength, seed): simulate the suite (S0-S1), run the
assumption checks from data alone (S1), split by episode (discovery half / estimation half), and run
the graph arms on the discovery half (S2-S3):

- ``bic``: the BIC-optimal parent set per target (one learned graph);
- ``jpcmci``: J-PCMCI+ with the logger as context (one learned graph; needs the ``discovery`` extra);
- ``posterior``: the smallest family with posterior mass >= 1 - alpha / N per target;
- ``icp``: the parent sets of size <= d accepted by the invariance test at alpha / N per target;
- ``oracle``: the true parent sets.

A family is *valid* when, for every target, some member contains all true parents (superset rule).
Results go to ``results/e1/structure_arms.csv``.

    uv run --extra discovery python experiments/e1_sysadmin_graph_arms/structure_arms.py
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from tqdm import tqdm

from scm_to_mdp.assumptions import episode_split, profile_assumptions
from scm_to_mdp.data import collect
from scm_to_mdp.envs import SysAdminRing
from scm_to_mdp.loggers import make_logger
from scm_to_mdp.structure import (
    best_bic,
    candidate_sets,
    contains_superset,
    credible_family,
    icp_family,
    posterior,
)

SUITES = {  # name -> list of (family, kwargs); the total episodes are shared equally
    "S-A aliasing": [("aliasing", {"offset": -1, "eps": 0.05})],
    "S-B two aliasing": [("aliasing", {"offset": -1, "eps": 0.05}), ("aliasing", {"offset": 1, "eps": 0.05})],
    "S-C softmax+lazy": [("softmax_q", {"temperature": 0.5}), ("lazy", {"p_noop": 0.9})],
    "S-D two specialists": [("specialist", {"region": (0, 1), "eps": 0.05}),
                            ("specialist", {"region": (2, 3), "eps": 0.05})],
}
EDGES = {"strong": 0.60, "weak": 0.85}
COLUMNS = ["suite", "episodes", "edge", "seed", "arm", "valid", "family_size", "exact", "missed_parent",
           "gate_p", "markov_p", "stationarity_p"]


def run_one(suite: str, episodes: int, edge: str, seed: int, alpha: float, d: int, use_jpcmci: bool):
    env = SysAdminRing(n=4, p_bad=EDGES[edge])
    specs = SUITES[suite]
    loggers = [make_logger(f, env, **kw) for f, kw in specs]
    per = [episodes // len(specs)] * len(specs)
    data = collect(env, loggers, per, horizon=30, seed=seed)
    report = profile_assumptions(env, data)
    disc, _ = episode_split(data, 0.5, seed=seed)
    cols = disc.columns(env)
    inputs = env.state_vars + env.action_flags
    cands = candidate_sets(inputs, d)
    truth = env.true_parents()
    a_t = alpha / env.n
    families: dict[str, dict[str, list]] = {"bic": {}, "posterior": {}, "icp": {}, "oracle": {}}
    for target, parents in truth.items():
        families["bic"][target] = [best_bic(cols, target, cands)]
        families["posterior"][target] = credible_family(posterior(cols, target, cands), 1 - a_t)
        families["icp"][target] = icp_family(cols, target, cands, disc.regime, a_t)
        families["oracle"][target] = [parents]
    if use_jpcmci:
        from scm_to_mdp.discovery import jpcmci_parents

        res = jpcmci_parents(env, disc, alpha=0.01)
        families["jpcmci"] = {t: [res.parents[t]] for t in truth}
    rows = []
    s1 = {"gate_p": min(report.gate.values()), "markov_p": min(report.markov.values()),
          "stationarity_p": min(report.stationarity.values())}
    for arm, fam in families.items():
        valid = all(contains_superset(fam[t], truth[t]) for t in truth)
        single = arm in ("bic", "jpcmci", "oracle")
        exact = single and all(set(fam[t][0]) == set(truth[t]) for t in truth)
        missed = single and any(not set(truth[t]) <= set(fam[t][0]) for t in truth)
        size = sum(len(fam[t]) for t in truth) / len(truth)
        rows.append({"suite": suite, "episodes": episodes, "edge": edge, "seed": seed, "arm": arm,
                     "valid": int(valid), "family_size": size, "exact": int(exact),
                     "missed_parent": int(missed), **s1})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, nargs="+", default=[40, 100, 400], help="total per suite")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.1, help="family-wise level over the targets")
    ap.add_argument("--d", type=int, default=3, help="declared in-degree bound")
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--no-jpcmci", action="store_true", help="skip J-PCMCI+ (no Tigramite)")
    ap.add_argument("--out", type=Path, default=Path("results/e1/structure_arms.csv"))
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()
    use_jp = not args.no_jpcmci
    if use_jp:
        try:
            import tigramite  # noqa: F401
        except ImportError:
            print("Tigramite not installed: skipping J-PCMCI+ (uv sync --extra discovery)")
            use_jp = False

    jobs = list(itertools.product(args.suites, args.episodes, EDGES, range(args.seeds)))
    rows = []
    bar = tqdm(jobs, desc="structure", unit="run", disable=args.no_progress)
    for suite, episodes, edge, seed in bar:
        bar.set_postfix_str(f"{suite}, {episodes} ep, {edge}, seed {seed}")
        rows += run_one(suite, episodes, edge, seed, args.alpha, args.d, use_jp)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(jobs)} runs; wrote {args.out}")


if __name__ == "__main__":
    main()
