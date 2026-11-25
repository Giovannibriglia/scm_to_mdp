"""Consistency family (step S2 -> S3): the candidate parent sets the pooled data do not contradict, with a
stated level.

For every next-state variable x' the candidates are the sets of exactly k state variables (plus every
action column, always a parent, as in the KWIK learners of kwik_structure.py). A candidate c is discarded
when its L1 ball is disjoint from the ball of one of its extensions c + Z, Z up to `extension_size` other
state variables: if c contained the true parents, both would estimate the same conditional. This is the
superset test of Strehl, Diuk, Littman (AAAI 2007) in the pairwise form of Rosenberg and Mansour (NeurIPS
2021, "consistent scopes"), run offline on the logged transitions by NBN's agreement rule
(nbn.structure.agreement_set, with extension sets: NBN issue #333).

What holds, for all state variables at once with probability >= 1 - delta, if every x' has at most k
state-variable parents, the logged state is Markov and no logger reads hidden information:
    * a candidate that contains the true parents is never discarded, so the survivors' union ("union"
      view) contains the true parents: a sound, possibly large, graph;
    * with extension_size = k, a candidate that does not contain the true parents is discarded wherever
      the data cover two rows that differ enough; when that has happened for all of them, the variables
      common to the survivors ("intersection" view) are exactly the true parents.
The intersection can miss a parent when a wrong candidate survives (too few data, or a parent that the
loggers never vary); the union cannot. Nothing is assumed about the strength of the parents' effect.

Unlike strehl_sl (the paper's rule: the first m samples of a setting, one fixed accuracy), every visit is
used and the radius of a ball shrinks with its count (sampling="trajectory": valid for counts that depend
on the data).

fit(...) returns the chosen view as a fixed graph (Candidates), so the usual S4 learns its tables.
"""

import inspect
import itertools
from dataclasses import dataclass, field

from nbn.structure import agreement_set

from scm_to_mdp.core import NEXT_MARK, Candidates, Dataset
from scm_to_mdp.discovery import oracle

VIEWS = ("union", "intersection")


@dataclass
class ConsistencyFamily:
    """Per state variable v: survivors[v] (the kept candidates, state variables only), common[v] (the
    state variables in every survivor), refuted[v] (every candidate discarded: the bound k is contradicted,
    or a logger is confounded; the view then falls back to every state variable), n_families[v]."""

    survivors: dict[str, list[tuple[str, ...]]]
    common: dict[str, tuple[str, ...]]
    refuted: dict[str, bool]
    n_families: dict[str, int]
    params: dict = field(default_factory=dict)

    def parents(self, dataset: Dataset, view: str = "union") -> dict[str, tuple[str, ...]]:
        """One parent set per next-state variable, plus the action columns."""
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
        out = {}
        for v in dataset.state_vars:
            if self.refuted[v]:
                chosen = set(dataset.state_vars)
            elif view == "union":
                chosen = set().union(*self.survivors[v])
            else:
                chosen = set(self.common[v])
            out[v] = tuple(s for s in dataset.state_vars if s in chosen) + tuple(dataset.action_vars)
        return out


def family(
    dataset: Dataset,
    k: int,
    extension_size: int | None = None,
    delta: float = 0.05,
    sampling: str = "trajectory",
    max_families: int = 1 << 16,
    max_rows: int = 1 << 22,
) -> ConsistencyFamily:
    """The kept candidates of every next-state variable. k: the declared bound on the number of
    state-variable parents; extension_size: default k (the pairwise test); delta: shared by the state
    variables (delta / n each)."""
    if "extension_size" not in inspect.signature(agreement_set).parameters:
        raise RuntimeError("consistency_family needs nbn with agreement_set(extension_size=...) (NBN #333)")
    if k < 0:
        raise ValueError(f"k must be >= 0, not {k}")
    states, actions = list(dataset.state_vars), tuple(dataset.action_vars)
    k = min(k, len(states))
    e = max(1, k if extension_size is None else extension_size)
    variables, data = dataset.variables, dataset.tensors()
    candidates = [(*c, *actions) for c in itertools.combinations(states, k)]
    out = ConsistencyFamily(
        {}, {}, {}, {}, {"k": k, "extension_size": e, "delta": delta, "sampling": sampling}
    )
    for v in states:
        res = agreement_set(
            variables, data, v + NEXT_MARK, list(dataset.inputs), delta=delta / len(states),
            sampling=sampling, candidates=candidates, extension_size=e, max_families=max_families,
            max_rows=max_rows,
        )  # fmt: skip
        out.survivors[v] = [tuple(p for p in s if p not in actions) for s in res.survivors]
        out.common[v] = tuple(p for p in res.common_parents if p not in actions)
        out.refuted[v] = bool(res.assumption_refuted)
        out.n_families[v] = int(res.n_families)
    return out


def fit(
    dataset: Dataset,
    k: int,
    extension_size: int | None = None,
    delta: float = 0.05,
    view: str = "union",
    sampling: str = "trajectory",
    same_step: bool = False,
    max_families: int = 1 << 16,
    max_rows: int = 1 << 22,
) -> Candidates:
    """The chosen view of the consistency family as one graph. same_step must be False: the rule has no
    same-step parents (the option exists so that the S4 union bound counts the slice-t inputs only)."""
    if same_step:
        raise ValueError("consistency_family has no same-step parents (same_step must be False)")
    fam = family(dataset, k, extension_size, delta, sampling, max_families, max_rows)
    out = oracle.fit(dataset, fam.parents(dataset, view))
    out.report.update(
        view=view, level=1.0 - delta, refuted=[v for v, r in fam.refuted.items() if r],
        survivors={v: len(s) for v, s in fam.survivors.items()}, n_families=fam.n_families, **fam.params,
    )  # fmt: skip
    return out
