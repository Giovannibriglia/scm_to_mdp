"""Parent families of the factors (step S4): which rows of a family's table every (s, a) uses.

A family is the parent tuple of x_i' in a fixed order: the slice-t inputs (dataset.inputs order),
then the same-step parents x_j' (dataset.next_vars order). NBN lists the rows of a table in C order over
that tuple, so a row is (code of the slice-t values) * R + (code of the same-step values), with R the
number of joint values of the same-step parents (1 if none).
"""

import numpy as np

from scm_to_mdp.core import NEXT_MARK, Dataset, encode


def family(dataset: Dataset, parents) -> tuple[str, ...]:
    """The parent set as a tuple: slice-t inputs, then same-step parents (one name for one family)."""
    chosen = set(parents)
    return tuple(x for x in dataset.inputs + dataset.next_vars if x in chosen)


def same_step(dataset: Dataset, parents: tuple[str, ...]) -> tuple[int, ...]:
    """Indices (into dataset.state_vars) of the same-step parents of a family."""
    return tuple(
        dataset.state_vars.index(p.removesuffix(NEXT_MARK)) for p in parents if p.endswith(NEXT_MARK)
    )


def rows(dataset: Dataset, parents: tuple[str, ...], X: np.ndarray) -> np.ndarray:
    """[S, K, R] the table row of the family at every state s, action k and joint value r of its
    same-step parents (C order). X [S, K, n_inputs]: data.slice_inputs(dataset)."""
    column = {name: j for j, name in enumerate(dataset.inputs)}
    now = [p for p in parents if not p.endswith(NEXT_MARK)]
    S, K = X.shape[:2]
    code = (
        encode(X[..., [column[p] for p in now]], [dataset.cards[p] for p in now])
        if now
        else np.zeros((S, K), dtype=np.int64)
    )
    R = int(np.prod([dataset.cards[dataset.state_vars[j]] for j in same_step(dataset, parents)]))
    return code[..., None] * R + np.arange(R)[None, None, :]


def seen_pairs(dataset, families: dict, counts, X) -> np.ndarray:
    """[S, K] bool: the pairs (s, a_k) at which every factor's parent row has data (for every family of a
    factor, and every row of its same-step parents). families: state var -> list of parent tuples; counts:
    NBN HypothesisSet.counts; X: slice_inputs(dataset). A pair outside it has a factor with interval [0, 1]
    (the "seen pairs only" option of the solvers, tool/methods.py)."""
    S, K = X.shape[:2]
    seen = np.ones((S, K), dtype=bool)
    for v, v_next in zip(dataset.state_vars, dataset.next_vars, strict=True):
        for parents in families[v]:
            has = (counts[(v_next, parents)].sum(-1) > 0).numpy()  # [parent rows]
            seen &= has[rows(dataset, parents, X)].all(-1)  # rows: [S, K, R]
    return seen


def seen_info(seen: np.ndarray) -> dict:
    """The row columns of a seen mask: the share of pairs with data, and of states with no such action."""
    return {"seen_pairs": float(seen.mean()), "states_no_seen": float((~seen.any(1)).mean())}
