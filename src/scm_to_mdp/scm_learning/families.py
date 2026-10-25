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
