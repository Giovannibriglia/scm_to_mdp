"""Declared tyings (tied.Template) for the pMDP experiments: which CPD rows share one Bernoulli parameter.

What a template declares is domain knowledge (an assumption, checked by tied.tying_test); what it
learns comes from the data and the graph of S2-S3 (DYNOTEARS's, or the true one for the reference arm).

SysAdmin (envs/sysadmin.py; 4 classes, every machine the same):
    repaired   the action repairs machine i
    up_ok      machine i up, every neighbour up
    up_deg     machine i up, some neighbour down
    down       machine i down, not repaired
  declared: a machine's own state and the action are parents of its next state; each machine has one
  repair value of `reboot`, no two the same. learned: the neighbours (the other machines among the
  graph's parents of m_i') and the repair values (the one-to-one assignment that best explains
  P(m_i' = 1 | m_i = 0, reboot = v)).

Stock trading (envs/stock_trading.py; every sector and every stock the same):
    buy, sell, hold_owned, hold_not     own_j' after buying sector j, selling it, or neither (owned or
                                        not): 4 classes for every sector
    stock_p<p>_r<r>                     a stock with p stock parents of which r are rising: one class per
                                        (p, r) that occurs
  declared: own_j' depends on own_j and the trade only, through those four roles; a stock's chance to
  rise depends only on how many of its stock parents rise. learned: the buy and sell values of every
  sector (one-to-one assignments, as SysAdmin's repairs) and the stock parents (the graph's parents of
  s_j_k' among the stocks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from scm_to_mdp.core import NEXT_MARK
from scm_to_mdp.data import slice_inputs


def _assign(score: np.ndarray) -> np.ndarray:
    """The one-to-one assignment of rows (variables) to columns (action values) of maximal total score."""
    rows, cols = linear_sum_assignment(-score)
    out = np.empty(score.shape[0], dtype=np.int64)
    out[rows] = cols
    return out


def _log_rate(df, var: str, own_value: int, action: str, values: np.ndarray, target: int) -> np.ndarray:
    """log of the Laplace-smoothed rate of var' == target among the rows with var == own_value, per value."""
    sel = df[var].values == own_value
    out = np.zeros(len(values))
    for j, v in enumerate(values):
        rows = sel & (df[action].values == v)
        k, n = int((df[var + NEXT_MARK].values[rows] == target).sum()), int(rows.sum())
        out[j] = np.log((k + 1) / (n + 2))
    return out


class _Columns:
    """Read a variable's column from a table (rows) or from the model grid (slice_inputs)."""

    def __init__(self, data=None, df=None):
        if df is not None:
            self.get = lambda name: df[name].values
        else:
            X = slice_inputs(data)
            col = {name: j for j, name in enumerate(data.inputs)}
            self.get = lambda name: X[..., col[name]]


# ---- SysAdmin -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SysAdminTemplate:
    state_vars: list[str]
    neighbours: dict[str, list[str]]
    repair: dict[str, int]
    action: str = "reboot"
    n_classes: int = 4
    class_names: list[str] = field(default_factory=lambda: ["repaired", "up_ok", "up_deg", "down"])

    @classmethod
    def learn(cls, data, parents: dict) -> SysAdminTemplate:
        ms = list(data.state_vars)
        nb = {m: sorted(p for p in parents[m] if p in ms and p != m) for m in ms}
        values = np.arange(data.cards["reboot"])
        score = np.stack([_log_rate(data.df, m, 0, "reboot", values, 1) for m in ms])
        repair = dict(zip(ms, (int(values[c]) for c in _assign(score)), strict=True))
        return cls(ms, nb, repair)

    def _classes(self, get) -> np.ndarray:
        out = []
        for m in self.state_vars:
            own = get(m)
            down = np.zeros(own.shape, dtype=bool)
            for j in self.neighbours[m]:
                down |= get(j) == 0
            repaired = get(self.action) == self.repair[m]
            out.append(np.where(repaired, 0, np.where(own == 1, np.where(down, 2, 1), 3)))
        return np.stack(out, axis=-1)

    def classes_of_rows(self, df) -> np.ndarray:
        return self._classes(_Columns(df=df).get)

    def classes_of_model(self, data) -> np.ndarray:
        return self._classes(_Columns(data=data).get)

    def untied_keys(self, df) -> np.ndarray:
        out = []
        for m in self.state_vars:
            key = df[m].values + 2 * (df[self.action].values == self.repair[m])
            for b, j in enumerate(self.neighbours[m]):
                key = key + 4 * (2**b) * df[j].values
            out.append(key)
        return np.stack(out, axis=1)


# ---- Stock trading --------------------------------------------------------------------------------
@dataclass(frozen=True)
class StockTemplate:
    state_vars: list[str]
    buy: dict[str, int]  # own_j -> its buy value of `trade`
    sell: dict[str, int]
    stock_parents: dict[str, list[str]]  # s_j_k -> its learned stock parents
    offsets: dict[int, int]  # number of stock parents p -> the class of (p, r = 0)
    n_classes: int
    class_names: list[str]
    action: str = "trade"

    @classmethod
    def learn(cls, data, parents: dict) -> StockTemplate:
        vars_ = list(data.state_vars)
        owns = [v for v in vars_ if v.startswith("own")]
        stocks = [v for v in vars_ if not v.startswith("own")]
        values = np.arange(data.cards["trade"])
        buy = _assign(np.stack([_log_rate(data.df, o, 0, "trade", values, 1) for o in owns]))
        sell = _assign(np.stack([_log_rate(data.df, o, 1, "trade", values, 0) for o in owns]))
        sp = {s: [p for p in parents[s] if p in stocks] for s in stocks}
        names = ["buy", "sell", "hold_owned", "hold_not"]
        offsets = {}
        for p in sorted({len(v) for v in sp.values()}):
            offsets[p] = len(names)
            names += [f"stock_p{p}_r{r}" for r in range(p + 1)]
        return cls(
            vars_,
            dict(zip(owns, (int(values[c]) for c in buy), strict=True)),
            dict(zip(owns, (int(values[c]) for c in sell), strict=True)),
            sp,
            offsets,
            len(names),
            names,
        )

    def _classes(self, get) -> np.ndarray:
        out = []
        for v in self.state_vars:
            if v in self.buy:
                trade = get(self.action)
                out.append(np.where(trade == self.buy[v], 0, np.where(trade == self.sell[v], 1,
                                                                       np.where(get(v) == 1, 2, 3))))  # fmt: skip
            else:
                ps = self.stock_parents[v]
                rising = sum((get(p) for p in ps), start=np.zeros_like(get(self.action)))
                out.append(self.offsets[len(ps)] + rising)
        return np.stack(out, axis=-1)

    def classes_of_rows(self, df) -> np.ndarray:
        return self._classes(_Columns(df=df).get)

    def classes_of_model(self, data) -> np.ndarray:
        return self._classes(_Columns(data=data).get)

    def untied_keys(self, df) -> np.ndarray:
        out, K = [], int(df[self.action].max()) + 1
        for v in self.state_vars:
            if v in self.buy:
                out.append(df[v].values * K + df[self.action].values)
            else:
                key = np.zeros(len(df), dtype=np.int64)
                for b, p in enumerate(self.stock_parents[v]):
                    key = key + (2**b) * df[p].values
                out.append(key)
        return np.stack(out, axis=1)
