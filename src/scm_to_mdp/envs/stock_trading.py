"""Stock trading (Strehl, Diuk, Littman, "Efficient structure learning in factored-state MDPs", AAAI
2007; the version of Schnitzer, Abate, Parker, AAAI 2026, App. E.4 and their PRISM models
models/stockmarket/stock_trading_<N>_<M>.prism, Zenodo 10.5281/zenodo.17580296, CC BY 4.0).

N economic sectors with M stocks each. State: own_j (does the agent own sector j) and s_j_k (is stock
k of sector j rising) for every sector and stock, all 0/1. Action (one categorical data column trade,
one trade per step): 0 = do nothing, 1 + j = buy sector j, 1 + N + j = sell sector j (= the env action).

    own_j'  = 1 after buy_j, 0 after sell_j, own_j otherwise
    s_j_k'  = 1 w.p. base + scale * (rising stocks of sector j now) / M     (base 0.1, scale 0.8: the
              values of their experiments, imdpcomp/Experiment.java), independently for every stock
    reward  = sum over owned sectors of (rising - falling stocks), in the current state (their PRISM
              reward is clipped at 0; we keep the unclipped reward of the appendix's description)

The true parents of s_j_k' are the stocks of sector j; of own_j', own_j and trade.
Episodes start with nothing owned and every stock falling (their init). No time variable: the
episode length is the sampling horizon and the value is discounted.

News (optional, ours; news=p): a hidden variable drawn afresh at every step, news ~ Bernoulli(p). It
hits ONE stock, s0_0: while it is on, s0_0 rises with its probability multiplied by (1 - news_effect).
One child only: once the hidden news is averaged out, the observed next-state variables stay
independent given the observed (state, action), so the true graph's factored model is exact (a news
on several stocks would make them dependent through it). The "confounded" logger (Env.behaviour)
sells sector 0 during news w.p. strength: rows where sector 0 is kept then come mostly from calm
steps, and the data make holding it look better than it is (as the storm of envs/sysadmin.py).
"""

import numpy as np

from scm_to_mdp.core import Reward, encode
from scm_to_mdp.envs.base import Env

BASE, SCALE = 0.1, 0.8


class StockTrading(Env):
    def __init__(
        self,
        n_sectors: int,
        n_stocks: int,
        seed: int = 0,
        base: float = BASE,
        scale: float = SCALE,
        news: float | None = None,
        news_effect: float = 0.6,
        max_model_gb: float | None = None,
    ):
        self.N, self.M, self.base, self.scale = n_sectors, n_stocks, base, scale
        self.news, self.news_effect = news, news_effect
        if max_model_gb is not None:
            self.max_model_gb = max_model_gb
        self.rng = np.random.default_rng(seed)
        self.sectors = [[f"own{j}"] + [f"s{j}_{k}" for k in range(n_stocks)] for j in range(n_sectors)]
        self.state_vars = [v for sector in self.sectors for v in sector] + (
            ["news"] if news is not None else []
        )
        self.action_vars = ["trade"]
        self.cards = dict.fromkeys(self.state_vars, 2) | {"trade": 1 + 2 * n_sectors}
        if news is not None:
            self.hidden_vars, self.confounder = ("news",), "news"
        self.n_actions = 1 + 2 * n_sectors
        self.policies = {"random": self.random_policy}
        self.build_model()

    def _own(self, j: int) -> int:
        return j * (1 + self.M)

    def _stock(self, j: int, k: int) -> int:
        return j * (1 + self.M) + 1 + k

    # ---- behaviour policies ------------------------------------------------------------------
    def random_policy(self, states, rng):
        return rng.integers(0, self.n_actions, len(states))

    # ---- the model ---------------------------------------------------------------------------
    def next_distributions(self, states, actions):
        E, n = len(states), len(self.state_vars)
        q = np.zeros((E, n))  # P(variable' = 1)
        for j in range(self.N):
            own = states[:, self._own(j)]
            q[:, self._own(j)] = np.where(
                actions == 1 + j, 1.0, np.where(actions == 1 + self.N + j, 0.0, own)
            )
            rising = sum(states[:, self._stock(j, k)] for k in range(self.M))
            p = self.base + self.scale * rising / self.M
            for k in range(self.M):
                q[:, self._stock(j, k)] = p
            if j == 0 and self.news is not None:  # the news hits s0_0 only
                q[:, self._stock(0, 0)] = np.where(states[:, -1] == 1, p * (1 - self.news_effect), p)
        if self.news is not None:
            q[:, -1] = self.news  # drawn afresh, whatever happened
        return np.stack([1 - q, q], axis=-1)

    def action_columns(self, actions):
        return np.asarray(actions, dtype=np.int64)[:, None]  # trade = the env action

    def reward(self, B, actions):
        payoff = np.zeros(len(B))
        for j in range(self.N):
            rising = sum(B[:, self._stock(j, k)] for k in range(self.M))
            payoff += B[:, self._own(j)] * (2 * rising - self.M)  # rising - falling
        return Reward.immediate(np.repeat(payoff[:, None], len(actions), axis=1))

    # ---- other tasks on the same dynamics (reward transfer, S9) ----------------------------------
    def alternative_rewards(self) -> dict:
        """Other tasks on the same market (reward transfer; the news is never read):
        transaction_cost  the original payoff, minus 1 for every buy or sell
        loss_aversion     a falling stock of an owned sector costs 2 (a rising one still earns 1)
        sector0_only      only sector 0 counts (x N, the same scale): its stock s0_0 is the one hit by news"""

        def task(rise, fall, sectors, scale=1.0, trade_cost=0.0):
            def reward(B, actions):
                payoff = np.zeros(len(B))
                for j in sectors:
                    rising = sum(B[:, self._stock(j, k)] for k in range(self.M))
                    payoff += B[:, self._own(j)] * (rise * rising - fall * (self.M - rising))
                trades = (actions[:, 0] > 0).astype(float)
                return Reward.immediate(scale * payoff[:, None] - trade_cost * trades[None, :])

            return reward

        every = range(self.N)
        return {
            "transaction_cost": task(1.0, 1.0, every, trade_cost=1.0),
            "loss_aversion": task(1.0, 2.0, every),
            "sector0_only": task(1.0, 1.0, [0], scale=float(self.N)),
        }

    def true_parents(self):
        truth = {}
        for j in range(self.N):
            truth[f"own{j}"] = {f"own{j}", "trade"}
            for k in range(self.M):
                truth[f"s{j}_{k}"] = {f"s{j}_{i}" for i in range(self.M)} | (
                    {"news"} if (j, k) == (0, 0) and self.news is not None else set()
                )
        if self.news is not None:
            truth["news"] = set()
        inputs = self.state_vars + self.action_vars
        return {v: tuple(x for x in inputs if x in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        start = np.zeros(len(self.B))
        if self.news is None:
            start[0] = 1.0  # nothing owned, every stock falling (state 0 in C order)
            return start
        calm = np.zeros(len(self.state_vars), dtype=np.int64)
        stormy = calm.copy()
        stormy[-1] = 1
        start[encode(calm[None], self.state_cards)] = 1 - self.news
        start[encode(stormy[None], self.state_cards)] = self.news
        return start

    def confounded_probs(self, expert, strength):
        """During news: w.p. strength sell sector 0 (env action 1 + N)."""
        news = self.B[:, -1] == 1
        sell = np.zeros_like(expert)
        sell[:, 1 + self.N] = 1.0
        return np.where(news[:, None], strength * sell + (1 - strength) * expert, expert)

    def state_label(self, values):
        return " ".join(
            "".join(str(int(values[self._own(j) + i])) for i in range(1 + self.M)) for j in range(self.N)
        )
