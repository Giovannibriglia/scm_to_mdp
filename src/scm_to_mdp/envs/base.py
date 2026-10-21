"""What every environment provides (pipeline step S0, and its behaviour policies for S1).

An environment is a factored MDP: given the state and the action, the next value of every state
variable is drawn independently from next_distributions(). From that one function the base class
builds the exact model (to evaluate policies) and the simulator (to produce data). The learner
never sees the model: it gets only the table returned by sample() and what read_dataset needs
(action columns, numbers of values, reward).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from scm_to_mdp.core import NEXT_MARK, Dataset, Reward, all_states, encode

# a behaviour policy: (states [E, n], rng) -> env actions [E]
BehaviourPolicy = Callable[[np.ndarray, np.random.Generator], np.ndarray]


class Env:
    state_vars: list[str]
    action_vars: list[str]  # the action columns of the data
    cards: dict[str, int]  # number of values of every state and action variable
    n_actions: int  # env actions are 0..n_actions-1
    policies: dict[str, BehaviourPolicy]  # the named behaviour policies of this env
    rng: np.random.Generator

    # ---- to implement in every environment --------------------------------------------------
    def next_distributions(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """[E, n, C]: P(x_i' = v | state, action) for a batch of states [E, n] and env actions [E]."""
        raise NotImplementedError

    def action_columns(self, actions: np.ndarray) -> np.ndarray:
        """[E, n_a] the data columns of a batch of env actions."""
        raise NotImplementedError

    def reward(self, B: np.ndarray, actions: np.ndarray) -> Reward:
        """The reward in the learner's terms, for states B and action vectors (data columns)."""
        raise NotImplementedError

    def true_parents(self, dataset: Dataset) -> dict[str, tuple[str, ...]]:
        """The true slice-t parents of every next-state variable, in data column names."""
        raise NotImplementedError

    def start_distribution(self) -> np.ndarray:
        """[S] the distribution of the first state of an episode."""
        raise NotImplementedError

    # ---- shared ------------------------------------------------------------------------------
    @property
    def state_cards(self) -> list[int]:
        return [self.cards[v] for v in self.state_vars]

    def build_model(self):
        """Set self.B, self.P [S, A, S'], self.true_reward and self.start from next_distributions."""
        self.B = all_states(self.state_cards)  # [S, n]
        S, A = len(self.B), self.n_actions
        states = np.repeat(self.B, A, axis=0)  # every (s, a) pair, s major
        actions = np.tile(np.arange(A), S)
        dist = self.next_distributions(states, actions)  # [S * A, n, C]
        P = np.ones((S * A, S))
        for i in range(self.B.shape[1]):  # the variables move independently: product of factors
            P *= dist[:, i, self.B[:, i]]  # [S * A, S']
        self.P = P.reshape(S, A, S)
        self.true_reward = self.reward(self.B, self.action_columns(np.arange(A)))
        self.start = self.start_distribution()

    def env_actions(self, dataset: Dataset) -> np.ndarray:
        """[K] the env action of every action vector of the dataset (to evaluate learned policies)."""
        columns = self.action_columns(np.arange(self.n_actions))
        lookup = {tuple(c): a for a, c in enumerate(columns)}
        return np.array([lookup[tuple(a)] for a in dataset.actions], dtype=np.int64)

    def state_label(self, values: np.ndarray) -> str:
        """A short name of a state, for pictures (default: its values, e.g. "101")."""
        return "".join(str(int(v)) for v in values)

    def is_terminal(self, states: np.ndarray) -> np.ndarray:
        return self.true_reward.terminal[encode(states, self.state_cards)]

    def initial_states(self, n_episodes: int) -> np.ndarray:
        index = self.rng.choice(len(self.B), size=n_episodes, p=self.start)
        return self.B[index]

    def step(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Sample the next states [E, n]. One uniform number per variable: x_i' is the largest
        value v with u_i < P(x_i' >= v) (inverse of the upper tail; for a 0/1 variable: 1 if u < P(1))."""
        dist = self.next_distributions(states, actions)  # [E, n, C]
        tail = np.cumsum(dist[..., ::-1], axis=-1)[..., ::-1]  # tail[..., v] = P(x' >= v)
        u = self.rng.random(states.shape)
        return (u[..., None] < tail[..., 1:]).sum(-1).astype(np.int64)

    def to_columns(self, states, actions, next_states) -> pd.DataFrame:
        """One block of the data table for a batch of transitions."""
        block = {v: states[:, i] for i, v in enumerate(self.state_vars)}
        block |= dict(zip(self.action_vars, self.action_columns(actions).T, strict=True))
        block |= {v + NEXT_MARK: next_states[:, i] for i, v in enumerate(self.state_vars)}
        return pd.DataFrame(block)

    def sample(self, n_episodes: int, horizon: int, policy: str | BehaviourPolicy = "random") -> pd.DataFrame:
        """Run n_episodes episodes of at most horizon steps with a behaviour policy (a name in
        self.policies, or a function); an episode stops when it reaches a terminal state.
        Returns every transition as one table, indexed by (episode, t)."""
        act = self.policies[policy] if isinstance(policy, str) else policy
        s = self.initial_states(n_episodes)
        alive = np.ones(n_episodes, dtype=bool)
        episode = np.arange(n_episodes)
        blocks = []
        for t in range(horizon):
            if not alive.any():
                break
            a = act(s, self.rng)
            s2 = self.step(s, a)
            block = self.to_columns(s[alive], a[alive], s2[alive])
            block.index = pd.MultiIndex.from_arrays(
                [episode[alive], np.full(alive.sum(), t)], names=["episode", "t"]
            )
            blocks.append(block)
            alive &= ~self.is_terminal(s2)
            s = s2
        return pd.concat(blocks)
