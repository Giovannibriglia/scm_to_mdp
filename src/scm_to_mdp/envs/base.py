"""What every environment provides (pipeline step S0, and its behaviour policies for S1).

An environment is a factored MDP: given the state and the action, the next value of every state
variable is drawn independently from next_distributions(). From that one function the base class
builds the exact model (to evaluate policies) and the simulator (to produce data). The learner
never sees the model: it gets only the table returned by sample() and the DataSpec (spec()).
ground_truth() packs the exact model as an MDPGroundTruth, the input of the evaluation (S9).

The dense model has S x A x S numbers: building it is refused above max_model_gb (ModelTooLarge).

Behaviour policies shared by every environment (behaviour(name, gamma, ...), exact tables over the
full state, see scm_to_mdp/behaviour.py):
    "uniform"     every action equally likely;
    "expert"      the best policy over the observed state, with a uniform action w.p. epsilon;
    "confounded"  the expert, except in states where the environment's confounder (a hidden
                  variable that also changes the dynamics) is "on": there it takes the
                  environment's confounded action w.p. strength (confounded_probs).
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from scm_to_mdp import behaviour
from scm_to_mdp.core import NEXT_MARK, Reward, all_states, encode
from scm_to_mdp.data import DataSpec
from scm_to_mdp.ground_truth import MAX_MODEL_GB, MDPGroundTruth, dense_transitions

# a behaviour policy: (states [E, n], rng) -> env actions [E]
BehaviourPolicy = Callable[[np.ndarray, np.random.Generator], np.ndarray]


class Env:
    state_vars: list[str]
    action_vars: list[str]  # the action columns of the data
    cards: dict[str, int]  # number of values of every state and action variable
    n_actions: int  # env actions are 0..n_actions-1
    policies: dict[str, BehaviourPolicy]  # the named behaviour policies of this env
    rng: np.random.Generator
    max_model_gb: float = MAX_MODEL_GB  # refuse to build a dense true model above this
    hidden_vars: tuple[str, ...] = ()  # state variables in the true model but not in the data
    confounder: str | None = None  # the variable the "confounded" logger reads (hidden or not)

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

    def true_parents(self) -> dict[str, tuple[str, ...]]:
        """The true slice-t parents of every next-state variable (state and action column names,
        in the order state_vars + action_vars)."""
        raise NotImplementedError

    def start_distribution(self) -> np.ndarray:
        """[S] the distribution of the first state of an episode."""
        raise NotImplementedError

    def confounded_probs(self, expert: np.ndarray, strength: float) -> np.ndarray:
        """[S, A] the "confounded" logger from the expert's table [S, A]: in the states where the
        confounder is on, the confounded action w.p. strength. Only environments with a confounder."""
        raise NotImplementedError(f"{type(self).__name__} has no confounder")

    # ---- shared ------------------------------------------------------------------------------
    @property
    def state_cards(self) -> list[int]:
        return [self.cards[v] for v in self.state_vars]

    def build_model(self):
        """Set self.B, self.P [S, A, S'], self.true_reward and self.start from next_distributions."""
        self.B = all_states(self.state_cards)  # [S, n]
        self.P = dense_transitions(self.B, self.n_actions, self.next_distributions, self.max_model_gb)
        self.true_reward = self.reward(self.B, self.action_columns(np.arange(self.n_actions)))
        self.start = self.start_distribution()

    def full_spec(self) -> DataSpec:
        """Every state variable of the true model, with the reward."""
        return DataSpec(list(self.state_vars), list(self.action_vars), dict(self.cards), self.reward)

    def spec(self) -> DataSpec:
        """What the learner is told: the observed variables, numbers of values, action columns and
        the reward (which may not depend on a hidden variable)."""
        if not self.hidden_vars:
            return self.full_spec()
        observed = [v for v in self.state_vars if v not in self.hidden_vars]
        cards = {v: c for v, c in self.cards.items() if v not in self.hidden_vars}
        return DataSpec(observed, list(self.action_vars), cards, self.observed_reward)

    def observed_reward(self, B_observed: np.ndarray, actions: np.ndarray) -> Reward:
        """The reward over observed states: the hidden variables set to 0 (the reward does not read them)."""
        full = np.zeros((len(B_observed), len(self.state_vars)), dtype=np.int64)
        observed = [i for i, v in enumerate(self.state_vars) if v not in self.hidden_vars]
        full[:, observed] = B_observed
        return self.reward(full, actions)

    def hide(self, df: pd.DataFrame) -> pd.DataFrame:
        """The table without the hidden variables (what the learner gets)."""
        return df.drop(columns=[c for v in self.hidden_vars for c in (v, v + NEXT_MARK)])

    def ground_truth(self) -> MDPGroundTruth:
        """The exact model of this environment, for the evaluation (built once, then cached)."""
        if getattr(self, "_truth", None) is None:
            self._truth = self._make_truth()
        return self._truth

    def _make_truth(self) -> MDPGroundTruth:
        return MDPGroundTruth(
            spec=self.full_spec(),
            P=self.P,
            actions=self.action_columns(np.arange(self.n_actions)),
            start=self.start,
            true_parents=self.true_parents(),
            state_names=[self.state_label(b) for b in self.B],
            observed=self.spec() if self.hidden_vars else None,
        )

    def behaviour(self, name: str, gamma: float, epsilon: float = 0.05, strength: float = 0.95) -> np.ndarray:
        """[S, A] a shared behaviour policy as an exact table over the full states (module docstring)."""
        truth = self.ground_truth()
        if name == "uniform":
            return behaviour.uniform(truth)
        expert = behaviour.expert(truth, gamma, epsilon)
        if name == "expert":
            return expert
        if name == "confounded":
            return self.confounded_probs(expert, strength)
        raise ValueError(f"behaviour must be 'uniform', 'expert' or 'confounded', not {name!r}")

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

    def sample(
        self, n_episodes: int, horizon: int, policy: str | BehaviourPolicy | np.ndarray = "random"
    ) -> pd.DataFrame:
        """Run n_episodes episodes of at most horizon steps with a behaviour policy (a name in
        self.policies, a function, or a table [S, A] from behaviour()); an episode stops when it
        reaches a terminal state.
        Returns every transition as one table, indexed by (episode, t)."""
        if isinstance(policy, np.ndarray):  # a table [S, A] over the full states (behaviour())
            policy = behaviour.sampler(policy, self.state_cards)
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
