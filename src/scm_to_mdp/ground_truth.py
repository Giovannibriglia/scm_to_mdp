"""The true MDP of a problem, used only to evaluate (step S9). Learning never sees it.

You fill an MDPGroundTruth from your environment:

    truth = MDPGroundTruth(
        spec=DataSpec(state_vars, action_vars, cards, reward),   # what the learner is told
        P=P,                # [S, A, S'] true transition probabilities; states in C order over state_vars
        actions=actions,    # [A, n_a]  the action columns of every true action (row a = action a)
        start=start,        # [S]       the start distribution
        true_parents=None,  # optional {x: parents of x'} (for SHD / SID and the oracle methods)
    )

States are every combination of the state variables in C order (the first variable is the most
significant digit), the order of scm_to_mdp.core.all_states.

Hidden variables: P may describe more state variables than the data show (e.g. a confounder). Give the
learner's spec as `observed` (its state variables are a subset of spec.state_vars, its action variables the
same). A learned policy acts on the observed state; it is evaluated in the full model, every full state
acting as its observed part. V* is then the optimal value of an agent that also sees the hidden variables
(an upper reference, not reachable in general); observed_optimal() gives the best policy over the observed
state and its value (the target of the evaluation; see its docstring for when it is exact). For a factored
environment, MDPGroundTruth.from_factored builds P from the next-value distribution of every variable, after
checking that the dense array fits in memory (a dense P grows as S^2 * A: 2^15 states with 8 actions need
~140 GB).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from scm_to_mdp.core import Dataset, Reward, all_states, encode
from scm_to_mdp.data import DataError, DataSpec

MAX_MODEL_GB = 4.0  # default: refuse a dense true model whose construction needs more memory


class ModelTooLarge(MemoryError):
    """The dense true model would not fit in the memory budget."""


def dense_model_gb(S: int, A: int) -> float:
    """Peak memory (GB) of building P [S, A, S] as float64 by products of factors (two arrays)."""
    return 2 * S * A * S * 8 / 1e9


def dense_transitions(
    B: np.ndarray, n_actions: int, next_distributions: Callable, max_gb: float = MAX_MODEL_GB
) -> np.ndarray:
    """P [S, A, S'] = prod_i P(x_i' | s, a), from next_distributions(states [E, n], actions [E]) ->
    [E, n, C] (the variables move independently given the state and the action)."""
    S, A = len(B), n_actions
    need = dense_model_gb(S, A)
    if need > max_gb:
        raise ModelTooLarge(
            f"the dense true model has {S} states x {A} actions and needs ~{need:.1f} GB "
            f"(limit {max_gb:g} GB); use a smaller problem or raise max_gb"
        )
    states = np.repeat(B, A, axis=0)  # every (s, a) pair, s major
    dist = next_distributions(states, np.tile(np.arange(A), S))  # [S * A, n, C]
    P = np.ones((S * A, S))
    for i in range(B.shape[1]):
        P *= dist[:, i, B[:, i]]  # factor i at the value of x_i in every next state s'
    return P.reshape(S, A, S)


@dataclass(frozen=True)
class MDPGroundTruth:
    spec: DataSpec
    P: np.ndarray  # [S, A, S']
    actions: np.ndarray  # [A, n_a]
    start: np.ndarray  # [S]
    true_parents: dict[str, tuple[str, ...]] | None = None
    state_names: list[str] | None = None  # optional short names of the states, for pictures
    observed: DataSpec | None = None  # the learner's spec when some state variables are hidden
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self.validate()

    # ---- what it is made of ------------------------------------------------------------------
    @property
    def B(self) -> np.ndarray:
        """[S, n] every state as a vector of values."""
        if "B" not in self._cache:
            self._cache["B"] = all_states([self.spec.cards[v] for v in self.spec.state_vars])
        return self._cache["B"]

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @property
    def reward(self) -> Reward:
        """The reward over the true actions."""
        if "reward" not in self._cache:
            self._cache["reward"] = self.spec.reward(self.B, self.actions)
        return self._cache["reward"]

    @property
    def learner_spec(self) -> DataSpec:
        """What the learner is told: the observed spec (all of spec if nothing is hidden)."""
        return self.spec if self.observed is None else self.observed

    @property
    def hidden(self) -> list[str]:
        return [v for v in self.spec.state_vars if v not in self.learner_spec.state_vars]

    def observed_index(self) -> np.ndarray:
        """[S] the index of the observed part of every full state (the identity if nothing is hidden)."""
        if "observed_index" not in self._cache:
            obs = self.learner_spec
            columns = [self.spec.state_vars.index(v) for v in obs.state_vars]
            self._cache["observed_index"] = encode(self.B[:, columns], [obs.cards[v] for v in obs.state_vars])
        return self._cache["observed_index"]

    def state_name(self, s: int) -> str:
        if self.state_names is not None:
            return self.state_names[s]
        return "".join(str(int(v)) for v in self.B[s])

    # ---- checks ------------------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError if the arrays do not describe an MDP over the spec's variables."""
        spec, S = self.spec, self.spec.n_states
        A = len(self.actions)
        if self.P.shape != (S, A, S):
            raise ValueError(f"P must have shape [S, A, S] = [{S}, {A}, {S}], found {list(self.P.shape)}")
        if self.actions.shape != (A, len(spec.action_vars)):
            raise ValueError(
                f"actions must have shape [A, {len(spec.action_vars)}] (one column per action var)"
            )
        for j, a in enumerate(spec.action_vars):
            if self.actions[:, j].min() < 0 or self.actions[:, j].max() >= spec.cards[a]:
                raise ValueError(f"actions: column {a!r} has values outside 0..{spec.cards[a] - 1}")
        if len({tuple(a) for a in self.actions}) != A:
            raise ValueError("actions: two true actions have the same action columns")
        if (self.P < -1e-12).any() or not np.allclose(self.P.sum(-1), 1.0, atol=1e-6):
            raise ValueError("P: every row P[s, a, :] must be a probability vector")
        if self.start.shape != (S,) or (self.start < 0).any() or not np.isclose(self.start.sum(), 1.0):
            raise ValueError(f"start must be a probability vector over the {S} states")
        r = self.reward
        if r.R.shape != (S, A) or r.r_next.shape != (S,) or r.terminal.shape != (S,):
            raise ValueError("spec.reward(B, actions) must return R [S, A], r_next [S] and terminal [S]")
        if self.true_parents is not None:
            inputs = set(spec.state_vars + spec.action_vars)
            if set(self.true_parents) != set(spec.state_vars):
                raise ValueError("true_parents must have one entry per state variable")
            for v, ps in self.true_parents.items():
                if not set(ps) <= inputs:
                    raise ValueError(f"true_parents[{v!r}] has names that are not state or action variables")
        if self.observed is not None:
            obs = self.observed
            if not set(obs.state_vars) <= set(spec.state_vars) or obs.action_vars != spec.action_vars:
                raise ValueError(
                    "observed: its state variables must be a subset of spec's, its actions the same"
                )
            if any(obs.cards[v] != spec.cards[v] for v in obs.state_vars + obs.action_vars):
                raise ValueError("observed: numbers of values must match spec")
            seen = obs.reward(all_states([obs.cards[v] for v in obs.state_vars]), self.actions)
            idx = self.observed_index()
            if not (np.allclose(seen.R[idx], r.R) and np.allclose(seen.r_next[idx], r.r_next)):
                raise ValueError("the reward may not depend on a hidden variable (the learner must know it)")

    def check_data(self, df) -> None:
        """Raise DataError if the table does not match the learner's spec or uses an action the truth
        lacks."""
        self.learner_spec.check(df)
        known = {tuple(a) for a in self.actions}
        unknown = {a for a in df[self.spec.action_vars].itertuples(index=False, name=None)} - known
        if unknown:
            raise DataError(f"the data use action vectors the true MDP does not have: {sorted(unknown)[:5]}")

    # ---- what evaluation needs ---------------------------------------------------------------
    def actions_of(self, dataset: Dataset) -> np.ndarray:
        """[K] the true action index of every action vector of the dataset."""
        lookup = {tuple(a): i for i, a in enumerate(self.actions)}
        return np.array([lookup[tuple(a)] for a in dataset.actions], dtype=np.int64)

    def parents_in(self, dataset: Dataset) -> dict[str, tuple[str, ...]]:
        """The true parents of the observed variables, in the order of the dataset's inputs (a hidden
        parent is left out: no graph over the observed columns can contain it)."""
        if self.true_parents is None:
            raise ValueError("this ground truth has no true_parents")
        return {v: tuple(x for x in dataset.inputs if x in self.true_parents[v]) for v in dataset.state_vars}

    def optimal_values(self, gamma: float) -> np.ndarray:
        """V*[s] (cached per gamma)."""
        key = ("V*", gamma)
        if key not in self._cache:
            from scm_to_mdp.solving import value_iteration

            self._cache[key] = value_iteration.solve(self.P, self.reward, gamma).V
        return self._cache[key]

    # ---- the best an agent can do on the observed state --------------------------------------
    def hidden_prior(self) -> np.ndarray:
        """[S] the start-distribution weight of the hidden part of every full state (all ones when
        nothing is hidden). Used as P(hidden | observed) in the observed MDP below."""
        if not self.hidden:
            return np.ones(self.n_states)
        columns = [self.spec.state_vars.index(v) for v in self.hidden]
        h = encode(self.B[:, columns], [self.spec.cards[v] for v in self.hidden])
        prior = np.bincount(h, weights=self.start, minlength=int(h.max()) + 1)
        return prior[h]

    def observed_next(self) -> np.ndarray:
        """[S, A, S_obs] P(observed part of s' | s, a): the true kernel summed over the hidden part of s'."""
        if "observed_next" not in self._cache:
            idx = self.observed_index()
            S_obs = self.learner_spec.n_states
            out = np.zeros((self.n_states, len(self.actions), S_obs))
            for a in range(len(self.actions)):
                np.add.at(out[:, a].T, idx, self.P[:, a].T)  # out[s, a, o'] += P[s, a, s'] for s' -> o'
            self._cache["observed_next"] = out
        return self._cache["observed_next"]

    def observed_optimal(self, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """(pi_obs [S_obs], V [S]): the policy over the observed state that is optimal in the observed
        MDP, and its exact value in the full model, per full state. The observed MDP averages the
        kernel over the hidden part with weight P(hidden) of the start distribution (hidden_prior).
        This is exact when the hidden variables are drawn afresh at every step, independently of the
        rest (then no policy over the observed state does better). For a hidden variable that
        persists over the episode it is a lower reference: a policy over the observed state could
        do better by inferring the variable from the history. Nothing hidden: V*, pi*."""
        key = ("pi_obs", gamma)
        if key not in self._cache:
            from scm_to_mdp.solving import value_iteration

            if not self.hidden:
                solution = value_iteration.solve(self.P, self.reward, gamma)
                self._cache[key] = (solution.pi, self.optimal_values(gamma))
            else:
                idx, w = self.observed_index(), self.hidden_prior()
                obs = self.learner_spec
                S_obs = obs.n_states
                weight = np.zeros(S_obs)
                np.add.at(weight, idx, w)
                P_obs = np.zeros((S_obs, len(self.actions), S_obs))
                np.add.at(P_obs, idx, w[:, None, None] * self.observed_next())
                P_obs /= weight[:, None, None]
                reward = obs.reward(all_states([obs.cards[v] for v in obs.state_vars]), self.actions)
                pi_obs = value_iteration.solve(P_obs, reward, gamma).pi
                V = value_iteration.policy_value(self.P, self.reward, pi_obs[idx], gamma)
                self._cache[key] = (pi_obs, V)
        return self._cache[key]

    def random_policy_values(self, gamma: float) -> np.ndarray:
        """V[s] of the policy that picks a true action uniformly at random (cached per gamma)."""
        key = ("V_random", gamma)
        if key not in self._cache:
            P, r = self.P.mean(1), self.reward  # [S, S'], the kernel of the uniform policy
            goes_on = np.where(r.terminal, 0.0, 1.0)
            self._cache[key] = np.linalg.solve(
                np.eye(self.n_states) - gamma * P * goes_on[None, :], r.R.mean(1) + P @ r.r_next
            )
        return self._cache[key]

    def as_model(self):
        """The true MDP as an exact IntervalMDP (L == U) over the true actions: solve it with the
        "vi" solver to get V* (a check of the evaluation)."""
        from scm_to_mdp.core import IntervalMDP

        return IntervalMDP(self.P, self.P, self.reward)

    # ---- builders ----------------------------------------------------------------------------
    @classmethod
    def from_factored(
        cls,
        spec: DataSpec,
        next_distributions: Callable[[np.ndarray, np.ndarray], np.ndarray],
        actions: np.ndarray,
        start: np.ndarray,
        true_parents: dict | None = None,
        state_names: list[str] | None = None,
        max_gb: float = MAX_MODEL_GB,
    ) -> "MDPGroundTruth":
        """For a factored environment: next_distributions(states [E, n], true action index [E]) ->
        [E, n, C], the distribution of every next value (padded with 0 beyond a variable's card)."""
        B = all_states([spec.cards[v] for v in spec.state_vars])
        P = dense_transitions(B, len(actions), next_distributions, max_gb)
        return cls(
            spec, P, np.asarray(actions, dtype=np.int64), np.asarray(start, float), true_parents, state_names
        )


def state_index(spec: DataSpec, values: np.ndarray) -> np.ndarray:
    """Index of state vectors (last axis) in the order of MDPGroundTruth.P."""
    return encode(values, [spec.cards[v] for v in spec.state_vars])
