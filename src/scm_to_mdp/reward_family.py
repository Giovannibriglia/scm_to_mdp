"""Random rewards on the same dynamics (step S9): is the promise reward-independent?

The promise of a robust method rests on one event about the transitions only (robustness.py: the
true kernel lies in the learned set). On that event the promise is a lower bound for EVERY reward at
once, also one chosen after seeing the data. So each learned model is re-planned (not re-learned)
under N random rewards and every resulting promise is checked against the true value. A failure on
some rewards and not others points at the learned structure (a missing parent matters for one reward
and not for another), not at the confidence level.

Four families, `per_family` rewards each, drawn once per (size, seed) with their own random stream
(RewardFamily.seed): the same rewards for every method and number of transitions.

    dense    r_next(s') = the mean over the state variables of w_i[x_i'], w ~ U[0, 1] per variable
             and value, minus an action cost c * u_a (c ~ U[0, 0.2], u_a ~ U[0, 1] per action vector)
    sparse   1 on arrival in g random non-terminal states (g = 1, 2 or 3), which become terminal
    noise    R(s, a) ~ U[0, 1] independently for every observed state and action vector
    stress   1 on arrival in the g least-visited non-terminal states of the data (the full table of
             the run): a data-dependent reward, which the event above still covers

Every reward reads the observed state only (never a hidden variable) and keeps the original terminal
states (a new terminal set must contain the old one: the data stop there). Rewards lie in [-0.2, 1].

Per run, `metrics` summarises the N re-planned policies (columns rf_*, sweep.py):
    rf_rewards       N, the rewards evaluated (those whose re-planning finished)
    rf_safe_all      the promise held in every non-terminal state for every reward
    rf_safe_share    the share of rewards for which it held
    rf_value_mean, rf_value_min, rf_value_q10   the normalised true value (V^pi - V_rand) /
                     (V*_obs - V_rand) at the start, over the rewards: mean, worst, 10th percentile
    rf_<family>_safe_all, rf_<family>_value_mean   the same per family
A model-free method keeps its policy (methods.replan) and has no promise: only its values count.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from scm_to_mdp.core import Dataset, Reward

FAMILIES = ("dense", "sparse", "noise", "stress")


@dataclass(frozen=True)
class RewardFamily:
    """What the sweep evaluates (Experiment.reward_family; None: nothing)."""

    per_family: int = 5  # rewards per family
    families: tuple[str, ...] = FAMILIES
    n_transitions: tuple[int, ...] | None = None  # the numbers of transitions evaluated (None: every one)
    max_states: int | None = None  # only problems with at most this many observed states (None: all);
    # re-planning costs one solve per reward, which dominates on large problems
    seed: int = 2026  # the random stream of the rewards (independent of the data's)

    def __post_init__(self):
        unknown = set(self.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown reward families {sorted(unknown)}; choose among {FAMILIES}")

    def evaluated_at(self, n: int) -> bool:
        return self.n_transitions is None or n in self.n_transitions

    def covers(self, n_observed_states: int) -> bool:
        return self.max_states is None or n_observed_states <= self.max_states


@dataclass(frozen=True)
class Task:
    """One random reward, over the observed states and every TRUE action vector."""

    family: str
    R: np.ndarray  # [S_obs, A] reward of acting
    r_next: np.ndarray  # [S_obs] reward of arriving
    terminal: np.ndarray  # [S_obs] bool, the episode ends on arrival (contains the original terminals)

    def for_learner(self, true_actions: np.ndarray) -> Reward:
        """The reward over the learner's action vectors (true_actions: their true indices)."""
        return Reward(self.R[:, true_actions], self.r_next, self.terminal)

    def for_truth(self, observed_index: np.ndarray) -> Reward:
        """The same reward over the full states (observed_index: full state -> observed state)."""
        return Reward(self.R[observed_index], self.r_next[observed_index], self.terminal[observed_index])


def draw(family: RewardFamily, B: np.ndarray, cards: list[int], terminal: np.ndarray, n_actions: int,
         visits: np.ndarray, seed: int) -> list[Task]:  # fmt: skip
    """The tasks of one (size, seed). B [S_obs, n]: the observed states; terminal [S_obs]: the original
    terminal states; visits [S_obs]: how often the data visit each state (the stress family)."""
    rng = np.random.default_rng([family.seed, seed])
    S = len(B)
    live = np.flatnonzero(~terminal)
    tasks = []
    for name in family.families:
        for _ in range(family.per_family):
            R, r_next, end = np.zeros((S, n_actions)), np.zeros(S), terminal.copy()
            if name == "dense":
                w = [rng.uniform(0, 1, c) for c in cards]
                r_next = np.mean([w[i][B[:, i]] for i in range(len(cards))], axis=0)
                R = np.broadcast_to(
                    -rng.uniform(0, 0.2) * rng.uniform(0, 1, n_actions), (S, n_actions)
                ).copy()
            elif name == "noise":
                R = rng.uniform(0, 1, (S, n_actions))
            else:  # sparse or stress: a few goal states that pay 1 on arrival and end the episode
                g = min(int(rng.integers(1, 4)), len(live))
                if name == "sparse":
                    goals = rng.choice(live, size=g, replace=False)
                else:  # the least-visited live states (ties broken at random)
                    order = np.lexsort((rng.random(len(live)), visits[live]))
                    goals = live[order[:g]]
                r_next[goals] = 1.0
                end[goals] = True
            tasks.append(Task(name, R, r_next, end))
    return tasks


def metrics(outcomes: list[dict]) -> dict:
    """The rf_* columns from one dict per task: {"family", "safe" (bool or NaN), "value"}."""
    if not outcomes:
        return {"rf_rewards": 0}
    out = {"rf_rewards": len(outcomes)}
    groups = {"": outcomes} | {f"{f}_": [o for o in outcomes if o["family"] == f] for f in FAMILIES}
    for prefix, group in groups.items():
        if not group:
            continue
        safe = np.array([o["safe"] for o in group], dtype=float)
        value = np.array([o["value"] for o in group], dtype=float)
        has_promise = np.isfinite(safe).all()
        out[f"rf_{prefix}safe_all"] = bool(safe.all()) if has_promise else float("nan")
        out[f"rf_{prefix}value_mean"] = float(np.nanmean(value)) if np.isfinite(value).any() else float("nan")
        if prefix == "":
            out["rf_safe_share"] = float(safe.mean()) if has_promise else float("nan")
            finite = value[np.isfinite(value)]
            out["rf_value_min"] = float(finite.min()) if len(finite) else float("nan")
            out["rf_value_q10"] = float(np.quantile(finite, 0.1)) if len(finite) else float("nan")
    return out


COLUMNS = ["rf_rewards", "rf_safe_all", "rf_safe_share", "rf_value_mean", "rf_value_min", "rf_value_q10"]
COLUMNS += [f"rf_{f}_{m}" for f in FAMILIES for m in ("safe_all", "value_mean")]


def with_task(data: Dataset, task: Task, true_actions: np.ndarray) -> Dataset:
    """The dataset with the task's reward (the transitions unchanged)."""
    return dataclasses.replace(data, reward=task.for_learner(true_actions))
