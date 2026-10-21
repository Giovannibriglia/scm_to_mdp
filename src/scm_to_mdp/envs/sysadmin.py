"""SysAdmin: a network of N machines (Guestrin, Koller, Parr, Venkataraman, JAIR 2003,
https://arxiv.org/abs/1106.1822). Small, factored, stochastic, with a known sparse 2-slice graph.

State: machine i is up (1) or down (0). Env actions: 0 = no-op, i + 1 = repair machine i.
Machine i at the next step depends on itself, its neighbours and whether it is repaired:

    P(m_i' = 1) = 0.95                  if repaired
                = 0.90                  if up and every neighbour up
                = 0.60                  if up and some neighbour down
                = 0.05                  if down and not repaired

Topology (who is a neighbour of machine i; our settings):
    "ring"           one-way ring: machine i-1                  (in-degree 3, the default)
    "bidirectional"  machines i-1 and i+1                       (in-degree 4)
    "star"           machine 0 is the hub: every other machine has the hub as its neighbour;
                     the hub has none                           (in-degree 3, hub 2)

Reward: number of machines up, minus repair_cost when a repair is made. Every episode starts with
all machines up.

Data columns: m0..m{N-1}, a0..a{N-1}, m0'..m{N-1}'. The action is one-hot over a0..a{N-1}, and
a_j repairs machine repairs[j], a hidden random shuffle: the learner must find which is which.
"""

import numpy as np

from scm_to_mdp.core import Dataset, Reward
from scm_to_mdp.envs.base import Env

TOPOLOGIES = ("ring", "bidirectional", "star")


def p_up(m_self, any_neighbour_down, repaired):
    """P(m_i' = 1 | m_i, neighbours, repaired_i), element-wise."""
    return np.where(repaired, 0.95, np.where(m_self == 1, np.where(any_neighbour_down, 0.60, 0.90), 0.05))


def neighbours(n: int, topology: str) -> list[list[int]]:
    if topology == "ring":
        return [[(i - 1) % n] for i in range(n)]
    if topology == "bidirectional":
        return [sorted({(i - 1) % n, (i + 1) % n} - {i}) for i in range(n)]
    if topology == "star":
        return [[]] + [[0] for _ in range(1, n)]
    raise ValueError(f"topology must be one of {TOPOLOGIES}")


class SysAdmin(Env):
    def __init__(self, n: int, seed: int, topology: str = "ring", repair_cost: float = 0.4):
        self.n, self.topology, self.repair_cost = n, topology, repair_cost
        self.rng = np.random.default_rng(seed)
        self.repairs = [int(i) for i in self.rng.permutation(n)]  # hidden: a_j repairs machine repairs[j]
        self.neighbours = neighbours(n, topology)
        self.state_vars = [f"m{i}" for i in range(n)]
        self.action_vars = [f"a{j}" for j in range(n)]
        self.cards = dict.fromkeys(self.state_vars + self.action_vars, 2)
        self.n_actions = n + 1
        self.policies = {"random": self.random_policy, "repair_broken": self.repair_broken_policy}
        self.build_model()

    # ---- behaviour policies ------------------------------------------------------------------
    def random_policy(self, states, rng):
        """Uniform over the no-op and the N repairs."""
        return rng.integers(0, self.n + 1, len(states))

    def repair_broken_policy(self, states, rng):
        """With probability 0.6 repair a random broken machine (if any), otherwise a uniform action."""
        broken = states == 0
        pick = np.argmax(rng.random(states.shape) * broken, axis=1) + 1  # a random broken machine
        uniform = rng.integers(0, self.n + 1, len(states))
        return np.where(broken.any(1) & (rng.random(len(states)) < 0.6), pick, uniform)

    # ---- the model ---------------------------------------------------------------------------
    def next_distributions(self, states, actions):
        repaired = actions[:, None] == np.arange(self.n)[None, :] + 1  # [E, n]
        down = states == 0
        neighbour_down = np.stack([down[:, nb].any(1) for nb in self.neighbours], axis=1)  # [E, n]
        q = p_up(states, neighbour_down, repaired)  # [E, n]
        return np.stack([1 - q, q], axis=-1)  # [E, n, 2]

    def action_columns(self, actions):
        return np.stack([actions == self.repairs[j] + 1 for j in range(self.n)], axis=1).astype(np.int64)

    def start_distribution(self):
        start = np.zeros(len(self.B))
        start[-1] = 1.0  # all machines up (the last state in C order)
        return start

    def initial_states(self, n_episodes):
        return np.ones((n_episodes, self.n), dtype=np.int64)

    # ---- what the learner is given, and what the evaluation needs ----------------------------
    def reward(self, B, actions):
        return Reward.immediate(B.sum(1)[:, None] - self.repair_cost * (actions.sum(1)[None, :] > 0))

    def true_parents(self, dataset: Dataset):
        truth = {
            f"m{i}": {f"m{i}", f"a{self.repairs.index(i)}"} | {f"m{j}" for j in self.neighbours[i]}
            for i in range(self.n)
        }
        return {v: tuple(x for x in dataset.inputs if x in truth[v]) for v in dataset.state_vars}
