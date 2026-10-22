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

Storm (optional, ours; storm=p): a hidden variable drawn afresh at every step, storm ~ Bernoulli(p),
independent of everything else. It hits one machine, the "exposed" machine 0: during a storm, if
not repaired, it stays or comes up with probability multiplied by (1 - storm_effect). The storm is
never in the data. It acts on one variable only on purpose: a hidden cause of several next-state
variables would make them dependent given (state, action), so no product of factors over the
observed variables would be the right model even with an unconfounded logger. With one variable
the observed model stays a product, and only a logger that reads the storm creates a problem.
The "confounded" logger (Env.behaviour) repairs machine 0 during a storm w.p. strength: rows
where machine 0 is not repaired then come mostly from calm steps, so the data make machine 0 look
healthier than it is and understate what repairing it buys (an MDP with unobserved confounders,
Bareinboim, Causal AI, draft v0.27, Def. 9.4.4, p. 623).
"""

import numpy as np

from scm_to_mdp.core import Reward
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
    def __init__(
        self,
        n: int,
        seed: int,
        topology: str = "ring",
        repair_cost: float = 0.4,
        storm: float | None = None,
        storm_effect: float = 0.4,
        max_model_gb: float | None = None,
    ):
        self.n, self.topology, self.repair_cost = n, topology, repair_cost
        self.storm, self.storm_effect = storm, storm_effect
        if max_model_gb is not None:
            self.max_model_gb = max_model_gb
        self.rng = np.random.default_rng(seed)
        self.repairs = [int(i) for i in self.rng.permutation(n)]  # hidden: a_j repairs machine repairs[j]
        self.neighbours = neighbours(n, topology)
        self.machines = [f"m{i}" for i in range(n)]
        self.state_vars = self.machines + (["storm"] if storm is not None else [])
        self.action_vars = [f"a{j}" for j in range(n)]
        self.cards = dict.fromkeys(self.state_vars + self.action_vars, 2)
        if storm is not None:
            self.hidden_vars, self.confounder = ("storm",), "storm"
        self.n_actions = n + 1
        self.policies = {"random": self.random_policy, "repair_broken": self.repair_broken_policy}
        self.build_model()

    # ---- behaviour policies ------------------------------------------------------------------
    def random_policy(self, states, rng):
        """Uniform over the no-op and the N repairs."""
        return rng.integers(0, self.n + 1, len(states))

    def repair_broken_policy(self, states, rng):
        """With probability 0.6 repair a random broken machine (if any), otherwise a uniform action."""
        broken = states[:, : self.n] == 0
        pick = np.argmax(rng.random(states.shape) * broken, axis=1) + 1  # a random broken machine
        uniform = rng.integers(0, self.n + 1, len(states))
        return np.where(broken.any(1) & (rng.random(len(states)) < 0.6), pick, uniform)

    # ---- the model ---------------------------------------------------------------------------
    def confounded_probs(self, expert, strength):
        """During a storm: w.p. strength repair the exposed machine 0 (env action 1)."""
        storm = self.B[:, self.n] == 1
        repair = np.zeros_like(expert)
        repair[:, 1] = 1.0
        return np.where(storm[:, None], strength * repair + (1 - strength) * expert, expert)

    def next_distributions(self, states, actions):
        machines = states[:, : self.n]
        repaired = actions[:, None] == np.arange(self.n)[None, :] + 1  # [E, n]
        down = machines == 0
        neighbour_down = np.stack([down[:, nb].any(1) for nb in self.neighbours], axis=1)  # [E, n]
        q = p_up(machines, neighbour_down, repaired)  # [E, n]
        if self.storm is None:
            return np.stack([1 - q, q], axis=-1)  # [E, n, 2]
        hit = (states[:, self.n] == 1) & ~repaired[:, 0]  # the storm hits machine 0 only
        q[:, 0] = np.where(hit, q[:, 0] * (1 - self.storm_effect), q[:, 0])
        storm_next = np.full((len(states), 1), self.storm)  # drawn afresh, whatever happened
        q = np.concatenate([q, storm_next], axis=1)
        return np.stack([1 - q, q], axis=-1)  # [E, n + 1, 2]

    def action_columns(self, actions):
        return np.stack([actions == self.repairs[j] + 1 for j in range(self.n)], axis=1).astype(np.int64)

    def start_distribution(self):
        if self.storm is None:
            start = np.zeros(len(self.B))
            start[-1] = 1.0  # all machines up (the last state in C order)
            return start
        all_up = self.B[:, : self.n].all(1)
        return all_up * np.where(self.B[:, self.n] == 1, self.storm, 1 - self.storm)

    def initial_states(self, n_episodes):
        if self.storm is not None:
            return super().initial_states(n_episodes)
        return np.ones((n_episodes, self.n), dtype=np.int64)

    # ---- what the learner is given, and what the evaluation needs ----------------------------
    def reward(self, B, actions):
        up = B[:, : self.n].sum(1)  # machines only (not the storm)
        return Reward.immediate(up[:, None] - self.repair_cost * (actions.sum(1)[None, :] > 0))

    def true_parents(self):
        storm = {"storm"} if self.storm is not None else set()
        truth = {
            f"m{i}": {f"m{i}", f"a{self.repairs.index(i)}"}
            | {f"m{j}" for j in self.neighbours[i]}
            | (storm if i == 0 else set())
            for i in range(self.n)
        }
        if self.storm is not None:
            truth["storm"] = set()
        inputs = self.state_vars + self.action_vars
        return {v: tuple(x for x in inputs if x in truth[v]) for v in self.state_vars}
