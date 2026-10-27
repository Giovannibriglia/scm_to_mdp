"""Chemistry: objects whose colours react along a random causal graph of chosen density.

Inspired by the "chemistry" environment of Ke, Didolkar, Mittal, Goyal, Lajoie, Bauer, Rezende,
Bengio, Mozer and Pal, "Systematic Evaluation of Causal Discovery in Visual Model Based
Reinforcement Learning" (2021, https://arxiv.org/abs/2107.00848). It is NOT a reproduction: ours is
discrete and tabular, and an effect takes one step (the colour of a child at t+1 reacts to its
parents' colours at t), so every variable moves independently given (state, action) and the true
2-slice graph is exactly the reaction graph. The point is a knob: the density of the graph.

State: the colours x_0..x_{n-1} in {0..C-1} (default 6 objects, 3 colours: 729 states).
Graph (per seed): a random order of the objects; every pair (earlier -> later) is an edge w.p.
density; an object keeps at most max_parents parents (a random subset).
Action (one categorical data column push in 0..n): 0 = do nothing, j + 1 = push object j. Every x_j'
has push as a parent (only the value that pushes object j matters to it).
Next colour of object j:
    pushed:              (x_j + 1) mod C w.p. 0.9, else x_j
    not pushed, parents: (sum of the parents' colours + offset_j) mod C w.p. 0.7 (the reaction),
                         x_j w.p. 0.25, a uniform colour w.p. 0.05
    not pushed, no parent: x_j w.p. 0.95, a uniform colour w.p. 0.05
Reward: the share of objects in their target colour (targets random per seed), at every step.
Start: a uniform state. In experiments/domains.py the size is the number of objects and the
density is the grid's knob.

Catalyst (optional, ours; catalyst=p): a hidden variable drawn afresh at every step, catalyst ~
Bernoulli(p), independent of everything else. It drives one reaction, that of the "exposed" object (the
first object, in index order, that has parents): that object reacts w.p. REACTION_CATALYST while the
catalyst is on and REACTION_WITHOUT otherwise, instead of REACTION (it stays w.p. 1 - reaction - NOISE). One
object only, so that the observed model stays a product of factors (see envs/sysadmin.py, storm). The
catalyst is never in the data. The "confounded" logger (Env.behaviour) does nothing (waits for the reaction)
while the catalyst is on, w.p. strength: the data then make that reaction look faster than it is. With
density 0 no object has parents and the catalyst changes nothing: a negative control (no bias whatever the
logger).
"""

import numpy as np

from scm_to_mdp.core import Reward
from scm_to_mdp.envs.base import Env

PUSH_SUCCESS, REACTION, STAY, NOISE = 0.9, 0.7, 0.25, 0.05
REACTION_CATALYST, REACTION_WITHOUT = 0.95, 0.3  # the exposed object, with and without the catalyst


class Chemistry(Env):
    def __init__(self, density: float = 0.3, seed: int = 0, n_objects: int = 6, n_colours: int = 3,
                 max_parents: int = 2, catalyst: float | None = None,
                 max_model_gb: float | None = None):  # fmt: skip
        self.density, self.n, self.C = float(density), n_objects, n_colours
        self.catalyst = catalyst
        if max_model_gb is not None:
            self.max_model_gb = max_model_gb
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        graph_rng = np.random.default_rng([seed, 1])  # the graph and targets: fixed by the seed
        order = graph_rng.permutation(n_objects)
        self.parents = {int(j): [] for j in range(n_objects)}
        for a in range(n_objects):
            for b in range(a + 1, n_objects):
                if graph_rng.random() < self.density:
                    self.parents[int(order[b])].append(int(order[a]))
        for j, ps in self.parents.items():
            if len(ps) > max_parents:
                self.parents[j] = sorted(graph_rng.choice(ps, max_parents, replace=False).tolist())
            else:
                self.parents[j] = sorted(ps)
        self.offset = graph_rng.integers(0, n_colours, n_objects)
        self.target = graph_rng.integers(0, n_colours, n_objects)
        self.objects = [f"x{j}" for j in range(n_objects)]
        self.state_vars = self.objects + (["catalyst"] if catalyst is not None else [])
        self.action_vars = ["push"]
        self.cards = {**dict.fromkeys(self.objects, n_colours), "push": n_objects + 1}
        self.exposed = next(
            (j for j in range(n_objects) if self.parents[j]), None
        )  # what the catalyst speeds
        if catalyst is not None:
            self.cards = {**self.cards, "catalyst": 2}
            self.hidden_vars, self.confounder = ("catalyst",), "catalyst"
        self.n_actions = n_objects + 1  # 0 = nothing, j + 1 = push object j
        self.policies = {"random": self.random_policy}
        self.build_model()

    def random_policy(self, states, rng):
        return rng.integers(0, self.n_actions, len(states))

    def confounded_probs(self, expert, strength):
        """While the catalyst is on: w.p. strength do nothing (env action 0)."""
        wait = np.zeros_like(expert)
        wait[:, 0] = 1.0
        on = self.B[:, self.n] == 1
        return np.where(on[:, None], strength * wait + (1 - strength) * expert, expert)

    def next_distributions(self, states, actions):
        E, C = len(states), max(self.C, 2)
        dist = np.zeros((E, len(self.state_vars), C))
        rows = np.arange(E)
        on = states[:, self.n] == 1 if self.catalyst is not None else np.zeros(E, bool)
        for j in range(self.n):
            x = states[:, j]
            pushed = actions == j + 1
            d = np.full((E, C), NOISE / C)
            if self.parents[j]:
                reaction = (states[:, self.parents[j]].sum(1) + self.offset[j]) % C
                fast = np.full(E, REACTION)
                if self.catalyst is not None and j == self.exposed:
                    fast = np.where(on, REACTION_CATALYST, REACTION_WITHOUT)
                d[rows, reaction] += fast
                d[rows, x] += 1 - NOISE - fast  # STAY without the catalyst
            else:
                d[rows, x] += 1 - NOISE
            push = np.zeros((E, C))
            push[rows, (x + 1) % C] = PUSH_SUCCESS
            push[rows, x] += 1 - PUSH_SUCCESS
            dist[:, j, : self.C] = np.where(pushed[:, None], push, d)
        if self.catalyst is not None:
            dist[:, self.n, :2] = [1 - self.catalyst, self.catalyst]  # drawn afresh
        return dist

    def action_columns(self, actions):
        return np.asarray(actions, dtype=np.int64)[:, None]  # push = the env action

    def reward(self, B, actions):
        share = (B[:, : self.n] == self.target[None, :]).mean(1)  # objects only
        return Reward.immediate(np.repeat(share[:, None], len(actions), axis=1))

    # ---- other tasks on the same dynamics (reward transfer, S9) ----------------------------------
    def alternative_rewards(self) -> dict:
        """Other tasks on the same reactions (reward transfer; the catalyst is never read): three new
        target colourings, target_1..target_3, drawn from the seed (rng [seed, 2, i]); the reward is
        the share of objects in their target colour, as in the original."""

        def task(target):
            def reward(B, actions):
                share = (B[:, : self.n] == target[None, :]).mean(1)
                return Reward.immediate(np.repeat(share[:, None], len(actions), axis=1))

            return reward

        return {
            f"target_{i}": task(np.random.default_rng([self.seed, 2, i]).integers(0, self.C, self.n))
            for i in (1, 2, 3)
        }

    def true_parents(self):
        inputs = self.state_vars + self.action_vars
        catalyst = {"catalyst"} if self.catalyst is not None else set()
        truth = {
            f"x{j}": {f"x{j}", "push"}
            | {f"x{p}" for p in self.parents[j]}
            | (catalyst if j == self.exposed else set())
            for j in range(self.n)
        }
        if self.catalyst is not None:
            truth["catalyst"] = set()
        return {v: tuple(x for x in inputs if x in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        if self.catalyst is None:
            return np.full(len(self.B), 1.0 / len(self.B))
        on = self.B[:, self.n] == 1
        return np.where(on, self.catalyst, 1 - self.catalyst) / self.C**self.n

    def initial_states(self, n_episodes):
        if self.catalyst is not None:
            return super().initial_states(n_episodes)
        return self.B[self.rng.integers(0, len(self.B), n_episodes)]

    @property
    def n_edges(self) -> int:
        return sum(len(ps) for ps in self.parents.values())
