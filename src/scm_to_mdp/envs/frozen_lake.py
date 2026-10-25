"""Frozen Lake (Gymnasium's FrozenLake-v1: Towers et al., "Gymnasium: a standard interface for
reinforcement learning environments", 2024, https://arxiv.org/abs/2407.17032), with one or two
agents. Schnitzer, Abate, Parker (AAAI 2026, App. E.3) use a two-agent version as a factored MDP;
their PRISM model (two agents driven by one command, no holes, a cost per move) differs from Gym,
so this module follows Gym and adds the second agent with its own actions.

Map: Gym's "4x4" or "8x8" (S start, F frozen, H hole, G goal). Every agent starts on S.
Variables, one per dimension:
    state     x<i> (column 0..n-1) and y<i> (row 0..n-1, 0 = top) of every agent i
    action    a<i> in 0..3 for every agent: 0 left, 1 down, 2 right, 3 up (Gym's order)
Moves: slippery (Gym's is_slippery): the agent moves in the intended direction or in either
perpendicular one, 1/3 each; into a wall it stays. Not slippery: the intended move. An agent on a hole
or on the goal stays there (whatever it does).

A slip moves the row OR the column, in one draw: under Gym's law x<i>' and y<i>' are dependent given
(state, action). axes chooses the dynamics:
    "independent"  ours, the default: each axis keeps its marginal under Gym's law, and the axes are
                   drawn independently (from an inner cell: a diagonal slip w.p. 2/9, 1/9 each
                   way, and no move w.p. 2/9).
                   Every variable then moves independently given (state, action), like envs/drone.py.
    "joint"        Gym's law: the pair (y<i>', x<i>') has the joint law of the slip. The true graph
                   then has a same-step edge y<i>' -> x<i>' (a factorisation of the shared slip, not a
                   causal claim: x<i>' -> y<i>' gives the same kernel).
Without slip (and without wind) the moves are deterministic and both choices give Gym's law.

Episode end and reward (no "done" variable: the end is a set of states, Reward.terminal):
    one agent    the episode ends in a hole (reward 0) or on the goal (reward 1 on arrival)
    two agents   the episode ends when both agents are on a hole or the goal; it then pays the number
                 of agents on the goal (an agent that finishes first waits where it is). The agents
                 never interact: the true parents of x<i>', y<i>' are agent i's cell and move only.

Wind (optional, ours; wind=p): a hidden variable drawn afresh at every step, wind ~ Bernoulli(p). It
pushes agent 0 only, toward the holes of the lower rows. With axes "independent" it pushes the ROW
only: while it is on, y0' is, w.p. wind_effect, the row of a move down (else the row of the slip), and
x0' keeps its calm law. So x0' and y0' stay independent given the OBSERVED (state, action) even
though the wind is hidden (a wind acting on both axes would make them dependent once it is averaged
out). True parents: x0' <- x0, y0, a0; y0' <- x0, y0, a0, wind. With axes "joint", agent 0's move
(both axes) is mixed with "down" w.p. wind_effect, and the same-step edge carries the dependence.
The "confounded" logger (Env.behaviour) makes agent 0 move up during wind w.p. strength, so the data
rarely show agent 0's other moves under wind.
"""

import itertools

import numpy as np

from scm_to_mdp.core import NEXT_MARK, Reward, encode
from scm_to_mdp.envs.base import Env

MAPS = {  # Gymnasium's toy_text/frozen_lake.py, MAPS
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"],
}
MOVES = ["left", "down", "right", "up"]
STEP = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # (row, column)
AXES = ("joint", "independent")


class FrozenLake(Env):
    def __init__(
        self,
        map_name: str = "4x4",
        n_agents: int = 1,
        seed: int = 0,
        slippery: bool = True,
        wind: float | None = None,
        wind_effect: float = 0.5,
        axes: str = "independent",
        max_model_gb: float | None = None,
    ):
        if n_agents not in (1, 2):
            raise ValueError("n_agents must be 1 or 2")
        if axes not in AXES:
            raise ValueError(f"axes must be one of {AXES}")
        self.grid = MAPS[map_name]
        self.size, self.n_agents, self.slippery = len(self.grid), n_agents, slippery
        self.wind, self.wind_effect, self.axes = wind, wind_effect, axes
        if max_model_gb is not None:
            self.max_model_gb = max_model_gb
        self.rng = np.random.default_rng(seed)
        letters = "".join(self.grid)
        self.holes = np.array([c == "H" for c in letters])  # [cells], cell = row * n + column
        self.goal, self.start_cell = letters.index("G"), letters.index("S")
        self.ends = self.holes.copy()
        self.ends[self.goal] = True  # an agent here stays
        self.coords = [(f"x{i}", f"y{i}") for i in range(n_agents)]
        self.state_vars = [v for xy in self.coords for v in xy] + (["wind"] if wind is not None else [])
        self.action_vars = [f"a{i}" for i in range(n_agents)]
        self.cards = dict.fromkeys(self.state_vars, self.size) | dict.fromkeys(self.action_vars, 4)
        if wind is not None:
            self.cards["wind"] = 2
            self.hidden_vars, self.confounder = ("wind",), "wind"
        self.n_actions = 4**n_agents
        self.moves_of = np.array(list(itertools.product(range(4), repeat=n_agents)))  # env action -> moves
        self.policies = {"random": self.random_policy}
        self._landing = self._landing_table()  # [cells, 4 directions] -> cell
        self.build_model()

    @property
    def coupled(self) -> bool:
        """Are x<i>' and y<i>' dependent given (state, action) in this lake (axes "joint" with slip or
        wind)?"""
        return self.axes == "joint" and (self.slippery or self.wind is not None)

    def _landing_table(self) -> np.ndarray:
        n = self.size
        table = np.zeros((n * n, 4), dtype=np.int64)
        for cell in range(n * n):
            r, c = divmod(cell, n)
            for d, (dr, dc) in STEP.items():
                r2, c2 = min(max(r + dr, 0), n - 1), min(max(c + dc, 0), n - 1)
                table[cell, d] = cell if self.ends[cell] else r2 * n + c2
        return table

    def _cells(self, states: np.ndarray, i: int) -> np.ndarray:
        return states[:, 2 * i + 1] * self.size + states[:, 2 * i]  # row * n + column

    # ---- behaviour policies ------------------------------------------------------------------
    def random_policy(self, states, rng):
        return rng.integers(0, self.n_actions, len(states))

    # ---- the model ---------------------------------------------------------------------------
    def _direction_probs(self, move: np.ndarray) -> np.ndarray:
        """[E, 4] the probability of every direction actually taken for an intended move [E]."""
        E = len(move)
        out = np.zeros((E, 4))
        if self.slippery:
            for shift in (-1, 0, 1):
                out[np.arange(E), (move + shift) % 4] += 1 / 3
        else:
            out[np.arange(E), move] = 1.0
        return out

    def _cell_laws(self, states, actions, wind: bool = True) -> list[np.ndarray]:
        """[E, cells] the law of every agent's next cell (Gym's law; with the wind if wind)."""
        E, cells = len(states), self.size**2
        moves = self.moves_of[np.asarray(actions)]  # [E, n_agents]
        laws = []
        for i in range(self.n_agents):
            direction = self._direction_probs(moves[:, i])  # [E, 4]
            if wind and i == 0 and self.wind is not None:
                down = np.zeros_like(direction)
                down[:, 1] = 1.0
                windy = (states[:, -1] == 1)[:, None]
                direction = np.where(
                    windy, (1 - self.wind_effect) * direction + self.wind_effect * down, direction
                )
            land = self._landing[self._cells(states, i)]  # [E, 4]
            law = np.zeros((E, cells))
            for d in range(4):
                np.add.at(law, (np.arange(E), land[:, d]), direction[:, d])
            laws.append(law)
        return laws

    def next_distributions(self, states, actions):
        """The marginal of every variable (with axes "joint", coupled_next gives the pairs' joint law)."""
        E, n = len(states), self.size
        dist = np.zeros((E, len(self.state_vars), max(n, 2)))
        laws = self._cell_laws(states, actions)
        row_only = self.wind is not None and self.axes == "independent"  # the wind pushes the row only
        calm = self._cell_laws(states, actions, wind=False) if row_only else laws
        for i in range(self.n_agents):
            dist[:, 2 * i, :n] = calm[i].reshape(E, n, n).sum(1)  # x' (column)
            dist[:, 2 * i + 1, :n] = laws[i].reshape(E, n, n).sum(2)  # y' (row)
        if self.wind is not None:
            dist[:, -1, 0], dist[:, -1, 1] = 1 - self.wind, self.wind
        return dist

    def coupled_next(self, states, actions) -> dict:
        """{(index of y<i>, index of x<i>): [E, rows, columns]} the joint law of each agent's next
        cell, when the axes are coupled (see the module docstring); else {}."""
        if not self.coupled:
            return {}
        E, n = len(states), self.size
        laws = self._cell_laws(states, actions)
        return {(2 * i + 1, 2 * i): law.reshape(E, n, n) for i, law in enumerate(laws)}

    def action_columns(self, actions):
        return self.moves_of[np.asarray(actions)].astype(np.int64)  # [E, n_agents]: a<i> = agent i's move

    def _ended(self, B) -> np.ndarray:
        """[S, n_agents] is agent i on a hole or the goal."""
        return np.stack([self.ends[self._cells(B, i)] for i in range(self.n_agents)], axis=1)

    def reward(self, B, actions):
        S = len(B)
        on_goal = np.stack([self._cells(B, i) == self.goal for i in range(self.n_agents)], axis=1)
        terminal = self._ended(B).all(1)  # one agent: a hole or the goal; two: both have finished
        r_next = np.where(terminal, on_goal.sum(1), 0).astype(float)
        return Reward(np.zeros((S, len(actions))), r_next, terminal)

    # ---- other tasks on the same dynamics (reward transfer, S9) ----------------------------------
    def alternative_rewards(self) -> dict:
        """Other tasks on the same lake (reward transfer; the wind is never read):
        goal_top_right    reach the top-right cell instead of G
        goal_bottom_left  reach the leftmost free cell of the bottom row instead of G
        step_cost         the original goal, minus 0.01 per step
        One agent: reaching the new cell pays 1 and ends the episode; holes still end it with nothing,
        and so does G (the dynamics freeze an agent there: under the new task G is a trap).
        Two agents: both agents on the new cell at once pays 1 and ends the episode (a rendezvous); an
        agent on a hole or on G (where it stays) ends it with nothing."""
        n, letters = self.size, "".join(self.grid)
        bottom = next(c for c in range(n * (n - 1), n * n) if letters[c] == "F")
        original = self.reward

        def goal(cell):
            def reward(B, actions):
                arrived = np.stack([self._cells(B, i) == cell for i in range(self.n_agents)], axis=1).all(1)
                terminal = arrived | self._ended(B).any(1)
                return Reward(np.zeros((len(B), len(actions))), arrived.astype(float), terminal)

            return reward

        def step_cost(B, actions):
            r = original(B, actions)
            return Reward(r.R - 0.01, r.r_next, r.terminal)

        return {"goal_top_right": goal(n - 1), "goal_bottom_left": goal(bottom), "step_cost": step_cost}

    def true_parents(self):
        """Agent i's cell and move; agent 0's row also the wind (with axes "joint" its column too); with
        coupled axes, y<i>' is also a parent of x<i>' (a same-step edge, named with NEXT_MARK)."""
        truth = {}
        for i, (x, y) in enumerate(self.coords):
            own = {x, y, f"a{i}"}
            windy = {"wind"} if i == 0 and self.wind is not None else set()
            truth[x] = own | (windy if self.axes == "joint" else set())
            truth[y] = own | windy
            if self.coupled:
                truth[x].add(y + NEXT_MARK)
        if self.wind is not None:
            truth["wind"] = set()
        inputs = self.state_vars + self.action_vars + [v + NEXT_MARK for v in self.state_vars]
        return {v: tuple(c for c in inputs if c in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        start = np.zeros(len(self.B))
        r, c = divmod(self.start_cell, self.size)
        first = np.array([c, r] * self.n_agents + ([0] if self.wind is not None else []), dtype=np.int64)
        if self.wind is None:
            start[encode(first[None], self.state_cards)] = 1.0
            return start
        for w, p in ((0, 1 - self.wind), (1, self.wind)):
            first[-1] = w
            start[encode(first[None], self.state_cards)] = p
        return start

    def confounded_probs(self, expert, strength):
        """During wind: w.p. strength agent 0 moves up (agent 1 as the expert)."""
        windy = self.B[:, -1] == 1
        up = np.zeros_like(expert)
        for a, moves in enumerate(self.moves_of):
            if moves[0] == 3:
                same = [b for b, m in enumerate(self.moves_of) if (m[1:] == moves[1:]).all()]
                up[:, a] = expert[:, same].sum(1)
        return np.where(windy[:, None], strength * up + (1 - strength) * expert, expert)

    def state_label(self, values):
        cells = (f"{int(values[2 * i + 1])},{int(values[2 * i])}" for i in range(self.n_agents))  # row,col
        return " ".join(cells)
