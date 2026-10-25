"""Taxi (Dietterich, "Hierarchical Reinforcement Learning with the MAXQ Value Function
Decomposition", JAIR 13, 2000; https://arxiv.org/abs/cs/9905014, Sec. 3.1 and 7.1).

A taxi on a 5 x 5 grid with walls must pick up a passenger at one of four landmarks R, G, Y, B and
put the passenger down at a destination landmark. Reward (Sec. 3.1): -1 for every action, +20 more
for a successful delivery, -10 for an illegal pickup or putdown (we use -10 in total for it).

Variables, one per dimension:
    state     x (column 0..4), y (row 0..4, 0 = top), passenger (0..3 waiting at R, G, Y, B; 4 in the
              taxi), destination (0..3): 500 states, Gymnasium's Taxi-v3 state space
    action    act: 0 north, 1 south, 2 east, 3 west, 4 pickup, 5 putdown (no "do nothing")
Episode end (no "delivered" value; Gymnasium's Taxi-v3): a putdown at the destination with the
passenger in the taxi sets passenger = destination, and the states with passenger = destination are
terminal. Start (Taxi-v3): a uniform square, the passenger at a uniform landmark, the destination a
uniform OTHER landmark (Dietterich allows them equal; with this end rule such a start would be over).

Navigation (size): "deterministic" (Sec. 3.1), or "noisy" (the Fickle Taxi of Sec. 7.1, without its
change of destination): the move goes in the intended direction w.p. 0.8, and to its right or left
w.p. 0.1 each. A move into a wall leaves the taxi in place. A noisy move changes the row OR the column,
in one draw, so x' and y' are dependent given (state, action); axes chooses the dynamics:
    "independent"  ours, the default: each axis keeps its marginal, the axes are drawn independently
                   (a free noisy move: diagonal w.p. 0.16, 0.08 each way, and no move w.p.
                   0.16)
    "joint"        the paper's law: (y', x') has the joint law of the move; the true graph has a
                   same-step edge y' -> x' (a factorisation of the shared draw, not a causal claim)
Walls: those of the usual map (Figure 1; the same as Gymnasium's Taxi-v3):
    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
"""

import numpy as np

from scm_to_mdp.core import NEXT_MARK, Reward
from scm_to_mdp.envs.base import Env

LANDMARKS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B as (row, column)
# walls between horizontally adjacent squares: (row, left column) blocks moves between left and left + 1
WALLS = {(0, 1), (1, 1), (3, 0), (3, 2), (4, 0), (4, 2)}
MOVES = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
RIGHT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}
LEFT_OF = {v: k for k, v in RIGHT_OF.items()}
ACTIONS = ["north", "south", "east", "west", "pickup", "putdown"]
IN_TAXI = 4
SIZES = {"deterministic": 0.0, "noisy": 0.2}
AXES = ("joint", "independent")


def step_square(square: int, direction: str) -> int:
    """The square after one move (in place if a wall or the border is in the way)."""
    r, c = divmod(square, 5)
    dr, dc = MOVES[direction]
    r2, c2 = r + dr, c + dc
    if not (0 <= r2 < 5 and 0 <= c2 < 5):
        return square
    if dc != 0 and (r, min(c, c2)) in WALLS:
        return square
    return r2 * 5 + c2


class Taxi(Env):
    def __init__(self, size: str = "deterministic", seed: int = 0, axes: str = "independent"):
        if size not in SIZES:
            raise ValueError(f"size must be one of {list(SIZES)}")
        if axes not in AXES:
            raise ValueError(f"axes must be one of {AXES}")
        self.noise, self.axes = SIZES[size], axes
        self.rng = np.random.default_rng(seed)
        self.state_vars = ["x", "y", "passenger", "destination"]
        self.action_vars = ["act"]
        self.cards = {"x": 5, "y": 5, "passenger": 5, "destination": 4, "act": len(ACTIONS)}
        self.n_actions = len(ACTIONS)
        self.policies = {"random": self.random_policy}
        self.landmark_square = np.array([r * 5 + c for r, c in LANDMARKS])
        self.build_model()

    @property
    def coupled(self) -> bool:
        """Are x' and y' dependent given (state, action) (noisy moves with axes "joint")?"""
        return self.axes == "joint" and self.noise > 0

    def random_policy(self, states, rng):
        return rng.integers(0, self.n_actions, len(states))

    def _square_law(self, states, actions) -> np.ndarray:
        """[E, 25] the law of the taxi's next square (row * 5 + column)."""
        E = len(states)
        square = states[:, 1] * 5 + states[:, 0]
        law = np.zeros((E, 25))
        for e in range(E):
            name = ACTIONS[actions[e]]
            if name in MOVES:
                outcomes = [
                    (name, 1 - self.noise),
                    (RIGHT_OF[name], self.noise / 2),
                    (LEFT_OF[name], self.noise / 2),
                ]
                for direction, p in outcomes:
                    law[e, step_square(int(square[e]), direction)] += p
            else:
                law[e, square[e]] = 1.0
        return law

    def next_distributions(self, states, actions):
        E = len(states)
        x, y, passenger, destination = (states[:, i] for i in range(4))
        dist = np.zeros((E, 4, 5))
        grid = self._square_law(states, actions).reshape(E, 5, 5)  # [E, row, column]
        dist[:, 0] = grid.sum(1)  # x'
        dist[:, 1] = grid.sum(2)  # y'
        square = y * 5 + x
        waiting = passenger < IN_TAXI
        at_passenger = waiting & (square == self.landmark_square[np.minimum(passenger, 3)])
        picks = (actions == ACTIONS.index("pickup")) & at_passenger
        drops = (actions == ACTIONS.index("putdown")) & (passenger == IN_TAXI)
        drops &= square == self.landmark_square[destination]
        new = np.where(picks, IN_TAXI, np.where(drops, destination, passenger))
        dist[np.arange(E), 2, new] = 1.0
        dist[np.arange(E), 3, destination] = 1.0  # the destination never changes
        return dist

    def coupled_next(self, states, actions) -> dict:
        """{(index of y, index of x): [E, rows, columns]} the joint law of the next square, when the
        axes are coupled; else {}."""
        if not self.coupled:
            return {}
        return {(1, 0): self._square_law(states, actions).reshape(len(states), 5, 5)}

    def action_columns(self, actions):
        return np.asarray(actions, dtype=np.int64)[:, None]  # act = the env action

    def reward(self, B, actions):
        """R[s, k] for action vectors k: -1, or -10 if illegal, or +19 for a delivery."""
        square, passenger, destination = B[:, 1] * 5 + B[:, 0], B[:, 2], B[:, 3]
        at_passenger = (passenger < IN_TAXI) & (square == self.landmark_square[np.minimum(passenger, 3)])
        can_drop = (passenger == IN_TAXI) & (square == self.landmark_square[destination])
        R = np.full((len(B), len(actions)), -1.0)
        for k, vector in enumerate(actions):
            name = ACTIONS[int(vector[0])]
            if name == "pickup":
                R[:, k] = np.where(at_passenger, -1.0, -10.0)
            elif name == "putdown":
                R[:, k] = np.where(can_drop, 19.0, -10.0)
        return Reward(R, np.zeros(len(B)), passenger == destination)

    # ---- other tasks on the same dynamics (reward transfer, S9; ours) ---------------------------------
    def alternative_rewards(self) -> dict:
        """Other tasks for the same taxi (ours):
        pickup_only   the episode ends when the passenger is in the taxi (+19 for the pickup, -1 per
                      action, -10 for an illegal pickup or putdown): half of the original task
        costly_moves  the original task, with -3 for every move instead of -1 (pickup and putdown
                      cost -1 as before): fewer detours, the same structure"""
        original = self.reward

        def pickup_only(B, actions):
            r = original(B, actions)
            square, passenger = B[:, 1] * 5 + B[:, 0], B[:, 2]
            at_passenger = (passenger < IN_TAXI) & (square == self.landmark_square[np.minimum(passenger, 3)])
            R = r.R.copy()
            for k, vector in enumerate(actions):
                if ACTIONS[int(vector[0])] == "pickup":
                    R[:, k] = np.where(at_passenger, 19.0, -10.0)
                elif ACTIONS[int(vector[0])] == "putdown":
                    R[:, k] = -10.0
            return Reward(R, r.r_next, r.terminal | (passenger == IN_TAXI))

        def costly_moves(B, actions):
            r = original(B, actions)
            moves = np.array([ACTIONS[int(v[0])] in MOVES for v in actions])
            return Reward(np.where(moves[None, :], r.R - 2.0, r.R), r.r_next, r.terminal)

        return {"pickup_only": pickup_only, "costly_moves": costly_moves}

    def true_parents(self):
        """x' depends on y too (the walls between columns); y' only on y and the move; with coupled
        axes, y' is also a parent of x' (a same-step edge, named with NEXT_MARK)."""
        truth = {
            "x": {"x", "y", "act"} | ({"y" + NEXT_MARK} if self.coupled else set()),
            "y": {"y", "act"},
            "passenger": {"passenger", "x", "y", "destination", "act"},
            "destination": {"destination"},
        }
        inputs = self.state_vars + self.action_vars + [v + NEXT_MARK for v in self.state_vars]
        return {v: tuple(c for c in inputs if c in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        passenger, destination = self.B[:, 2], self.B[:, 3]
        start = ((passenger < IN_TAXI) & (passenger != destination)).astype(float)
        return start / start.sum()

    def state_label(self, values):
        x, y, passenger, destination = (int(v) for v in values)
        where = "RGYB"[passenger] if passenger < IN_TAXI else "T"
        return f"({y},{x}) p:{where} d:{'RGYB'[destination]}"
