"""Taxi (Dietterich, "Hierarchical Reinforcement Learning with the MAXQ Value Function
Decomposition", JAIR 13, 2000; https://arxiv.org/abs/cs/9905014, Sec. 3.1 and 7.1).

A taxi on a 5 x 5 grid with walls must pick up a passenger at one of four landmarks R, G, Y, B and
put the passenger down at a destination landmark. Actions (data columns, exactly one on): north,
south, east, west, pickup, putdown. Reward (Sec. 3.1): -1 for every action, +20 more for a
successful delivery, -10 for an illegal pickup or putdown (we use -10 in total for it). The episode
ends at the delivery. Start: the taxi on a uniform square; the passenger at a uniform landmark;
the destination a uniform landmark (it may be the passenger's: the taxi must still pick up and put
down, Sec. 3.1).

Navigation (size): "deterministic" (Sec. 3.1), or "noisy" (the Fickle Taxi of Sec. 7.1, without its
change of destination): the move goes in the intended direction w.p. 0.8, and to its right or left
w.p. 0.1 each. A move into a wall leaves the taxi in place.

State (three variables, 600 states): taxi (square 0..24, row-major from the top-left corner),
passenger (0..3 waiting at R, G, Y, B; 4 in the taxi; 5 delivered = terminal), destination (0..3).
The taxi square is ONE variable so that the noisy moves (which change the row OR the column) keep
the model a product of independent factors. Difference from the paper: a "delivered" value, so
that the end of the episode is a state (Dietterich's 500 states have no terminal state).
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

from scm_to_mdp.core import Reward
from scm_to_mdp.envs.base import Env

LANDMARKS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B as (row, column)
# walls between horizontally adjacent squares: (row, left column) blocks moves between left and left + 1
WALLS = {(0, 1), (1, 1), (3, 0), (3, 2), (4, 0), (4, 2)}
MOVES = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
RIGHT_OF = {"north": "east", "east": "south", "south": "west", "west": "north"}
LEFT_OF = {v: k for k, v in RIGHT_OF.items()}
ACTIONS = ["north", "south", "east", "west", "pickup", "putdown"]
IN_TAXI, DELIVERED = 4, 5
SIZES = {"deterministic": 0.0, "noisy": 0.2}


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
    def __init__(self, size: str = "deterministic", seed: int = 0):
        if size not in SIZES:
            raise ValueError(f"size must be one of {list(SIZES)}")
        self.noise = SIZES[size]
        self.rng = np.random.default_rng(seed)
        self.state_vars = ["taxi", "passenger", "destination"]
        self.action_vars = list(ACTIONS)
        self.cards = {"taxi": 25, "passenger": 6, "destination": 4, **dict.fromkeys(ACTIONS, 2)}
        self.n_actions = 6
        self.policies = {"random": self.random_policy}
        self.landmark_square = np.array([r * 5 + c for r, c in LANDMARKS])
        self.build_model()

    def random_policy(self, states, rng):
        return rng.integers(0, 6, len(states))

    def next_distributions(self, states, actions):
        E = len(states)
        taxi, passenger, destination = states[:, 0], states[:, 1], states[:, 2]
        dist = np.zeros((E, 3, 25))
        # the taxi's square
        for e in range(E):
            name = ACTIONS[actions[e]]
            if name in MOVES:
                outcomes = [
                    (name, 1 - self.noise),
                    (RIGHT_OF[name], self.noise / 2),
                    (LEFT_OF[name], self.noise / 2),
                ]
                for direction, p in outcomes:
                    dist[e, 0, step_square(int(taxi[e]), direction)] += p
            else:
                dist[e, 0, taxi[e]] = 1.0
        # the passenger
        waiting = passenger < IN_TAXI
        at_passenger = waiting & (taxi == self.landmark_square[np.minimum(passenger, 3)])
        picks = (actions == ACTIONS.index("pickup")) & at_passenger
        drops = (actions == ACTIONS.index("putdown")) & (passenger == IN_TAXI)
        drops &= taxi == self.landmark_square[destination]
        new = np.where(picks, IN_TAXI, np.where(drops, DELIVERED, passenger))
        dist[np.arange(E), 1, new] = 1.0
        dist[np.arange(E), 2, destination] = 1.0  # the destination never changes
        return dist

    def action_columns(self, actions):
        return np.eye(6, dtype=np.int64)[actions]

    def reward(self, B, actions):
        """R[s, k] for action vectors k (one-hot): -1, or -10 if illegal, or +19 for a delivery."""
        taxi, passenger, destination = B[:, 0], B[:, 1], B[:, 2]
        at_passenger = (passenger < IN_TAXI) & (taxi == self.landmark_square[np.minimum(passenger, 3)])
        can_drop = (passenger == IN_TAXI) & (taxi == self.landmark_square[destination])
        R = np.full((len(B), len(actions)), -1.0)
        for k, vector in enumerate(actions):
            name = ACTIONS[int(np.argmax(vector))]
            if name == "pickup":
                R[:, k] = np.where(at_passenger, -1.0, -10.0)
            elif name == "putdown":
                R[:, k] = np.where(can_drop, 19.0, -10.0)
        return Reward(R, np.zeros(len(B)), passenger == DELIVERED)

    def true_parents(self):
        truth = {
            "taxi": {"taxi", "north", "south", "east", "west"},
            "passenger": {"passenger", "taxi", "destination", "pickup", "putdown"},
            "destination": {"destination"},
        }
        inputs = self.state_vars + self.action_vars
        return {v: tuple(x for x in inputs if x in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        start = (self.B[:, 1] < IN_TAXI).astype(float)  # passenger waiting, any square and destination
        return start / start.sum()

    def state_label(self, values):
        taxi, passenger, destination = (int(v) for v in values)
        where = "RGYB"[passenger] if passenger < IN_TAXI else ("T" if passenger == IN_TAXI else "done")
        return f"{divmod(taxi, 5)} p:{where} d:{'RGYB'[destination]}"
