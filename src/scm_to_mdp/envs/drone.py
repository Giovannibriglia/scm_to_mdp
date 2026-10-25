"""Drone delivery in a 3D grid (Badings et al., "Scenario-based verification of uncertain parametric
MDPs", STTT 2022; the version of Schnitzer, Abate, Parker, AAAI 2026, App. E.5, and their PRISM model
models/drone_multislip/drone.prism, Zenodo 10.5281/zenodo.17580296, CC BY 4.0).

One or two drones in an L x L x L grid; drone i's state is x<i>, y<i>, z<i> in 0..L-1, every drone
starts at (0, 0, 0). Action (one categorical data column d<i> per drone, no "do nothing"):
0 front (+x), 1 back (-x), 2 right (+y), 3 left (-y), 4 up (+z), 5 down (-z). Wind makes the
drones drift (p = 0.2, their value).

Their model moves y together with the intended axis in one random draw (e.g. "front": x+1 and y-1
together w.p. p). Our state variables must be independent given (state, action), so we keep their
per-axis MARGINALS and draw the axes independently (a deviation from their joint law):
    front / back   x +-1 w.p. 1 - p, unchanged w.p. p;  y: -1, +1, -2 w.p. p each, unchanged 1 - 3p
    up / down      z +-1 w.p. 1 - p, unchanged w.p. p;  y drifts as for front / back
    right / left   y +-1 w.p. 1 - 2p, +-2 w.p. p, -+1 w.p. p
every move clipped to the grid. True parents: x<i>' <- x<i>, d<i>; y<i>' <- y<i>, d<i>; z<i>' <- z<i>,
d<i> (only some values of d<i> matter to each axis: front / back to x, right / left to y, up / down to
z). The drones never interact.

Regions (their layout, scaled to L): a wall at x = L // 2 with openings, crashing into it ends the
episode with nothing; the cells z = 0, y = 0 and 2 <= y <= L - 2 of the wall are solid. The target is
x >= L - 1 - t, y <= t, z >= L - 1 - t (t = 1 for L >= 5, else 0); the episode ends with reward 1 when
every drone is in it (the probability of a safe delivery, discounted).

Gust (optional, ours; gust=p): a hidden variable drawn afresh at every step. While it is on, drone
0's y is pushed down by one more cell w.p. gust_effect. The "confounded" logger (Env.behaviour) makes
drone 0 climb during gusts w.p. strength.
"""

import itertools

import numpy as np

from scm_to_mdp.core import Reward, encode
from scm_to_mdp.envs.base import Env

MOVES = ["front", "back", "right", "left", "up", "down"]
P_WIND = 0.2


class Drone(Env):
    def __init__(
        self,
        size: int,
        n_drones: int = 1,
        seed: int = 0,
        wind: float = P_WIND,
        gust: float | None = None,
        gust_effect: float = 0.5,
        max_model_gb: float | None = None,
    ):
        if n_drones not in (1, 2):
            raise ValueError("n_drones must be 1 or 2")
        self.L, self.n_drones, self.p = size, n_drones, wind
        self.gust, self.gust_effect = gust, gust_effect
        if max_model_gb is not None:
            self.max_model_gb = max_model_gb
        self.rng = np.random.default_rng(seed)
        self.coords = [[f"x{i}", f"y{i}", f"z{i}"] for i in range(n_drones)]
        self.state_vars = [v for c in self.coords for v in c] + (["gust"] if gust is not None else [])
        self.action_vars = [f"d{i}" for i in range(n_drones)]
        self.cards = dict.fromkeys([v for c in self.coords for v in c], size) | dict.fromkeys(
            self.action_vars, len(MOVES)
        )
        if gust is not None:
            self.cards["gust"] = 2
            self.hidden_vars, self.confounder = ("gust",), "gust"
        self.n_actions = 6**n_drones
        self.moves_of = np.array(list(itertools.product(range(6), repeat=n_drones)))
        self.t = 1 if size >= 5 else 0
        self.policies = {"random": self.random_policy}
        self.build_model()

    # ---- behaviour policies ------------------------------------------------------------------
    def random_policy(self, states, rng):
        return rng.integers(0, self.n_actions, len(states))

    # ---- regions -----------------------------------------------------------------------------
    def wall(self, x, y, z) -> np.ndarray:
        L = self.L
        return (x == L // 2) & ((z == 0) | (y == 0) | ((y >= 2) & (y <= L - 2)))

    def in_target(self, x, y, z) -> np.ndarray:
        L, t = self.L, self.t
        return (x >= L - 1 - t) & (y <= t) & (z >= L - 1 - t)

    # ---- the model ---------------------------------------------------------------------------
    def _shifted(self, values, shifts_probs):
        out = np.zeros((len(values), self.L))
        for shift, prob in shifts_probs:
            np.add.at(out, (np.arange(len(values)), np.clip(values + shift, 0, self.L - 1)), prob)
        return out

    def next_distributions(self, states, actions):
        # What follows an end state never counts (Reward.backup; sampling stops there), so the
        # dynamics need no special case for the wall or the target.
        E, p = len(states), self.p
        C = max(self.cards[v] for v in self.state_vars)
        dist = np.zeros((E, len(self.state_vars), C))
        moves = self.moves_of[actions]
        one = np.ones(E)
        drift = [(-1, p * one), (1, p * one), (-2, p * one), (0, (1 - 3 * p) * one)]
        for i in range(self.n_drones):
            x, y, z = states[:, 3 * i], states[:, 3 * i + 1], states[:, 3 * i + 2]
            m = moves[:, i]
            for axis, values, plus, minus in ((0, x, 0, 1), (2, z, 4, 5)):
                step = np.where(m == plus, 1, np.where(m == minus, -1, 0))
                moved = self._shifted(values, [(step, (1 - p) * one), (0, p * one)])
                still = self._shifted(values, [(0, one)])
                dist[:, 3 * i + axis, : self.L] = np.where((step != 0)[:, None], moved, still)
            sign = np.where(m == 2, 1, -1)
            lateral = self._shifted(y, [(sign, (1 - 2 * p) * one), (2 * sign, p * one), (-sign, p * one)])
            y_dist = np.where(((m == 2) | (m == 3))[:, None], lateral, self._shifted(y, drift))
            if i == 0 and self.gust is not None:
                pushed = np.zeros_like(y_dist)
                pushed[:, :-1] = 0.0
                pushed[:, 0] = y_dist[:, 0] + y_dist[:, 1]
                pushed[:, 1:-1] = y_dist[:, 2:]
                pushed[:, -1] = 0.0
                gusty = (states[:, -1] == 1)[:, None]
                y_dist = np.where(gusty, (1 - self.gust_effect) * y_dist + self.gust_effect * pushed, y_dist)
            dist[:, 3 * i + 1, : self.L] = y_dist
        if self.gust is not None:
            dist[:, -1, 0], dist[:, -1, 1] = 1 - self.gust, self.gust
        return dist

    def action_columns(self, actions):
        return self.moves_of[np.asarray(actions)].astype(np.int64)  # [E, n_drones]: d<i> = drone i's move

    def reward(self, B, actions):
        crashed = np.zeros(len(B), dtype=bool)
        arrived = np.ones(len(B), dtype=bool)
        for i in range(self.n_drones):
            x, y, z = B[:, 3 * i], B[:, 3 * i + 1], B[:, 3 * i + 2]
            crashed |= self.wall(x, y, z)
            arrived &= self.in_target(x, y, z)
        arrived &= ~crashed
        return Reward(np.zeros((len(B), len(actions))), arrived.astype(float), arrived | crashed)

    # ---- other tasks on the same dynamics (reward transfer, S9) ----------------------------------
    def alternative_rewards(self) -> dict:
        """Other tasks in the same grid (reward transfer; the gust is never read). The wall still ends
        the episode with nothing; the episode ends with 1 when every drone is in the task's target:
        target_far_y   x >= L-1-t, y >= L-1-t, z >= L-1-t (the other side of the wall's openings)
        target_low     x >= L-1-t, y <= t, z <= t (near the ground)
        step_cost      the original target, minus 0.01 per step
        The old target is then an ordinary region: the data never show what follows it (the episodes
        ended there), a factored model still knows it from the other cells, a flat one does not."""
        L, t = self.L, self.t
        original = self.reward

        def target(inside):
            def reward(B, actions):
                crashed = np.zeros(len(B), dtype=bool)
                arrived = np.ones(len(B), dtype=bool)
                for i in range(self.n_drones):
                    x, y, z = B[:, 3 * i], B[:, 3 * i + 1], B[:, 3 * i + 2]
                    crashed |= self.wall(x, y, z)
                    arrived &= inside(x, y, z)
                arrived &= ~crashed
                return Reward(np.zeros((len(B), len(actions))), arrived.astype(float), arrived | crashed)

            return reward

        def step_cost(B, actions):
            r = original(B, actions)
            return Reward(r.R - 0.01, r.r_next, r.terminal)

        far = L - 1 - t
        return {
            "target_far_y": target(lambda x, y, z: (x >= far) & (y >= far) & (z >= far)),
            "target_low": target(lambda x, y, z: (x >= far) & (y <= t) & (z <= t)),
            "step_cost": step_cost,
        }

    def true_parents(self):
        truth = {}
        for i in range(self.n_drones):
            truth[f"x{i}"] = {f"x{i}", f"d{i}"}
            truth[f"y{i}"] = {f"y{i}", f"d{i}"}
            truth[f"z{i}"] = {f"z{i}", f"d{i}"}
        if self.gust is not None:
            truth["y0"] |= {"gust"}
            truth["gust"] = set()
        inputs = self.state_vars + self.action_vars
        return {v: tuple(c for c in inputs if c in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        start = np.zeros(len(self.B))
        first = np.zeros(len(self.state_vars), dtype=np.int64)
        if self.gust is None:
            start[encode(first[None], self.state_cards)] = 1.0
            return start
        for g, prob in ((0, 1 - self.gust), (1, self.gust)):
            first[-1] = g
            start[encode(first[None], self.state_cards)] = prob
        return start

    def confounded_probs(self, expert, strength):
        """During a gust: w.p. strength drone 0 climbs (the other drone as the expert)."""
        gusty = self.B[:, -1] == 1
        climb = np.zeros_like(expert)
        for a, moves in enumerate(self.moves_of):
            if moves[0] == 4:
                same = [b for b, m in enumerate(self.moves_of) if (m[1:] == moves[1:]).all()]
                climb[:, a] = expert[:, same].sum(1)
        return np.where(gusty[:, None], strength * climb + (1 - strength) * expert, expert)

    def state_label(self, values):
        return " ".join("".join(str(int(v)) for v in values[3 * i : 3 * i + 3]) for i in range(self.n_drones))
