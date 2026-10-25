"""Aircraft collision avoidance on a grid (after Kochenderfer, "Decision Making Under Uncertainty",
MIT Press 2015; the version of Schnitzer, Abate, Parker, AAAI 2026, App. E.1, and their PRISM model
models/aircraft_collision_multislip/aircraft_10x20_resolution_3.prism, Zenodo 10.5281/zenodo.17580296,
CC BY 4.0).

Two aircraft fly toward each other: the own aircraft from column 0 to column W, the intruder from
column W to column 0, one column per step each, so the intruder's column is always W - x and is not a
state variable. State: x (own column, 0..W), y (own altitude, 0..H), iy (intruder altitude, 0..H).
Action (one categorical data column move): 0 stay, 1 up, 2 down (= the env action).

Own altitude (their probabilities): up / down move one cell w.p. r = 0.8, two cells w.p. 0.1, three
w.p. 0.05, and stay w.p. 0.05; stay drifts by +-1, +-2, +-3 cells w.p. 0.05 each and keeps the
altitude w.p. 0.7 (all clipped to 0..H). Intruder altitude (their adversary's up / down move): +-1
w.p. 0.2 each, +-2 w.p. 0.1 each, +-3 w.p. 0.05 each, unchanged w.p. 0.3. Differences from their
model, so that it is an MDP with a fixed known stochastic intruder:
  - the intruder ignores the own aircraft's action (theirs moves on the same action label);
  - proximity (optional, on by default; their Sec. 2 example): while the intruder is near
    (|x - (W - x)| <= NEAR_X and |y - iy| <= NEAR_Y), an intended move succeeds w.p. r_near = 0.5
    instead of 0.8 (the rest stays), so the own altitude depends on x and iy too.
Collision: |x - (W - x)| <= 1 and |y - iy| <= 1: the episode ends with nothing. Reaching column W
ends it with reward 1 (the probability of crossing safely, discounted).

Turbulence (optional, ours; turbulence=p): a hidden variable drawn afresh at every step. While it is
on, an intended move of the own aircraft succeeds w.p. r * (1 - turbulence_effect) (the rest stays).
The "confounded" logger (Env.behaviour) stays during turbulence w.p. strength, so the data rarely
show climbs and descents in turbulence and make them look more reliable than they are.
"""

import numpy as np

from scm_to_mdp.core import Reward, encode
from scm_to_mdp.envs.base import Env

R_MOVE, D2, D3 = 0.8, 0.1, 0.05  # their r, d2, d3
R_NEAR = 0.5
DRIFT = 0.05  # their drift1 = drift2 = drift3; no drift w.p. 0.7
P_INTRUDER, D2_INTRUDER, D3_INTRUDER = 0.2, 0.1, 0.05
NEAR_X, NEAR_Y = 3, 2


class Aircraft(Env):
    def __init__(
        self,
        width: int,
        height: int,
        seed: int = 0,
        proximity: bool = True,
        turbulence: float | None = None,
        turbulence_effect: float = 0.5,
        max_model_gb: float | None = None,
    ):
        self.W, self.H, self.proximity = width, height, proximity
        self.turbulence, self.turbulence_effect = turbulence, turbulence_effect
        if max_model_gb is not None:
            self.max_model_gb = max_model_gb
        self.rng = np.random.default_rng(seed)
        self.state_vars = ["x", "y", "iy"] + (["turbulence"] if turbulence is not None else [])
        self.action_vars = ["move"]
        self.cards = {"x": width + 1, "y": height + 1, "iy": height + 1, "move": 3}
        if turbulence is not None:
            self.cards["turbulence"] = 2
            self.hidden_vars, self.confounder = ("turbulence",), "turbulence"
        self.n_actions = 3
        self.policies = {"random": self.random_policy}
        self.build_model()

    # ---- behaviour policies ------------------------------------------------------------------
    def random_policy(self, states, rng):
        return rng.integers(0, 3, len(states))

    # ---- the model ---------------------------------------------------------------------------
    def _shifted(self, values, shifts_probs, card):
        """[E, card] the distribution of clip(values + shift) for a list of (shift, prob [E])."""
        out = np.zeros((len(values), card))
        for shift, prob in shifts_probs:
            np.add.at(out, (np.arange(len(values)), np.clip(values + shift, 0, card - 1)), prob)
        return out

    def collision(self, x, y, iy) -> np.ndarray:
        return (np.abs(2 * x - self.W) <= 1) & (np.abs(y - iy) <= 1)

    def next_distributions(self, states, actions):
        E = len(states)
        x, y, iy = states[:, 0], states[:, 1], states[:, 2]
        C = max(self.cards[v] for v in self.state_vars)
        dist = np.zeros((E, len(self.state_vars), C))
        # The episode ends on arrival in a collision or in column W, and what follows an end state
        # never counts (Reward.backup; sampling stops there): the dynamics below need no special case
        # for it, which keeps the true parents those of the flight itself.
        # own column: one step forward
        dist[np.arange(E), 0, np.minimum(x + 1, self.W)] = 1.0
        # own altitude
        r = np.full(E, R_MOVE)
        if self.proximity:
            near = (np.abs(2 * x - self.W) <= NEAR_X) & (np.abs(y - iy) <= NEAR_Y)
            r = np.where(near, R_NEAR, r)
        if self.turbulence is not None:
            r = np.where(states[:, -1] == 1, r * (1 - self.turbulence_effect), r)
        sign = np.where(actions == 1, 1, -1)
        moving = [(sign * 1, r), (sign * 2, np.full(E, D2)), (sign * 3, np.full(E, D3)), (0, 1 - r - D2 - D3)]
        drifting = [(s, np.full(E, DRIFT)) for s in (-3, -2, -1, 1, 2, 3)] + [(0, np.full(E, 1 - 6 * DRIFT))]
        move_dist = self._shifted(y, moving, self.H + 1)
        drift_dist = self._shifted(y, drifting, self.H + 1)
        dist[:, 1, : self.H + 1] = np.where((actions == 0)[:, None], drift_dist, move_dist)
        # intruder altitude
        sizes = ((1, P_INTRUDER), (2, D2_INTRUDER), (3, D3_INTRUDER))
        jumps = [(sign * s, np.full(E, prob)) for s, prob in sizes for sign in (1, -1)]
        jumps.append((0, np.full(E, 1 - 2 * (P_INTRUDER + D2_INTRUDER + D3_INTRUDER))))
        dist[:, 2, : self.H + 1] = self._shifted(iy, jumps, self.H + 1)
        if self.turbulence is not None:
            dist[:, -1, 0], dist[:, -1, 1] = 1 - self.turbulence, self.turbulence
        return dist

    def action_columns(self, actions):
        return np.asarray(actions, dtype=np.int64)[:, None]  # move = the env action

    def reward(self, B, actions):
        x, y, iy = B[:, 0], B[:, 1], B[:, 2]
        crashed = self.collision(x, y, iy)
        arrived = (x == self.W) & ~crashed
        return Reward(np.zeros((len(B), len(actions))), arrived.astype(float), arrived | crashed)

    # ---- other tasks on the same dynamics (reward transfer, S9) ----------------------------------
    def alternative_rewards(self) -> dict:
        """Other tasks in the same airspace (reward transfer; the turbulence is never read). The end
        states (collision, reaching column W) are unchanged:
        low_band    reaching column W pays 1 only at altitude <= H // 3 (else 0)
        high_band   reaching column W pays 1 only at altitude >= H - H // 3 (else 0)
        fuel_cost   the original reward, minus 0.02 for every climb or descent"""
        original = self.reward

        def band(keep):
            def reward(B, actions):
                r = original(B, actions)
                return Reward(r.R, r.r_next * keep(B[:, 1]), r.terminal)

            return reward

        def fuel_cost(B, actions):
            r = original(B, actions)
            return Reward(r.R - 0.02 * (actions[:, 0] > 0)[None, :], r.r_next, r.terminal)

        third = self.H // 3
        return {
            "low_band": band(lambda y: y <= third),
            "high_band": band(lambda y: y >= self.H - third),
            "fuel_cost": fuel_cost,
        }

    def true_parents(self):
        own = {"y", "move"} | ({"x", "iy"} if self.proximity else set())
        truth = {"x": {"x"}, "y": own, "iy": {"iy"}}
        if self.turbulence is not None:
            truth["y"] |= {"turbulence"}
            truth["turbulence"] = set()
        inputs = self.state_vars + self.action_vars
        return {v: tuple(c for c in inputs if c in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        start = np.zeros(len(self.B))
        first = np.array([0, self.H // 2, self.H // 2] + ([0] if self.turbulence is not None else []))
        if self.turbulence is None:
            start[encode(first[None], self.state_cards)] = 1.0
            return start
        for t, p in ((0, 1 - self.turbulence), (1, self.turbulence)):
            first[-1] = t
            start[encode(first[None], self.state_cards)] = p
        return start

    def confounded_probs(self, expert, strength):
        """During turbulence: w.p. strength stay (env action 0)."""
        rough = self.B[:, -1] == 1
        stay = np.zeros_like(expert)
        stay[:, 0] = 1.0
        return np.where(rough[:, None], strength * stay + (1 - strength) * expert, expert)

    def state_label(self, values):
        return f"x{int(values[0])} y{int(values[1])} i{int(values[2])}"
