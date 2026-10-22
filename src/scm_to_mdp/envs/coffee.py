"""The coffee robot (Boutilier and colleagues), a classic factored MDP with a hand-written 2-slice graph.

Source of the domain: R. Dearden and C. Boutilier, "Integrating Planning and Execution in Stochastic
Domains", UAI 1994 (https://arxiv.org/abs/1302.6799), Sec. 2, Fig. 1-2; also used by Boutilier,
Dearden and Goldszmidt (AIJ 2000) and the SPUDD paper (Hoey et al., UAI 1999).

A robot must buy coffee at the shop and deliver it to the user in the office; it gets wet if it moves
in the rain without an umbrella. State (all 0/1): office (the robot is in the office, else at the
shop), wet, umbrella, rain, hrc (the robot has coffee), huc (the user has coffee). Actions (data
columns, exactly one on): move, buy_coffee, deliver_coffee, get_umbrella.

Numbers. The paper's text gives two: get_umbrella in the office gives the umbrella w.p. 0.9; a move
in the rain from the office ends "not office, wet" w.p. 0.81, which we read as two independent
effects of 0.9 each (move succeeds 0.9; wet 0.9 when raining without umbrella). Its figures (the
full tables and the reward) are not in the text; every other number below is OURS:
    buy_coffee at the shop: hrc w.p. 0.9;  deliver_coffee: hrc is lost, and huc w.p. 0.9 when in the
    office with coffee;  rain, wet, umbrella and huc otherwise keep their value;
    reward on arrival: 0.9 * huc + 0.1 * (not wet);  start: in the office, dry, no umbrella, no
    coffee, rain w.p. 0.5.

Sizes: "coffee" (6 variables, 64 states, 4 actions) and "coffee+snack", which adds the snack of the
paper's experiments (hrs, hus; actions buy_snack, deliver_snack with the same numbers as coffee;
reward 0.6 * huc + 0.3 * hus + 0.1 * (not wet); 8 variables, 256 states, 6 actions).
Every variable moves independently given (state, action): the model is a product of factors.
"""

import numpy as np

from scm_to_mdp.core import Reward, encode
from scm_to_mdp.envs.base import Env

MOVE_SUCCESS, WET_IN_RAIN, UMBRELLA_SUCCESS = 0.9, 0.9, 0.9  # from the paper (see above)
BUY_SUCCESS, DELIVER_SUCCESS = 0.9, 0.9  # ours
SIZES = ("coffee", "coffee+snack")


class Coffee(Env):
    def __init__(self, size: str = "coffee", seed: int = 0):
        if size not in SIZES:
            raise ValueError(f"size must be one of {SIZES}")
        self.items = ["coffee"] + (["snack"] if size == "coffee+snack" else [])
        self.rng = np.random.default_rng(seed)
        self.state_vars = ["office", "wet", "umbrella", "rain"]
        for item in self.items:
            self.state_vars += [f"hr{item[0]}", f"hu{item[0]}"]  # robot has / user has
        self.action_vars = ["move", "get_umbrella"]
        for item in self.items:
            self.action_vars += [f"buy_{item}", f"deliver_{item}"]
        self.cards = dict.fromkeys(self.state_vars + self.action_vars, 2)
        self.n_actions = len(self.action_vars)  # env action a = the action column a is on
        self.policies = {"random": self.random_policy}
        self.build_model()

    def random_policy(self, states, rng):
        return rng.integers(0, self.n_actions, len(states))

    def x(self, states, name):
        return states[:, self.state_vars.index(name)]

    def next_distributions(self, states, actions):
        E, n = len(states), len(self.state_vars)
        act = {name: actions == j for j, name in enumerate(self.action_vars)}
        office, wet, umbrella, rain = (self.x(states, v) == 1 for v in ["office", "wet", "umbrella", "rain"])
        p = {}  # P(next value = 1) for every variable
        p["office"] = np.where(act["move"], np.where(office, 1 - MOVE_SUCCESS, MOVE_SUCCESS), office)
        p["wet"] = np.where(wet, 1.0, np.where(act["move"] & rain & ~umbrella, WET_IN_RAIN, 0.0))
        p["umbrella"] = np.where(umbrella, 1.0, np.where(act["get_umbrella"] & office, UMBRELLA_SUCCESS, 0.0))
        p["rain"] = rain.astype(float)
        for item in self.items:
            hr, hu = f"hr{item[0]}", f"hu{item[0]}"
            has_r, has_u = self.x(states, hr) == 1, self.x(states, hu) == 1
            buy, deliver = act[f"buy_{item}"], act[f"deliver_{item}"]
            p[hr] = np.where(deliver, 0.0, np.where(has_r, 1.0, np.where(buy & ~office, BUY_SUCCESS, 0.0)))
            p[hu] = np.where(has_u, 1.0, np.where(deliver & office & has_r, DELIVER_SUCCESS, 0.0))
        q = np.stack([p[v] for v in self.state_vars], axis=1).astype(float)  # [E, n]
        return np.stack([1 - q, q], axis=-1).reshape(E, n, 2)

    def action_columns(self, actions):
        return np.eye(self.n_actions, dtype=np.int64)[actions]

    def reward(self, B, actions):
        weights = {"coffee": 0.9} if len(self.items) == 1 else {"coffee": 0.6, "snack": 0.3}
        value = 0.1 * (B[:, self.state_vars.index("wet")] == 0)
        for item, w in weights.items():
            value = value + w * B[:, self.state_vars.index(f"hu{item[0]}")]
        S = len(B)
        return Reward(np.zeros((S, len(actions))), value.astype(float), np.zeros(S, dtype=bool))

    def true_parents(self):
        truth = {
            "office": {"office", "move"},
            "wet": {"wet", "move", "rain", "umbrella"},
            "umbrella": {"umbrella", "get_umbrella", "office"},
            "rain": {"rain"},
        }
        for item in self.items:
            hr, hu = f"hr{item[0]}", f"hu{item[0]}"
            truth[hr] = {hr, f"buy_{item}", f"deliver_{item}", "office"}
            truth[hu] = {hu, f"deliver_{item}", "office", hr}
        inputs = self.state_vars + self.action_vars
        return {v: tuple(x for x in inputs if x in truth[v]) for v in self.state_vars}

    def start_distribution(self):
        start = np.zeros(len(self.B))
        for rain in (0, 1):
            values = dict.fromkeys(self.state_vars, 0) | {"office": 1, "rain": rain}
            start[self.index(values)] = 0.5
        return start

    def index(self, values: dict) -> int:
        return int(encode(np.array([values[v] for v in self.state_vars]), self.state_cards))

    def state_label(self, values):
        x = dict(zip(self.state_vars, values, strict=True))
        flags = [("O" if x["office"] else "S"), "W" if x["wet"] else "", "U" if x["umbrella"] else "",
                 "R" if x["rain"] else ""]  # fmt: skip
        flags += [k.upper() for k in self.state_vars[4:] if x[k]]
        return "".join(f for f in flags if f)
