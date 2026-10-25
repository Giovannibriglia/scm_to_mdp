"""The coffee robot (Boutilier and colleagues), a classic factored MDP with a hand-written 2-slice graph.

Source of the domain: R. Dearden and C. Boutilier, "Integrating Planning and Execution in Stochastic
Domains", UAI 1994 (https://arxiv.org/abs/1302.6799), Sec. 2, Fig. 1-2; also used by Boutilier,
Dearden and Goldszmidt (AIJ 2000) and the SPUDD paper (Hoey et al., UAI 1999).

A robot must buy coffee at the shop and deliver it to the user in the office; it gets wet if it moves
in the rain without an umbrella. State (all 0/1): office (the robot is in the office, else at the
shop), wet, umbrella, rain, hrc (the robot has coffee), huc (the user has coffee).
Action (one categorical data column act): 0 move, 1 get_umbrella, 2 buy_coffee, 3 deliver_coffee
(then buy_snack, deliver_snack); no "do nothing" in the domain.

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
        self.action_names = ["move", "get_umbrella"]  # the values of act, in order
        for item in self.items:
            self.action_names += [f"buy_{item}", f"deliver_{item}"]
        self.action_vars = ["act"]
        self.cards = dict.fromkeys(self.state_vars, 2) | {"act": len(self.action_names)}
        self.n_actions = len(self.action_names)  # env action a = value a of act
        self.policies = {"random": self.random_policy}
        self.build_model()

    def random_policy(self, states, rng):
        return rng.integers(0, self.n_actions, len(states))

    def x(self, states, name):
        return states[:, self.state_vars.index(name)]

    def next_distributions(self, states, actions):
        E, n = len(states), len(self.state_vars)
        act = {name: actions == j for j, name in enumerate(self.action_names)}
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
        return np.asarray(actions, dtype=np.int64)[:, None]  # act = the env action

    def reward(self, B, actions):
        weights = {"coffee": 0.9} if len(self.items) == 1 else {"coffee": 0.6, "snack": 0.3}
        value = 0.1 * (B[:, self.state_vars.index("wet")] == 0)
        for item, w in weights.items():
            value = value + w * B[:, self.state_vars.index(f"hu{item[0]}")]
        S = len(B)
        return Reward(np.zeros((S, len(actions))), value.astype(float), np.zeros(S, dtype=bool))

    # ---- other tasks on the same dynamics (reward transfer, S9; ours) ---------------------------------
    def alternative_rewards(self) -> dict:
        """Other tasks for the same robot (ours; reward on arrival, as the original):
        coffee_only  1 when the user has coffee, whatever the robot's state (getting wet is free)
        dry_delivery 1 when the user has coffee AND the robot is dry (in the rain the umbrella becomes
                     necessary; the original reward values dryness at 0.1 only)
        snack_only   (coffee+snack) 1 when the user has the snack"""

        def on_arrival(value):
            def reward(B, actions):
                S = len(B)
                return Reward(np.zeros((S, len(actions))), value(B).astype(float), np.zeros(S, dtype=bool))

            return reward

        column = self.state_vars.index
        tasks = {
            "coffee_only": on_arrival(lambda B: B[:, column("huc")] == 1),
            "dry_delivery": on_arrival(lambda B: (B[:, column("huc")] == 1) & (B[:, column("wet")] == 0)),
        }
        if "snack" in self.items:
            tasks["snack_only"] = on_arrival(lambda B: B[:, column("hus")] == 1)
        return tasks

    def true_parents(self):
        truth = {
            "office": {"office", "act"},
            "wet": {"wet", "act", "rain", "umbrella"},
            "umbrella": {"umbrella", "act", "office"},
            "rain": {"rain"},
        }
        for item in self.items:
            hr, hu = f"hr{item[0]}", f"hu{item[0]}"
            truth[hr] = {hr, "act", "office"}
            truth[hu] = {hu, "act", "office", hr}
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
