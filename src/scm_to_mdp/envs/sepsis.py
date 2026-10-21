"""Sepsis simulator of Oberst & Sontag, "Counterfactual Off-Policy Evaluation with Gumbel-Max
Structural Causal Models", ICML 2019 (https://arxiv.org/abs/1905.05824; code, MIT licence:
https://github.com/clinicalml/gumbel-max-scm, sepsisSimDiabetes/MDP.py). Re-implemented here as
next-value distributions per variable, so the exact model and the simulator come from one function.

State (values; the normal value in brackets):
    diabetic (0/1, constant in an episode)
    hr       heart rate        0 low, 1 normal, 2 high           [1]
    sysbp    blood pressure    0 low, 1 normal, 2 high           [1]
    o2       oxygen            0 low, 1 normal                   [1]
    glucose  0..4: very low, low, normal, high, very high        [2]
    abx_on, vaso_on, vent_on   treatment given at the last step (0/1)
Actions: three binary treatments, data columns antibiotics, ventilation, vasopressors; env action
index 4 * antibiotics + 2 * ventilation + vasopressors (as in the original code).

One step applies, in this order (each probability is a separate random draw):
    antibiotics  on: hr high -> normal and sysbp high -> normal, w.p. 0.5 each.
                 just stopped: hr normal -> high and sysbp normal -> high, w.p. 0.1 each.
    ventilation  on: o2 low -> normal w.p. 0.7.  just stopped: o2 normal -> low w.p. 0.1.
    vasopressors on, non-diabetic: w.p. 0.7 sysbp up by one (low -> normal, normal -> high).
                 on, diabetic: sysbp normal -> high w.p. 0.9; low -> normal w.p. 0.5, low -> high
                 w.p. 0.4; glucose up by one w.p. 0.5.
                 just stopped: sysbp down by one w.p. 0.1 (non-diabetic) or 0.05 (diabetic).
    fluctuation  a vital that no treatment touched this step moves down / up by one w.p. 0.1 each
                 (glucose of a diabetic: 0.3 each). Glucose is untouched by antibiotics and
                 ventilation, and does not fluctuate while vasopressors are on.
    The treatment flags become the action. Every vital moves on its own draws, so the variables
    are independent given (state, action): the model is a product of factors.

Reward (on arrival): -1 and the episode ends (death) when at least 3 vitals are abnormal; +1 and
the episode ends (discharge) when every vital is normal and no treatment is on; 0 otherwise.
Start: hr, sysbp ~ (.25, .5, .25), o2 ~ (.2, .8), glucose ~ (.05, .15, .6, .15, .05), or
(.01, .05, .15, .6, .19) for a diabetic; treatments off; diabetic w.p. p_diabetes; drawn again
while the state is terminal.

Difference from the original code: for a non-diabetic, glucose "up by one" is min(1, g + 1) there,
which sends every level >= 1 to 1 (a bug; the comment says "fluctuate +/- 1"). Here it is
min(4, g + 1). glucose_bug=True restores the original behaviour.

Sizes (our reduced versions; a vital left out is always normal and never changes; death needs
min(3, number of vitals) abnormal vitals):
    small   hr, sysbp; no diabetes                     72 states
    medium  hr, sysbp, o2; no diabetes                144 states
    full    hr, sysbp, o2, glucose; diabetes observed  1440 states (the original)
    custom  Sepsis(vitals=(...), diabetes=True / False)
"""

import numpy as np

from scm_to_mdp.core import Dataset, Reward, encode
from scm_to_mdp.envs.base import Env

VITALS = {"hr": 3, "sysbp": 3, "o2": 2, "glucose": 5}  # number of values
NORMAL = {"hr": 1, "sysbp": 1, "o2": 1, "glucose": 2}
START = {
    "hr": [0.25, 0.5, 0.25],
    "sysbp": [0.25, 0.5, 0.25],
    "o2": [0.2, 0.8],
    "glucose": [0.05, 0.15, 0.6, 0.15, 0.05],
    "glucose_diabetic": [0.01, 0.05, 0.15, 0.6, 0.19],
}
TREATMENT_FLAGS = ["abx_on", "vaso_on", "vent_on"]
ACTIONS = ["antibiotics", "ventilation", "vasopressors"]
SIZES = {
    "small": {"vitals": ("hr", "sysbp"), "diabetes": False},
    "medium": {"vitals": ("hr", "sysbp", "o2"), "diabetes": False},
    "full": {"vitals": ("hr", "sysbp", "o2", "glucose"), "diabetes": True},
}


# ---- one stage of a step as a transition matrix M[e, v, v'] for a batch of E rows --------------
def stay(E: int, card: int) -> np.ndarray:
    return np.repeat(np.eye(card)[None], E, axis=0)


def stage(E: int, card: int, moves: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    """One random draw: from value `frm`, go to `to` with probability p[e] (else stay)."""
    M = stay(E, card)
    for frm, to, p in moves:
        M[:, frm, frm] -= p
        M[:, frm, to] += p
    return M


def fluctuate(
    E: int, card: int, p_down: np.ndarray, p_up: np.ndarray, up_cap: int | None = None
) -> np.ndarray:
    """Down by one w.p. p_down, up by one (to at most up_cap) w.p. p_up, clipped at the ends."""
    up_cap = card - 1 if up_cap is None else up_cap
    moves = []
    for v in range(card):
        moves += [(v, max(0, v - 1), p_down), (v, min(up_cap, v + 1), p_up)]
    return stage(E, card, [m for m in moves if m[0] != m[1]])


class Sepsis(Env):
    def __init__(
        self,
        size: str = "full",
        seed: int = 0,
        vitals: tuple[str, ...] | None = None,
        diabetes: bool | None = None,
        p_diabetes: float = 0.2,
        glucose_bug: bool = False,
    ):
        if size not in SIZES and size != "custom":
            raise ValueError(f"size must be one of {list(SIZES)} or 'custom'")
        spec = SIZES.get(size, SIZES["full"])  # "custom" starts from full
        self.vitals = [v for v in VITALS if v in (spec["vitals"] if vitals is None else vitals)]
        self.diabetes = spec["diabetes"] if diabetes is None else diabetes
        self.p_diabetes = p_diabetes if self.diabetes else 0.0
        self.glucose_bug = glucose_bug
        self.death = min(3, len(self.vitals))  # abnormal vitals for death
        self.rng = np.random.default_rng(seed)
        self.state_vars = (["diabetic"] if self.diabetes else []) + self.vitals + TREATMENT_FLAGS
        self.action_vars = list(ACTIONS)
        self.cards = {"diabetic": 2, **VITALS, **dict.fromkeys(TREATMENT_FLAGS + ACTIONS, 2)}
        self.cards = {v: self.cards[v] for v in self.state_vars + self.action_vars}
        self.n_actions = 8
        self.policies = {"random": self.random_policy}
        self.build_model()

    # ---- behaviour policies ------------------------------------------------------------------
    def random_policy(self, states, rng):
        return rng.integers(0, 8, len(states))

    # ---- the model ---------------------------------------------------------------------------
    def column(self, states: np.ndarray, name: str) -> np.ndarray:
        if name in self.state_vars:
            return states[:, self.state_vars.index(name)]
        return np.zeros(len(states), dtype=np.int64)  # diabetic when diabetes is off

    def next_distributions(self, states, actions):
        E = len(states)
        abx, vent, vaso = actions // 4, (actions // 2) % 2, actions % 2  # this step's treatments
        diab = self.column(states, "diabetic") == 1
        f = lambda x: x.astype(float)  # noqa: E731

        abx_starts_or_on = abx == 1
        abx_stopped = (abx == 0) & (self.column(states, "abx_on") == 1)
        vent_on = vent == 1
        vent_stopped = (vent == 0) & (self.column(states, "vent_on") == 1)
        vaso_on = vaso == 1
        vaso_stopped = (vaso == 0) & (self.column(states, "vaso_on") == 1)
        abx_still = ~abx_starts_or_on & ~abx_stopped  # antibiotics did not touch hr / sysbp

        M = {}
        # heart rate: antibiotics, then fluctuation
        M["hr"] = (
            stage(E, 3, [(2, 1, 0.5 * f(abx_starts_or_on))])
            @ stage(E, 3, [(1, 2, 0.1 * f(abx_stopped))])
            @ fluctuate(E, 3, 0.1 * f(abx_still), 0.1 * f(abx_still))
        )
        # blood pressure: antibiotics, then vasopressors, then fluctuation
        on_nd, on_d = vaso_on & ~diab, vaso_on & diab
        falls = 0.1 * f(vaso_stopped & ~diab) + 0.05 * f(vaso_stopped & diab)
        sysbp_still = abx_still & ~vaso_on & ~vaso_stopped
        M["sysbp"] = (
            stage(E, 3, [(2, 1, 0.5 * f(abx_starts_or_on))])
            @ stage(E, 3, [(1, 2, 0.1 * f(abx_stopped))])
            @ stage(
                E,
                3,
                [
                    (0, 1, 0.7 * f(on_nd) + 0.5 * f(on_d)),
                    (0, 2, 0.4 * f(on_d)),
                    (1, 2, 0.7 * f(on_nd) + 0.9 * f(on_d)),
                ],
            )
            @ stage(E, 3, [(1, 0, falls), (2, 1, falls)])
            @ fluctuate(E, 3, 0.1 * f(sysbp_still), 0.1 * f(sysbp_still))
        )
        # oxygen: ventilation, then fluctuation
        o2_still = ~vent_on & ~vent_stopped
        M["o2"] = (
            stage(E, 2, [(0, 1, 0.7 * f(vent_on))])
            @ stage(E, 2, [(1, 0, 0.1 * f(vent_stopped))])
            @ fluctuate(E, 2, 0.1 * f(o2_still), 0.1 * f(o2_still))
        )
        # glucose: vasopressors raise it (diabetic only); otherwise it fluctuates. The "up" move of
        # a non-diabetic is built apart, because the original code caps it at 1 (glucose_bug).
        down = 0.1 * f(~vaso_on & ~diab) + 0.3 * f(~vaso_on & diab)
        up_d, up_nd = 0.3 * f(~vaso_on & diab), 0.1 * f(~vaso_on & ~diab)
        up_nd_moves = fluctuate(E, 5, 0 * up_nd, up_nd, up_cap=1 if self.glucose_bug else 4) - stay(E, 5)
        fluct = fluctuate(E, 5, down, up_d) + up_nd_moves
        M["glucose"] = stage(E, 5, [(v, v + 1, 0.5 * f(on_d)) for v in range(4)]) @ fluct

        C = max(self.state_cards)
        dist = np.zeros((E, len(self.state_vars), C))
        for i, v in enumerate(self.state_vars):
            if v in M:  # a vital: row of its transition matrix at the current value
                dist[:, i, : VITALS[v]] = M[v][np.arange(E), states[:, i]]
            else:  # deterministic: treatment flags become the action, diabetes stays
                new = {"abx_on": abx, "vent_on": vent, "vaso_on": vaso, "diabetic": states[:, i]}[v]
                dist[np.arange(E), i, new] = 1.0
        return dist

    def action_columns(self, actions):
        return np.stack([actions // 4, (actions // 2) % 2, actions % 2], axis=1).astype(np.int64)

    def start_distribution(self):
        """The original start (independent marginals, treatments off) conditioned on not terminal."""
        p = np.ones(len(self.B))
        diab = self.B[:, self.state_vars.index("diabetic")] if self.diabetes else np.zeros(len(self.B), int)
        if self.diabetes:
            p *= np.where(diab == 1, self.p_diabetes, 1 - self.p_diabetes)
        for v in self.vitals:
            x = self.B[:, self.state_vars.index(v)]
            if v == "glucose":
                p *= np.where(
                    diab == 1, np.array(START["glucose_diabetic"])[x], np.array(START["glucose"])[x]
                )
            else:
                p *= np.array(START[v])[x]
        for flag in TREATMENT_FLAGS:
            p *= self.B[:, self.state_vars.index(flag)] == 0
        p *= ~self.true_reward.terminal
        return p / p.sum()

    def state_label(self, values):
        """e.g. "D hr:H bp:N o2:L gl:N | A·V" (diabetic, vitals as L/N/H, treatments on)."""
        names = {"hr": "LNH", "sysbp": "LNH", "o2": "LN", "glucose": ["LL", "L", "N", "H", "HH"]}
        short = {"hr": "hr", "sysbp": "bp", "o2": "o2", "glucose": "gl"}
        x = dict(zip(self.state_vars, values, strict=True))
        vitals = " ".join(f"{short[v]}:{names[v][x[v]]}" for v in self.vitals)
        on = "·".join(t[0].upper() for t in TREATMENT_FLAGS if x[t])
        return ("D " if x.get("diabetic") else "") + vitals + (f" | {on}" if on else "")

    # ---- what the learner is given, and what the evaluation needs ----------------------------
    def reward(self, B, actions):
        abnormal = sum(B[:, self.state_vars.index(v)] != NORMAL[v] for v in self.vitals)
        treated = B[:, [self.state_vars.index(t) for t in TREATMENT_FLAGS]].any(1)
        r_next = np.where(abnormal >= self.death, -1.0, np.where((abnormal == 0) & ~treated, 1.0, 0.0))
        return Reward(np.zeros((len(B), len(actions))), r_next, r_next != 0)

    def true_parents(self, dataset: Dataset):
        d = ("diabetic",) if self.diabetes else ()
        truth = {
            "hr": ("hr", "antibiotics", "abx_on"),
            "sysbp": ("sysbp", "antibiotics", "abx_on", "vasopressors", "vaso_on") + d,
            "o2": ("o2", "ventilation", "vent_on"),
            "glucose": ("glucose", "vasopressors") + d,
            "abx_on": ("antibiotics",),
            "vent_on": ("ventilation",),
            "vaso_on": ("vasopressors",),
            "diabetic": ("diabetic",),
        }
        return {v: tuple(x for x in dataset.inputs if x in truth[v]) for v in dataset.state_vars}


def state_index(env: Sepsis, **values) -> int:
    """Index of the state given by name=value (helper for tests and examples)."""
    vector = np.array([values[v] for v in env.state_vars])
    return int(encode(vector, env.state_cards))
