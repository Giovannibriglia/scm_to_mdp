"""The nine domains of experiments 1 and 2, as one table: how to build each environment, its sizes,
its discount and its fixed settings (S0, S1). Both experiments read it, so the domains stay identical.

The sizes are the ones calibrated on 2026-10-26 (experiments/trials_wrong/calibrate_sizes.py, kept on
disk only): every size fits the common memory limit, and the oracle (the true observed MDP) solves it.
A domain with a hidden exogenous variable (SysAdmin's storm, Chemistry's catalyst, Drone's gust,
Aircraft's turbulence, Stock's news, Frozen Lake's wind) draws it on at every step with probability
HIDDEN; the learner never sees it, and the true observed MDP averages it out.

    domain        sizes                                  knob                      gamma
    sysadmin      N = 2..11 machines (ring)              topology = ring           0.90
    chemistry     2..7 objects                           density = 1.0             0.90
    aircraft      10 grids, 4x3 .. 18x13                 proximity = on            0.95
    drone         side 3..12, one drone                  gust                      0.95
    frozenlake    1 agent: maps 4..32; 2 agents: 4..7    agents = 1 | 2            0.95
    stock         9 layouts, 1x2 .. 4x2                  scale = 0.8               0.90
    sepsis        2, 3, 4 vitals                         diabetes = observed       0.90
    coffee        coffee, coffee+snack                   variant = original        0.90
    taxi          deterministic, noisy navigation        variant = original        0.99

make_env functions live at module level: the grid's worker processes import them by name.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from scm_to_mdp.envs import (
    Aircraft,
    Chemistry,
    Coffee,
    Drone,
    FrozenLake,
    Sepsis,
    StockTrading,
    SysAdmin,
    Taxi,
)

HIDDEN = 0.1  # P(hidden exogenous variable on) at every step
MODEL_GB = 12.0  # the largest dense true model an environment may build


@dataclass(frozen=True)
class Domain:
    title: str
    make_env: Callable[[Any, Any, int], Any]  # (size, knob, seed) -> Env
    sizes: list | dict  # a list, or {knob: list} when each knob has its own sizes
    size_name: str
    knobs: list
    knob_name: str
    gamma: float
    constants: dict = field(default_factory=dict)  # the settings make_env uses (written to grid.json)
    sl_spibb_parents: int = 3  # SL-SPIBB: the size of its candidate parent sets


# ---- environments ----------------------------------------------------------------------------------
STORM_EFFECT = 0.8  # a storm multiplies machine 0's P(up) by 1 - 0.8


def sysadmin(n, topology: str, seed: int) -> SysAdmin:
    return SysAdmin(int(n), seed, topology=str(topology), storm=HIDDEN, storm_effect=STORM_EFFECT)


def chemistry(n_objects, density: str, seed: int) -> Chemistry:
    return Chemistry(
        float(density), seed, n_objects=int(n_objects), n_colours=3, max_parents=2, catalyst=HIDDEN
    )


TURBULENCE_EFFECT = 0.5


def aircraft(size: str, proximity: str, seed: int) -> Aircraft:
    width, height = (int(v) for v in str(size).split("x"))
    return Aircraft(width, height, seed, proximity=proximity == "on", turbulence=HIDDEN,
                    turbulence_effect=TURBULENCE_EFFECT, max_model_gb=MODEL_GB)  # fmt: skip


GUST_EFFECT = 0.5


def drone(size: str, gust: str, seed: int) -> Drone:
    side, drones = (int(v) for v in str(size).split("-"))
    return Drone(side, drones, seed, gust=HIDDEN if gust == "gust" else None, gust_effect=GUST_EFFECT,
                 max_model_gb=MODEL_GB)  # fmt: skip


WIND_EFFECT = 0.5


def frozenlake(map_name: str, agents: str, seed: int) -> FrozenLake:
    return FrozenLake(str(map_name), int(agents), seed, slippery=True, wind=HIDDEN, wind_effect=WIND_EFFECT,
                      axes="independent", max_model_gb=MODEL_GB)  # fmt: skip


NEWS_EFFECT = 0.6


def stock(size: str, scale, seed: int) -> StockTrading:
    sectors, stocks = (int(v) for v in str(size).split("x"))
    return StockTrading(sectors, stocks, seed, scale=float(scale), news=HIDDEN, news_effect=NEWS_EFFECT,
                        max_model_gb=MODEL_GB)  # fmt: skip


VITALS = {2: ("hr", "sysbp"), 3: ("hr", "sysbp", "o2"), 4: ("hr", "sysbp", "o2", "glucose")}
DIABETES_EFFECT = 1.5  # the diabetic response, scaled away from the non-diabetic one (1 = original)
P_DIABETES = 0.2


def sepsis(n_vitals, diabetes: str, seed: int) -> Sepsis:
    return Sepsis("custom", seed, vitals=VITALS[int(n_vitals)], diabetes=diabetes != "absent",
                  hide_diabetes=diabetes == "hidden", diabetes_effect=DIABETES_EFFECT,
                  p_diabetes=P_DIABETES)  # fmt: skip


def coffee(size: str, variant: str, seed: int) -> Coffee:
    return Coffee(size, seed)


def taxi(size: str, variant: str, seed: int) -> Taxi:
    return Taxi(size, seed, axes="independent")


DOMAINS = {
    "sysadmin": Domain("SysAdmin", sysadmin, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "N", ["ring"], "topology", 0.9,
                       {"storm": HIDDEN, "storm_effect": STORM_EFFECT}),
    "chemistry": Domain("Chemistry", chemistry, [2, 3, 4, 5, 6, 7], "objects", ["1.0"], "density", 0.9,
                        {"catalyst": HIDDEN, "n_colours": 3, "max_parents": 2}),
    "aircraft": Domain("Aircraft", aircraft,
                       ["4x3", "5x4", "6x4", "8x5", "8x6", "10x7", "12x8", "14x10", "16x12", "18x13"],
                       "width x height", ["on"], "proximity", 0.95,
                       {"turbulence": HIDDEN, "turbulence_effect": TURBULENCE_EFFECT}),
    "drone": Domain("Drone", drone, ["3-1", "4-1", "5-1", "6-1", "7-1", "8-1", "9-1", "10-1", "11-1", "12-1"],
                    "side-drones", ["gust"], "gust", 0.95, {"gust": HIDDEN, "gust_effect": GUST_EFFECT}),
    "frozenlake": Domain("Frozen Lake", frozenlake,
                         {"1": ["4x4", "8x8", "16x16", "32x32"], "2": ["4x4", "5x5", "6x6", "7x7"]},
                         "map", ["1", "2"], "agents", 0.95, {"wind": HIDDEN, "wind_effect": WIND_EFFECT},
                         sl_spibb_parents=2),
    "stock": Domain("Stock trading", stock, ["1x2", "1x3", "1x4", "2x2", "1x6", "2x3", "3x2", "2x4", "4x2"],
                    "sectors x stocks", [0.8], "scale", 0.9, {"news": HIDDEN, "news_effect": NEWS_EFFECT}),
    "sepsis": Domain("Sepsis", sepsis, list(VITALS), "vitals", ["observed"], "diabetes", 0.9,
                     {"p_diabetes": P_DIABETES, "diabetes_effect": DIABETES_EFFECT}, sl_spibb_parents=4),
    "coffee": Domain("Coffee robot", coffee, ["coffee", "coffee+snack"], "domain", ["original"], "variant",
                     0.9),
    "taxi": Domain("Taxi", taxi, ["deterministic", "noisy"], "navigation", ["original"], "variant", 0.99),
}  # fmt: skip
