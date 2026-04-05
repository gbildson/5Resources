"""Strategy archetype definitions and weak-supervision labeling heuristics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import Building, NUM_RESOURCES
from .state import GameState


@dataclass(frozen=True)
class StrategyArchetype:
    key: str
    title: str
    description: str


ARCHETYPES: tuple[StrategyArchetype, ...] = (
    StrategyArchetype(
        key="ows_city_engine",
        title="OWS City Engine",
        description="Prioritize ore-wheat-sheep/dev tempo into city growth and largest army pressure.",
    ),
    StrategyArchetype(
        key="road_expansion",
        title="Road Expansion",
        description="Prioritize wood-brick tempo, expansion control, and settlement count growth.",
    ),
    StrategyArchetype(
        key="port_conversion",
        title="Port Conversion",
        description="Exploit 2:1 or 3:1 ports with concentrated production to convert into flexible builds.",
    ),
    StrategyArchetype(
        key="dev_army_pressure",
        title="Dev Army Pressure",
        description="Lean on dev-card acquisition and timely knight plays to win largest army and control robber.",
    ),
    StrategyArchetype(
        key="hybrid_flexible",
        title="Hybrid Flexible",
        description="Maintain balanced production and adapt between city, road, and dev lines by board context.",
    ),
    StrategyArchetype(
        key="city_vp_line",
        title="City Plus VP Line",
        description="Push city upgrades and VP tempo while limiting overcommitment to knight race.",
    ),
)
STRATEGY_KEYS: tuple[str, ...] = tuple(a.key for a in ARCHETYPES)


def archetype_definitions() -> list[dict[str, str]]:
    return [{"key": a.key, "title": a.title, "description": a.description} for a in ARCHETYPES]


def _vertex_resource_pips(state: GameState, vertex: int) -> np.ndarray:
    out = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for h in state.topology.vertex_to_hexes[int(vertex)]:
        if h < 0:
            continue
        terrain = int(state.hex_terrain[int(h)])
        if terrain <= 0:
            continue
        out[terrain - 1] += float(state.hex_pip_count[int(h)])
    return out


def _production_profile(state: GameState, player: int) -> np.ndarray:
    prof = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) != int(player):
            continue
        b = int(state.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        prof += mult * _vertex_resource_pips(state, v)
    return prof


def infer_strategy_scores(state: GameState, player: int) -> dict[str, float]:
    p = int(player)
    prof = _production_profile(state, p)
    wood, brick, sheep, wheat, ore = [float(x) for x in prof.tolist()]

    roads_built = float(max(0, 15 - int(state.roads_left[p])))
    settles_on_board = float(
        sum(
            1
            for v in range(len(state.vertex_owner))
            if int(state.vertex_owner[v]) == p and int(state.vertex_building[v]) == int(Building.SETTLEMENT)
        )
    )
    cities_on_board = float(
        sum(
            1
            for v in range(len(state.vertex_owner))
            if int(state.vertex_owner[v]) == p and int(state.vertex_building[v]) == int(Building.CITY)
        )
    )
    has_generic_port = float(int(state.has_port[p, 0]) == 1)
    has_specific_port = float(np.count_nonzero(state.has_port[p, 1:]) > 0)

    dev_hidden = state.dev_cards_hidden[p] + state.dev_cards_bought_this_turn[p]
    dev_total = float(dev_hidden.sum())
    knight_hidden = float(dev_hidden[0])
    vp_hidden = float(dev_hidden[4] + int(state.vp_cards_held[p]))
    knights_played = float(state.knights_played[p])
    public_vp = float(state.public_vp[p])

    scores = {
        # High ore/wheat/sheep with dev usage and city propensity.
        "ows_city_engine": 0.55 * min(ore, wheat) + 0.25 * sheep + 0.25 * cities_on_board + 0.20 * dev_total + 0.10 * knights_played,
        # Wood/brick + roads/settlements.
        "road_expansion": 0.60 * min(wood, brick) + 0.30 * roads_built + 0.25 * settles_on_board,
        # Port ownership + concentrated production.
        "port_conversion": 0.50 * has_specific_port + 0.25 * has_generic_port + 0.25 * float(max(prof)) + 0.10 * dev_total,
        # Dev+knight pressure.
        "dev_army_pressure": 0.45 * (knights_played + knight_hidden) + 0.30 * dev_total + 0.15 * sheep + 0.10 * ore,
        # Balanced profile and mixed piece development.
        "hybrid_flexible": 0.30 * float(np.count_nonzero(prof > 0.0)) + 0.20 * float(np.mean(prof)) + 0.20 * min(settles_on_board, 3.0) + 0.15 * cities_on_board + 0.10 * dev_total,
        # City + VP progression.
        "city_vp_line": 0.50 * cities_on_board + 0.30 * vp_hidden + 0.20 * public_vp + 0.20 * min(ore, wheat) - 0.10 * knights_played,
    }
    # Ensure non-negative prior weights for normalization.
    return {k: float(max(0.0, v)) for k, v in scores.items()}


def infer_strategy_distribution(state: GameState, player: int) -> dict[str, float]:
    scores = infer_strategy_scores(state, player)
    total = float(sum(scores.values()))
    if total <= 1e-8:
        uniform = 1.0 / max(1, len(scores))
        return {k: uniform for k in scores.keys()}
    return {k: float(v / total) for k, v in scores.items()}


def infer_primary_strategy(state: GameState, player: int) -> str:
    dist = infer_strategy_distribution(state, player)
    if not dist:
        return "hybrid_flexible"
    return max(dist.items(), key=lambda kv: kv[1])[0]


def strategy_target_vector(state: GameState, player: int) -> np.ndarray:
    dist = infer_strategy_distribution(state, player)
    return np.asarray([float(dist.get(k, 0.0)) for k in STRATEGY_KEYS], dtype=np.float32)

