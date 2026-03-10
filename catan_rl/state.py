"""Game state dataclass aligned with catan_ds_o46.md."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    Building,
    INITIAL_DEV_DECK,
    NUM_EDGES,
    NUM_HEXES,
    NUM_PLAYERS,
    NUM_PORTS,
    NUM_RESOURCES,
    NUM_VERTICES,
    Phase,
)
from .topology import Topology, build_topology

NUMBER_TOKENS_STANDARD = np.asarray([2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12], dtype=np.int64)


@dataclass
class GameState:
    # Static board
    hex_terrain: np.ndarray
    hex_number: np.ndarray
    hex_pip_count: np.ndarray
    port_type: np.ndarray
    port_vertices: np.ndarray
    topology: Topology

    # Dynamic board
    robber_hex: int
    vertex_building: np.ndarray
    vertex_owner: np.ndarray
    edge_road: np.ndarray
    edge_owner: np.ndarray

    # Player state
    resources: np.ndarray
    resource_total: np.ndarray
    dev_cards_hidden: np.ndarray
    dev_cards_bought_this_turn: np.ndarray
    knights_played: np.ndarray
    vp_cards_held: np.ndarray
    settlements_left: np.ndarray
    cities_left: np.ndarray
    roads_left: np.ndarray
    has_port: np.ndarray
    longest_road_length: np.ndarray
    has_longest_road: np.ndarray
    has_largest_army: np.ndarray
    actual_vp: np.ndarray
    public_vp: np.ndarray
    # Compact public-memory signals (human-trackable approximations).
    public_recent_gain: np.ndarray
    public_recent_spend: np.ndarray
    public_recent_rob_unknown_gain: np.ndarray
    public_recent_rob_unknown_loss: np.ndarray
    public_trade_offers: np.ndarray
    public_trade_accepts: np.ndarray
    public_trade_rejects: np.ndarray

    # Dev deck
    dev_deck_remaining: int
    dev_deck_composition: np.ndarray

    # Turn state
    current_player: int
    turn_number: int
    phase: int
    dice_roll: int
    has_rolled: bool
    dev_card_played_this_turn: bool
    free_roads_remaining: int
    must_discard: np.ndarray
    setup_round: int
    setup_forward: bool
    setup_player_order_index: int
    setup_settlement_vertex: int

    # Trade state
    trade_offer_give: np.ndarray
    trade_offer_want: np.ndarray
    trade_proposer: int
    trade_responses: np.ndarray

    winner: int

    def copy(self) -> "GameState":
        data = {}
        for k, v in self.__dict__.items():
            data[k] = v.copy() if isinstance(v, np.ndarray) else v
        return GameState(**data)


def _has_adjacent_six_or_eight(nums: np.ndarray, topo: Topology) -> bool:
    for h in np.flatnonzero(np.isin(nums, [6, 8])):
        for n in topo.hex_to_hexes[int(h)]:
            if n >= 0 and nums[int(n)] in (6, 8):
                return True
    return False


def _roll_hex_numbers(rng: np.random.Generator, hex_terrain: np.ndarray, topo: Topology) -> tuple[np.ndarray, np.ndarray]:
    non_desert = np.flatnonzero(hex_terrain != 0)
    for _ in range(5000):
        tokens = NUMBER_TOKENS_STANDARD.copy()
        rng.shuffle(tokens)
        nums = np.zeros(NUM_HEXES, dtype=np.int64)
        nums[non_desert] = tokens
        if _has_adjacent_six_or_eight(nums, topo):
            continue
        pips = np.zeros(NUM_HEXES, dtype=np.int64)
        for h in non_desert:
            n = int(nums[int(h)])
            pips[int(h)] = 6 - abs(7 - n)
        return nums, pips
    raise RuntimeError("Failed to sample a valid token layout with non-adjacent 6/8 tiles")


def new_game_state(seed: int | None = None) -> GameState:
    rng = np.random.default_rng(seed)
    topo = build_topology()

    # Official terrain distribution:
    # desert x1, wood x4, brick x3, sheep x4, wheat x4, ore x3.
    terrain_pool = [0] + [1] * 4 + [2] * 3 + [3] * 4 + [4] * 4 + [5] * 3
    rng.shuffle(terrain_pool)
    hex_terrain = np.asarray(terrain_pool[:NUM_HEXES], dtype=np.int64)
    desert_hex = int(np.where(hex_terrain == 0)[0][0])
    hex_number, hex_pip_count = _roll_hex_numbers(rng, hex_terrain, topo)
    hex_number[desert_hex] = 0
    hex_pip_count[desert_hex] = 0

    return GameState(
        hex_terrain=hex_terrain,
        hex_number=hex_number,
        hex_pip_count=hex_pip_count,
        port_type=topo.port_type.copy(),
        port_vertices=topo.port_vertices.copy(),
        topology=topo,
        robber_hex=desert_hex,
        vertex_building=np.full(NUM_VERTICES, Building.EMPTY, dtype=np.int64),
        vertex_owner=np.full(NUM_VERTICES, -1, dtype=np.int64),
        edge_road=np.zeros(NUM_EDGES, dtype=np.int64),
        edge_owner=np.full(NUM_EDGES, -1, dtype=np.int64),
        resources=np.zeros((NUM_PLAYERS, NUM_RESOURCES), dtype=np.int64),
        resource_total=np.zeros(NUM_PLAYERS, dtype=np.int64),
        dev_cards_hidden=np.zeros((NUM_PLAYERS, 5), dtype=np.int64),
        dev_cards_bought_this_turn=np.zeros((NUM_PLAYERS, 5), dtype=np.int64),
        knights_played=np.zeros(NUM_PLAYERS, dtype=np.int64),
        vp_cards_held=np.zeros(NUM_PLAYERS, dtype=np.int64),
        settlements_left=np.full(NUM_PLAYERS, 5, dtype=np.int64),
        cities_left=np.full(NUM_PLAYERS, 4, dtype=np.int64),
        roads_left=np.full(NUM_PLAYERS, 15, dtype=np.int64),
        has_port=np.zeros((NUM_PLAYERS, 6), dtype=np.int64),
        longest_road_length=np.zeros(NUM_PLAYERS, dtype=np.int64),
        has_longest_road=np.zeros(NUM_PLAYERS, dtype=np.int64),
        has_largest_army=np.zeros(NUM_PLAYERS, dtype=np.int64),
        actual_vp=np.zeros(NUM_PLAYERS, dtype=np.int64),
        public_vp=np.zeros(NUM_PLAYERS, dtype=np.int64),
        public_recent_gain=np.zeros((NUM_PLAYERS, NUM_RESOURCES), dtype=np.float32),
        public_recent_spend=np.zeros((NUM_PLAYERS, NUM_RESOURCES), dtype=np.float32),
        public_recent_rob_unknown_gain=np.zeros(NUM_PLAYERS, dtype=np.float32),
        public_recent_rob_unknown_loss=np.zeros(NUM_PLAYERS, dtype=np.float32),
        public_trade_offers=np.zeros(NUM_PLAYERS, dtype=np.float32),
        public_trade_accepts=np.zeros(NUM_PLAYERS, dtype=np.float32),
        public_trade_rejects=np.zeros(NUM_PLAYERS, dtype=np.float32),
        dev_deck_remaining=sum(INITIAL_DEV_DECK),
        dev_deck_composition=np.asarray(INITIAL_DEV_DECK, dtype=np.int64),
        current_player=0,
        turn_number=0,
        phase=Phase.SETUP_SETTLEMENT,
        dice_roll=0,
        has_rolled=False,
        dev_card_played_this_turn=False,
        free_roads_remaining=0,
        must_discard=np.zeros(NUM_PLAYERS, dtype=np.int64),
        setup_round=1,
        setup_forward=True,
        setup_player_order_index=0,
        setup_settlement_vertex=-1,
        trade_offer_give=np.zeros(NUM_RESOURCES, dtype=np.int64),
        trade_offer_want=np.zeros(NUM_RESOURCES, dtype=np.int64),
        trade_proposer=-1,
        trade_responses=np.zeros(NUM_PLAYERS, dtype=np.int64),
        winner=-1,
    )
