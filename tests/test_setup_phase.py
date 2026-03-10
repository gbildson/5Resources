from catan_rl.actions import Action, CATALOG
from catan_rl.constants import Building, Phase
from catan_rl.env import CatanEnv


def _first_legal(mask, kind: str) -> int:
    for a_id in CATALOG.legal_ids(mask):
        action = CATALOG.decode(int(a_id))
        if action.kind == kind:
            return int(a_id)
    raise AssertionError(f"No legal action found for {kind}")


def test_setup_snake_order_and_phase_transitions():
    env = CatanEnv(seed=101)
    _, info = env.reset(seed=101)
    player_history = []
    phase_history = []

    # 16 setup placements: 8 settlements + 8 roads.
    for _ in range(16):
        player_history.append(env.state.current_player)
        phase_history.append(env.state.phase)
        mask = info["action_mask"]
        if env.state.phase == Phase.SETUP_SETTLEMENT:
            action_id = _first_legal(mask, "PLACE_SETTLEMENT")
        else:
            action_id = _first_legal(mask, "PLACE_ROAD")
        res = env.step(action_id)
        info = {"action_mask": res.info["action_mask"]}

    assert env.state.setup_round == 0
    assert env.state.phase == Phase.PRE_ROLL
    assert env.state.current_player == 0
    assert env.state.turn_number == 1

    # Settlement turns follow snake order: 0,1,2,3,3,2,1,0
    settlement_players = [p for p, ph in zip(player_history, phase_history) if ph == Phase.SETUP_SETTLEMENT]
    assert settlement_players == [0, 1, 2, 3, 3, 2, 1, 0]


def test_setup_road_must_touch_latest_settlement():
    env = CatanEnv(seed=202)
    _, info = env.reset(seed=202)
    assert env.state.phase == Phase.SETUP_SETTLEMENT

    settlement_id = _first_legal(info["action_mask"], "PLACE_SETTLEMENT")
    settlement_action = CATALOG.decode(settlement_id)
    placed_vertex = settlement_action.params[0]
    res = env.step(settlement_id)
    mask = res.info["action_mask"]
    assert env.state.phase == Phase.SETUP_ROAD

    legal_roads = []
    for a_id in CATALOG.legal_ids(mask):
        action = CATALOG.decode(int(a_id))
        if action.kind == "PLACE_ROAD":
            legal_roads.append(action.params[0])
    assert legal_roads, "Expected at least one legal setup road"

    # All legal setup roads should include the newly placed settlement endpoint.
    for edge_id in legal_roads:
        u, v = env.state.topology.edge_to_vertices[edge_id]
        assert placed_vertex in (int(u), int(v))


def test_setup_distance_rule_blocks_adjacent_settlement():
    env = CatanEnv(seed=303)
    _, info = env.reset(seed=303)

    settlement_id = _first_legal(info["action_mask"], "PLACE_SETTLEMENT")
    placed_vertex = CATALOG.decode(settlement_id).params[0]
    res = env.step(settlement_id)

    # Move into next player's settlement placement.
    road_id = _first_legal(res.info["action_mask"], "PLACE_ROAD")
    res = env.step(road_id)
    assert env.state.phase == Phase.SETUP_SETTLEMENT

    adjacent = [v for v in env.state.topology.vertex_to_vertices[placed_vertex] if v >= 0]
    mask = res.info["action_mask"]
    for a_id in CATALOG.legal_ids(mask):
        action = CATALOG.decode(int(a_id))
        if action.kind != "PLACE_SETTLEMENT":
            continue
        assert action.params[0] not in adjacent

    # Sanity check that the placed settlement is present.
    assert env.state.vertex_building[placed_vertex] == Building.SETTLEMENT
