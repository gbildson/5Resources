from catan_rl.actions import Action, CATALOG
from catan_rl.constants import Building, Phase, Resource
from catan_rl.env import CatanEnv


def _set_robber_phase(env: CatanEnv, player: int = 0) -> None:
    env.state.setup_round = 0
    env.state.current_player = player
    env.state.phase = Phase.MOVE_ROBBER
    env.state.has_rolled = True


def test_move_robber_skips_rob_phase_when_no_eligible_victim():
    env = CatanEnv(seed=1234)
    env.reset(seed=1234)
    _set_robber_phase(env, player=0)

    target_hex = 1
    vertices = env.state.topology.hex_to_vertices[target_hex]
    env.state.vertex_owner[vertices[0]] = 0
    env.state.vertex_building[vertices[0]] = Building.SETTLEMENT
    env.state.vertex_owner[vertices[1]] = 1
    env.state.vertex_building[vertices[1]] = Building.SETTLEMENT
    env.state.resources[1, :] = 0
    env.state.resource_total[1] = 0

    move_id = CATALOG.encode(Action("MOVE_ROBBER", (target_hex,)))
    env.step(move_id)

    assert env.state.phase == Phase.MAIN
    assert env.state.current_player == 0
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert "ROB_PLAYER" not in legal_kinds


def test_robber_rob_targets_are_only_adjacent_players_with_cards():
    env = CatanEnv(seed=2233)
    env.reset(seed=2233)
    _set_robber_phase(env, player=0)

    target_hex = 2
    vertices = env.state.topology.hex_to_vertices[target_hex]

    env.state.vertex_owner[vertices[0]] = 1
    env.state.vertex_building[vertices[0]] = Building.SETTLEMENT
    env.state.vertex_owner[vertices[1]] = 2
    env.state.vertex_building[vertices[1]] = Building.SETTLEMENT

    env.state.resources[1, Resource.WOOD] = 1
    env.state.resource_total[1] = 1
    env.state.resources[2, :] = 0
    env.state.resource_total[2] = 0
    env.state.resources[3, Resource.BRICK] = 2
    env.state.resource_total[3] = 2

    move_id = CATALOG.encode(Action("MOVE_ROBBER", (target_hex,)))
    env.step(move_id)
    assert env.state.phase == Phase.ROB_PLAYER

    mask = env.action_mask()
    legal_rob_targets = set()
    for a_id in CATALOG.legal_ids(mask):
        action = CATALOG.decode(int(a_id))
        if action.kind == "ROB_PLAYER":
            legal_rob_targets.add(action.params[0])
    assert legal_rob_targets == {1}
