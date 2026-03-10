import numpy as np

from catan_rl.actions import Action, CATALOG
from catan_rl.constants import Phase, Resource
from catan_rl.env import CatanEnv


def _set_main_phase(env: CatanEnv, player: int = 0) -> None:
    env.state.setup_round = 0
    env.state.phase = Phase.MAIN
    env.state.current_player = player
    env.state.has_rolled = True


def test_trade_phase_accept_executes_and_returns_turn_to_proposer():
    env = CatanEnv(seed=77)
    env.reset(seed=77)
    _set_main_phase(env, player=0)

    env.state.resources[0, Resource.WOOD] = 1
    env.state.resource_total[0] = 1
    env.state.resources[1, Resource.BRICK] = 1
    env.state.resource_total[1] = 1

    propose_id = CATALOG.encode(Action("PROPOSE_TRADE", (Resource.WOOD, 1, Resource.BRICK, 1)))
    res = env.step(propose_id)
    assert env.state.phase == Phase.TRADE_PROPOSED
    assert env.state.current_player == 1
    assert res.info["action_mask"].sum() > 0

    accept_id = CATALOG.encode(Action("ACCEPT_TRADE"))
    env.step(accept_id)
    assert env.state.phase == Phase.MAIN
    assert env.state.current_player == 0
    assert env.state.resources[0, Resource.BRICK] == 1
    assert env.state.resources[1, Resource.WOOD] == 1


def test_trade_phase_rejections_rotate_responders_then_end():
    env = CatanEnv(seed=88)
    env.reset(seed=88)
    _set_main_phase(env, player=0)
    env.state.resources[0, Resource.WOOD] = 1
    env.state.resource_total[0] = 1

    propose_id = CATALOG.encode(Action("PROPOSE_TRADE", (Resource.WOOD, 1, Resource.BRICK, 1)))
    env.step(propose_id)
    assert env.state.current_player == 1

    reject_id = CATALOG.encode(Action("REJECT_TRADE"))
    env.step(reject_id)
    assert env.state.phase == Phase.TRADE_PROPOSED
    assert env.state.current_player == 2
    env.step(reject_id)
    assert env.state.current_player == 3
    env.step(reject_id)
    assert env.state.phase == Phase.MAIN
    assert env.state.current_player == 0


def test_discard_phase_uses_pending_discard_counts_and_returns_to_roller():
    env = CatanEnv(seed=99)
    env.reset(seed=99)
    env.state.phase = Phase.DISCARD
    env.state.current_player = 1
    env.state.must_discard[:] = np.array([0, 1, 1, 0], dtype=np.int64)
    env.state.resources[1] = np.array([5, 0, 0, 0, 0], dtype=np.int64)
    env.state.resource_total[1] = 5
    env.state.resources[2] = np.array([0, 4, 0, 0, 0], dtype=np.int64)
    env.state.resource_total[2] = 4
    env._discard_remaining[:] = np.array([0, 5, 4, 0], dtype=np.int64)
    env._discard_roller = 3

    discard_p1 = CATALOG.encode(Action("DISCARD", (5, 0, 0, 0, 0)))
    env.step(discard_p1)
    assert env.state.current_player == 2
    assert env.state.must_discard[1] == 0

    discard_p2 = CATALOG.encode(Action("DISCARD", (0, 4, 0, 0, 0)))
    env.step(discard_p2)
    assert env.state.phase == Phase.MOVE_ROBBER
    assert env.state.current_player == 3


def test_trade_proposal_limit_per_turn_blocks_second_proposal():
    env = CatanEnv(seed=111, allow_player_trade=True, trade_action_mode="full", max_player_trade_proposals_per_turn=1)
    env.reset(seed=111)
    _set_main_phase(env, player=0)

    env.state.resources[0, Resource.WOOD] = 2
    env.state.resource_total[0] = 2

    propose_id = CATALOG.encode(Action("PROPOSE_TRADE", (Resource.WOOD, 1, Resource.BRICK, 1)))
    env.step(propose_id)
    # Force turn back to proposer main to simulate another attempt in same turn.
    env.state.phase = Phase.MAIN
    env.state.current_player = 0
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert "PROPOSE_TRADE" not in legal_kinds
