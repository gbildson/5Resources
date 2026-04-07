from catan_rl.constants import Phase, Resource
from catan_rl.env import CatanEnv
from catan_rl.strategy_metrics import (
    opponent_danger_targets,
    strategic_aux_target_weights,
    strategic_aux_targets,
    strategic_evaluator_snapshot,
)


def test_trade_responder_aux_targets_masked_outside_trade_response():
    env = CatanEnv(seed=7)
    env.reset(seed=7)
    state = env.state
    player = int(state.current_player)

    snapshot = strategic_evaluator_snapshot(state, player)
    aux = strategic_aux_targets(snapshot, state, player)
    weights = strategic_aux_target_weights(state, player)

    assert aux["trade_accept_value"] == 0.0
    assert aux["trade_accept_immediate_build_gain"] == 0.0
    assert aux["trade_accept_should_take"] == 0.0
    assert weights["trade_accept_value"] == 0.0
    assert weights["trade_accept_immediate_build_gain"] == 0.0
    assert weights["trade_accept_should_take"] == 0.0


def test_trade_responder_aux_targets_active_on_pending_response():
    env = CatanEnv(seed=11)
    env.reset(seed=11)
    state = env.state
    state.phase = Phase.TRADE_PROPOSED
    state.current_player = 0
    state.trade_proposer = 1
    state.trade_responses[:] = 0
    state.trade_offer_give[:] = 0
    state.trade_offer_want[:] = 0
    state.trade_offer_give[Resource.WOOD] = 1
    state.trade_offer_want[Resource.ORE] = 1
    state.public_vp[:] = [2, 0, 4, 5]
    state.resources[:] = 0
    state.resources[0, Resource.BRICK] = 1
    state.resources[0, Resource.SHEEP] = 1
    state.resources[0, Resource.WHEAT] = 1
    state.resources[0, Resource.ORE] = 1
    state.resources[1, Resource.WOOD] = 1
    state.resource_total[:] = state.resources.sum(axis=1)

    snapshot = strategic_evaluator_snapshot(state, 0)
    aux = strategic_aux_targets(snapshot, state, 0)
    weights = strategic_aux_target_weights(state, 0)

    assert weights["trade_accept_value"] == 1.0
    assert weights["trade_accept_immediate_build_gain"] == 1.0
    assert weights["trade_accept_should_take"] == 1.0
    assert aux["trade_accept_value"] >= 0.0
    assert aux["trade_accept_immediate_build_gain"] == 1.0
    assert aux["trade_accept_should_take"] == float(aux["trade_accept_value"] > 0.0)


def test_opponent_danger_targets_follow_relative_seat_order():
    env = CatanEnv(seed=19)
    env.reset(seed=19)
    state = env.state
    state.public_vp[:] = [2, 4, 8, 6]
    state.resource_total[:] = [2, 3, 9, 7]
    state.longest_road_length[:] = [1, 2, 6, 4]
    state.knights_played[:] = [0, 1, 3, 1]
    state.has_longest_road[:] = [0, 0, 1, 0]
    state.has_largest_army[:] = [0, 0, 1, 0]
    state.dev_cards_hidden[:] = 0
    state.dev_cards_hidden[2, 0] = 2
    state.dev_cards_bought_this_turn[:] = 0

    p0 = opponent_danger_targets(state, 0)
    p1 = opponent_danger_targets(state, 1)

    assert p0["opponent_danger_opp2"] > p0["opponent_danger_opp3"] > p0["opponent_danger_opp1"]
    assert p1["opponent_danger_opp1"] > p1["opponent_danger_opp2"] > p1["opponent_danger_opp3"]
