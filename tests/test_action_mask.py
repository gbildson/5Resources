from catan_rl.actions import CATALOG
from catan_rl.constants import Phase, Resource
from catan_rl.env import CatanEnv


def test_preroll_only_allows_roll_or_dev():
    env = CatanEnv(seed=42)
    env.reset(seed=42)
    env.state.setup_round = 0
    env.state.phase = Phase.PRE_ROLL
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert "ROLL_DICE" in legal_kinds
    assert "END_TURN" not in legal_kinds
    assert "PLACE_SETTLEMENT" not in legal_kinds


def test_decode_encode_roundtrip():
    for i, action in enumerate(CATALOG.actions):
        j = CATALOG.encode(action)
        assert i == j


def test_main_action_cap_forces_end_turn_only():
    env = CatanEnv(seed=5, max_main_actions_per_turn=0)
    env.reset(seed=5)
    env.state.setup_round = 0
    env.state.phase = Phase.MAIN
    env.state.current_player = 0
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert legal_kinds == {"END_TURN"}


def test_trade_draft_action_cap_only_allows_submit_or_cancel():
    env = CatanEnv(seed=15, max_main_actions_per_turn=1, allow_player_trade=True, trade_action_mode="full")
    env.reset(seed=15)
    env.state.setup_round = 0
    env.state.phase = Phase.TRADE_DRAFT
    env.state.current_player = 0
    env.state.trade_proposer = 0
    env.state.trade_offer_give[Resource.WOOD] = 1
    env.state.trade_offer_want[Resource.BRICK] = 1
    env.state.resources[0, Resource.WOOD] = 1
    env.state.resource_total[0] = 1
    env.state.resources[1, Resource.BRICK] = 1
    env.state.resource_total[1] = 1
    env.state.public_recent_gain[1, Resource.BRICK] = 1.0
    env._main_actions_this_turn = 1
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert legal_kinds == {"PROPOSE_TRADE", "CANCEL_TRADE"}


def test_disable_player_trade_masks_trade_actions():
    env = CatanEnv(seed=6, allow_player_trade=False)
    env.reset(seed=6)
    env.state.setup_round = 0
    env.state.phase = Phase.MAIN
    env.state.current_player = 0
    env.state.resources[0, 0] = 2
    env.state.resource_total[0] = 2
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert "TRADE_ADD_GIVE" not in legal_kinds
    assert "TRADE_ADD_WANT" not in legal_kinds
    assert "PROPOSE_TRADE" not in legal_kinds


def test_guided_trade_mask_allows_draft_but_not_submit_in_main():
    env = CatanEnv(seed=12, allow_player_trade=True, trade_action_mode="guided")
    env.reset(seed=12)
    env.state.setup_round = 0
    env.state.phase = Phase.MAIN
    env.state.current_player = 0
    env.state.has_rolled = True
    # Draft actions are available, but submit only appears after entering draft.
    env.state.resources[0, Resource.WOOD] = 1
    env.state.resources[0, Resource.BRICK] = 1
    env.state.resource_total[0] = 2
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert "TRADE_ADD_GIVE" in legal_kinds
    assert "TRADE_ADD_WANT" in legal_kinds
    assert "PROPOSE_TRADE" not in legal_kinds
    assert "END_TURN" in legal_kinds


def test_full_trade_mode_keeps_trade_draft_actions_legal():
    env = CatanEnv(seed=13, allow_player_trade=True, trade_action_mode="full")
    env.reset(seed=13)
    env.state.setup_round = 0
    env.state.phase = Phase.MAIN
    env.state.current_player = 0
    env.state.has_rolled = True
    env.state.resources[0, Resource.WOOD] = 1
    env.state.resource_total[0] = 1
    mask = env.action_mask()
    legal_kinds = {CATALOG.decode(i).kind for i in CATALOG.legal_ids(mask)}
    assert "TRADE_ADD_GIVE" in legal_kinds
    assert "TRADE_ADD_WANT" in legal_kinds
