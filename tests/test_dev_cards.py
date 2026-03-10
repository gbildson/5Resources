from catan_rl.actions import Action, CATALOG
from catan_rl.constants import DevCard, Phase, Resource
from catan_rl.env import CatanEnv


def _set_main_phase(env: CatanEnv, player: int = 0) -> None:
    env.state.setup_round = 0
    env.state.phase = Phase.MAIN
    env.state.current_player = player
    env.state.has_rolled = True
    env.state.dev_card_played_this_turn = False


def test_bought_dev_card_not_playable_same_turn():
    env = CatanEnv(seed=31)
    env.reset(seed=31)
    _set_main_phase(env, player=0)

    env.state.resources[0] = [0, 0, 1, 1, 1]
    env.state.resource_total[0] = 3
    env.state.dev_deck_composition[:] = [1, 0, 0, 0, 0]
    env.state.dev_deck_remaining = 1

    buy_id = CATALOG.encode(Action("BUY_DEV_CARD"))
    env.step(buy_id)

    assert env.state.dev_cards_bought_this_turn[0, DevCard.KNIGHT] == 1
    assert env.state.dev_cards_hidden[0, DevCard.KNIGHT] == 0

    knight_id = CATALOG.encode(Action("PLAY_KNIGHT"))
    mask = env.action_mask()
    assert mask[knight_id] == 0


def test_only_one_dev_card_can_be_played_per_turn():
    env = CatanEnv(seed=32)
    env.reset(seed=32)
    _set_main_phase(env, player=0)

    env.state.dev_cards_hidden[0, DevCard.YEAR_OF_PLENTY] = 1
    env.state.dev_cards_hidden[0, DevCard.MONOPOLY] = 1

    yop_id = CATALOG.encode(Action("PLAY_YEAR_OF_PLENTY", (Resource.WOOD, Resource.BRICK)))
    env.step(yop_id)
    assert env.state.dev_card_played_this_turn

    monopoly_id = CATALOG.encode(Action("PLAY_MONOPOLY", (Resource.ORE,)))
    mask = env.action_mask()
    assert mask[monopoly_id] == 0


def test_bought_dev_card_becomes_playable_next_turn():
    env = CatanEnv(seed=33)
    env.reset(seed=33)
    _set_main_phase(env, player=0)

    env.state.resources[0] = [0, 0, 1, 1, 1]
    env.state.resource_total[0] = 3
    env.state.dev_deck_composition[:] = [1, 0, 0, 0, 0]
    env.state.dev_deck_remaining = 1

    buy_id = CATALOG.encode(Action("BUY_DEV_CARD"))
    env.step(buy_id)
    end_id = CATALOG.encode(Action("END_TURN"))
    env.step(end_id)

    assert env.state.dev_cards_hidden[0, DevCard.KNIGHT] == 1
    assert env.state.dev_cards_bought_this_turn[0, DevCard.KNIGHT] == 0
