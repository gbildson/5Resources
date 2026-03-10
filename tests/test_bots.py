import numpy as np

from catan_rl.actions import Action, CATALOG
from catan_rl.bots import HeuristicAgent, MetaStrategyHeuristicAgent, RandomLegalAgent, StrategyHeuristicAgent


def test_bots_fallback_to_end_turn_on_empty_mask():
    empty_mask = np.zeros(CATALOG.size(), dtype=np.int8)
    end_turn_id = CATALOG.encode(Action("END_TURN"))

    rand_bot = RandomLegalAgent(seed=1)
    heur_bot = HeuristicAgent(seed=2)
    dummy_obs = np.zeros(4, dtype=np.float32)

    assert rand_bot.act(dummy_obs, empty_mask) == end_turn_id
    assert heur_bot.act(dummy_obs, empty_mask) == end_turn_id


def test_strategy_heuristics_choose_different_profiles():
    mask = np.zeros(CATALOG.size(), dtype=np.int8)
    road_id = CATALOG.encode(Action("PLACE_ROAD", (0,)))
    dev_id = CATALOG.encode(Action("BUY_DEV_CARD"))
    end_turn_id = CATALOG.encode(Action("END_TURN"))
    mask[road_id] = 1
    mask[dev_id] = 1
    mask[end_turn_id] = 1
    dummy_obs = np.zeros(4, dtype=np.float32)

    road_bot = StrategyHeuristicAgent(strategy="road_builder", seed=11)
    ows_bot = StrategyHeuristicAgent(strategy="full_ows", seed=22)
    assert road_bot.act(dummy_obs, mask) == road_id
    assert ows_bot.act(dummy_obs, mask) == dev_id


def test_meta_strategy_agent_handles_empty_mask():
    empty_mask = np.zeros(CATALOG.size(), dtype=np.int8)
    end_turn_id = CATALOG.encode(Action("END_TURN"))
    bot = MetaStrategyHeuristicAgent(seed=3)
    dummy_obs = np.zeros(4, dtype=np.float32)
    assert bot.act(dummy_obs, empty_mask) == end_turn_id
