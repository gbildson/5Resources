import numpy as np

from catan_rl.encoding import encode_observation
from catan_rl.state import new_game_state


ENGINEERED_FEATURE_DIM = 75
TRADE_BLOCK_DIM = 15


def test_trade_fields_are_relative_to_current_player():
    state = new_game_state(seed=7)
    state.current_player = 2
    state.trade_offer_give[:] = np.asarray([1, 0, 2, 0, 0], dtype=np.int64)
    state.trade_offer_want[:] = np.asarray([0, 1, 0, 0, 1], dtype=np.int64)
    state.trade_proposer = 1
    state.trade_responses[:] = np.asarray([0, 1, 2, 1], dtype=np.int64)

    obs = encode_observation(state, full_info=True)
    trade_tail = obs[-(ENGINEERED_FEATURE_DIM + TRADE_BLOCK_DIM) : -ENGINEERED_FEATURE_DIM]

    expected_offer = np.asarray([1, 0, 2, 0, 0, 0, 1, 0, 0, 1], dtype=np.float32) / 4.0
    expected_proposer = np.asarray([1.0], dtype=np.float32)  # (1 - 2) % 4 == 3, then / 3 => 1.0
    expected_responses = np.asarray([2, 1, 0, 1], dtype=np.float32) / 2.0

    np.testing.assert_allclose(trade_tail[:10], expected_offer)
    np.testing.assert_allclose(trade_tail[10:11], expected_proposer)
    np.testing.assert_allclose(trade_tail[11:], expected_responses)


def test_trade_proposer_none_is_preserved():
    state = new_game_state(seed=9)
    state.current_player = 3
    state.trade_proposer = -1
    obs = encode_observation(state, full_info=True)
    trade_tail = obs[-(ENGINEERED_FEATURE_DIM + TRADE_BLOCK_DIM) : -ENGINEERED_FEATURE_DIM]
    assert np.isclose(trade_tail[10], -1.0 / 3.0)


def test_observation_includes_compact_engineered_block():
    state = new_game_state(seed=11)
    obs = encode_observation(state, full_info=True)
    assert obs.shape[0] == 1301
