import numpy as np

from catan_rl.env import CatanEnv


def test_longest_road_not_awarded_on_new_tie_without_previous_holder():
    env = CatanEnv(seed=1)
    env.reset(seed=1)
    env.state.has_longest_road[:] = 0
    env._road_lengths = lambda: np.array([5, 5, 2, 1], dtype=np.int64)  # type: ignore[method-assign]

    env._update_vp_and_achievements()

    assert env.state.has_longest_road.tolist() == [0, 0, 0, 0]


def test_longest_road_previous_holder_keeps_on_tie():
    env = CatanEnv(seed=2)
    env.reset(seed=2)
    env.state.has_longest_road[:] = np.array([1, 0, 0, 0], dtype=np.int64)
    env._road_lengths = lambda: np.array([6, 6, 3, 2], dtype=np.int64)  # type: ignore[method-assign]

    env._update_vp_and_achievements()

    assert env.state.has_longest_road.tolist() == [1, 0, 0, 0]


def test_longest_road_transfers_on_unique_higher_contender():
    env = CatanEnv(seed=3)
    env.reset(seed=3)
    env.state.has_longest_road[:] = np.array([1, 0, 0, 0], dtype=np.int64)
    env._road_lengths = lambda: np.array([5, 6, 1, 1], dtype=np.int64)  # type: ignore[method-assign]

    env._update_vp_and_achievements()

    assert env.state.has_longest_road.tolist() == [0, 1, 0, 0]


def test_largest_army_previous_holder_keeps_on_tie():
    env = CatanEnv(seed=4)
    env.reset(seed=4)
    env.state.has_largest_army[:] = np.array([0, 1, 0, 0], dtype=np.int64)
    env.state.knights_played[:] = np.array([2, 3, 3, 1], dtype=np.int64)
    env._road_lengths = lambda: np.zeros(4, dtype=np.int64)  # type: ignore[method-assign]

    env._update_vp_and_achievements()

    assert env.state.has_largest_army.tolist() == [0, 1, 0, 0]
