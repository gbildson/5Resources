from catan_rl.training.self_play import _truncation_reward_for_player
from catan_rl.env import CatanEnv


def test_truncation_reward_gives_plus_one_to_leader_and_minus_one_to_nonleader():
    env = CatanEnv(seed=17)
    env.reset(seed=17)
    env.state.actual_vp[:] = [8, 6, 8, 5]

    assert _truncation_reward_for_player(env, 0) == 1.0
    assert _truncation_reward_for_player(env, 2) == 1.0
    assert _truncation_reward_for_player(env, 1) == -1.0
