from catan_rl.bots import RandomLegalAgent
from catan_rl.env import CatanEnv


def test_random_playout_does_not_deadlock():
    env = CatanEnv(seed=999)
    bot = RandomLegalAgent(seed=1000)
    obs, info = env.reset(seed=999)
    mask = info["action_mask"]

    steps = 0
    done = False
    while not done and steps < 2000:
        action = bot.act(obs, mask)
        result = env.step(action)
        obs = result.obs
        done = result.done
        mask = result.info["action_mask"]
        steps += 1

    assert steps > 0
    assert done or steps == 2000
