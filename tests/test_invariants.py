import numpy as np

from catan_rl.bots import RandomLegalAgent
from catan_rl.env import CatanEnv
from catan_rl.rules import check_invariants


def test_state_invariants_hold_during_random_play():
    env = CatanEnv(seed=123)
    agent = RandomLegalAgent(seed=456)
    obs, info = env.reset(seed=123)
    mask = info["action_mask"]

    for _ in range(400):
        action = agent.act(obs, mask)
        res = env.step(action)
        errs = check_invariants(env.state)
        assert not errs, f"Invariant violations: {errs}"
        obs = res.obs
        mask = res.info["action_mask"]
        if res.done:
            obs, info = env.reset()
            mask = info["action_mask"]


def test_action_mask_has_at_least_one_legal_action():
    env = CatanEnv(seed=10)
    env.reset(seed=10)
    for _ in range(120):
        mask = env.action_mask()
        assert np.any(mask == 1)
        action = int(np.flatnonzero(mask)[0])
        env.step(action)
