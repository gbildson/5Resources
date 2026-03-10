from catan_rl.actions import Action, CATALOG
from catan_rl.constants import Phase
from catan_rl.env import CatanEnv


def test_illegal_action_step_returns_action_mask_in_info():
    env = CatanEnv(seed=123)
    env.reset(seed=123)
    env.state.setup_round = 0
    env.state.phase = Phase.PRE_ROLL
    env.state.current_player = 0

    illegal_id = CATALOG.encode(Action("END_TURN"))
    res = env.step(illegal_id)

    assert res.info["illegal_action"] is True
    assert "action_mask" in res.info
    assert res.info["action_mask"].shape[0] == CATALOG.size()
