source .venv/bin/activate && python - <<'PY'
import numpy as np
import torch

from catan_rl.env import CatanEnv
from catan_rl.eval import tournament
from catan_rl.training.ppo import PolicyValueNet
from catan_rl.training.wrappers import PolicyAgent

CKPT_CHAMP = "artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt"
CKPT_META  = "artifacts_schedule_realboard_vs_meta_mix_v1/best_checkpoint.pt"  # or v2 best

GAMES_PER_SEAT = 20
BASE_SEED = 9900
ENV_KW = dict(
    max_main_actions_per_turn=10,
    allow_player_trade=True,
    trade_action_mode="guided",
    max_player_trade_proposals_per_turn=1,
)

def load_agent(path):
    env = CatanEnv(seed=0, **ENV_KW)
    obs, info = env.reset(seed=0)
    model = PolicyValueNet(obs_dim=obs.shape[0], action_dim=info["action_mask"].shape[0])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return PolicyAgent(model)

champ = load_agent(CKPT_CHAMP)
meta = load_agent(CKPT_META)

def seat_eval(primary, other, label):
    strict_rates = []
    win_rates = []
    for seat in range(4):
        agents = [other, other, other, other]
        agents[seat] = primary
        stats = tournament(
            agents,
            num_games=GAMES_PER_SEAT,
            base_seed=BASE_SEED + seat * 1000,
            max_steps=2200,
            env_kwargs=ENV_KW,
        )
        strict = stats["strict_win_rates"][seat]
        win = stats["win_rates"][seat]
        strict_rates.append(strict)
        win_rates.append(win)
        print(f"{label} seat {seat}: strict={strict:.3f} win={win:.3f}")
    print(f"{label} mean: strict={np.mean(strict_rates):.3f} win={np.mean(win_rates):.3f}\n")

seat_eval(champ, meta, "CHAMP_vs_META")
seat_eval(meta, champ, "META_vs_CHAMP")
PY
