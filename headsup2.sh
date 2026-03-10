source .venv/bin/activate && python - <<'PY'
import math
import numpy as np
import torch

from catan_rl.env import CatanEnv
from catan_rl.eval import tournament
from catan_rl.training.ppo import PolicyValueNet
from catan_rl.training.wrappers import PolicyAgent

CKPT_CHAMP = "artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt"
CKPT_META  = "artifacts_schedule_realboard_vs_meta_mix_v1/best_checkpoint.pt"  # change if desired

GAMES_PER_SEAT = 50
BASE_SEED = 19900
ENV_KW = dict(
    max_main_actions_per_turn=10,
    allow_player_trade=True,
    trade_action_mode="guided",
    max_player_trade_proposals_per_turn=1,
)

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def load_agent(path):
    env = CatanEnv(seed=0, **ENV_KW)
    obs, info = env.reset(seed=0)
    model = PolicyValueNet(obs_dim=obs.shape[0], action_dim=info["action_mask"].shape[0])
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return PolicyAgent(model)

champ = load_agent(CKPT_CHAMP)
meta = load_agent(CKPT_META)

def head_to_head(primary, other, label):
    strict_rates = []
    win_rates = []
    total_terminal = 0
    total_truncated = 0
    strict_wins = 0.0
    total_games = 0

    print(f"\n=== {label} ===")
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
        sr = float(stats["strict_win_rates"][seat])
        wr = float(stats["win_rates"][seat])
        strict_rates.append(sr)
        win_rates.append(wr)

        seat_terminal = int(stats["terminal_games"])
        seat_trunc = int(stats["truncated_games"])
        total_terminal += seat_terminal
        total_truncated += seat_trunc

        # strict_wins is float but should be integer-like for terminal single winners
        seat_strict_wins = float(stats["strict_wins"][seat])
        strict_wins += seat_strict_wins
        total_games += GAMES_PER_SEAT

        print(
            f"seat {seat}: strict={sr:.3f} win={wr:.3f} "
            f"terminal={seat_terminal} truncated={seat_trunc} strict_wins={seat_strict_wins:.1f}"
        )

    mean_strict = float(np.mean(strict_rates))
    mean_win = float(np.mean(win_rates))
    lo, hi = wilson_ci(strict_wins, total_games)

    print(f"{label} mean: strict={mean_strict:.3f} win={mean_win:.3f}")
    print(
        f"{label} aggregate: strict_wins={strict_wins:.1f}/{total_games} "
        f"95%CI=[{lo:.3f}, {hi:.3f}] terminal={total_terminal} truncated={total_truncated}"
    )

head_to_head(champ, meta, "CHAMP_vs_META")
head_to_head(meta, champ, "META_vs_CHAMP")
PY
