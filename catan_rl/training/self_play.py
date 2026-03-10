"""Self-play rollout collection and PPO training loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..bots import RandomLegalAgent
from ..env import CatanEnv
from ..eval import tournament
from .ppo import PPOConfig, PPOTrainer, PolicyValueNet, compute_gae


@dataclass
class TrainConfig:
    total_updates: int = 50
    rollout_steps: int = 2048
    eval_every: int = 5
    seed: int = 7
    max_episode_steps: int = 2000
    truncation_leader_reward: bool = True
    reward_shaping_vp: float = 0.0
    reward_shaping_resource: float = 0.0


def _truncation_reward_for_player(env: CatanEnv, player: int) -> float:
    top_vp = int(env.state.actual_vp.max(initial=0))
    leaders = set(int(p) for p in np.flatnonzero(env.state.actual_vp == top_vp))
    return 1.0 if player in leaders else -1.0


def collect_rollout(
    env: CatanEnv,
    trainer: PPOTrainer,
    steps: int,
    max_episode_steps: int = 2000,
    truncation_leader_reward: bool = True,
    reward_shaping_vp: float = 0.0,
    reward_shaping_resource: float = 0.0,
) -> dict:
    obs, info = env.reset()
    mask = info["action_mask"]
    buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones", "masks")}
    episode_steps = 0
    for _ in range(steps):
        player = env.state.current_player
        prev_public_vp = int(env.state.public_vp[player])
        prev_resource_total = int(env.state.resource_total[player])
        action, logp, value = trainer.act(obs, mask)
        res = env.step(action)
        reward = float(res.reward)
        delta_vp = int(env.state.public_vp[player]) - prev_public_vp
        delta_resources = int(env.state.resource_total[player]) - prev_resource_total
        reward += reward_shaping_vp * float(delta_vp)
        reward += reward_shaping_resource * float(delta_resources)

        done = bool(res.done)
        episode_steps += 1
        if not done and episode_steps >= max_episode_steps:
            done = True
            if truncation_leader_reward:
                reward += _truncation_reward_for_player(env, player)

        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["logp"].append(logp)
        buf["values"].append(value)
        buf["rewards"].append(reward)
        buf["dones"].append(float(done))
        buf["masks"].append(mask)
        if done:
            obs, info = env.reset()
            mask = info["action_mask"]
            episode_steps = 0
        else:
            obs = res.obs
            mask = res.info["action_mask"]
    for k in ("obs", "masks"):
        buf[k] = np.asarray(buf[k], dtype=np.float32)
    for k in ("actions",):
        buf[k] = np.asarray(buf[k], dtype=np.int64)
    for k in ("logp", "values", "rewards", "dones"):
        buf[k] = np.asarray(buf[k], dtype=np.float32)
    adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], trainer.cfg.gamma, trainer.cfg.lam)
    buf["advantages"] = adv
    buf["returns"] = ret
    return buf


def train_self_play(config: TrainConfig, ppo_cfg: PPOConfig) -> tuple[PolicyValueNet, list[dict]]:
    rng = np.random.default_rng(config.seed)
    env = CatanEnv(seed=config.seed)
    obs, _ = env.reset(seed=config.seed)
    model = PolicyValueNet(obs_dim=obs.shape[0], action_dim=env.action_mask().shape[0])
    trainer = PPOTrainer(model, ppo_cfg)
    history: list[dict] = []

    for update in range(1, config.total_updates + 1):
        env.seed = int(rng.integers(0, 1_000_000))
        batch = collect_rollout(
            env,
            trainer,
            config.rollout_steps,
            max_episode_steps=config.max_episode_steps,
            truncation_leader_reward=config.truncation_leader_reward,
            reward_shaping_vp=config.reward_shaping_vp,
            reward_shaping_resource=config.reward_shaping_resource,
        )
        stats = trainer.update(batch)
        row = {"update": update, **stats}
        if update % config.eval_every == 0:
            # Quick baseline benchmark.
            from .wrappers import PolicyAgent

            policy_agent = PolicyAgent(model)
            opp = RandomLegalAgent(seed=config.seed + update)
            eval_stats = tournament([policy_agent, opp, opp, opp], num_games=20, base_seed=config.seed + update * 100)
            row.update({"eval_win_rate": eval_stats["win_rates"][0], "eval_avg_turns": eval_stats["avg_turns"]})
        history.append(row)
    return model, history


def save_checkpoint(model: PolicyValueNet, path: str) -> None:
    torch.save(model.state_dict(), path)
