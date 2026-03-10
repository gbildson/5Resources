"""Masked PPO implementation for the Catan environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for PPO training. Install torch first.") from exc


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 256
    minibatch_size: int = 64


class PPOTrainer:
    def __init__(self, model: PolicyValueNet, config: PPOConfig):
        self.model = model
        self.cfg = config
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(mask <= 0, -1e9)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> tuple[int, float, float]:
        self.model.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
            logits, value = self.model(obs_t)
            logits = self._masked_logits(logits, mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), float(value.item())

    def update(self, batch: dict) -> dict:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        act = torch.as_tensor(batch["actions"], dtype=torch.int64)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
        ret = torch.as_tensor(batch["returns"], dtype=torch.float32)
        adv = torch.as_tensor(batch["advantages"], dtype=torch.float32)
        mask = torch.as_tensor(batch["masks"], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        logs = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        self.model.train()
        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx)
            for i in range(0, n, self.cfg.minibatch_size):
                mb = idx[i : i + self.cfg.minibatch_size]
                logits, value = self.model(obs[mb])
                logits = self._masked_logits(logits, mask[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - old_logp[mb])
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                policy_loss = -torch.min(ratio * adv[mb], clipped * adv[mb]).mean()
                value_loss = ((value - ret[mb]) ** 2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                logs["policy_loss"] += float(policy_loss.item())
                logs["value_loss"] += float(value_loss.item())
                logs["entropy"] += float(entropy.item())
                updates += 1

        for k in logs:
            logs[k] /= max(1, updates)
        return logs


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t]
        nextvalue = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values[: len(advantages)]
    return advantages, returns
