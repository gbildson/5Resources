"""Behavior cloning utilities for warm-starting PPO policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..bots import HeuristicAgent
from ..env import CatanEnv


@dataclass
class BCDataset:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray


def collect_bc_dataset(steps: int, seed: int, env_kwargs: dict | None = None) -> BCDataset:
    env = CatanEnv(seed=seed, **(env_kwargs or {}))
    teacher = HeuristicAgent(seed=seed + 1)
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]

    obs_buf = []
    mask_buf = []
    act_buf = []
    for _ in range(steps):
        action = teacher.act(obs, mask)
        obs_buf.append(obs)
        mask_buf.append(mask)
        act_buf.append(action)
        res = env.step(action)
        obs = res.obs
        mask = res.info["action_mask"]
        if res.done:
            obs, info = env.reset()
            mask = info["action_mask"]

    return BCDataset(
        obs=np.asarray(obs_buf, dtype=np.float32),
        masks=np.asarray(mask_buf, dtype=np.float32),
        actions=np.asarray(act_buf, dtype=np.int64),
    )


def pretrain_policy_with_bc(
    model,
    dataset: BCDataset,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> dict:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n = dataset.obs.shape[0]
    idx = np.arange(n)
    logs = {"bc_loss": 0.0, "bc_acc": 0.0}
    updates = 0

    model.train()
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            mb = idx[i : i + batch_size]
            obs_t = torch.as_tensor(dataset.obs[mb], dtype=torch.float32)
            mask_t = torch.as_tensor(dataset.masks[mb], dtype=torch.float32)
            act_t = torch.as_tensor(dataset.actions[mb], dtype=torch.int64)

            logits, _ = model(obs_t)
            logits = logits.masked_fill(mask_t <= 0, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, act_t)

            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == act_t).float().mean().item()
            logs["bc_loss"] += float(loss.item())
            logs["bc_acc"] += float(acc)
            updates += 1

    for k in logs:
        logs[k] /= max(1, updates)
    logs["dataset_size"] = int(n)
    return logs
