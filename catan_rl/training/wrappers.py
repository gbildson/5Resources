"""Policy wrappers to use torch models as evaluation agents."""

from __future__ import annotations

import numpy as np
import torch

from ..bots import Agent


class PolicyAgent(Agent):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model(obs_t)
            logits = logits.masked_fill(mask_t <= 0, -1e9)
            action = int(torch.argmax(logits, dim=-1).item())
        return action
