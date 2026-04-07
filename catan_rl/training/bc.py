"""Behavior cloning utilities for warm-starting PPO policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch

from ..bots import HeuristicAgent
from ..env import CatanEnv
from ..search import SearchDecision, choose_search_action, export_search_decisions_jsonl
from ..strategy_archetypes import strategy_target_vector
from ..strategy_metrics import AUX_LABEL_KEYS, strategic_aux_target_weights, strategic_aux_targets, strategic_evaluator_snapshot


@dataclass
class BCDataset:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    aux_targets: dict[str, np.ndarray]
    aux_target_weights: dict[str, np.ndarray]
    strategy_targets: np.ndarray


@dataclass
class SearchDistillDataset:
    obs: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    aux_targets: dict[str, np.ndarray]
    aux_target_weights: dict[str, np.ndarray]
    strategy_targets: np.ndarray
    metadata: dict


def collect_bc_dataset(steps: int, seed: int, env_kwargs: dict | None = None) -> BCDataset:
    env = CatanEnv(seed=seed, **(env_kwargs or {}))
    teacher = HeuristicAgent(seed=seed + 1)
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]

    obs_buf = []
    mask_buf = []
    act_buf = []
    aux_buf = {k: [] for k in AUX_LABEL_KEYS}
    aux_weight_buf = {k: [] for k in AUX_LABEL_KEYS}
    strategy_buf = []
    for _ in range(steps):
        state_before = env.state.copy()
        player = int(state_before.current_player)
        snap = strategic_evaluator_snapshot(state_before, player)
        aux = strategic_aux_targets(snap, state_before, player)
        aux_weights = strategic_aux_target_weights(state_before, player)
        strategy_buf.append(strategy_target_vector(state_before, player))
        action = teacher.act(obs, mask)
        obs_buf.append(obs)
        mask_buf.append(mask)
        act_buf.append(action)
        for k in AUX_LABEL_KEYS:
            aux_buf[k].append(float(aux[k]))
            aux_weight_buf[k].append(float(aux_weights[k]))
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
        aux_targets={k: np.asarray(v, dtype=np.float32) for k, v in aux_buf.items()},
        aux_target_weights={k: np.asarray(v, dtype=np.float32) for k, v in aux_weight_buf.items()},
        strategy_targets=np.asarray(strategy_buf, dtype=np.float32),
    )


def _model_action(model, obs: np.ndarray, mask: np.ndarray) -> int:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(obs_t)
        logits = logits.masked_fill(mask_t <= 0, -1e9)
        return int(torch.argmax(logits, dim=-1).item())


def collect_search_distill_dataset(
    *,
    steps: int,
    seed: int,
    env_kwargs: dict | None = None,
    teacher_model=None,
    setup_search: bool = True,
    robber_search: bool = True,
    top_k: int = 5,
    setup_rollout_steps: int = 0,
    robber_rollout_steps: int = 0,
    export_decisions_jsonl: str | None = None,
) -> SearchDistillDataset:
    """Collect offline labels from selective search for setup + robber phases."""
    env = CatanEnv(seed=seed, **(env_kwargs or {}))
    fallback = HeuristicAgent(seed=seed + 17)
    rng = np.random.default_rng(seed + 23)

    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]

    obs_buf = []
    mask_buf = []
    act_buf = []
    aux_buf = {k: [] for k in AUX_LABEL_KEYS}
    aux_weight_buf = {k: [] for k in AUX_LABEL_KEYS}
    strategy_buf = []
    decisions: list[SearchDecision] = []
    while len(obs_buf) < int(steps):
        state_before = env.state.copy()
        player = int(state_before.current_player)

        d = choose_search_action(
            env,
            obs,
            mask,
            setup_search=bool(setup_search),
            robber_search=bool(robber_search),
            top_k=int(top_k),
            setup_rollout_steps=int(setup_rollout_steps),
            robber_rollout_steps=int(robber_rollout_steps),
            rng=rng,
        )
        if d is not None:
            snap = strategic_evaluator_snapshot(state_before, player)
            aux = strategic_aux_targets(snap, state_before, player)
            aux_weights = strategic_aux_target_weights(state_before, player)
            obs_buf.append(obs)
            mask_buf.append(mask)
            act_buf.append(int(d.selected_action))
            strategy_buf.append(strategy_target_vector(state_before, player))
            for k in AUX_LABEL_KEYS:
                aux_buf[k].append(float(aux[k]))
                aux_weight_buf[k].append(float(aux_weights[k]))
            decisions.append(d)
            action = int(d.selected_action)
        else:
            action = _model_action(teacher_model, obs, mask) if teacher_model is not None else int(fallback.act(obs, mask))

        res = env.step(int(action))
        if res.done:
            obs, info = env.reset()
            mask = info["action_mask"]
        else:
            obs = res.obs
            mask = res.info["action_mask"]

    if export_decisions_jsonl:
        export_search_decisions_jsonl(export_decisions_jsonl, decisions)

    return SearchDistillDataset(
        obs=np.asarray(obs_buf, dtype=np.float32),
        masks=np.asarray(mask_buf, dtype=np.float32),
        actions=np.asarray(act_buf, dtype=np.int64),
        aux_targets={k: np.asarray(v, dtype=np.float32) for k, v in aux_buf.items()},
        aux_target_weights={k: np.asarray(v, dtype=np.float32) for k, v in aux_weight_buf.items()},
        strategy_targets=np.asarray(strategy_buf, dtype=np.float32),
        metadata={
            "steps": int(steps),
            "setup_search": bool(setup_search),
            "robber_search": bool(robber_search),
            "top_k": int(top_k),
            "setup_rollout_steps": int(setup_rollout_steps),
            "robber_rollout_steps": int(robber_rollout_steps),
            "decision_count": int(len(decisions)),
        },
    )


def save_search_distill_dataset(path: str | Path, dataset: SearchDistillDataset) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        obs=dataset.obs,
        masks=dataset.masks,
        actions=dataset.actions,
        strategy_targets=dataset.strategy_targets,
        **{f"auxw_{k}": v for k, v in dataset.aux_target_weights.items()},
        **{f"aux_{k}": v for k, v in dataset.aux_targets.items()},
    )
    meta_path = Path(str(p) + ".meta.json")
    meta_path.write_text(json.dumps(dataset.metadata, indent=2), encoding="utf-8")


def pretrain_policy_with_search_distill(
    *,
    model,
    dataset: SearchDistillDataset,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    aux_coef: float = 0.0,
    strategy_coef: float = 0.0,
) -> dict:
    return pretrain_policy_with_bc(
        model=model,
        dataset=BCDataset(
            obs=dataset.obs,
            masks=dataset.masks,
            actions=dataset.actions,
            aux_targets=dataset.aux_targets,
            aux_target_weights=dataset.aux_target_weights,
            strategy_targets=dataset.strategy_targets,
        ),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        aux_coef=aux_coef,
        strategy_coef=strategy_coef,
    )


def pretrain_policy_with_bc(
    model,
    dataset: BCDataset,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    aux_coef: float = 0.0,
    strategy_coef: float = 0.0,
) -> dict:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n = dataset.obs.shape[0]
    idx = np.arange(n)
    logs = {"bc_loss": 0.0, "bc_acc": 0.0, "bc_aux_loss": 0.0, "bc_strategy_loss": 0.0}
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
            aux_loss = torch.zeros((), dtype=torch.float32, device=obs_t.device)
            strategy_loss = torch.zeros((), dtype=torch.float32, device=obs_t.device)
            if aux_coef > 0.0 and hasattr(model, "predict_aux"):
                pred_aux = model.predict_aux(obs_t)
                losses = []
                for k in AUX_LABEL_KEYS:
                    tgt = torch.as_tensor(dataset.aux_targets[k][mb], dtype=torch.float32, device=obs_t.device)
                    tgt_weight = torch.as_tensor(dataset.aux_target_weights[k][mb], dtype=torch.float32, device=obs_t.device)
                    if k in pred_aux:
                        denom = torch.clamp(tgt_weight.sum(), min=1.0)
                        losses.append((((pred_aux[k] - tgt) ** 2) * tgt_weight).sum() / denom)
                if losses:
                    aux_loss = torch.stack(losses).mean()
                    loss = loss + float(aux_coef) * aux_loss
            if strategy_coef > 0.0 and hasattr(model, "predict_strategy"):
                pred_strategy = model.predict_strategy(obs_t)
                tgt_strategy = torch.as_tensor(dataset.strategy_targets[mb], dtype=torch.float32, device=obs_t.device)
                strategy_loss = -(tgt_strategy * torch.log(torch.clamp(pred_strategy, min=1e-8))).sum(dim=-1).mean()
                loss = loss + float(strategy_coef) * strategy_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == act_t).float().mean().item()
            logs["bc_loss"] += float(loss.item())
            logs["bc_acc"] += float(acc)
            logs["bc_aux_loss"] += float(aux_loss.item())
            logs["bc_strategy_loss"] += float(strategy_loss.item())
            updates += 1

    for k in logs:
        logs[k] /= max(1, updates)
    logs["dataset_size"] = int(n)
    logs["bc_aux_coef"] = float(aux_coef)
    logs["bc_strategy_coef"] = float(strategy_coef)
    return logs
