"""Selective search modules for setup and robber decisions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .actions import CATALOG
from .bots import HeuristicAgent
from .constants import Building, Phase
from .env import CatanEnv
from .strategy_metrics import (
    anti_leader_rob_quality,
    robber_block_quality,
    road_frontier_metrics,
    setup_choice_metrics,
    setup_settlement_score,
)


@dataclass
class RankedAction:
    action_id: int
    score: float
    features: dict[str, float]


@dataclass
class SearchDecision:
    phase: int
    player: int
    selected_action: int
    top_k: list[RankedAction]


def _clone_env(env: CatanEnv, seed: int | None = None) -> CatanEnv:
    sim = CatanEnv(
        seed=int(seed if seed is not None else env.seed),
        full_info_obs=bool(env.full_info_obs),
        max_main_actions_per_turn=env.max_main_actions_per_turn,
        allow_player_trade=env.allow_player_trade,
        trade_action_mode=env.trade_action_mode,
        max_player_trade_proposals_per_turn=env.max_player_trade_proposals_per_turn,
    )
    sim.state = env.state.copy()
    sim._discard_remaining = env._discard_remaining.copy()
    sim._discard_roller = int(env._discard_roller)
    sim._main_actions_this_turn = int(env._main_actions_this_turn)
    sim._trade_proposals_this_turn = int(env._trade_proposals_this_turn)
    sim._last_mask = None if env._last_mask is None else env._last_mask.copy()
    return sim


def _shallow_rollout_value(
    env: CatanEnv,
    first_action: int,
    perspective_player: int,
    steps: int,
    rng: np.random.Generator,
) -> float:
    sim = _clone_env(env, seed=int(rng.integers(0, 1_000_000)))
    agent = HeuristicAgent(seed=int(rng.integers(0, 1_000_000)))
    res = sim.step(int(first_action))
    if res.done:
        return float(sim.state.actual_vp[perspective_player] - np.mean(sim.state.actual_vp))
    obs = res.obs
    mask = res.info["action_mask"]
    for _ in range(max(0, int(steps))):
        action = int(agent.act(obs, mask))
        res = sim.step(action)
        if res.done:
            break
        obs = res.obs
        mask = res.info["action_mask"]
    centered_vp = float(sim.state.actual_vp[perspective_player] - np.mean(sim.state.actual_vp))
    return float(np.clip(centered_vp / 10.0, -1.0, 1.0))


def rank_setup_actions(
    env: CatanEnv,
    mask: np.ndarray,
    *,
    top_k: int = 5,
    rollout_steps: int = 0,
    rollout_weight: float = 0.25,
    rng: np.random.Generator | None = None,
) -> list[RankedAction]:
    state = env.state
    player = int(state.current_player)
    legal_ids = np.flatnonzero(mask > 0)
    out: list[RankedAction] = []
    rrng = rng if rng is not None else np.random.default_rng()
    for action_id in legal_ids:
        action = CATALOG.decode(int(action_id))
        if int(state.phase) == int(Phase.SETUP_SETTLEMENT) and action.kind == "PLACE_SETTLEMENT":
            (vertex,) = action.params
            metrics = setup_choice_metrics(state, player, int(vertex))
            base_score = float(setup_settlement_score(state, player, int(vertex)))
            percentile = float(metrics.get("percentile", 0.0))
            score = base_score + 2.0 * percentile
            if rollout_steps > 0:
                lookahead = _shallow_rollout_value(env, int(action_id), player, int(rollout_steps), rrng)
                score += float(rollout_weight) * float(lookahead)
            out.append(
                RankedAction(
                    action_id=int(action_id),
                    score=float(score),
                    features={
                        "base_score": float(base_score),
                        "percentile": percentile,
                        "rollout": float((score - base_score - 2.0 * percentile) / max(rollout_weight, 1e-8))
                        if rollout_steps > 0
                        else 0.0,
                    },
                )
            )
        elif int(state.phase) == int(Phase.SETUP_ROAD) and action.kind == "PLACE_ROAD":
            (edge,) = action.params
            m = road_frontier_metrics(state, player, int(edge))
            base_score = float(2.0 * m["connected_gain"] + 1.25 * m["best_site_roads_delta"])
            score = base_score
            if rollout_steps > 0:
                lookahead = _shallow_rollout_value(env, int(action_id), player, int(rollout_steps), rrng)
                score += float(rollout_weight) * float(lookahead)
            out.append(
                RankedAction(
                    action_id=int(action_id),
                    score=float(score),
                    features={
                        "connected_gain": float(m["connected_gain"]),
                        "best_site_roads_delta": float(m["best_site_roads_delta"]),
                    },
                )
            )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: max(1, int(top_k))]


def _retaliation_risk(state, acting_player: int, hex_id: int) -> float:
    risk = 0.0
    for v in state.topology.hex_to_vertices[int(hex_id)]:
        owner = int(state.vertex_owner[int(v)])
        if owner < 0 or owner == int(acting_player):
            continue
        b = int(state.vertex_building[int(v)])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        risk += mult * float(state.hex_pip_count[int(hex_id)])
    return float(np.clip(risk / 12.0, 0.0, 1.0))


def rank_robber_actions(
    env: CatanEnv,
    mask: np.ndarray,
    *,
    top_k: int = 5,
    include_retaliation_risk: bool = True,
    rollout_steps: int = 0,
    rollout_weight: float = 0.2,
    threat_dev_card_weight: float = 0.7,
    rng: np.random.Generator | None = None,
) -> list[RankedAction]:
    state = env.state
    player = int(state.current_player)
    legal_ids = np.flatnonzero(mask > 0)
    out: list[RankedAction] = []
    rrng = rng if rng is not None else np.random.default_rng()
    for action_id in legal_ids:
        action = CATALOG.decode(int(action_id))
        if int(state.phase) == int(Phase.MOVE_ROBBER) and action.kind == "MOVE_ROBBER":
            (hex_id,) = action.params
            block = robber_block_quality(state, player, int(hex_id), float(threat_dev_card_weight))
            risk = _retaliation_risk(state, player, int(hex_id)) if include_retaliation_risk else 0.0
            score = float(block["quality_score"]) - 0.25 * float(risk)
            if rollout_steps > 0:
                score += float(rollout_weight) * _shallow_rollout_value(
                    env, int(action_id), player, int(rollout_steps), rrng
                )
            out.append(
                RankedAction(
                    action_id=int(action_id),
                    score=float(score),
                    features={
                        "block_quality": float(block["quality_score"]),
                        "retaliation_risk": float(risk),
                    },
                )
            )
        elif int(state.phase) == int(Phase.ROB_PLAYER) and action.kind == "ROB_PLAYER":
            (target_player,) = action.params
            q = anti_leader_rob_quality(state, player, int(target_player), float(threat_dev_card_weight))
            target_cards = float(q["target_resource_total"])
            score = float(q["quality_score"]) + 0.1 * float(np.clip(target_cards / 7.0, 0.0, 1.0))
            out.append(
                RankedAction(
                    action_id=int(action_id),
                    score=float(score),
                    features={
                        "target_is_leader": float(1.0 if bool(q["robbed_leader"]) else 0.0),
                        "target_cards": target_cards,
                    },
                )
            )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: max(1, int(top_k))]


def choose_search_action(
    env: CatanEnv,
    obs: np.ndarray,
    mask: np.ndarray,
    *,
    setup_search: bool = True,
    robber_search: bool = True,
    top_k: int = 5,
    setup_rollout_steps: int = 0,
    robber_rollout_steps: int = 0,
    rng: np.random.Generator | None = None,
) -> SearchDecision | None:
    phase = int(env.state.phase)
    ranked: list[RankedAction] = []
    if setup_search and phase in (int(Phase.SETUP_SETTLEMENT), int(Phase.SETUP_ROAD)):
        ranked = rank_setup_actions(
            env,
            mask,
            top_k=int(top_k),
            rollout_steps=int(setup_rollout_steps),
            rng=rng,
        )
    elif robber_search and phase in (int(Phase.MOVE_ROBBER), int(Phase.ROB_PLAYER)):
        ranked = rank_robber_actions(
            env,
            mask,
            top_k=int(top_k),
            rollout_steps=int(robber_rollout_steps),
            rng=rng,
        )
    if not ranked:
        return None
    return SearchDecision(
        phase=phase,
        player=int(env.state.current_player),
        selected_action=int(ranked[0].action_id),
        top_k=ranked,
    )


def export_search_decisions_jsonl(path: str | Path, decisions: list[SearchDecision]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for d in decisions:
            row = {
                "phase": int(d.phase),
                "player": int(d.player),
                "selected_action": int(d.selected_action),
                "top_k": [
                    {"action_id": int(x.action_id), "score": float(x.score), "features": {k: float(v) for k, v in x.features.items()}}
                    for x in d.top_k
                ],
            }
            f.write(json.dumps(row) + "\n")

