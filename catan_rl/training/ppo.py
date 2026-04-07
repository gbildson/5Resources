"""Masked PPO implementation for the Catan environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for PPO training. Install torch first.") from exc

from ..constants import NUM_PLAYERS, NUM_VERTICES, Phase
from ..encoding import observation_slices, phase_one_hot_slice
from ..actions import CATALOG
from ..graph_features import (
    global_feature_start_index,
    graph_observation_to_entities,
    player_feature_dim,
    topology_tensors_for_device,
)
from ..strategy_archetypes import ARCHETYPES
from ..strategy_metrics import AUX_LABEL_KEYS


def _build_aux_heads(hidden: int) -> nn.ModuleDict:
    return nn.ModuleDict({k: nn.Linear(hidden, 1) for k in AUX_LABEL_KEYS})


def _build_trade_action_masks() -> dict[str, torch.Tensor]:
    n = CATALOG.size()
    gen = torch.zeros((n,), dtype=torch.bool)
    resp = torch.zeros((n,), dtype=torch.bool)
    for i, a in enumerate(CATALOG.actions):
        if a.kind in {
            "TRADE_ADD_GIVE",
            "TRADE_ADD_WANT",
            "TRADE_REMOVE_GIVE",
            "TRADE_REMOVE_WANT",
            "PROPOSE_TRADE",
            "CANCEL_TRADE",
            "BANK_TRADE",
        }:
            gen[i] = True
        elif a.kind in {"ACCEPT_TRADE", "REJECT_TRADE"}:
            resp[i] = True
    return {"trade_generation_mask": gen, "trade_response_mask": resp}


def _phase_ids_from_obs(obs: torch.Tensor) -> torch.Tensor:
    phase_oh = obs[:, phase_one_hot_slice()]
    return torch.argmax(phase_oh, dim=-1)


def _apply_trade_split_logits(
    obs: torch.Tensor,
    logits: torch.Tensor,
    generation_logits: torch.Tensor,
    response_logits: torch.Tensor,
    trade_generation_mask: torch.Tensor,
    trade_response_mask: torch.Tensor,
) -> torch.Tensor:
    phase_ids = _phase_ids_from_obs(obs)
    main_rows = (phase_ids == int(Phase.MAIN)) | (phase_ids == int(Phase.TRADE_DRAFT))
    resp_rows = phase_ids == int(Phase.TRADE_PROPOSED)
    if torch.any(main_rows) and torch.any(trade_generation_mask):
        tmp = logits[main_rows]
        tmp[:, trade_generation_mask] = generation_logits[main_rows][:, trade_generation_mask]
        logits[main_rows] = tmp
    if torch.any(resp_rows) and torch.any(trade_response_mask):
        tmp = logits[resp_rows]
        tmp[:, trade_response_mask] = response_logits[resp_rows][:, trade_response_mask]
        logits[resp_rows] = tmp
    return logits


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
        self.trade_generation_head = nn.Linear(hidden, action_dim)
        self.trade_response_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.aux_heads = _build_aux_heads(hidden)
        for k, v in _build_trade_action_masks().items():
            self.register_buffer(k, v)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(obs)
        logits = self.policy_head(x)
        logits = _apply_trade_split_logits(
            obs,
            logits,
            self.trade_generation_head(x),
            self.trade_response_head(x),
            self.trade_generation_mask,
            self.trade_response_mask,
        )
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def predict_aux(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._encode(obs)
        return {k: head(x).squeeze(-1) for k, head in self.aux_heads.items()}


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.fc1(x))
        y = self.fc2(y)
        return self.act(x + y)


class ResidualPolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 512, blocks: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden)
        self.input_act = nn.ReLU()
        self.blocks = nn.ModuleList(ResidualMLPBlock(hidden) for _ in range(max(1, blocks)))
        self.policy_head = nn.Linear(hidden, action_dim)
        self.trade_generation_head = nn.Linear(hidden, action_dim)
        self.trade_response_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.aux_heads = _build_aux_heads(hidden)
        for k, v in _build_trade_action_masks().items():
            self.register_buffer(k, v)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_act(self.input_layer(obs))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(obs)
        logits = self.policy_head(x)
        logits = _apply_trade_split_logits(
            obs,
            logits,
            self.trade_generation_head(x),
            self.trade_response_head(x),
            self.trade_generation_mask,
            self.trade_response_mask,
        )
        value = self.value_head(x).squeeze(-1)
        return logits, value

    def predict_aux(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._encode(obs)
        return {k: head(x).squeeze(-1) for k, head in self.aux_heads.items()}


class PhaseAwareResidualPolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 512, blocks: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden)
        self.input_act = nn.ReLU()
        self.blocks = nn.ModuleList(ResidualMLPBlock(hidden) for _ in range(max(1, blocks)))
        self.phase_heads = nn.ModuleDict(
            {
                "setup": nn.Linear(hidden, action_dim),
                "main": nn.Linear(hidden, action_dim),
                "robber": nn.Linear(hidden, action_dim),
                "trade": nn.Linear(hidden, action_dim),
                "dev": nn.Linear(hidden, action_dim),
            }
        )
        self.trade_generation_head = nn.Linear(hidden, action_dim)
        self.trade_response_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.aux_heads = _build_aux_heads(hidden)
        self._phase_slice = phase_one_hot_slice()
        for k, v in _build_trade_action_masks().items():
            self.register_buffer(k, v)
        self._phase_to_head = {
            int(Phase.SETUP_SETTLEMENT): "setup",
            int(Phase.SETUP_ROAD): "setup",
            int(Phase.PRE_ROLL): "main",
            int(Phase.DICE_ROLLED): "main",
            int(Phase.DISCARD): "main",
            int(Phase.MOVE_ROBBER): "robber",
            int(Phase.ROB_PLAYER): "robber",
            int(Phase.MAIN): "main",
            int(Phase.TRADE_DRAFT): "trade",
            int(Phase.TRADE_PROPOSED): "trade",
            int(Phase.YEAR_OF_PLENTY): "dev",
            int(Phase.MONOPOLY): "dev",
            int(Phase.ROAD_BUILDING): "main",
            int(Phase.GAME_OVER): "main",
        }

    def _phase_ids(self, obs: torch.Tensor) -> torch.Tensor:
        phase_oh = obs[:, self._phase_slice]
        return torch.argmax(phase_oh, dim=-1)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_act(self.input_layer(obs))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(obs)

        all_logits = {name: head(x) for name, head in self.phase_heads.items()}
        phase_ids = self._phase_ids(obs)
        logits = all_logits["main"].clone()

        for phase_id, head_name in self._phase_to_head.items():
            mask = phase_ids == int(phase_id)
            if torch.any(mask):
                logits[mask] = all_logits[head_name][mask]
        logits = _apply_trade_split_logits(
            obs,
            logits,
            self.trade_generation_head(x),
            self.trade_response_head(x),
            self.trade_generation_mask,
            self.trade_response_mask,
        )

        value = self.value_head(x).squeeze(-1)
        return logits, value

    def predict_aux(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._encode(obs)
        return {k: head(x).squeeze(-1) for k, head in self.aux_heads.items()}


class StrategyPhaseAwareResidualPolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 512, blocks: int = 4):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden)
        self.input_act = nn.ReLU()
        self.blocks = nn.ModuleList(ResidualMLPBlock(hidden) for _ in range(max(1, blocks)))
        self.phase_heads = nn.ModuleDict(
            {
                "setup": nn.Linear(hidden, action_dim),
                "main": nn.Linear(hidden, action_dim),
                "robber": nn.Linear(hidden, action_dim),
                "trade": nn.Linear(hidden, action_dim),
                "dev": nn.Linear(hidden, action_dim),
            }
        )
        self.trade_generation_head = nn.Linear(hidden, action_dim)
        self.trade_response_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.aux_heads = _build_aux_heads(hidden)
        self.strategy_dim = len(ARCHETYPES)
        self.strategy_head = nn.Linear(hidden, self.strategy_dim)
        self.strategy_embed = nn.Linear(self.strategy_dim, hidden)
        self._phase_slice = phase_one_hot_slice()
        for k, v in _build_trade_action_masks().items():
            self.register_buffer(k, v)
        self._phase_to_head = {
            int(Phase.SETUP_SETTLEMENT): "setup",
            int(Phase.SETUP_ROAD): "setup",
            int(Phase.PRE_ROLL): "main",
            int(Phase.DICE_ROLLED): "main",
            int(Phase.DISCARD): "main",
            int(Phase.MOVE_ROBBER): "robber",
            int(Phase.ROB_PLAYER): "robber",
            int(Phase.MAIN): "main",
            int(Phase.TRADE_DRAFT): "trade",
            int(Phase.TRADE_PROPOSED): "trade",
            int(Phase.YEAR_OF_PLENTY): "dev",
            int(Phase.MONOPOLY): "dev",
            int(Phase.ROAD_BUILDING): "main",
            int(Phase.GAME_OVER): "main",
        }

    def _phase_ids(self, obs: torch.Tensor) -> torch.Tensor:
        phase_oh = obs[:, self._phase_slice]
        return torch.argmax(phase_oh, dim=-1)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_act(self.input_layer(obs))
        for block in self.blocks:
            x = block(x)
        return x

    def _condition_with_strategy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        strat_logits = self.strategy_head(x)
        strat_probs = torch.softmax(strat_logits, dim=-1)
        # Strategy embedding nudges policy representation toward coherent tactical style.
        x_cond = x + self.strategy_embed(strat_probs)
        return x_cond, strat_probs

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(obs)
        x_cond, _ = self._condition_with_strategy(x)

        all_logits = {name: head(x_cond) for name, head in self.phase_heads.items()}
        phase_ids = self._phase_ids(obs)
        logits = all_logits["main"].clone()

        for phase_id, head_name in self._phase_to_head.items():
            mask = phase_ids == int(phase_id)
            if torch.any(mask):
                logits[mask] = all_logits[head_name][mask]
        logits = _apply_trade_split_logits(
            obs,
            logits,
            self.trade_generation_head(x_cond),
            self.trade_response_head(x_cond),
            self.trade_generation_mask,
            self.trade_response_mask,
        )

        value = self.value_head(x).squeeze(-1)
        return logits, value

    def predict_aux(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._encode(obs)
        return {k: head(x).squeeze(-1) for k, head in self.aux_heads.items()}

    def predict_strategy(self, obs: torch.Tensor) -> torch.Tensor:
        x = self._encode(obs)
        _, strat_probs = self._condition_with_strategy(x)
        return strat_probs


def _build_action_entity_maps() -> dict[str, torch.Tensor]:
    n = CATALOG.size()
    settle = torch.full((n,), -1, dtype=torch.long)
    city = torch.full((n,), -1, dtype=torch.long)
    road = torch.full((n,), -1, dtype=torch.long)
    robber = torch.full((n,), -1, dtype=torch.long)
    rob_player = torch.full((n,), -1, dtype=torch.long)
    trade_mask = torch.zeros((n,), dtype=torch.bool)
    for i, a in enumerate(CATALOG.actions):
        if a.kind == "PLACE_SETTLEMENT":
            settle[i] = int(a.params[0])
        elif a.kind == "PLACE_CITY":
            city[i] = int(a.params[0])
        elif a.kind == "PLACE_ROAD":
            road[i] = int(a.params[0])
        elif a.kind == "MOVE_ROBBER":
            robber[i] = int(a.params[0])
        elif a.kind == "ROB_PLAYER":
            rob_player[i] = int(a.params[0])
        elif a.kind in {
            "TRADE_ADD_GIVE",
            "TRADE_ADD_WANT",
            "TRADE_REMOVE_GIVE",
            "TRADE_REMOVE_WANT",
            "PROPOSE_TRADE",
            "CANCEL_TRADE",
            "ACCEPT_TRADE",
            "REJECT_TRADE",
            "BANK_TRADE",
        }:
            trade_mask[i] = True
    return {
        "settle_idx": settle,
        "city_idx": city,
        "road_idx": road,
        "robber_idx": robber,
        "rob_player_idx": rob_player,
        "trade_mask": trade_mask,
    }


def _gather_neighbor_mean(src: torch.Tensor, nbr_idx: torch.Tensor) -> torch.Tensor:
    # src: [B, N, D], nbr_idx: [T, K] with -1 padding.
    idx = torch.clamp(nbr_idx, min=0)
    gathered = src[:, idx, :]  # [B, T, K, D]
    mask = (nbr_idx >= 0).to(src.dtype).unsqueeze(0).unsqueeze(-1)  # [1, T, K, 1]
    summed = (gathered * mask).sum(dim=2)
    denom = torch.clamp(mask.sum(dim=2), min=1.0)
    return summed / denom


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.v_update = nn.Linear(hidden * 4, hidden)
        self.e_update = nn.Linear(hidden * 2, hidden)
        self.h_update = nn.Linear(hidden * 2, hidden)
        self.p_update = nn.Linear(hidden * 2, hidden)
        self.act = nn.ReLU()

    def forward(
        self,
        hex_h: torch.Tensor,
        vertex_h: torch.Tensor,
        edge_h: torch.Tensor,
        player_h: torch.Tensor,
        topo: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vv = _gather_neighbor_mean(vertex_h, topo["vertex_to_vertices"])
        vh = _gather_neighbor_mean(hex_h, topo["vertex_to_hexes"])
        ve = _gather_neighbor_mean(edge_h, topo["vertex_to_edges"])
        vertex_h = self.act(self.v_update(torch.cat([vertex_h, vv, vh, ve], dim=-1)))

        ev = _gather_neighbor_mean(vertex_h, topo["edge_to_vertices"])
        edge_h = self.act(self.e_update(torch.cat([edge_h, ev], dim=-1)))

        hh = _gather_neighbor_mean(hex_h, topo["hex_to_hexes"])
        hv = _gather_neighbor_mean(vertex_h, topo["hex_to_vertices"])
        hex_h = self.act(self.h_update(torch.cat([hex_h, hh + hv], dim=-1)))

        pooled = (
            hex_h.mean(dim=1, keepdim=True)
            + vertex_h.mean(dim=1, keepdim=True)
            + edge_h.mean(dim=1, keepdim=True)
        ).expand_as(player_h)
        player_h = self.act(self.p_update(torch.cat([player_h, pooled], dim=-1)))
        return hex_h, vertex_h, edge_h, player_h


class GraphEntityPolicyValueNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 256,
        layers: int = 3,
        hybrid_mode: bool = True,
        phase_aware: bool = False,
        settlement_context_aware: bool = False,
        road_context_aware: bool = False,
        setup_adapter_enabled: bool = False,
    ):
        super().__init__()
        self.hybrid_mode = bool(hybrid_mode)
        self.phase_aware = bool(phase_aware)
        self.settlement_context_aware = bool(settlement_context_aware)
        self.road_context_aware = bool(road_context_aware)
        self.setup_adapter_enabled = bool(setup_adapter_enabled)
        self.hex_proj = nn.Linear(9, hidden)
        self.vertex_proj = nn.Linear(10, hidden)
        self.edge_proj = nn.Linear(5, hidden)
        self.player_proj = nn.Linear(player_feature_dim(), hidden)
        self.global_proj = nn.Linear(obs_dim - global_feature_start_index(), hidden)
        self.obs_proj = nn.Linear(obs_dim, hidden) if self.hybrid_mode else None
        self.vertex_settlement_proj = nn.Linear(3, hidden) if self.settlement_context_aware else None
        self.edge_road_proj = nn.Linear(5, hidden) if self.road_context_aware else None
        self.vertex_setup_adapter = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            if self.setup_adapter_enabled
            else None
        )
        self.layers = nn.ModuleList(GraphMessageLayer(hidden) for _ in range(max(1, layers)))
        fusion_in = hidden * (6 if self.hybrid_mode else 5)
        self.fusion = nn.Sequential(nn.Linear(fusion_in, hidden), nn.ReLU())
        self.global_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.aux_heads = _build_aux_heads(hidden)
        self.trade_generation_bias = nn.Linear(hidden * 2, 1)
        self.trade_response_bias = nn.Linear(hidden * 2, 1)

        self.settlement_head = nn.Linear(hidden, 1)
        self.settlement_setup1_head = nn.Linear(hidden, 1) if self.settlement_context_aware else None
        self.settlement_setup2_head = nn.Linear(hidden, 1) if self.settlement_context_aware else None
        self.settlement_main_head = nn.Linear(hidden, 1) if self.settlement_context_aware else None
        self.city_head = nn.Linear(hidden, 1)
        self.road_head = nn.Linear(hidden, 1)
        self.robber_head = nn.Linear(hidden, 1)
        self.rob_player_head = nn.Linear(hidden, 1)
        self.phase_heads = None
        self._phase_to_head: dict[int, str] = {}
        if self.phase_aware:
            self.phase_heads = nn.ModuleDict(
                {
                    "setup": nn.Linear(hidden, action_dim),
                    "main": nn.Linear(hidden, action_dim),
                    "robber": nn.Linear(hidden, action_dim),
                    "trade": nn.Linear(hidden, action_dim),
                    "dev": nn.Linear(hidden, action_dim),
                }
            )
            self._phase_to_head = {
                int(Phase.SETUP_SETTLEMENT): "setup",
                int(Phase.SETUP_ROAD): "setup",
                int(Phase.PRE_ROLL): "main",
                int(Phase.DICE_ROLLED): "main",
                int(Phase.DISCARD): "main",
                int(Phase.MOVE_ROBBER): "robber",
                int(Phase.ROB_PLAYER): "robber",
                int(Phase.MAIN): "main",
                int(Phase.TRADE_DRAFT): "trade",
                int(Phase.TRADE_PROPOSED): "trade",
                int(Phase.YEAR_OF_PLENTY): "dev",
                int(Phase.MONOPOLY): "dev",
                int(Phase.ROAD_BUILDING): "main",
                int(Phase.GAME_OVER): "main",
            }

        maps = _build_action_entity_maps()
        for k, v in maps.items():
            self.register_buffer(k, v)
        for k, v in _build_trade_action_masks().items():
            self.register_buffer(k, v)

    def _topology(self, device: torch.device) -> dict[str, torch.Tensor]:
        return topology_tensors_for_device(device)

    @staticmethod
    def _setup_round_row_masks(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sl = observation_slices()
        vb = obs[:, sl["vertex_building_owner_onehot"]].reshape(obs.shape[0], NUM_VERTICES, 1 + 2 * NUM_PLAYERS)
        own_settlement_or_city_count = (vb[:, :, 1] + vb[:, :, 2]).sum(dim=1)
        phase_ids = _phase_ids_from_obs(obs)
        setup_rows = phase_ids == int(Phase.SETUP_SETTLEMENT)
        round1_rows = setup_rows & (own_settlement_or_city_count < 0.5)
        round2_rows = setup_rows & ~round1_rows
        return round1_rows, round2_rows

    def _compute_settlement_vertex_features(
        self,
        obs: torch.Tensor,
        ent,
        topo: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Per-vertex context for settlement decisions: [quality, top6_flag, top6_rank_signal]."""
        b = obs.shape[0]
        vertex_block = ent.vertex_feat[:, :, : (1 + 2 * NUM_PLAYERS)]
        # hex_feat layout: [terrain_onehot(6), number(1), pip(1), robber(1)]
        hex_terrain_oh = ent.hex_feat[:, :, :6]
        hex_pip = ent.hex_feat[:, :, 7]
        edge_owner_oh = ent.edge_feat

        occupancy = 1.0 - vertex_block[:, :, 0]
        self_owned_building = (vertex_block[:, :, 1] > 0.5) | (vertex_block[:, :, 2] > 0.5)

        v2v = topo["vertex_to_vertices"]
        v2h = topo["vertex_to_hexes"]
        v2e = topo["vertex_to_edges"]

        nbr_idx = torch.clamp(v2v, min=0)
        nbr_occ = occupancy[:, nbr_idx]
        nbr_valid = (v2v >= 0).to(occupancy.dtype).unsqueeze(0)
        nbr_occ_any = (nbr_occ * nbr_valid).amax(dim=2)
        distance_rule_open = (occupancy < 0.5) & (nbr_occ_any < 0.5)

        hex_idx = torch.clamp(v2h, min=0)
        h_valid = (v2h >= 0).to(hex_pip.dtype).unsqueeze(0).unsqueeze(-1)
        pip_g = hex_pip[:, hex_idx].unsqueeze(-1)
        terrain_g = hex_terrain_oh[:, hex_idx, :]
        # Resource channels are terrain indices 1..5 (skip desert 0).
        vertex_res_pips = (pip_g * terrain_g[:, :, :, 1:6] * h_valid).sum(dim=2)
        total_pips = vertex_res_pips.sum(dim=-1)
        diversity = (vertex_res_pips > 0.0).to(total_pips.dtype).sum(dim=-1)
        weighted = (
            1.00 * vertex_res_pips[:, :, 0]
            + 1.00 * vertex_res_pips[:, :, 1]
            + 1.00 * vertex_res_pips[:, :, 2]
            + 1.25 * vertex_res_pips[:, :, 3]
            + 1.25 * vertex_res_pips[:, :, 4]
        )
        base = total_pips + 0.70 * diversity + 0.08 * weighted

        own_mask = self_owned_building.to(vertex_res_pips.dtype).unsqueeze(-1)
        own_prod = (vertex_res_pips * own_mask).sum(dim=1)
        missing_mask = (own_prod <= 1.0).to(vertex_res_pips.dtype).unsqueeze(1)
        complement = (vertex_res_pips * missing_mask).sum(dim=-1)

        existing_res_mask = (own_prod > 0.0).unsqueeze(1)
        cand_res_mask = vertex_res_pips > 0.0
        overlap_count = (existing_res_mask & cand_res_mask).to(base.dtype).sum(dim=-1)
        wheat_pips = vertex_res_pips[:, :, 3]
        own_wheat_weak = (own_prod[:, 3].unsqueeze(1) <= 1.0).to(base.dtype)
        wheat_bonus = 0.18 * wheat_pips + 0.28 * wheat_pips * own_wheat_weak
        wheat_penalty = 0.75 * (wheat_pips <= 0.0).to(base.dtype)

        e_idx = torch.clamp(v2e, min=0)
        own_edge = edge_owner_oh[:, :, 1] > 0.5
        e_valid = (v2e >= 0).unsqueeze(0)
        own_edge_adj = (own_edge[:, e_idx] & e_valid).any(dim=2)
        connected = own_edge_adj | self_owned_building

        # Port settle utility: prefer candidates near useful ports (especially 1 road away).
        port_type = topo["port_type"]
        v_port_dist2 = topo["vertex_port_dist2"]  # [V, P], values in {0,1,2,3}; 3 means farther than 2.
        combined_prod = own_prod.unsqueeze(1) + vertex_res_pips  # [B, V, R]
        port_best = torch.zeros_like(base)
        for p in range(int(port_type.shape[0])):
            t = int(port_type[p].item())
            if t == 0:
                desirability = torch.full_like(base, 0.45)
            else:
                strength = torch.clamp(combined_prod[:, :, t - 1] / 8.0, 0.0, 1.0)
                desirability = 0.35 + 0.35 * strength
            d = v_port_dist2[:, p]
            dist_w = torch.where(
                d == 0,
                torch.full_like(base, 0.35),
                torch.where(d == 1, torch.ones_like(base), torch.where(d == 2, torch.full_like(base, 0.55), torch.zeros_like(base))),
            )
            port_best = torch.maximum(port_best, desirability * dist_w)
        port_bonus = 0.35 * port_best

        setup_round1_score = base + wheat_bonus - wheat_penalty + 0.15 * port_bonus
        setup_round2_score = base + 0.22 * complement - 0.10 * overlap_count + wheat_bonus - wheat_penalty + port_bonus
        main_score = base + 0.20 * complement + 0.60 * connected.to(base.dtype) + port_bonus

        round1_rows, round2_rows = self._setup_round_row_masks(obs)
        score = main_score
        if torch.any(round1_rows):
            score = torch.where(round1_rows.unsqueeze(1), setup_round1_score, score)
        if torch.any(round2_rows):
            score = torch.where(round2_rows.unsqueeze(1), setup_round2_score, score)

        quality = torch.clamp(score / 25.0, 0.0, 1.0) * distance_rule_open.to(score.dtype)
        top6_flag = torch.zeros_like(quality)
        top6_rank = torch.zeros_like(quality)
        k = min(6, quality.shape[1])
        rank_vals = torch.linspace(1.0, 1.0 / max(1, k), steps=k, device=quality.device, dtype=quality.dtype)
        for bi in range(b):
            open_mask = distance_rule_open[bi]
            if not torch.any(open_mask):
                continue
            s = quality[bi].clone()
            s[~open_mask] = -1e9
            _, idx = torch.topk(s, k=k, largest=True, sorted=True)
            top6_flag[bi, idx] = 1.0
            top6_rank[bi, idx] = rank_vals

        return torch.stack([quality, top6_flag, top6_rank], dim=-1)

    def _compute_road_edge_features(
        self,
        obs: torch.Tensor,
        ent,
        topo: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Per-edge road context: [settle_unlock_gain, best_settle_delta, port_delta, dead_end_risk, connected_gain]."""
        vertex_block = ent.vertex_feat[:, :, : (1 + 2 * NUM_PLAYERS)]
        port_mask = ent.vertex_feat[:, :, -1] > 0.5
        edge_owner_oh = ent.edge_feat

        edge_empty = edge_owner_oh[:, :, 0] > 0.5
        own_edge = edge_owner_oh[:, :, 1] > 0.5
        self_owned_building = (vertex_block[:, :, 1] > 0.5) | (vertex_block[:, :, 2] > 0.5)
        occupancy = 1.0 - vertex_block[:, :, 0]

        v2v = topo["vertex_to_vertices"]
        v2e = topo["vertex_to_edges"]
        e2v = topo["edge_to_vertices"]

        nbr_idx = torch.clamp(v2v, min=0)
        nbr_occ = occupancy[:, nbr_idx]
        nbr_valid = (v2v >= 0).to(occupancy.dtype).unsqueeze(0)
        nbr_occ_any = (nbr_occ * nbr_valid).amax(dim=2)
        distance_rule_open = (occupancy < 0.5) & (nbr_occ_any < 0.5)

        e_idx = torch.clamp(v2e, min=0)
        e_valid = (v2e >= 0).unsqueeze(0)
        own_edge_adj = (own_edge[:, e_idx] & e_valid).any(dim=2)
        connected_vertex = own_edge_adj | self_owned_building
        before_connected_open_site = distance_rule_open & own_edge_adj

        settle_feat = self._compute_settlement_vertex_features(obs, ent, topo)
        settle_quality = settle_feat[:, :, 0]

        before_quality = torch.where(before_connected_open_site, settle_quality, torch.zeros_like(settle_quality))
        before_best = before_quality.max(dim=1).values

        u = e2v[:, 0]
        v = e2v[:, 1]
        u_open_site = distance_rule_open[:, u]
        v_open_site = distance_rule_open[:, v]
        u_before_conn = before_connected_open_site[:, u]
        v_before_conn = before_connected_open_site[:, v]
        edge_empty_f = edge_empty.to(settle_quality.dtype)

        settle_unlock_gain = (
            ((u_open_site & ~u_before_conn).to(settle_quality.dtype) + (v_open_site & ~v_before_conn).to(settle_quality.dtype))
            * edge_empty_f
            / 2.0
        )

        u_quality = torch.where(u_open_site, settle_quality[:, u], torch.zeros_like(settle_quality[:, u]))
        v_quality = torch.where(v_open_site, settle_quality[:, v], torch.zeros_like(settle_quality[:, v]))
        cand_best = torch.maximum(before_best.unsqueeze(1), torch.maximum(u_quality, v_quality))
        best_settle_delta = torch.clamp(cand_best - before_best.unsqueeze(1), 0.0, 1.0) * edge_empty_f

        u_conn = connected_vertex[:, u]
        v_conn = connected_vertex[:, v]
        connected_gain = (u_conn ^ v_conn).to(settle_quality.dtype) * edge_empty_f

        before_port_connected = (connected_vertex & port_mask).any(dim=1, keepdim=True)
        u_port = port_mask[:, u]
        v_port = port_mask[:, v]
        reaches_new_port = (~before_port_connected) & ((u_conn & v_port) | (v_conn & u_port))
        best_port_delta = reaches_new_port.to(settle_quality.dtype) * edge_empty_f

        open_adj_count = (
            edge_empty[:, e_idx].to(settle_quality.dtype) * e_valid.to(settle_quality.dtype)
        ).sum(dim=2)
        u_open_adj = open_adj_count[:, u]
        v_open_adj = open_adj_count[:, v]
        u_onward = torch.clamp(u_open_adj - edge_empty_f, min=0.0)
        v_onward = torch.clamp(v_open_adj - edge_empty_f, min=0.0)

        one = torch.ones_like(u_onward)
        half = torch.full_like(u_onward, 0.5)
        zero = torch.zeros_like(u_onward)
        risk_u = torch.where(u_onward <= 0.5, one, torch.where(u_onward <= 1.5, half, zero))
        risk_v = torch.where(v_onward <= 0.5, one, torch.where(v_onward <= 1.5, half, zero))
        new_u = (~u_conn) & v_conn
        new_v = (~v_conn) & u_conn
        dead_end_risk = torch.zeros_like(settle_unlock_gain)
        dead_end_risk = torch.where(new_u, risk_u, dead_end_risk)
        dead_end_risk = torch.where(new_v, torch.maximum(dead_end_risk, risk_v), dead_end_risk)
        dead_end_risk = dead_end_risk * edge_empty_f

        return torch.stack(
            [settle_unlock_gain, best_settle_delta, best_port_delta, dead_end_risk, connected_gain],
            dim=-1,
        )

    def _encode(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ent = graph_observation_to_entities(obs)
        topo = self._topology(obs.device)
        hex_h = torch.relu(self.hex_proj(ent.hex_feat))
        vertex_h = torch.relu(self.vertex_proj(ent.vertex_feat))
        if self.settlement_context_aware and self.vertex_settlement_proj is not None:
            settle_feat = self._compute_settlement_vertex_features(obs, ent, topo)
            vertex_h = vertex_h + torch.relu(self.vertex_settlement_proj(settle_feat))
        edge_h = torch.relu(self.edge_proj(ent.edge_feat))
        if self.road_context_aware and self.edge_road_proj is not None:
            road_feat = self._compute_road_edge_features(obs, ent, topo)
            edge_h = edge_h + torch.relu(self.edge_road_proj(road_feat))
        player_h = torch.relu(self.player_proj(ent.player_feat))
        global_h = torch.relu(self.global_proj(ent.global_feat))
        for layer in self.layers:
            hex_h, vertex_h, edge_h, player_h = layer(hex_h, vertex_h, edge_h, player_h, topo)
        setup_vertex_h = vertex_h
        if self.setup_adapter_enabled and self.vertex_setup_adapter is not None:
            setup_vertex_h = vertex_h + self.vertex_setup_adapter(vertex_h)
        pooled = [
            hex_h.mean(dim=1),
            vertex_h.mean(dim=1),
            edge_h.mean(dim=1),
            player_h.mean(dim=1),
            global_h,
        ]
        if self.hybrid_mode and self.obs_proj is not None:
            pooled.append(torch.relu(self.obs_proj(obs)))
        fused = self.fusion(torch.cat(pooled, dim=-1))
        return fused, hex_h, vertex_h, setup_vertex_h, edge_h, player_h

    def _apply_entity_action_scoring(
        self,
        obs: torch.Tensor,
        logits: torch.Tensor,
        vertex_h: torch.Tensor,
        setup_vertex_h: torch.Tensor,
        edge_h: torch.Tensor,
        hex_h: torch.Tensor,
        player_h: torch.Tensor,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        def _assign(idx_buf: torch.Tensor, emb: torch.Tensor, head: nn.Linear) -> None:
            valid = idx_buf >= 0
            if torch.any(valid):
                idx = idx_buf[valid]
                scores = head(emb[:, idx, :]).squeeze(-1)
                logits[:, valid] = scores

        if (
            self.settlement_context_aware
            and self.settlement_setup1_head is not None
            and self.settlement_setup2_head is not None
            and self.settlement_main_head is not None
        ):
            valid = self.settle_idx >= 0
            if torch.any(valid):
                idx = self.settle_idx[valid]
                s1 = self.settlement_setup1_head(setup_vertex_h[:, idx, :]).squeeze(-1)
                s2 = self.settlement_setup2_head(setup_vertex_h[:, idx, :]).squeeze(-1)
                sm = self.settlement_main_head(vertex_h[:, idx, :]).squeeze(-1)
                round1_rows, round2_rows = self._setup_round_row_masks(obs)
                main_rows = ~(round1_rows | round2_rows)
                settle_scores = sm
                if torch.any(round1_rows):
                    settle_scores = torch.where(round1_rows.unsqueeze(1), s1, settle_scores)
                if torch.any(round2_rows):
                    settle_scores = torch.where(round2_rows.unsqueeze(1), s2, settle_scores)
                if torch.any(main_rows):
                    settle_scores = torch.where(main_rows.unsqueeze(1), sm, settle_scores)
                logits[:, valid] = settle_scores
        else:
            _assign(self.settle_idx, vertex_h, self.settlement_head)
        _assign(self.city_idx, vertex_h, self.city_head)
        _assign(self.road_idx, edge_h, self.road_head)
        _assign(self.robber_idx, hex_h, self.robber_head)
        _assign(self.rob_player_idx, player_h, self.rob_player_head)

        trade_ctx = torch.cat([fused, player_h.mean(dim=1)], dim=-1)
        gen_scalar = self.trade_generation_bias(trade_ctx).squeeze(-1).unsqueeze(-1)
        resp_scalar = self.trade_response_bias(trade_ctx).squeeze(-1).unsqueeze(-1)
        gen_logits = logits.clone()
        resp_logits = logits.clone()
        if torch.any(self.trade_generation_mask):
            gen_logits[:, self.trade_generation_mask] = gen_logits[:, self.trade_generation_mask] + gen_scalar
        if torch.any(self.trade_response_mask):
            resp_logits[:, self.trade_response_mask] = resp_logits[:, self.trade_response_mask] + resp_scalar
        logits = _apply_trade_split_logits(
            obs,
            logits,
            gen_logits,
            resp_logits,
            self.trade_generation_mask,
            self.trade_response_mask,
        )
        return logits

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fused, hex_h, vertex_h, setup_vertex_h, edge_h, player_h = self._encode(obs)
        if self.phase_aware and self.phase_heads is not None:
            all_logits = {name: head(fused) for name, head in self.phase_heads.items()}
            phase_ids = _phase_ids_from_obs(obs)
            logits = all_logits["main"].clone()
            for phase_id, head_name in self._phase_to_head.items():
                mask = phase_ids == int(phase_id)
                if torch.any(mask):
                    logits[mask] = all_logits[head_name][mask]
        else:
            logits = self.global_head(fused)
        logits = self._apply_entity_action_scoring(obs, logits, vertex_h, setup_vertex_h, edge_h, hex_h, player_h, fused)
        value = self.value_head(fused).squeeze(-1)
        return logits, value

    def predict_aux(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        fused, _, _, _, _, _ = self._encode(obs)
        return {k: head(fused).squeeze(-1) for k, head in self.aux_heads.items()}

def build_policy_value_net(
    *,
    obs_dim: int,
    action_dim: int,
    model_arch: str = "mlp",
    hidden: int = 256,
    residual_blocks: int = 4,
) -> nn.Module:
    if model_arch == "mlp":
        return PolicyValueNet(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden)
    if model_arch == "residual_mlp":
        return ResidualPolicyValueNet(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, blocks=residual_blocks)
    if model_arch == "phase_aware_residual_mlp":
        return PhaseAwareResidualPolicyValueNet(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, blocks=residual_blocks)
    if model_arch == "strategy_phase_aware_residual_mlp":
        return StrategyPhaseAwareResidualPolicyValueNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            blocks=residual_blocks,
        )
    if model_arch == "graph_entity_hybrid":
        return GraphEntityPolicyValueNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            layers=max(1, residual_blocks),
            hybrid_mode=True,
            phase_aware=False,
        )
    if model_arch == "graph_entity_phase_aware_hybrid":
        return GraphEntityPolicyValueNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            layers=max(1, residual_blocks),
            hybrid_mode=True,
            phase_aware=True,
            settlement_context_aware=True,
            road_context_aware=True,
            setup_adapter_enabled=True,
        )
    if model_arch == "graph_entity_only":
        return GraphEntityPolicyValueNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden=hidden,
            layers=max(1, residual_blocks),
            hybrid_mode=False,
            phase_aware=False,
        )
    raise ValueError(f"Unknown model_arch: {model_arch}")


def infer_model_config_from_state_dict(state_dict: dict) -> dict[str, int | str]:
    if "hex_proj.weight" in state_dict and "vertex_proj.weight" in state_dict:
        hidden = int(state_dict["hex_proj.weight"].shape[0])
        block_idxs = {
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("layers.") and ".v_update.weight" in k and k.split(".")[1].isdigit()
        }
        if "phase_heads.setup.weight" in state_dict:
            arch = "graph_entity_phase_aware_hybrid"
        else:
            arch = "graph_entity_hybrid" if "obs_proj.weight" in state_dict else "graph_entity_only"
        return {
            "model_arch": arch,
            "hidden": hidden,
            "residual_blocks": max(1, len(block_idxs)),
        }
    if "strategy_head.weight" in state_dict:
        hidden = int(state_dict["input_layer.weight"].shape[0])
        block_idxs = {
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("blocks.") and ".fc1.weight" in k and k.split(".")[1].isdigit()
        }
        return {
            "model_arch": "strategy_phase_aware_residual_mlp",
            "hidden": hidden,
            "residual_blocks": max(1, len(block_idxs)),
        }
    if "phase_heads.setup.weight" in state_dict:
        hidden = int(state_dict["phase_heads.setup.weight"].shape[1])
        block_idxs = {
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("blocks.") and ".fc1.weight" in k and k.split(".")[1].isdigit()
        }
        return {
            "model_arch": "phase_aware_residual_mlp",
            "hidden": hidden,
            "residual_blocks": max(1, len(block_idxs)),
        }
    if "input_layer.weight" in state_dict:
        hidden = int(state_dict["input_layer.weight"].shape[0])
        block_idxs = {
            int(k.split(".")[1])
            for k in state_dict.keys()
            if k.startswith("blocks.") and ".fc1.weight" in k and k.split(".")[1].isdigit()
        }
        return {
            "model_arch": "residual_mlp",
            "hidden": hidden,
            "residual_blocks": max(1, len(block_idxs)),
        }
    if "backbone.0.weight" in state_dict:
        hidden = int(state_dict["backbone.0.weight"].shape[0])
        return {
            "model_arch": "mlp",
            "hidden": hidden,
            "residual_blocks": 0,
        }
    return {
        "model_arch": "mlp",
        "hidden": 256,
        "residual_blocks": 0,
    }


def load_policy_value_net(
    checkpoint: str | Path,
    *,
    obs_dim: int,
    action_dim: int,
    model_arch: str | None = None,
    hidden: int = 256,
    residual_blocks: int = 4,
    map_location: str = "cpu",
) -> nn.Module:
    state_dict = torch.load(str(checkpoint), map_location=map_location)
    inferred = infer_model_config_from_state_dict(state_dict)
    if model_arch is not None and str(model_arch) != str(inferred["model_arch"]):
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"requested model_arch='{model_arch}' but checkpoint looks like '{inferred['model_arch']}'. "
            "Pass the matching --model-arch value, or omit it in eval/trace for auto-inference."
        )
    arch = inferred["model_arch"] if model_arch is None else model_arch
    h = int(inferred["hidden"]) if model_arch is None else int(hidden)
    blocks = int(inferred["residual_blocks"]) if model_arch is None else int(residual_blocks)
    model = build_policy_value_net(
        obs_dim=obs_dim,
        action_dim=action_dim,
        model_arch=str(arch),
        hidden=int(h),
        residual_blocks=int(blocks),
    )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "trade_generation_mask",
            "trade_response_mask",
            "trade_generation_head.",
            "trade_response_head.",
            "trade_generation_bias.",
            "trade_response_bias.",
            "aux_heads.",
            "vertex_settlement_proj.",
            "edge_road_proj.",
            "vertex_setup_adapter.",
            "settlement_setup1_head.",
            "settlement_setup2_head.",
            "settlement_main_head.",
        )
        allowed_missing = all(any(m.startswith(p) for p in allowed_missing_prefixes) for m in missing)
        if unexpected or not allowed_missing:
            raise ValueError(
                "Failed loading checkpoint into model. "
                f"requested=(arch={arch}, hidden={h}, blocks={blocks}) "
                f"inferred=(arch={inferred['model_arch']}, hidden={inferred['hidden']}, "
                f"blocks={inferred['residual_blocks']}) "
                f"checkpoint='{checkpoint}' "
                f"missing_keys={list(missing)} unexpected_keys={list(unexpected)}."
            ) from exc
    return model


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    aux_coef: float = 0.0
    strategy_coef: float = 0.0
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 256
    minibatch_size: int = 64
    setup_phase_loss_weight: float = 1.0
    max_grad_norm: float = 0.5


class PPOTrainer:
    def __init__(self, model: nn.Module, config: PPOConfig):
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
        aux_targets = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in batch.get("aux_targets", {}).items()
        }
        aux_target_weights = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in batch.get("aux_target_weights", {}).items()
        }
        strategy_targets = torch.as_tensor(batch["strategy_targets"], dtype=torch.float32) if "strategy_targets" in batch else None
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        phase_ids = _phase_ids_from_obs(obs)

        n = obs.shape[0]
        idx = np.arange(n)
        logs = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "aux_loss": 0.0, "strategy_loss": 0.0, "grad_norm": 0.0}
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
                policy_terms = -torch.min(ratio * adv[mb], clipped * adv[mb])
                mb_phase_ids = phase_ids[mb]
                setup_mask = (mb_phase_ids == int(Phase.SETUP_SETTLEMENT)) | (mb_phase_ids == int(Phase.SETUP_ROAD))
                if self.cfg.setup_phase_loss_weight != 1.0 and torch.any(setup_mask):
                    phase_weights = torch.ones_like(policy_terms)
                    phase_weights = torch.where(
                        setup_mask,
                        torch.full_like(policy_terms, float(self.cfg.setup_phase_loss_weight)),
                        phase_weights,
                    )
                    policy_loss = (policy_terms * phase_weights).sum() / torch.clamp(phase_weights.sum(), min=1.0)
                else:
                    policy_loss = policy_terms.mean()
                value_loss = ((value - ret[mb]) ** 2).mean()
                entropy = dist.entropy().mean()
                aux_loss = value.new_zeros(())
                strategy_loss = value.new_zeros(())
                if self.cfg.aux_coef > 0.0 and aux_targets and hasattr(self.model, "predict_aux"):
                    pred_aux = self.model.predict_aux(obs[mb])
                    losses = []
                    for k, tgt in aux_targets.items():
                        if k in pred_aux:
                            weights = aux_target_weights.get(k)
                            if weights is None:
                                losses.append(((pred_aux[k] - tgt[mb]) ** 2).mean())
                            else:
                                mb_weights = weights[mb]
                                denom = torch.clamp(mb_weights.sum(), min=1.0)
                                losses.append((((pred_aux[k] - tgt[mb]) ** 2) * mb_weights).sum() / denom)
                    if losses:
                        aux_loss = torch.stack(losses).mean()
                if self.cfg.strategy_coef > 0.0 and strategy_targets is not None and hasattr(self.model, "predict_strategy"):
                    pred_strategy = self.model.predict_strategy(obs[mb])
                    tgt = strategy_targets[mb]
                    strategy_loss = -(tgt * torch.log(torch.clamp(pred_strategy, min=1e-8))).sum(dim=-1).mean()
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    - self.cfg.ent_coef * entropy
                    + self.cfg.aux_coef * aux_loss
                    + self.cfg.strategy_coef * strategy_loss
                )

                self.optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg.max_grad_norm))
                self.optim.step()

                logs["policy_loss"] += float(policy_loss.item())
                logs["value_loss"] += float(value_loss.item())
                logs["entropy"] += float(entropy.item())
                logs["aux_loss"] += float(aux_loss.item())
                logs["strategy_loss"] += float(strategy_loss.item())
                logs["grad_norm"] += float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
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
