"""Graph/entity feature extraction from flat encoded observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import NUM_EDGES, NUM_HEXES, NUM_PLAYERS, NUM_VERTICES
from .encoding import encode_observation, observation_slices
from .state import GameState
from .topology import build_topology


@dataclass
class EntityBatch:
    hex_feat: torch.Tensor
    vertex_feat: torch.Tensor
    edge_feat: torch.Tensor
    player_feat: torch.Tensor
    global_feat: torch.Tensor


def _topology_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    topo = build_topology()
    # Precompute shortest vertex->port-edge endpoint distance, clipped at 3 (3 == "farther than 2").
    vertex_port_dist2 = np.full((NUM_VERTICES, len(topo.port_type)), 3, dtype=np.int64)
    for v in range(NUM_VERTICES):
        dist = np.full((NUM_VERTICES,), 10_000, dtype=np.int64)
        dist[v] = 0
        q = [v]
        head = 0
        while head < len(q):
            cur = q[head]
            head += 1
            d = int(dist[cur])
            if d >= 2:
                continue
            for n in topo.vertex_to_vertices[cur]:
                if n < 0:
                    continue
                ni = int(n)
                if d + 1 < int(dist[ni]):
                    dist[ni] = d + 1
                    q.append(ni)
        for p in range(len(topo.port_type)):
            a = int(topo.port_vertices[p, 0])
            b = int(topo.port_vertices[p, 1])
            vertex_port_dist2[v, p] = min(3, int(min(dist[a], dist[b])))
    return {
        "vertex_to_vertices": torch.as_tensor(topo.vertex_to_vertices, dtype=torch.long, device=device),
        "vertex_to_hexes": torch.as_tensor(topo.vertex_to_hexes, dtype=torch.long, device=device),
        "vertex_to_edges": torch.as_tensor(topo.vertex_to_edges, dtype=torch.long, device=device),
        "edge_to_vertices": torch.as_tensor(topo.edge_to_vertices, dtype=torch.long, device=device),
        "hex_to_vertices": torch.as_tensor(topo.hex_to_vertices, dtype=torch.long, device=device),
        "hex_to_hexes": torch.as_tensor(topo.hex_to_hexes, dtype=torch.long, device=device),
        "port_vertices": torch.as_tensor(topo.port_vertices, dtype=torch.long, device=device),
        "port_type": torch.as_tensor(topo.port_type, dtype=torch.long, device=device),
        "vertex_port_dist2": torch.as_tensor(vertex_port_dist2, dtype=torch.long, device=device),
    }


def graph_observation_to_entities(obs: torch.Tensor) -> EntityBatch:
    """Convert flat encoded observations [B, D] into entity feature blocks."""
    if obs.ndim != 2:
        raise ValueError(f"Expected obs rank-2 [B,D], got {tuple(obs.shape)}")
    sl = observation_slices()
    b = obs.shape[0]

    hex_terrain = obs[:, sl["hex_terrain_onehot"]].reshape(b, NUM_HEXES, 6)
    hex_number = obs[:, sl["hex_number"]].reshape(b, NUM_HEXES, 1)
    hex_pip = obs[:, sl["hex_pip"]].reshape(b, NUM_HEXES, 1)
    robber = obs[:, sl["robber_onehot"]].reshape(b, NUM_HEXES, 1)
    hex_feat = torch.cat([hex_terrain, hex_number, hex_pip, robber], dim=-1)

    vertex_block = obs[:, sl["vertex_building_owner_onehot"]].reshape(b, NUM_VERTICES, 1 + 2 * NUM_PLAYERS)
    port_mask = obs[:, sl["port_vertex_mask"]].reshape(b, NUM_VERTICES, 1)
    vertex_feat = torch.cat([vertex_block, port_mask], dim=-1)

    edge_owner = obs[:, sl["edge_owner_onehot"]].reshape(b, NUM_EDGES, 1 + NUM_PLAYERS)
    edge_feat = edge_owner

    player_parts = [
        obs[:, sl["self_resources"]],  # 5
        obs[:, sl["opp_resource_totals"]],  # 3
        obs[:, sl["self_dev_hidden"]],  # 5
        obs[:, sl["opp_knights_played"]],  # 3
        obs[:, sl["pieces_left"]],  # 12
        obs[:, sl["has_port"]],  # 24
        obs[:, sl["longest_road_length"]],  # 4
        obs[:, sl["has_longest_road_and_largest_army"]],  # 8
        obs[:, sl["public_vp"]],  # 4
    ]
    player_flat = torch.cat(player_parts, dim=-1)
    # Expand shared player context across player entities; ownership channels in vertex/edge remain player-specific.
    player_feat = player_flat.unsqueeze(1).expand(-1, NUM_PLAYERS, -1)

    global_parts = [
        obs[:, sl["phase_onehot"]],
        obs[:, sl["turn_features"]],
        obs[:, sl["trade_offer_give_want"]],
        obs[:, sl["trade_proposer"]],
        obs[:, sl["trade_responses"]],
        obs[:, sl["engineered_features"]],
    ]
    global_feat = torch.cat(global_parts, dim=-1)

    return EntityBatch(
        hex_feat=hex_feat,
        vertex_feat=vertex_feat,
        edge_feat=edge_feat,
        player_feat=player_feat,
        global_feat=global_feat,
    )


def graph_state_to_entities(state: GameState, perspective_player: int) -> EntityBatch:
    obs = encode_observation(state, perspective_player)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    return graph_observation_to_entities(obs_t)


def topology_tensors_for_device(device: torch.device) -> dict[str, torch.Tensor]:
    return _topology_tensors(device)


def player_feature_dim() -> int:
    sl = observation_slices()
    keys = [
        "self_resources",
        "opp_resource_totals",
        "self_dev_hidden",
        "opp_knights_played",
        "pieces_left",
        "has_port",
        "longest_road_length",
        "has_longest_road_and_largest_army",
        "public_vp",
    ]
    return sum((sl[k].stop - sl[k].start) for k in keys)  # type: ignore[operator]


def global_feature_start_index() -> int:
    return observation_slices()["phase_onehot"].start  # type: ignore[return-value]

