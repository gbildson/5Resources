"""Run a single game and print a human-readable action walkthrough."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

from catan_rl.actions import CATALOG, Action
from catan_rl.bots import HeuristicAgent, RandomLegalAgent
from catan_rl.constants import Building, Phase
from catan_rl.env import CatanEnv
from catan_rl.training.ppo import PolicyValueNet


TERRAIN_NAMES = {
    0: "DESERT",
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}
RESOURCE_NAMES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
PORT_NAMES = ["GENERIC", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
PLAYER_COLORS = ["#00c40d", "#ff00d9", "#0000FF", "#00FFFF"]
TERRAIN_SHORT = {
    0: "DS",
    1: "WD",
    2: "BR",
    3: "SH",
    4: "WH",
    5: "OR",
}


def _format_action(action: Action) -> str:
    if not action.params:
        return action.kind
    return f"{action.kind}{tuple(int(x) for x in action.params)}"


def _policy_choice_with_probs(model: PolicyValueNet, obs: np.ndarray, mask: np.ndarray) -> tuple[int, list[tuple[int, float]]]:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(obs_t)
        logits = logits.masked_fill(mask_t <= 0, -1e9)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    action_id = int(np.argmax(probs))
    top_ids = np.argsort(probs)[-5:][::-1]
    top = [(int(i), float(probs[i])) for i in top_ids if probs[i] > 0.0]
    return action_id, top


def _print_setup(env: CatanEnv) -> None:
    s = env.state
    print("=== Setup ===")
    print(f"seed={env.seed}")
    print(f"robber_hex={s.robber_hex}")
    print(f"hex_terrain={s.hex_terrain.tolist()}")
    print(f"hex_number={s.hex_number.tolist()}")
    print(f"port_type={s.port_type.tolist()}")
    print("=============\n")


def _agent_for_player(player: int, model_players: set[int], model: PolicyValueNet, seed: int):
    if player in model_players:
        return ("policy", None)
    # Alternate simple opponents for a bit of variety.
    if player % 2 == 0:
        return ("heuristic", HeuristicAgent(seed=seed + 100 + player))
    return ("random", RandomLegalAgent(seed=seed + 100 + player))


def _print_top_actions(top: Iterable[tuple[int, float]]) -> None:
    rendered = []
    for a_id, p in top:
        rendered.append(f"{_format_action(CATALOG.decode(a_id))}:{p:.3f}")
    print("  policy_top5:", " | ".join(rendered) if rendered else "n/a")


def _format_resources(resources: np.ndarray) -> str:
    parts = [f"{name[:2]}={int(resources[i])}" for i, name in enumerate(RESOURCE_NAMES)]
    return ", ".join(parts)


def _owned_vertices_for_player(env: CatanEnv, player: int, building_kind: int) -> list[int]:
    s = env.state
    out: list[int] = []
    for v in range(len(s.vertex_owner)):
        if int(s.vertex_owner[v]) == player and int(s.vertex_building[v]) == building_kind:
            out.append(v)
    return out


def _owned_roads_for_player(env: CatanEnv, player: int) -> list[str]:
    s = env.state
    roads: list[str] = []
    for e in range(len(s.edge_owner)):
        if int(s.edge_owner[e]) != player:
            continue
        u = int(s.topology.edge_to_vertices[e, 0])
        v = int(s.topology.edge_to_vertices[e, 1])
        roads.append(f"{e}({u}-{v})")
    return roads


def _print_final_board(env: CatanEnv) -> None:
    s = env.state
    print("\n=== Final Board Snapshot ===")
    print(f"phase={Phase(s.phase).name}")
    print(f"current_player={int(s.current_player)}")
    print(f"robber_hex={int(s.robber_hex)}")
    print("\nHexes:")
    for h in range(len(s.hex_terrain)):
        terrain = TERRAIN_NAMES.get(int(s.hex_terrain[h]), f"T{int(s.hex_terrain[h])}")
        number = int(s.hex_number[h])
        vertices = [int(v) for v in s.topology.hex_to_vertices[h]]
        robber_tag = " <ROBBER>" if h == int(s.robber_hex) else ""
        print(f"  hex={h:02d} terrain={terrain:6s} number={number:2d} vertices={vertices}{robber_tag}")

    print("\nPlayers:")
    for p in range(len(s.public_vp)):
        settlements = _owned_vertices_for_player(env, p, int(Building.SETTLEMENT))
        cities = _owned_vertices_for_player(env, p, int(Building.CITY))
        roads = _owned_roads_for_player(env, p)
        ports = [PORT_NAMES[i] for i in range(len(PORT_NAMES)) if int(s.has_port[p, i]) == 1]
        print(
            f"  P{p}: public_vp={int(s.public_vp[p])} actual_vp={int(s.actual_vp[p])} "
            f"knights={int(s.knights_played[p])} longest_road={int(s.longest_road_length[p])}"
        )
        print(
            f"      pieces_left(settle/city/road)="
            f"{int(s.settlements_left[p])}/{int(s.cities_left[p])}/{int(s.roads_left[p])}"
        )
        print(f"      resources=({_format_resources(s.resources[p])}) total={int(s.resource_total[p])}")
        print(f"      settlements={settlements if settlements else '[]'}")
        print(f"      cities={cities if cities else '[]'}")
        print(f"      roads={roads if roads else '[]'}")
        print(f"      ports={ports if ports else '[]'}")
    print("============================")


def _inspect_tile_label(terrain: int, number: int, has_robber: bool, idx: int) -> str:
    t = TERRAIN_SHORT.get(int(terrain), f"T{int(terrain)}")
    token = "--" if int(number) == 0 else f"{int(number):02d}"
    robber = "*" if has_robber else " "
    return f"[{idx:02d} {t} {token}{robber}]"


def _inspect_hex_rows() -> list[list[int]]:
    return [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18]]


def _inspect_edge_hex_map(env: CatanEnv) -> dict[tuple[int, int], list[int]]:
    edge_to_hexes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for h in range(len(env.state.topology.hex_to_vertices)):
        hv = env.state.topology.hex_to_vertices[h]
        for i in range(6):
            u = int(hv[i])
            v = int(hv[(i + 1) % 6])
            key = (u, v) if u < v else (v, u)
            edge_to_hexes[key].append(h)
    return edge_to_hexes


def _print_final_inspect_board(env: CatanEnv, show_topology: bool) -> None:
    s = env.state
    print("\n=== Inspect Board View ===")
    print("Hex map (index terrain token; * = robber):")
    for row_i, row in enumerate(_inspect_hex_rows()):
        indent = " " * (4 - row_i if row_i <= 2 else row_i)
        labels = [
            _inspect_tile_label(int(s.hex_terrain[h]), int(s.hex_number[h]), h == int(s.robber_hex), h) for h in row
        ]
        print(f"{indent}{' '.join(labels)}")
    print()

    edge_to_hexes = _inspect_edge_hex_map(env)
    print("Ports (type, vertices, adjacent hex):")
    for p in range(len(s.port_type)):
        a = int(s.port_vertices[p, 0])
        b = int(s.port_vertices[p, 1])
        key = (a, b) if a < b else (b, a)
        adjacent = edge_to_hexes.get(key, [])
        adj_str = ",".join(str(h) for h in adjacent) if adjacent else "?"
        print(f"  port={p:02d} type={PORT_NAMES[int(s.port_type[p])]:7s} edge=({a}-{b}) coast_hex={adj_str}")
    print()

    terrain_counts = [int((s.hex_terrain == i).sum()) for i in range(6)]
    tokens = sorted(int(n) for n in s.hex_number.tolist() if int(n) > 0)
    print("Terrain counts [DESERT, WOOD, BRICK, SHEEP, WHEAT, ORE]:", terrain_counts)
    print("Number tokens:", tokens)
    print(f"Robber hex: {int(s.robber_hex)}")

    print("\nPlayer occupancy (final state):")
    for p in range(len(s.public_vp)):
        settlements = _owned_vertices_for_player(env, p, int(Building.SETTLEMENT))
        cities = _owned_vertices_for_player(env, p, int(Building.CITY))
        roads = _owned_roads_for_player(env, p)
        print(f"  P{p}: settlements={settlements if settlements else []}")
        print(f"      cities={cities if cities else []}")
        print(f"      roads={roads if roads else []}")

    if show_topology:
        print("\nVertices (id -> adjacent hexes / adjacent vertices):")
        for v in range(len(s.topology.vertex_to_hexes)):
            hexes = [int(h) for h in s.topology.vertex_to_hexes[v] if int(h) >= 0]
            neigh = [int(u) for u in s.topology.vertex_to_vertices[v] if int(u) >= 0]
            print(f"  v{v:02d}: hexes={hexes} neigh={neigh}")
        print("\nEdges (id -> vertex endpoints):")
        for e in range(len(s.topology.edge_to_vertices)):
            u = int(s.topology.edge_to_vertices[e, 0])
            v = int(s.topology.edge_to_vertices[e, 1])
            print(f"  e{e:02d}: ({u}-{v})")
    print("===========================")


def _print_final_board_pycatan(env: CatanEnv) -> None:
    try:
        from pycatan import Player as PyCatanPlayer
        from pycatan.board import (
            BeginnerBoard,
            BoardRenderer,
            BuildingType as PyCatanBuildingType,
            HexType as PyCatanHexType,
            IntersectionBuilding as PyCatanIntersectionBuilding,
            PathBuilding as PyCatanPathBuilding,
        )
    except ImportError:
        print(
            "\n=== PyCatan Board Projection ===\n"
            "pycatan is not installed in this environment.\n"
            "Install it in your venv and re-run with --show-final-board-pycatan:\n"
            "  .venv/bin/pip install pycatan\n"
            "================================"
        )
        return

    s = env.state
    board = BeginnerBoard()
    players = [PyCatanPlayer() for _ in range(len(s.public_vp))]

    terrain_to_pycatan = {
        0: PyCatanHexType.DESERT,
        1: PyCatanHexType.FOREST,
        2: PyCatanHexType.HILLS,
        3: PyCatanHexType.PASTURE,
        4: PyCatanHexType.FIELDS,
        5: PyCatanHexType.MOUNTAINS,
    }
    hex_coords_order = [
        (4, -2),
        (3, 0),
        (2, 2),
        (0, 3),
        (-2, 4),
        (-3, 3),
        (-4, 2),
        (-3, 0),
        (-2, -2),
        (0, -3),
        (2, -4),
        (3, -3),
        (2, -1),
        (1, 1),
        (-1, 2),
        (-2, 1),
        (-1, -1),
        (1, -2),
        (0, 0),
    ]
    coords_to_hex = {(h.coords.q, h.coords.r): h for h in board.hexes.values()}
    if len(hex_coords_order) != len(s.hex_terrain):
        print("\n=== PyCatan Board Projection ===\nhex count mismatch; cannot project\n================================")
        return

    for h_idx, coord in enumerate(hex_coords_order):
        hex_obj = coords_to_hex[coord]
        terrain = int(s.hex_terrain[h_idx])
        hex_obj.hex_type = terrain_to_pycatan.get(terrain, PyCatanHexType.DESERT)
        token = int(s.hex_number[h_idx])
        hex_obj.token_number = None if terrain == 0 else token
    board.robber = coords_to_hex[hex_coords_order[int(s.robber_hex)]].coords

    # This is a visualization projection: index-based mapping preserves occupancy
    # counts but does not reproduce the synthetic topology geometry exactly.
    intersections_sorted = sorted(board.intersections.keys(), key=lambda c: (c.q, c.r))
    for v_idx, owner in enumerate(s.vertex_owner):
        p = int(owner)
        if p < 0:
            continue
        coord = intersections_sorted[v_idx]
        b_kind = int(s.vertex_building[v_idx])
        if b_kind == int(Building.SETTLEMENT):
            board.intersections[coord].building = PyCatanIntersectionBuilding(
                owner=players[p],
                building_type=PyCatanBuildingType.SETTLEMENT,
                coords=coord,
            )
        elif b_kind == int(Building.CITY):
            board.intersections[coord].building = PyCatanIntersectionBuilding(
                owner=players[p],
                building_type=PyCatanBuildingType.CITY,
                coords=coord,
            )

    def _path_key(path_coords: frozenset) -> tuple[tuple[int, int], tuple[int, int]]:
        pts = sorted(((c.q, c.r) for c in path_coords))
        return pts[0], pts[1]

    paths_sorted = sorted(board.paths.keys(), key=_path_key)
    for e_idx, owner in enumerate(s.edge_owner):
        p = int(owner)
        if p < 0:
            continue
        path_coords = paths_sorted[e_idx]
        board.paths[path_coords].building = PyCatanPathBuilding(
            owner=players[p],
            building_type=PyCatanBuildingType.ROAD,
            path_coords=set(path_coords),
        )

    player_color_map = {players[p]: PLAYER_COLORS[p % len(PLAYER_COLORS)] for p in range(len(players))}
    renderer = BoardRenderer(board=board, player_color_map=player_color_map)
    print("\n=== PyCatan Board Projection ===")
    print("Note: index-based projection onto pycatan geometry (visual aid only).")
    print(renderer.get_board_as_string())
    print("================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path (.pt)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--model-players",
        type=str,
        default="0",
        help="Comma-separated player ids controlled by model (e.g. '0' or '0,1,2,3')",
    )
    parser.add_argument("--max-main-actions-per-turn", type=int, default=10)
    parser.add_argument("--disable-player-trade", action="store_true")
    parser.add_argument("--trade-action-mode", choices=["guided", "full"], default="guided")
    parser.add_argument("--max-player-trade-proposals-per-turn", type=int, default=None)
    parser.add_argument("--enhanced-obs-features", action="store_true")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSONL output file for action trace")
    parser.add_argument(
        "--show-final-board",
        action="store_true",
        help="Print a detailed final board snapshot (hexes, pieces, roads, and resources).",
    )
    parser.add_argument(
        "--show-final-board-pycatan",
        action="store_true",
        help="Render a pycatan-style terminal board projection of the final state (optional dependency).",
    )
    parser.add_argument(
        "--show-final-inspect-board",
        action="store_true",
        help="Print inspect_board-style final board view, including ports, token counts, and player occupancy.",
    )
    parser.add_argument(
        "--show-final-inspect-topology",
        action="store_true",
        help="With --show-final-inspect-board, include full vertex/edge topology tables.",
    )
    args = parser.parse_args()

    model_players = set(int(x) for x in args.model_players.split(",") if x != "")
    env = CatanEnv(
        seed=args.seed,
        enhanced_obs_features=bool(args.enhanced_obs_features),
        max_main_actions_per_turn=args.max_main_actions_per_turn,
        allow_player_trade=not args.disable_player_trade,
        trade_action_mode=args.trade_action_mode,
        max_player_trade_proposals_per_turn=args.max_player_trade_proposals_per_turn,
    )
    obs, info = env.reset(seed=args.seed)
    mask = info["action_mask"]

    model = PolicyValueNet(obs_dim=obs.shape[0], action_dim=mask.shape[0])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    _print_setup(env)

    out_f = open(args.json_out, "w", encoding="utf-8") if args.json_out else None
    try:
        for step in range(1, args.max_steps + 1):
            if env.state.phase == Phase.GAME_OVER:
                break
            player = int(env.state.current_player)
            phase = Phase(env.state.phase).name
            legal_count = int(mask.sum())
            role, bot = _agent_for_player(player, model_players, model, args.seed)

            if role == "policy":
                action_id, top = _policy_choice_with_probs(model, obs, mask)
            else:
                action_id = int(bot.act(obs, mask))
                top = []
            action = CATALOG.decode(action_id)

            print(
                f"step={step:03d} player={player} role={role:9s} phase={phase:15s} "
                f"legal={legal_count:3d} action={_format_action(action)}"
            )
            if top:
                _print_top_actions(top)

            result = env.step(action_id)
            if out_f:
                out_f.write(
                    json.dumps(
                        {
                            "step": step,
                            "player": player,
                            "role": role,
                            "phase": phase,
                            "legal_count": legal_count,
                            "action_id": action_id,
                            "action": _format_action(action),
                            "top5": top,
                            "reward": float(result.reward),
                            "winner": int(result.info.get("winner", -1)),
                            "public_vp": env.state.public_vp.tolist(),
                            "actual_vp": env.state.actual_vp.tolist(),
                        }
                    )
                    + "\n"
                )
            obs = result.obs
            mask = result.info["action_mask"]
            if result.done:
                break

        print("\n=== Final State ===")
        print(f"winner={env.state.winner}")
        print(f"turn_number={env.state.turn_number}")
        print(f"public_vp={env.state.public_vp.tolist()}")
        print(f"actual_vp={env.state.actual_vp.tolist()}")
        print("===================")
        if args.show_final_board:
            _print_final_board(env)
        if args.show_final_inspect_board:
            _print_final_inspect_board(env, show_topology=args.show_final_inspect_topology)
        if args.show_final_board_pycatan:
            _print_final_board_pycatan(env)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()
