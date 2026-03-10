"""Visual board inspector for generated Catan states."""

from __future__ import annotations

import argparse
from collections import defaultdict

from catan_rl.constants import NUM_HEXES
from catan_rl.state import new_game_state


TERRAIN_SHORT = {
    0: "DS",
    1: "WD",
    2: "BR",
    3: "SH",
    4: "WH",
    5: "OR",
}

PORT_NAMES = {
    0: "3:1",
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}


def _tile_label(terrain: int, number: int, has_robber: bool, idx: int) -> str:
    t = TERRAIN_SHORT.get(int(terrain), f"T{int(terrain)}")
    token = "--" if int(number) == 0 else f"{int(number):02d}"
    robber = "*" if has_robber else " "
    return f"[{idx:02d} {t} {token}{robber}]"


def _hex_rows() -> list[list[int]]:
    # Matches axial generation order used in topology: row sizes 3-4-5-4-3.
    return [
        [0, 1, 2],
        [3, 4, 5, 6],
        [7, 8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18],
    ]


def _edge_hex_map(hex_to_vertices) -> dict[tuple[int, int], list[int]]:
    edge_to_hexes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for h in range(NUM_HEXES):
        hv = hex_to_vertices[h]
        for i in range(6):
            u = int(hv[i])
            v = int(hv[(i + 1) % 6])
            key = (u, v) if u < v else (v, u)
            edge_to_hexes[key].append(h)
    return edge_to_hexes


def _print_hex_map(state) -> None:
    print("Hex map (index terrain token; * = robber):")
    for row_i, row in enumerate(_hex_rows()):
        indent = " " * (4 - row_i if row_i <= 2 else row_i)
        labels = []
        for h in row:
            labels.append(
                _tile_label(
                    terrain=int(state.hex_terrain[h]),
                    number=int(state.hex_number[h]),
                    has_robber=(h == int(state.robber_hex)),
                    idx=h,
                )
            )
        print(f"{indent}{' '.join(labels)}")
    print()


def _print_ports(state) -> None:
    edge_to_hexes = _edge_hex_map(state.topology.hex_to_vertices)
    print("Ports (type, vertices, adjacent hex):")
    for p in range(len(state.port_type)):
        a = int(state.port_vertices[p, 0])
        b = int(state.port_vertices[p, 1])
        key = (a, b) if a < b else (b, a)
        adjacent = edge_to_hexes.get(key, [])
        adj_str = ",".join(str(h) for h in adjacent) if adjacent else "?"
        port_name = PORT_NAMES.get(int(state.port_type[p]), f"T{int(state.port_type[p])}")
        print(f"  port={p:02d} type={port_name:5s} edge=({a}-{b}) coast_hex={adj_str}")
    print()


def _print_counts(state) -> None:
    terrain_counts = [int((state.hex_terrain == i).sum()) for i in range(6)]
    print("Terrain counts [DESERT, WOOD, BRICK, SHEEP, WHEAT, ORE]:", terrain_counts)
    tokens = sorted(int(n) for n in state.hex_number.tolist() if int(n) > 0)
    print("Number tokens:", tokens)
    print(f"Robber hex: {int(state.robber_hex)}")
    print()


def _print_vertex_edge_tables(state) -> None:
    print("Vertices (id -> adjacent hexes / adjacent vertices):")
    for v in range(len(state.topology.vertex_to_hexes)):
        hexes = [int(h) for h in state.topology.vertex_to_hexes[v] if int(h) >= 0]
        neigh = [int(u) for u in state.topology.vertex_to_vertices[v] if int(u) >= 0]
        print(f"  v{v:02d}: hexes={hexes} neigh={neigh}")
    print()
    print("Edges (id -> vertex endpoints):")
    for e in range(len(state.topology.edge_to_vertices)):
        u = int(state.topology.edge_to_vertices[e, 0])
        v = int(state.topology.edge_to_vertices[e, 1])
        print(f"  e{e:02d}: ({u}-{v})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated Catan board layouts.")
    parser.add_argument("--seed", type=int, default=123, help="Seed for deterministic board generation.")
    parser.add_argument("--show-topology", action="store_true", help="Print full vertex/edge topology tables.")
    args = parser.parse_args()

    state = new_game_state(seed=args.seed)
    print(f"=== Board Inspect (seed={args.seed}) ===\n")
    _print_hex_map(state)
    _print_ports(state)
    _print_counts(state)
    if args.show_topology:
        _print_vertex_edge_tables(state)


if __name__ == "__main__":
    main()
