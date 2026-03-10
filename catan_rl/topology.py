"""Deterministic true Catan board topology tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import NUM_EDGES, NUM_HEXES, NUM_PORTS, NUM_VERTICES


@dataclass(frozen=True)
class Topology:
    vertex_to_hexes: np.ndarray
    vertex_to_edges: np.ndarray
    vertex_to_vertices: np.ndarray
    edge_to_vertices: np.ndarray
    hex_to_vertices: np.ndarray
    hex_to_hexes: np.ndarray
    port_vertices: np.ndarray
    port_type: np.ndarray
    port_vertex_mask: np.ndarray


def _pad_neighbors(items: list[list[int]], width: int) -> np.ndarray:
    arr = np.full((len(items), width), -1, dtype=np.int64)
    for i, values in enumerate(items):
        for j, v in enumerate(values[:width]):
            arr[i, j] = v
    return arr


def _hex_axial_coords(radius: int = 2) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r in range(-radius, radius + 1):
        qmin = max(-radius, -r - radius)
        qmax = min(radius, -r + radius)
        for q in range(qmin, qmax + 1):
            coords.append((q, r))
    return coords


def _axial_to_xy(q: int, r: int) -> tuple[float, float]:
    # Pointy-top axial coordinates.
    x = np.sqrt(3.0) * (q + 0.5 * r)
    y = 1.5 * r
    return float(x), float(y)


def _corner_points(cx: float, cy: float) -> list[tuple[float, float]]:
    # Clockwise corners around a pointy-top hex.
    angles = [30, 90, 150, 210, 270, 330]
    pts = []
    for deg in angles:
        rad = np.deg2rad(deg)
        pts.append((cx + np.cos(rad), cy + np.sin(rad)))
    return pts


def _canonical_point_key(x: float, y: float) -> tuple[float, float]:
    return (round(x, 6), round(y, 6))


def _build_hex_to_hexes(hex_coords: list[tuple[int, int]]) -> np.ndarray:
    coord_to_idx = {c: i for i, c in enumerate(hex_coords)}
    # Axial neighbor directions.
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    neighbors: list[list[int]] = []
    for q, r in hex_coords:
        row = []
        for dq, dr in directions:
            idx = coord_to_idx.get((q + dq, r + dr), -1)
            row.append(idx)
        neighbors.append(row)
    return _pad_neighbors(neighbors, 6)


def _select_port_edges(boundary_edges: list[int], edge_to_vertices: np.ndarray, vertex_xy: np.ndarray) -> np.ndarray:
    mid_angles = []
    for e in boundary_edges:
        u, v = edge_to_vertices[e]
        x = 0.5 * (vertex_xy[u, 0] + vertex_xy[v, 0])
        y = 0.5 * (vertex_xy[u, 1] + vertex_xy[v, 1])
        a = np.arctan2(y, x)
        if a < 0:
            a += 2 * np.pi
        mid_angles.append((e, float(a)))
    mid_angles.sort(key=lambda t: t[1])
    ordered_edges = [e for e, _ in mid_angles]
    # Pick 9 roughly even port edges around coastline.
    idxs = [int(i * len(ordered_edges) / NUM_PORTS) for i in range(NUM_PORTS)]
    selected = [ordered_edges[i] for i in idxs]
    return np.asarray(selected, dtype=np.int64)


def build_topology() -> Topology:
    hex_coords = _hex_axial_coords(radius=2)
    assert len(hex_coords) == NUM_HEXES
    hex_to_hexes = _build_hex_to_hexes(hex_coords)

    vertex_index: dict[tuple[float, float], int] = {}
    vertex_xy: list[tuple[float, float]] = []
    hex_to_vertices = np.zeros((NUM_HEXES, 6), dtype=np.int64)

    for h, (q, r) in enumerate(hex_coords):
        cx, cy = _axial_to_xy(q, r)
        corners = _corner_points(cx, cy)
        for i, (x, y) in enumerate(corners):
            key = _canonical_point_key(x, y)
            if key not in vertex_index:
                vertex_index[key] = len(vertex_xy)
                vertex_xy.append(key)
            hex_to_vertices[h, i] = vertex_index[key]

    vertex_xy_arr = np.asarray(vertex_xy, dtype=np.float64)
    assert vertex_xy_arr.shape[0] == NUM_VERTICES

    edge_index: dict[tuple[int, int], int] = {}
    edge_to_vertices_list: list[tuple[int, int]] = []
    edge_to_hexes: list[list[int]] = []
    for h in range(NUM_HEXES):
        hv = hex_to_vertices[h]
        for i in range(6):
            u = int(hv[i])
            v = int(hv[(i + 1) % 6])
            key = (u, v) if u < v else (v, u)
            if key not in edge_index:
                edge_index[key] = len(edge_to_vertices_list)
                edge_to_vertices_list.append(key)
                edge_to_hexes.append([h])
            else:
                edge_to_hexes[edge_index[key]].append(h)

    edge_to_vertices = np.asarray(edge_to_vertices_list, dtype=np.int64)
    assert edge_to_vertices.shape == (NUM_EDGES, 2)
    assert len(edge_to_hexes) == NUM_EDGES

    vertex_edges: list[list[int]] = [[] for _ in range(NUM_VERTICES)]
    vertex_neighbors: list[list[int]] = [[] for _ in range(NUM_VERTICES)]
    for e, (u, v) in enumerate(edge_to_vertices):
        vertex_edges[int(u)].append(e)
        vertex_edges[int(v)].append(e)
        vertex_neighbors[int(u)].append(int(v))
        vertex_neighbors[int(v)].append(int(u))

    for v in range(NUM_VERTICES):
        vertex_edges[v].sort()
        vertex_neighbors[v].sort()

    vertex_to_edges = _pad_neighbors(vertex_edges, 3)
    vertex_to_vertices = _pad_neighbors(vertex_neighbors, 3)

    vertex_hexes: list[list[int]] = [[] for _ in range(NUM_VERTICES)]
    for h in range(NUM_HEXES):
        for v in hex_to_vertices[h]:
            vertex_hexes[int(v)].append(h)
    for v in range(NUM_VERTICES):
        vertex_hexes[v].sort()
    vertex_to_hexes = _pad_neighbors(vertex_hexes, 3)

    boundary_edges = [e for e, hs in enumerate(edge_to_hexes) if len(hs) == 1]
    assert len(boundary_edges) == 30
    selected_port_edges = _select_port_edges(boundary_edges, edge_to_vertices, vertex_xy_arr)
    port_vertices = edge_to_vertices[selected_port_edges].copy()

    # 4x generic + 5x specific ports.
    port_type = np.asarray([0, 1, 2, 3, 4, 5, 0, 0, 0], dtype=np.int64)

    port_vertex_mask = np.zeros(NUM_VERTICES, dtype=np.int64)
    for a, b in port_vertices:
        port_vertex_mask[int(a)] = 1
        port_vertex_mask[int(b)] = 1

    return Topology(
        vertex_to_hexes=vertex_to_hexes,
        vertex_to_edges=vertex_to_edges,
        vertex_to_vertices=vertex_to_vertices,
        edge_to_vertices=edge_to_vertices,
        hex_to_vertices=hex_to_vertices,
        hex_to_hexes=hex_to_hexes,
        port_vertices=port_vertices,
        port_type=port_type,
        port_vertex_mask=port_vertex_mask,
    )
