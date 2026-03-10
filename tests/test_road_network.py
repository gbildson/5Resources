import numpy as np

from catan_rl.constants import Building
from catan_rl.env import CatanEnv


def _edge_lookup(env: CatanEnv) -> dict[tuple[int, int], int]:
    return {
        tuple(sorted((int(u), int(v)))): i
        for i, (u, v) in enumerate(env.state.topology.edge_to_vertices)
    }


def _find_simple_vertex_path(
    env: CatanEnv,
    edge_count: int,
    forbidden_vertices: set[int] | None = None,
) -> list[int]:
    forbidden = forbidden_vertices or set()
    topo = env.state.topology
    target_vertices = edge_count + 1

    def dfs(v: int, path: list[int], visited: set[int]) -> list[int] | None:
        if len(path) == target_vertices:
            return path.copy()
        for nxt in topo.vertex_to_vertices[v]:
            nxt = int(nxt)
            if nxt < 0 or nxt in visited or nxt in forbidden:
                continue
            visited.add(nxt)
            path.append(nxt)
            got = dfs(nxt, path, visited)
            if got is not None:
                return got
            path.pop()
            visited.remove(nxt)
        return None

    for start in range(len(topo.vertex_to_vertices)):
        if start in forbidden:
            continue
        path = [start]
        got = dfs(start, path, {start})
        if got is not None:
            return got
    raise AssertionError(f"Could not find simple path with {edge_count} edges")


def _path_to_edges(env: CatanEnv, vertices: list[int]) -> list[int]:
    edge_map = _edge_lookup(env)
    edges = []
    for a, b in zip(vertices[:-1], vertices[1:]):
        edges.append(edge_map[tuple(sorted((int(a), int(b))))])
    return edges


def test_longest_road_blocked_by_opponent_settlement():
    env = CatanEnv(seed=404)
    env.reset(seed=404)

    # Player 0 owns chain edges: (0-1), (1-2), (2-3)
    env.state.edge_owner[:] = -1
    env.state.edge_road[:] = 0
    for e in (0, 1, 2):
        env.state.edge_owner[e] = 0
        env.state.edge_road[e] = 1

    # No blockers -> length 3
    lengths = env._road_lengths()
    assert lengths[0] == 3

    # Opponent building at vertex 1 blocks continuation across that vertex.
    env.state.vertex_owner[1] = 1
    env.state.vertex_building[1] = Building.SETTLEMENT
    lengths = env._road_lengths()
    assert lengths[0] == 2


def test_branching_roads_do_not_overcount_without_cycle():
    env = CatanEnv(seed=405)
    env.reset(seed=405)

    # Build a "Y" at vertex 0:
    # edge 53: (53-0), edge 0: (0-1), edge 54: (0-27)
    env.state.edge_owner[:] = -1
    env.state.edge_road[:] = 0
    for e in (53, 0, 54):
        env.state.edge_owner[e] = 0
        env.state.edge_road[e] = 1

    lengths = env._road_lengths()
    # Best trail in this branch-only shape is 2.
    assert lengths[0] == 2


def test_longest_road_award_transfers_when_block_reduces_holder():
    env = CatanEnv(seed=406)
    env.reset(seed=406)
    env.state.edge_owner[:] = -1
    env.state.edge_road[:] = 0
    env.state.vertex_owner[:] = -1
    env.state.vertex_building[:] = Building.EMPTY

    # Player 0: length-6 chain.
    p0_vertices = _find_simple_vertex_path(env, edge_count=6)
    p0_edges = _path_to_edges(env, p0_vertices)
    for e in p0_edges:
        env.state.edge_owner[e] = 0
        env.state.edge_road[e] = 1

    # Player 1: disjoint length-5 chain.
    p1_vertices = _find_simple_vertex_path(env, edge_count=5, forbidden_vertices=set(p0_vertices))
    p1_edges = _path_to_edges(env, p1_vertices)
    for e in p1_edges:
        env.state.edge_owner[e] = 1
        env.state.edge_road[e] = 1

    env._update_vp_and_achievements()
    assert env.state.has_longest_road[0] == 1

    # Block player 0 in the middle so p1 should take longest road.
    mid_vertex = int(p0_vertices[len(p0_vertices) // 2])
    env.state.vertex_owner[mid_vertex] = 1
    env.state.vertex_building[mid_vertex] = Building.SETTLEMENT
    env._update_vp_and_achievements()
    assert env.state.has_longest_road[1] == 1
