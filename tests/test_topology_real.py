import numpy as np

from catan_rl.constants import NUM_EDGES, NUM_HEXES, NUM_PORTS, NUM_VERTICES
from catan_rl.state import new_game_state
from catan_rl.topology import build_topology


def test_topology_matches_true_catan_counts():
    topo = build_topology()
    assert topo.hex_to_vertices.shape == (NUM_HEXES, 6)
    assert topo.hex_to_hexes.shape == (NUM_HEXES, 6)
    assert topo.edge_to_vertices.shape == (NUM_EDGES, 2)
    assert topo.vertex_to_hexes.shape == (NUM_VERTICES, 3)
    assert topo.vertex_to_edges.shape == (NUM_VERTICES, 3)
    assert topo.vertex_to_vertices.shape == (NUM_VERTICES, 3)
    assert topo.port_vertices.shape == (NUM_PORTS, 2)
    vertex_degrees = (topo.vertex_to_edges >= 0).sum(axis=1)
    assert set(vertex_degrees.tolist()).issubset({2, 3})
    hex_incidence = (topo.vertex_to_hexes >= 0).sum(axis=1)
    assert int((hex_incidence == 1).sum()) == 18
    assert int((hex_incidence == 2).sum()) == 12
    assert int((hex_incidence == 3).sum()) == 24
    # Hex adjacency should be symmetric.
    for h in range(NUM_HEXES):
        for n in topo.hex_to_hexes[h]:
            n = int(n)
            if n < 0:
                continue
            assert h in [int(x) for x in topo.hex_to_hexes[n] if int(x) >= 0]


def test_boundary_and_port_structure():
    topo = build_topology()
    edge_key_to_id = {
        tuple(sorted((int(u), int(v)))): i for i, (u, v) in enumerate(topo.edge_to_vertices)
    }
    edge_hex_count = np.zeros(NUM_EDGES, dtype=np.int64)
    for h in range(NUM_HEXES):
        hv = topo.hex_to_vertices[h]
        for i in range(6):
            key = tuple(sorted((int(hv[i]), int(hv[(i + 1) % 6]))))
            edge_hex_count[edge_key_to_id[key]] += 1

    boundary_edges = np.flatnonzero(edge_hex_count == 1)
    assert boundary_edges.size == 30

    boundary_edge_set = set(int(e) for e in boundary_edges)
    port_edge_ids = set()
    for a, b in topo.port_vertices:
        key = tuple(sorted((int(a), int(b))))
        e = edge_key_to_id[key]
        port_edge_ids.add(e)
        assert e in boundary_edge_set
    assert len(port_edge_ids) == NUM_PORTS


def test_official_terrain_distribution():
    state = new_game_state(seed=123)
    counts = np.bincount(state.hex_terrain, minlength=6).tolist()
    # [desert, wood, brick, sheep, wheat, ore]
    assert counts == [1, 4, 3, 4, 4, 3]
    assert int((state.hex_number == 0).sum()) == 1
    assert sorted(int(n) for n in state.hex_number.tolist() if int(n) > 0) == [
        2,
        3,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
        12,
    ]


def test_no_adjacent_six_eight_tokens_across_seeds():
    for seed in range(200):
        state = new_game_state(seed=seed)
        hot_hexes = set(int(h) for h in np.flatnonzero(np.isin(state.hex_number, [6, 8])))
        for h in hot_hexes:
            for n in state.topology.hex_to_hexes[h]:
                n = int(n)
                if n >= 0:
                    assert n not in hot_hexes
