"""Strategy-oriented metrics for setup and road frontier analysis."""

from __future__ import annotations

from collections import deque

import numpy as np

from .constants import Building, DevCard, NUM_PLAYERS, NUM_RESOURCES, Phase, RESOURCE_COSTS
from .state import GameState


def _is_distance_rule_open(state: GameState, vertex: int) -> bool:
    if int(state.vertex_building[int(vertex)]) != int(Building.EMPTY):
        return False
    for n in state.topology.vertex_to_vertices[int(vertex)]:
        if n >= 0 and int(state.vertex_building[int(n)]) != int(Building.EMPTY):
            return False
    return True


def _vertex_resource_pips(state: GameState, vertex: int, *, robber_aware: bool) -> np.ndarray:
    out = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for h in state.topology.vertex_to_hexes[int(vertex)]:
        if h < 0:
            continue
        if robber_aware and int(h) == int(state.robber_hex):
            continue
        terrain = int(state.hex_terrain[int(h)])
        if terrain <= 0:
            continue
        out[terrain - 1] += float(state.hex_pip_count[int(h)])
    return out


def _self_production_profile(state: GameState, player: int) -> np.ndarray:
    prof = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) != int(player):
            continue
        b = int(state.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        prof += mult * _vertex_resource_pips(state, v, robber_aware=False)
    return prof


def _global_resource_pips(state: GameState) -> np.ndarray:
    out = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for h in range(len(state.hex_terrain)):
        terrain = int(state.hex_terrain[h])
        if terrain <= 0:
            continue
        out[terrain - 1] += float(state.hex_pip_count[h])
    return out


def setup_settlement_score(state: GameState, player: int, vertex: int) -> float:
    def _port_proximity_bonus(candidate: int, vec: np.ndarray) -> float:
        # Small bonus for useful nearby ports; one-road-away is weighted above on-port.
        inf = 10_000
        dist = np.full(len(state.vertex_owner), inf, dtype=np.int32)
        dist[int(candidate)] = 0
        dq: deque[int] = deque([int(candidate)])
        while dq:
            v = int(dq.popleft())
            d = int(dist[v])
            if d >= 2:
                continue
            for n in state.topology.vertex_to_vertices[v]:
                if n < 0:
                    continue
                ni = int(n)
                if d + 1 < int(dist[ni]):
                    dist[ni] = d + 1
                    dq.append(ni)

        best = 0.0
        for p in range(len(state.port_type)):
            a = int(state.port_vertices[p, 0])
            b = int(state.port_vertices[p, 1])
            d = min(int(dist[a]), int(dist[b]))
            if d > 2:
                continue
            port_t = int(state.port_type[p])
            if port_t == 0:
                desirability = 0.45
            else:
                own = float(vec[port_t - 1])
                desirability = 0.35 + 0.35 * min(own, 8.0) / 8.0
            dist_w = 0.35 if d == 0 else (1.0 if d == 1 else 0.55)
            best = max(best, desirability * dist_w)
        return 0.45 * best

    self_prod = _self_production_profile(state, int(player))
    global_pips = _global_resource_pips(state)
    vec = _vertex_resource_pips(state, int(vertex), robber_aware=False)
    pips_total = float(vec.sum())
    diversity = float(np.count_nonzero(vec > 0))
    ows_synergy = float(min(vec[2], vec[3], vec[4]))  # sheep/wheat/ore
    road_synergy = float(min(vec[0], vec[1]))  # wood/brick
    weights = np.asarray([1.0, 1.0, 1.0, 1.25, 1.25], dtype=np.float32)
    if vec[0] > 0.0 and vec[1] > 0.0:
        # Boost wood+brick only when they co-occur on the same candidate vertex.
        weights[0] = 1.10
        weights[1] = 1.10
    weighted = float(np.sum(vec * weights))
    missing_mask = (self_prod <= 1.0).astype(np.float32)
    complement = float((vec * missing_mask).sum())
    scarcity = float(np.sum(vec / np.maximum(1.0, global_pips)))
    wheat_bonus = 0.35 * float(vec[3])
    if float(self_prod[3]) <= 1.0:
        wheat_bonus += 0.55 * float(vec[3])
    wheat_penalty = 1.20 if float(vec[3]) <= 0.0 else 0.0
    port_bonus = _port_proximity_bonus(vertex, vec)
    return (
        pips_total
        + 0.7 * diversity
        + 0.25 * (ows_synergy + road_synergy)
        + 0.08 * weighted
        + 0.15 * complement
        + 2.5 * scarcity
        + wheat_bonus
        - wheat_penalty
        + port_bonus
    )


def setup_choice_metrics(state: GameState, player: int, vertex: int) -> dict:
    scores: list[tuple[int, float]] = []
    for v in range(len(state.vertex_owner)):
        if not _is_distance_rule_open(state, v):
            continue
        scores.append((int(v), float(setup_settlement_score(state, int(player), int(v)))))
    if not scores:
        return {
            "candidate_count": 0,
            "rank": -1,
            "percentile": 0.0,
            "score": 0.0,
            "best_score": 0.0,
            "score_gap_to_best": 0.0,
            "total_pips": 0.0,
            "diversity": 0,
        }
    scores.sort(key=lambda x: x[1], reverse=True)
    by_vertex = {v: s for v, s in scores}
    rank = 1 + sum(1 for v, s in scores if s > by_vertex.get(int(vertex), -1e9))
    best_score = float(scores[0][1])
    cur_score = float(by_vertex.get(int(vertex), 0.0))
    vec = _vertex_resource_pips(state, int(vertex), robber_aware=False)
    return {
        "candidate_count": int(len(scores)),
        "rank": int(rank),
        "percentile": float((len(scores) - rank) / max(1, len(scores) - 1)),
        "score": cur_score,
        "best_score": best_score,
        "score_gap_to_best": float(best_score - cur_score),
        "total_pips": float(vec.sum()),
        "diversity": int(np.count_nonzero(vec > 0)),
    }


def _frontier_costs(state: GameState, player: int, extra_edges: set[int] | None = None) -> np.ndarray:
    topo = state.topology
    n_vertices = len(state.vertex_owner)
    inf = 10_000
    dist = np.full(n_vertices, inf, dtype=np.int32)
    dq: deque[int] = deque()
    extra = extra_edges or set()

    for v in range(n_vertices):
        owned_building = int(state.vertex_owner[v]) == int(player) and int(state.vertex_building[v]) != int(Building.EMPTY)
        has_road = any(
            e >= 0 and (int(state.edge_owner[int(e)]) == int(player) or int(e) in extra)
            for e in topo.vertex_to_edges[v]
        )
        if owned_building or has_road:
            dist[v] = 0
            dq.append(v)

    while dq:
        v = int(dq.popleft())
        base = int(dist[v])
        for e in topo.vertex_to_edges[v]:
            if e < 0:
                continue
            owner = int(state.edge_owner[int(e)])
            if owner >= 0 and owner != int(player):
                continue
            w = 0 if (owner == int(player) or int(e) in extra) else 1
            u = int(topo.edge_to_vertices[int(e), 0])
            t = int(topo.edge_to_vertices[int(e), 1])
            nxt = t if u == v else u
            cand = base + w
            if cand < int(dist[nxt]):
                dist[nxt] = cand
                if w == 0:
                    dq.appendleft(nxt)
                else:
                    dq.append(nxt)
    return dist.astype(np.float32)


def _roads_needed_to_reach_site(state: GameState, player: int, vertex: int, frontier_cost: np.ndarray) -> int | None:
    if any(e >= 0 and int(state.edge_owner[int(e)]) == int(player) for e in state.topology.vertex_to_edges[int(vertex)]):
        return 0
    best = 10_000
    for e in state.topology.vertex_to_edges[int(vertex)]:
        if e < 0:
            continue
        owner = int(state.edge_owner[int(e)])
        if owner >= 0 and owner != int(player):
            continue
        u = int(state.topology.edge_to_vertices[int(e), 0])
        v = int(state.topology.edge_to_vertices[int(e), 1])
        other = v if u == int(vertex) else u
        cand = int(frontier_cost[other]) + 1
        best = min(best, cand)
    return None if best > 6 else best


def _count_connected_settlement_opportunities(state: GameState, player: int, extra_edges: set[int] | None = None) -> int:
    extra = extra_edges or set()
    topo = state.topology
    count = 0
    for v in range(len(state.vertex_owner)):
        if not _is_distance_rule_open(state, v):
            continue
        connected = False
        for e in topo.vertex_to_edges[v]:
            if e < 0:
                continue
            if int(state.edge_owner[int(e)]) == int(player) or int(e) in extra:
                connected = True
                break
        if connected:
            count += 1
    return count


def road_frontier_metrics(state: GameState, player: int, edge: int) -> dict:
    if int(state.edge_owner[int(edge)]) >= 0:
        return {
            "connected_before": 0,
            "connected_after": 0,
            "connected_gain": 0,
            "best_site_roads_before": -1,
            "best_site_roads_after": -1,
            "best_site_roads_delta": 0,
        }
    before_connected = _count_connected_settlement_opportunities(state, int(player))
    after_connected = _count_connected_settlement_opportunities(state, int(player), extra_edges={int(edge)})

    frontier_before = _frontier_costs(state, int(player))
    frontier_after = _frontier_costs(state, int(player), extra_edges={int(edge)})
    best_before: int | None = None
    best_after: int | None = None
    for v in range(len(state.vertex_owner)):
        if not _is_distance_rule_open(state, v):
            continue
        rb = _roads_needed_to_reach_site(state, int(player), int(v), frontier_before)
        ra = _roads_needed_to_reach_site(state, int(player), int(v), frontier_after)
        if rb is not None:
            best_before = rb if best_before is None else min(best_before, rb)
        if ra is not None:
            best_after = ra if best_after is None else min(best_after, ra)
    if best_before is None:
        best_before = -1
    if best_after is None:
        best_after = -1
    delta = 0 if (best_before < 0 or best_after < 0) else int(best_before - best_after)
    return {
        "connected_before": int(before_connected),
        "connected_after": int(after_connected),
        "connected_gain": int(after_connected - before_connected),
        "best_site_roads_before": int(best_before),
        "best_site_roads_after": int(best_after),
        "best_site_roads_delta": int(delta),
    }


def road_port_reach_metrics(state: GameState, player: int, edge: int) -> dict:
    """
    Estimate how a road changes distance (in roads needed) to any port-access vertex.
    Lower is better; returns positive delta when the candidate edge improves access.
    """
    if int(state.edge_owner[int(edge)]) >= 0:
        return {
            "best_port_roads_before": -1,
            "best_port_roads_after": -1,
            "best_port_roads_delta": 0,
        }
    port_vertices: set[int] = set()
    for p in range(len(state.port_vertices)):
        a = int(state.port_vertices[p, 0])
        b = int(state.port_vertices[p, 1])
        port_vertices.add(a)
        port_vertices.add(b)
    if not port_vertices:
        return {
            "best_port_roads_before": -1,
            "best_port_roads_after": -1,
            "best_port_roads_delta": 0,
        }

    frontier_before = _frontier_costs(state, int(player))
    frontier_after = _frontier_costs(state, int(player), extra_edges={int(edge)})
    best_before: int | None = None
    best_after: int | None = None
    for v in sorted(port_vertices):
        rb = _roads_needed_to_reach_site(state, int(player), int(v), frontier_before)
        ra = _roads_needed_to_reach_site(state, int(player), int(v), frontier_after)
        if rb is not None:
            best_before = rb if best_before is None else min(best_before, rb)
        if ra is not None:
            best_after = ra if best_after is None else min(best_after, ra)
    if best_before is None:
        best_before = -1
    if best_after is None:
        best_after = -1
    delta = 0 if (best_before < 0 or best_after < 0) else int(best_before - best_after)
    return {
        "best_port_roads_before": int(best_before),
        "best_port_roads_after": int(best_after),
        "best_port_roads_delta": int(delta),
    }


def city_upgrade_metrics(state: GameState, player: int, vertex: int) -> dict:
    if int(state.vertex_owner[int(vertex)]) != int(player) or int(state.vertex_building[int(vertex)]) != int(Building.SETTLEMENT):
        return {
            "is_upgrade_candidate": False,
            "city_delta_pips": 0.0,
            "city_delta_score": 0.0,
            "rank": -1,
            "candidate_count": 0,
        }
    deltas: list[tuple[int, float]] = []
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) != int(player) or int(state.vertex_building[v]) != int(Building.SETTLEMENT):
            continue
        delta = float(_vertex_resource_pips(state, int(v), robber_aware=True).sum())
        deltas.append((int(v), delta))
    deltas.sort(key=lambda x: x[1], reverse=True)
    by_vertex = {v: d for v, d in deltas}
    cur = float(by_vertex.get(int(vertex), 0.0))
    rank = 1 + sum(1 for _, d in deltas if d > cur)
    return {
        "is_upgrade_candidate": True,
        "city_delta_pips": cur,
        "city_delta_score": float(np.clip(cur / 15.0, 0.0, 1.0)),
        "rank": int(rank),
        "candidate_count": int(len(deltas)),
    }


def _blocked_production_on_hex(state: GameState, hex_id: int, player: int) -> float:
    blocked = 0.0
    for v in state.topology.hex_to_vertices[int(hex_id)]:
        if int(state.vertex_owner[v]) != int(player):
            continue
        b = int(state.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        blocked += float(state.hex_pip_count[int(hex_id)]) * mult
    return blocked


def opponent_leader_set(state: GameState, player: int, threat_dev_card_weight: float = 0.7) -> set[int]:
    opponents = [p for p in range(len(state.public_vp)) if p != int(player)]
    if not opponents:
        return set()
    dev_counts = (state.dev_cards_hidden + state.dev_cards_bought_this_turn).sum(axis=1).astype(np.float32)
    threats = state.public_vp.astype(np.float32) + float(threat_dev_card_weight) * dev_counts
    top = float(np.max(threats[opponents]))
    return set(p for p in opponents if float(threats[p]) == top)


def robber_block_quality(state: GameState, player: int, hex_id: int, threat_dev_card_weight: float = 0.7) -> dict:
    leaders = opponent_leader_set(state, int(player), float(threat_dev_card_weight))
    blocked_leader = float(sum(_blocked_production_on_hex(state, int(hex_id), p) for p in leaders))
    blocked_self = float(_blocked_production_on_hex(state, int(hex_id), int(player)))
    raw = blocked_leader - blocked_self
    return {
        "leader_count": int(len(leaders)),
        "blocked_leader_pips": blocked_leader,
        "blocked_self_pips": blocked_self,
        "block_delta_pips": raw,
        "quality_score": float(np.clip(raw / 10.0, -1.0, 1.0)),
    }


def anti_leader_rob_quality(state: GameState, player: int, target_player: int, threat_dev_card_weight: float = 0.7) -> dict:
    leaders = opponent_leader_set(state, int(player), float(threat_dev_card_weight))
    target = int(target_player)
    robbed_leader = bool(target in leaders)
    target_cards = int(state.resource_total[target]) if 0 <= target < NUM_PLAYERS else 0
    return {
        "leader_count": int(len(leaders)),
        "target_player": target,
        "robbed_leader": robbed_leader,
        "target_resource_total": target_cards,
        "quality_score": float(1.0 if robbed_leader else -1.0),
    }


def race_pressure_metrics(state: GameState, player: int) -> dict:
    p = int(player)
    lr = state.longest_road_length.astype(np.float32)
    kn = state.knights_played.astype(np.float32)
    my_lr = float(lr[p])
    my_kn = float(kn[p])
    opp_lr_top = float(np.max(np.delete(lr, p))) if NUM_PLAYERS > 1 else my_lr
    opp_kn_top = float(np.max(np.delete(kn, p))) if NUM_PLAYERS > 1 else my_kn
    return {
        "longest_road_gap": float(opp_lr_top - my_lr),
        "largest_army_gap": float(opp_kn_top - my_kn),
        "longest_road_pressure": float(np.clip((opp_lr_top - my_lr) / 10.0, -1.0, 1.0)),
        "largest_army_pressure": float(np.clip((opp_kn_top - my_kn) / 6.0, -1.0, 1.0)),
    }


def build_readiness_estimates(state: GameState, player: int, *, robber_aware: bool = True) -> dict:
    p = int(player)
    resources = state.resources[p].astype(np.float32)
    per_turn_prod = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) != p:
            continue
        b = int(state.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        per_turn_prod += mult * _vertex_resource_pips(state, int(v), robber_aware=bool(robber_aware)) / 36.0

    out: dict[str, dict[str, float]] = {}
    for key in ("road", "settlement", "city", "dev"):
        cost = np.asarray(RESOURCE_COSTS[key], dtype=np.float32)
        deficit = np.maximum(cost - resources, 0.0)
        total_deficit = float(deficit.sum())
        if total_deficit <= 0.0:
            est_turns = 0.0
        else:
            turns = []
            for r in range(NUM_RESOURCES):
                need = float(deficit[r])
                if need <= 0.0:
                    continue
                rate = float(per_turn_prod[r])
                turns.append(need / max(rate, 1e-3))
            est_turns = float(max(turns)) if turns else 0.0
        out[key] = {
            "total_deficit": total_deficit,
            "can_build_now": float(1.0 if total_deficit == 0.0 else 0.0),
            "estimated_turns": float(np.clip(est_turns, 0.0, 50.0)),
        }
    return out


def _estimate_trade_build_value(state: GameState, player: int, resources: np.ndarray) -> float:
    """Approximate value from current hand and near-term build readiness."""
    p = int(player)
    resources = np.asarray(resources, dtype=np.float32)
    per_turn_prod = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) != p:
            continue
        b = int(state.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        per_turn_prod += mult * _vertex_resource_pips(state, int(v), robber_aware=True) / 36.0

    score = 0.0
    for key in ("road", "settlement", "city", "dev"):
        cost = np.asarray(RESOURCE_COSTS[key], dtype=np.float32)
        deficit = np.maximum(cost - resources, 0.0)
        total_deficit = float(deficit.sum())
        if total_deficit <= 0.0:
            ready = 1.0
            eta = 0.0
        else:
            turns = []
            for r in range(NUM_RESOURCES):
                need = float(deficit[r])
                if need <= 0.0:
                    continue
                rate = float(per_turn_prod[r])
                turns.append(need / max(rate, 1e-3))
            eta = float(max(turns)) if turns else 0.0
            ready = float(np.clip(1.0 - eta / 15.0, 0.0, 1.0))
        if key == "city":
            w = 1.2
        elif key == "settlement":
            w = 1.0
        elif key == "dev":
            w = 0.55
        else:
            w = 0.65
        score += w * (ready + (1.0 if total_deficit <= 0.0 else 0.0))
    # Light hand-size pressure term to reduce discard-risk by preferring value density.
    score += float(np.clip(8.0 - float(resources.sum()), -3.0, 3.0)) * 0.03
    return float(score)


def _trade_affordable_build_flags(state: GameState, player: int, resources: np.ndarray) -> dict[str, bool]:
    p = int(player)
    resources = np.asarray(resources, dtype=np.float32)
    out: dict[str, bool] = {}
    if int(state.roads_left[p]) > 0:
        out["road"] = bool(np.all(resources >= np.asarray(RESOURCE_COSTS["road"], dtype=np.float32)))
    if int(state.settlements_left[p]) > 0:
        out["settlement"] = bool(np.all(resources >= np.asarray(RESOURCE_COSTS["settlement"], dtype=np.float32)))
    if int(state.cities_left[p]) > 0:
        out["city"] = bool(np.all(resources >= np.asarray(RESOURCE_COSTS["city"], dtype=np.float32)))
    if int(state.dev_deck_remaining) > 0:
        out["dev"] = bool(np.all(resources >= np.asarray(RESOURCE_COSTS["dev"], dtype=np.float32)))
    return out


def _trade_immediate_unlock_bonus(state: GameState, player: int, before: np.ndarray, after: np.ndarray) -> float:
    before_flags = _trade_affordable_build_flags(state, int(player), before)
    after_flags = _trade_affordable_build_flags(state, int(player), after)
    bonus = 0.0
    weights = {"city": 0.28, "settlement": 0.24, "dev": 0.14, "road": 0.10}
    for key, weight in weights.items():
        if after_flags.get(key, False) and not before_flags.get(key, False):
            bonus += float(weight)
    return float(np.clip(bonus, 0.0, 0.65))


def _trade_hand_unstuck_bonus(state: GameState, player: int, before: np.ndarray, after: np.ndarray) -> float:
    p = int(player)
    build_costs: list[np.ndarray] = []
    if int(state.roads_left[p]) > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["road"], dtype=np.float32))
    if int(state.settlements_left[p]) > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["settlement"], dtype=np.float32))
    if int(state.cities_left[p]) > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["city"], dtype=np.float32))
    if int(state.dev_deck_remaining) > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["dev"], dtype=np.float32))
    if not build_costs:
        return 0.0

    def _closest_deficit_metrics(resources: np.ndarray) -> tuple[float, float]:
        best_total = 1e9
        best_types = 1e9
        for cost in build_costs:
            deficit = np.maximum(cost - resources, 0.0)
            best_total = min(best_total, float(deficit.sum()))
            best_types = min(best_types, float(np.count_nonzero(deficit > 0.0)))
        return best_total, best_types

    before_total, before_types = _closest_deficit_metrics(np.asarray(before, dtype=np.float32))
    after_total, after_types = _closest_deficit_metrics(np.asarray(after, dtype=np.float32))
    bonus = 0.0
    bonus += 0.08 * float(np.clip(before_total - after_total, 0.0, 2.0))
    bonus += 0.05 * float(np.clip(before_types - after_types, 0.0, 2.0))
    return float(np.clip(bonus, 0.0, 0.26))


def _trade_proposer_penalty_coef(state: GameState, proposer: int) -> float:
    proposer = int(proposer)
    dev_counts = (state.dev_cards_hidden + state.dev_cards_bought_this_turn).sum(axis=1).astype(np.float32)
    public_vp = state.public_vp.astype(np.float32)
    threat = public_vp + 0.7 * dev_counts
    other_idxs = [p for p in range(NUM_PLAYERS) if p != proposer]
    best_other_public_vp = float(np.max(public_vp[other_idxs])) if other_idxs else 0.0
    best_other_threat = float(np.max(threat[other_idxs])) if other_idxs else 0.0
    proposer_public_vp = float(public_vp[proposer])
    proposer_threat = float(threat[proposer])
    proposer_dev_count = float(dev_counts[proposer])

    if proposer_public_vp >= (best_other_public_vp + 2.0) or (proposer_public_vp + proposer_dev_count) > 7.0:
        return 0.65
    if proposer_threat >= (best_other_threat + 0.75) or proposer_public_vp > best_other_public_vp:
        return 0.45
    if proposer_public_vp <= (best_other_public_vp - 2.0) and proposer_threat <= (best_other_threat - 0.5):
        return 0.18
    return 0.30


def trade_accept_should_avoid_proposer(state: GameState, responder: int) -> bool:
    """Public-info veto for trades that materially help an already threatening proposer."""
    proposer = int(state.trade_proposer)
    p = int(responder)
    if proposer < 0 or proposer == p:
        return False
    proposer_public_vp = int(state.public_vp[proposer])
    others_public_vp = [int(state.public_vp[q]) for q in range(len(state.public_vp)) if q != proposer]
    best_other_public_vp = max(others_public_vp, default=0)
    clear_public_leader = proposer_public_vp >= (best_other_public_vp + 2)
    proposer_dev_count = int(np.sum(state.dev_cards_hidden[proposer] + state.dev_cards_bought_this_turn[proposer]))
    vp_plus_devs_threat = (proposer_public_vp + proposer_dev_count) > 7
    return bool(clear_public_leader or vp_plus_devs_threat)


def trade_accept_immediate_build_gain(state: GameState, responder: int) -> bool:
    """Whether accepting the pending trade immediately unlocks more build options."""
    p = int(responder)
    proposer = int(state.trade_proposer)
    if proposer < 0:
        return False
    give = state.trade_offer_give.astype(np.float32)
    want = state.trade_offer_want.astype(np.float32)
    before = state.resources[p].astype(np.float32)
    if np.any(before < want):
        return False
    after = before - want + give
    before_flags = _trade_affordable_build_flags(state, p, before)
    after_flags = _trade_affordable_build_flags(state, p, after)
    if sum(int(v) for v in after_flags.values()) > sum(int(v) for v in before_flags.values()):
        return True
    for key in ("settlement", "city", "dev", "road"):
        if after_flags.get(key, False) and not before_flags.get(key, False):
            return True
    return False


def _coerce_trade_vectors(*offer_args) -> tuple[np.ndarray, np.ndarray]:
    if len(offer_args) == 2:
        give = np.asarray(offer_args[0], dtype=np.float32)
        want = np.asarray(offer_args[1], dtype=np.float32)
    elif len(offer_args) == 4:
        give_r, give_n, want_r, want_n = (int(x) for x in offer_args)
        give = np.zeros(NUM_RESOURCES, dtype=np.float32)
        want = np.zeros(NUM_RESOURCES, dtype=np.float32)
        give[give_r] = float(give_n)
        want[want_r] = float(want_n)
    else:
        raise TypeError("trade offer must be provided as (give_vec, want_vec) or (give_r, give_n, want_r, want_n)")
    return give, want


def trade_offer_value(
    state: GameState,
    proposer: int,
    *offer_args,
) -> float:
    p = int(proposer)
    give, want = _coerce_trade_vectors(*offer_args)
    if give.shape[0] != NUM_RESOURCES or want.shape[0] != NUM_RESOURCES:
        return -1.0
    if np.any(give < 0) or np.any(want < 0):
        return -1.0
    if int(give.sum()) <= 0 or int(want.sum()) <= 0:
        return -1.0
    if np.any((give > 0) & (want > 0)):
        return -1.0
    before = state.resources[p].astype(np.float32)
    if np.any(before < give):
        return -1.0
    after = before - give + want
    v_before = _estimate_trade_build_value(state, p, before)
    v_after = _estimate_trade_build_value(state, p, after)
    # Favor more efficient ratios unless hand is near discard threshold.
    ratio_term = float(want.sum()) / max(1.0, float(give.sum()))
    near_discard = float(before.sum()) > 7.0
    efficiency_bonus = 0.15 * ratio_term + (0.10 if near_discard else 0.0)
    return float(np.clip((v_after - v_before) + efficiency_bonus, -1.0, 1.0))


def trade_offer_counterparty_value(
    state: GameState,
    proposer: int,
    *offer_args,
) -> float:
    """
    Utility proxy for how acceptable a proposed offer is to potential responders.
    Returns the best responder utility among legal responders (those who can pay "want").
    """
    p = int(proposer)
    give, want = _coerce_trade_vectors(*offer_args)
    if give.shape[0] != NUM_RESOURCES or want.shape[0] != NUM_RESOURCES:
        return -1.0
    if np.any(give < 0) or np.any(want < 0):
        return -1.0
    if int(give.sum()) <= 0 or int(want.sum()) <= 0:
        return -1.0
    if np.any((give > 0) & (want > 0)):
        return -1.0

    best = -1.0
    found = False
    for r in range(NUM_PLAYERS):
        if r == p:
            continue
        before = state.resources[r].astype(np.float32)
        if np.any(before < want):
            continue
        found = True
        after = before - want + give
        v_before = _estimate_trade_build_value(state, r, before)
        v_after = _estimate_trade_build_value(state, r, after)
        util = float(np.clip(v_after - v_before, -1.0, 1.0))
        if util > best:
            best = util
    return float(best if found else -1.0)


def bank_trade_value(
    state: GameState,
    player: int,
    give_r: int,
    give_n: int,
    want_r: int,
) -> float:
    p = int(player)
    give_r = int(give_r)
    give_n = int(give_n)
    want_r = int(want_r)
    if give_r == want_r or give_n <= 0:
        return -1.0
    before = state.resources[p].astype(np.float32)
    if before[give_r] < give_n:
        return -1.0
    after = before.copy()
    after[give_r] -= float(give_n)
    after[want_r] += 1.0
    v_before = _estimate_trade_build_value(state, p, before)
    v_after = _estimate_trade_build_value(state, p, after)
    return float(np.clip((v_after - v_before), -1.0, 1.0))


def trade_accept_value(state: GameState, responder: int) -> float:
    """Utility for ACCEPT_TRADE versus REJECT_TRADE for current responder."""
    p = int(responder)
    proposer = int(state.trade_proposer)
    if proposer < 0:
        return 0.0
    give = state.trade_offer_give.astype(np.float32)
    want = state.trade_offer_want.astype(np.float32)
    # Responder would give "want" and receive "give".
    before = state.resources[p].astype(np.float32)
    if np.any(before < want):
        return -1.0
    after = before - want + give
    v_before = _estimate_trade_build_value(state, p, before)
    v_after = _estimate_trade_build_value(state, p, after)
    own_gain = float(v_after - v_before)
    own_gain += _trade_immediate_unlock_bonus(state, p, before, after)
    own_gain += _trade_hand_unstuck_bonus(state, p, before, after)
    # Penalize feeding an immediate strong move to proposer, scaled by table threat.
    prop_before = state.resources[proposer].astype(np.float32)
    prop_after = prop_before - give + want
    proposer_gain = _estimate_trade_build_value(state, proposer, prop_after) - _estimate_trade_build_value(
        state, proposer, prop_before
    )
    penalty_coef = _trade_proposer_penalty_coef(state, proposer)
    return float(np.clip(own_gain - float(penalty_coef) * float(proposer_gain), -1.0, 1.0))


def _relative_opponents(values: np.ndarray, player: int) -> np.ndarray:
    p = int(player)
    return np.asarray([values[(p + offset) % NUM_PLAYERS] for offset in range(1, NUM_PLAYERS)], dtype=np.float32)


def _trade_response_target_active(state: GameState, player: int) -> bool:
    p = int(player)
    proposer = int(state.trade_proposer)
    return bool(
        int(state.phase) == int(Phase.TRADE_PROPOSED)
        and proposer >= 0
        and proposer != p
        and int(state.trade_responses[p]) == 0
    )


def opponent_danger_targets(state: GameState, player: int, *, threat_dev_card_weight: float = 0.7) -> dict[str, float]:
    p = int(player)
    public_vp = state.public_vp.astype(np.float32)
    dev_counts = (state.dev_cards_hidden + state.dev_cards_bought_this_turn).sum(axis=1).astype(np.float32)
    road_len = state.longest_road_length.astype(np.float32)
    knight_len = state.knights_played.astype(np.float32)
    resource_total = state.resource_total.astype(np.float32)
    has_road = state.has_longest_road.astype(np.float32)
    has_army = state.has_largest_army.astype(np.float32)

    scores = np.zeros(NUM_PLAYERS, dtype=np.float32)
    self_vp = float(public_vp[p])
    self_road = float(road_len[p])
    self_knights = float(knight_len[p])
    for q in range(NUM_PLAYERS):
        if q == p:
            continue
        vp_term = np.clip(float(public_vp[q]) / 10.0, 0.0, 1.0)
        dev_term = np.clip(float(dev_counts[q]) / 5.0, 0.0, 1.0)
        road_term = np.clip(float(road_len[q]) / 10.0, 0.0, 1.0)
        knight_term = np.clip(float(knight_len[q]) / 6.0, 0.0, 1.0)
        card_term = np.clip(float(resource_total[q]) / 12.0, 0.0, 1.0)
        vp_lead_term = np.clip((float(public_vp[q]) - self_vp + 2.0) / 6.0, 0.0, 1.0)
        road_race_term = np.clip((float(road_len[q]) - self_road + 2.0) / 6.0, 0.0, 1.0)
        army_race_term = np.clip((float(knight_len[q]) - self_knights + 1.0) / 4.0, 0.0, 1.0)
        score = (
            0.34 * vp_term
            + 0.15 * dev_term * float(threat_dev_card_weight)
            + 0.10 * road_term
            + 0.08 * knight_term
            + 0.08 * card_term
            + 0.14 * vp_lead_term
            + 0.06 * road_race_term
            + 0.03 * army_race_term
            + 0.01 * float(has_road[q])
            + 0.01 * float(has_army[q])
        )
        if float(public_vp[q] + threat_dev_card_weight * dev_counts[q]) >= 8.0:
            score += 0.05
        scores[q] = float(np.clip(score, 0.0, 1.0))

    rel = _relative_opponents(scores, p)
    return {f"opponent_danger_opp{i + 1}": float(rel[i]) for i in range(NUM_PLAYERS - 1)}


def strategic_evaluator_snapshot(state: GameState, player: int, *, threat_dev_card_weight: float = 0.7) -> dict:
    p = int(player)
    city_candidates = []
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) == p and int(state.vertex_building[v]) == int(Building.SETTLEMENT):
            city_candidates.append(city_upgrade_metrics(state, p, int(v)))
    city_top = max((float(x["city_delta_score"]) for x in city_candidates), default=0.0)

    robber_hex_score = robber_block_quality(state, p, int(state.robber_hex), float(threat_dev_card_weight))
    race = race_pressure_metrics(state, p)
    readiness = build_readiness_estimates(state, p)
    return {
        "city_top_delta_score": city_top,
        "robber_block_quality": robber_hex_score,
        "race_pressure": race,
        "build_readiness": readiness,
        "opponent_danger": opponent_danger_targets(state, p, threat_dev_card_weight=float(threat_dev_card_weight)),
    }


AUX_LABEL_KEYS = (
    "city_top_delta_score",
    "robber_block_quality",
    "longest_road_pressure",
    "largest_army_pressure",
    "ready_road_turns",
    "ready_settlement_turns",
    "ready_city_turns",
    "ready_dev_turns",
    "trade_accept_value",
    "trade_accept_immediate_build_gain",
    "trade_accept_should_take",
    "opponent_danger_opp1",
    "opponent_danger_opp2",
    "opponent_danger_opp3",
)


def strategic_aux_targets(snapshot: dict, state: GameState | None = None, player: int | None = None) -> dict[str, float]:
    out = {
        "city_top_delta_score": float(snapshot["city_top_delta_score"]),
        "robber_block_quality": float(snapshot["robber_block_quality"]["quality_score"]),
        "longest_road_pressure": float(snapshot["race_pressure"]["longest_road_pressure"]),
        "largest_army_pressure": float(snapshot["race_pressure"]["largest_army_pressure"]),
        "ready_road_turns": float(snapshot["build_readiness"]["road"]["estimated_turns"]),
        "ready_settlement_turns": float(snapshot["build_readiness"]["settlement"]["estimated_turns"]),
        "ready_city_turns": float(snapshot["build_readiness"]["city"]["estimated_turns"]),
        "ready_dev_turns": float(snapshot["build_readiness"]["dev"]["estimated_turns"]),
        "trade_accept_value": 0.0,
        "trade_accept_immediate_build_gain": 0.0,
        "trade_accept_should_take": 0.0,
    }
    out.update({k: float(v) for k, v in snapshot["opponent_danger"].items()})
    if state is not None and player is not None and _trade_response_target_active(state, int(player)):
        accept_value = float(trade_accept_value(state, int(player)))
        immediate_gain = float(1.0 if trade_accept_immediate_build_gain(state, int(player)) else 0.0)
        out["trade_accept_value"] = accept_value
        out["trade_accept_immediate_build_gain"] = immediate_gain
        out["trade_accept_should_take"] = float(1.0 if accept_value > 0.0 else 0.0)
    return out


def strategic_aux_target_weights(state: GameState, player: int) -> dict[str, float]:
    weights = {k: 1.0 for k in AUX_LABEL_KEYS}
    if not _trade_response_target_active(state, int(player)):
        for k in (
            "trade_accept_value",
            "trade_accept_immediate_build_gain",
            "trade_accept_should_take",
        ):
            weights[k] = 0.0
    return weights


def is_setup_settlement_phase(state: GameState) -> bool:
    return int(state.phase) == int(Phase.SETUP_SETTLEMENT)


_DEV_TRACKED = (
    ("KNIGHT", int(DevCard.KNIGHT), "PLAY_KNIGHT"),
    ("YEAR_OF_PLENTY", int(DevCard.YEAR_OF_PLENTY), "PLAY_YEAR_OF_PLENTY"),
    ("MONOPOLY", int(DevCard.MONOPOLY), "PLAY_MONOPOLY"),
)


def init_dev_timing_tracker() -> dict:
    opportunity_turns: dict[int, dict[str, set[int]]] = {}
    play_turns: dict[int, dict[str, set[int]]] = {}
    for p in range(NUM_PLAYERS):
        opportunity_turns[p] = {k: set() for k, _, _ in _DEV_TRACKED}
        play_turns[p] = {k: set() for k, _, _ in _DEV_TRACKED}
    return {
        "opportunity_turns": opportunity_turns,
        "play_turns": play_turns,
    }


def _dev_play_opportunity(state: GameState, player: int, card_name: str, card_idx: int) -> bool:
    if int(state.dev_card_played_this_turn):
        return False
    if int(state.dev_cards_hidden[int(player), int(card_idx)]) <= 0:
        return False
    if card_name == "KNIGHT":
        return int(state.phase) in (int(Phase.PRE_ROLL), int(Phase.MAIN))
    return int(state.phase) == int(Phase.MAIN)


def record_dev_timing_step(tracker: dict, state_before: GameState, player: int, action_kind: str) -> None:
    p = int(player)
    turn = int(state_before.turn_number)
    for card_name, card_idx, play_action in _DEV_TRACKED:
        if _dev_play_opportunity(state_before, p, card_name, card_idx):
            tracker["opportunity_turns"][p][card_name].add(turn)
        if action_kind == play_action:
            tracker["play_turns"][p][card_name].add(turn)


def summarize_dev_timing(tracker: dict) -> dict:
    per_player: dict[int, dict[str, dict[str, float | int]]] = {}
    for p in range(NUM_PLAYERS):
        by_card: dict[str, dict[str, float | int]] = {}
        for card_name, _, _ in _DEV_TRACKED:
            opp = sorted(int(x) for x in tracker["opportunity_turns"][p][card_name])
            plays = sorted(int(x) for x in tracker["play_turns"][p][card_name])
            first_play = int(plays[0]) if plays else -1
            if first_play >= 0:
                held_before = int(sum(1 for t in opp if t < first_play and t not in set(plays)))
            else:
                held_before = int(len(opp))
            by_card[card_name] = {
                "opportunity_turns": int(len(opp)),
                "play_turns": int(len(plays)),
                "first_play_turn": int(first_play),
                "held_turns_before_first_play": int(held_before),
                "play_rate_when_available": float(len(plays) / len(opp)) if opp else 0.0,
            }
        per_player[p] = by_card

    def _mean_for(card_name: str, key: str) -> float:
        vals = [float(per_player[p][card_name][key]) for p in range(NUM_PLAYERS)]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "per_player": per_player,
        "summary": {
            "knight_mean_play_rate_when_available": _mean_for("KNIGHT", "play_rate_when_available"),
            "knight_mean_held_turns_before_first_play": _mean_for("KNIGHT", "held_turns_before_first_play"),
            "yop_mean_play_rate_when_available": _mean_for("YEAR_OF_PLENTY", "play_rate_when_available"),
            "monopoly_mean_play_rate_when_available": _mean_for("MONOPOLY", "play_rate_when_available"),
        },
    }

