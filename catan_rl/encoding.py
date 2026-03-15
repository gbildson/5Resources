"""Observation encoding with relative player indexing."""

from __future__ import annotations

from collections import deque

import numpy as np

from .constants import Building, NUM_PLAYERS, NUM_RESOURCES, Phase, RESOURCE_COSTS
from .state import GameState


def _rotate_players(arr: np.ndarray, current_player: int) -> np.ndarray:
    idx = [(current_player + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
    return arr[idx]


def _expansion_potential(state: GameState, player: int) -> tuple[int, int]:
    topo = state.topology
    # Settlement opportunities connected to own road, ignoring resource costs.
    settle_count = 0
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_building[v]) != int(Building.EMPTY):
            continue
        blocked = False
        for n in topo.vertex_to_vertices[v]:
            if n >= 0 and int(state.vertex_building[int(n)]) != int(Building.EMPTY):
                blocked = True
                break
        if blocked:
            continue
        if any(e >= 0 and int(state.edge_owner[int(e)]) == player for e in topo.vertex_to_edges[v]):
            settle_count += 1

    # Road frontier opportunities, ignoring resource costs.
    road_count = 0
    for e in range(len(state.edge_owner)):
        if int(state.edge_owner[e]) >= 0:
            continue
        u = int(topo.edge_to_vertices[e, 0])
        v = int(topo.edge_to_vertices[e, 1])
        if (
            (int(state.vertex_owner[u]) == player and int(state.vertex_building[u]) != int(Building.EMPTY))
            or (int(state.vertex_owner[v]) == player and int(state.vertex_building[v]) != int(Building.EMPTY))
        ):
            road_count += 1
            continue
        if any(x >= 0 and int(state.edge_owner[int(x)]) == player for x in topo.vertex_to_edges[u]):
            road_count += 1
            continue
        if any(x >= 0 and int(state.edge_owner[int(x)]) == player for x in topo.vertex_to_edges[v]):
            road_count += 1
            continue
    return settle_count, road_count


def _is_distance_rule_open(state: GameState, vertex: int) -> bool:
    if int(state.vertex_building[vertex]) != int(Building.EMPTY):
        return False
    for n in state.topology.vertex_to_vertices[vertex]:
        if n >= 0 and int(state.vertex_building[int(n)]) != int(Building.EMPTY):
            return False
    return True


def _vertex_resource_pips(state: GameState, vertex: int, *, robber_aware: bool) -> np.ndarray:
    out = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for h in state.topology.vertex_to_hexes[vertex]:
        if h < 0:
            continue
        if robber_aware and int(h) == int(state.robber_hex):
            continue
        terrain = int(state.hex_terrain[h])
        if terrain <= 0:
            continue
        out[terrain - 1] += float(state.hex_pip_count[h])
    return out


def _global_resource_pips(state: GameState) -> np.ndarray:
    out = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for h in range(len(state.hex_terrain)):
        terrain = int(state.hex_terrain[h])
        if terrain <= 0:
            continue
        out[terrain - 1] += float(state.hex_pip_count[h])
    return out


def _self_production_profile(state: GameState, player: int) -> np.ndarray:
    prof = np.zeros(NUM_RESOURCES, dtype=np.float32)
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) != player:
            continue
        b = int(state.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        prof += mult * _vertex_resource_pips(state, v, robber_aware=False)
    return prof


def _top3(values: list[float]) -> np.ndarray:
    if not values:
        return np.zeros(3, dtype=np.float32)
    top = sorted(values, reverse=True)[:3]
    while len(top) < 3:
        top.append(0.0)
    return np.asarray(top, dtype=np.float32)


def _setup_settlement_score(
    state: GameState,
    vertex: int,
    self_prod: np.ndarray,
    global_pips: np.ndarray,
) -> float:
    vec = _vertex_resource_pips(state, vertex, robber_aware=False)
    pips_total = float(vec.sum())
    diversity = float(np.count_nonzero(vec > 0))
    ows_synergy = float(min(vec[2], vec[3], vec[4]))  # sheep/wheat/ore
    road_synergy = float(min(vec[0], vec[1]))  # wood/brick
    weighted = float(1.2 * vec[3] + 1.15 * vec[4] + 1.0 * vec[2] + 0.9 * vec[0] + 0.9 * vec[1])
    missing_mask = (self_prod <= 1.0).astype(np.float32)
    complement = float((vec * missing_mask).sum())
    scarcity = float(np.sum(vec / np.maximum(1.0, global_pips)))
    return pips_total + 0.7 * diversity + 0.25 * (ows_synergy + road_synergy) + 0.08 * weighted + 0.15 * complement + 2.5 * scarcity


def _frontier_costs(state: GameState, player: int) -> np.ndarray:
    topo = state.topology
    n_vertices = len(state.vertex_owner)
    inf = 10_000
    dist = np.full(n_vertices, inf, dtype=np.int32)
    dq: deque[int] = deque()
    for v in range(n_vertices):
        owned_building = int(state.vertex_owner[v]) == player and int(state.vertex_building[v]) != int(Building.EMPTY)
        has_road = any(e >= 0 and int(state.edge_owner[int(e)]) == player for e in topo.vertex_to_edges[v])
        if owned_building or has_road:
            dist[v] = 0
            dq.append(v)

    while dq:
        v = dq.popleft()
        base = int(dist[v])
        for e in topo.vertex_to_edges[v]:
            if e < 0:
                continue
            owner = int(state.edge_owner[int(e)])
            if owner >= 0 and owner != player:
                continue
            w = 0 if owner == player else 1
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
    if any(e >= 0 and int(state.edge_owner[int(e)]) == player for e in state.topology.vertex_to_edges[vertex]):
        return 0
    best = 10_000
    for e in state.topology.vertex_to_edges[vertex]:
        if e < 0:
            continue
        owner = int(state.edge_owner[int(e)])
        if owner >= 0 and owner != player:
            continue
        u = int(state.topology.edge_to_vertices[int(e), 0])
        v = int(state.topology.edge_to_vertices[int(e), 1])
        other = v if u == vertex else u
        cand = int(frontier_cost[other]) + 1
        best = min(best, cand)
    return None if best > 2 else best


def _site_contention(state: GameState, player: int, vertex: int) -> float:
    opp_road_adj = 0.0
    for e in state.topology.vertex_to_edges[vertex]:
        if e >= 0 and int(state.edge_owner[int(e)]) >= 0 and int(state.edge_owner[int(e)]) != player:
            opp_road_adj += 1.0
    opp_building_adj = 0.0
    for n in state.topology.vertex_to_vertices[vertex]:
        if n >= 0 and int(state.vertex_owner[int(n)]) >= 0 and int(state.vertex_owner[int(n)]) != player:
            opp_building_adj += 1.0
    return opp_road_adj + 0.5 * opp_building_adj


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
            if int(state.edge_owner[int(e)]) == player or int(e) in extra:
                connected = True
                break
        if connected:
            count += 1
    return count


def _road_candidates(state: GameState, player: int) -> list[int]:
    topo = state.topology
    out: list[int] = []
    for e in range(len(state.edge_owner)):
        if int(state.edge_owner[e]) >= 0:
            continue
        u = int(topo.edge_to_vertices[e, 0])
        v = int(topo.edge_to_vertices[e, 1])
        own_building_touch = (
            (int(state.vertex_owner[u]) == player and int(state.vertex_building[u]) != int(Building.EMPTY))
            or (int(state.vertex_owner[v]) == player and int(state.vertex_building[v]) != int(Building.EMPTY))
        )
        own_road_touch = any(x >= 0 and int(state.edge_owner[int(x)]) == player for x in topo.vertex_to_edges[u]) or any(
            x >= 0 and int(state.edge_owner[int(x)]) == player for x in topo.vertex_to_edges[v]
        )
        if own_building_touch or own_road_touch:
            out.append(e)
    return out


def _compute_engineered_features(state: GameState, current_player: int) -> np.ndarray:
    cp = current_player
    s = state
    out: list[float] = []

    # Build-distance / immediate action readiness for self.
    self_res = s.resources[cp]
    for key in ("road", "settlement", "city", "dev"):
        cost = np.asarray(RESOURCE_COSTS[key], dtype=np.int64)
        deficit = int(np.maximum(cost - self_res, 0).sum())
        out.append(min(deficit, 6) / 6.0)
        out.append(1.0 if deficit == 0 else 0.0)

    # Bank rates for self resources (2/3/4 mapped to [0,1]).
    has_generic_port = int(s.has_port[cp, 0]) == 1
    for r in range(NUM_RESOURCES):
        has_specific = int(s.has_port[cp, r + 1]) == 1
        rate = 2 if has_specific else (3 if has_generic_port else 4)
        out.append((rate - 2.0) / 2.0)

    # Pip-weighted production potential by player/resource (robber-aware).
    prod = np.zeros((NUM_PLAYERS, NUM_RESOURCES), dtype=np.float32)
    for h in range(len(s.hex_terrain)):
        if h == int(s.robber_hex):
            continue
        terrain = int(s.hex_terrain[h])
        if terrain <= 0:
            continue
        r = terrain - 1
        pips = float(s.hex_pip_count[h])
        if pips <= 0:
            continue
        for v in s.topology.hex_to_vertices[h]:
            owner = int(s.vertex_owner[v])
            if owner < 0:
                continue
            mult = 2.0 if int(s.vertex_building[v]) == int(Building.CITY) else 1.0
            prod[owner, r] += pips * mult

    prod_rot = _rotate_players(prod, cp)
    out.extend((np.clip(prod_rot[0] / 30.0, 0.0, 1.0)).tolist())  # self per-resource opportunity
    prod_total_rot = prod_rot.sum(axis=1)
    out.extend((np.clip(prod_total_rot / 60.0, 0.0, 1.0)).tolist())  # all-player opportunity totals

    totals_rot = _rotate_players(s.resource_total, cp)
    out.extend((np.clip(totals_rot / 20.0, 0.0, 1.0)).astype(np.float32).tolist())  # compact visible-card proxy
    out.extend((totals_rot > 7).astype(np.float32).tolist())  # discard pressure flags

    # Short-horizon public memory: visible gains/spends and unknown robber-card flow.
    gain_rot = _rotate_players(s.public_recent_gain, cp)
    spend_rot = _rotate_players(s.public_recent_spend, cp)
    recent_net_opp = np.clip((gain_rot[1:] - spend_rot[1:]) / 10.0, -1.0, 1.0)
    out.extend(recent_net_opp.astype(np.float32).reshape(-1).tolist())  # 3 * 5

    unknown_rob_net = s.public_recent_rob_unknown_gain - s.public_recent_rob_unknown_loss
    unknown_rob_net_rot = _rotate_players(unknown_rob_net, cp)
    out.extend(np.clip(unknown_rob_net_rot[1:] / 5.0, -1.0, 1.0).astype(np.float32).tolist())  # 3

    # Robber target quality proxies for opponents.
    rob_hex = int(s.robber_hex)
    blocked_prod = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for v in s.topology.hex_to_vertices[rob_hex]:
        owner = int(s.vertex_owner[v])
        if owner < 0:
            continue
        mult = 2.0 if int(s.vertex_building[v]) == int(Building.CITY) else 1.0
        blocked_prod[owner] += float(s.hex_pip_count[rob_hex]) * mult
    blocked_prod_rot = _rotate_players(blocked_prod, cp)
    out.extend(np.clip(blocked_prod_rot[1:] / 20.0, 0.0, 1.0).astype(np.float32).tolist())  # 3
    out.extend(np.clip(totals_rot[1:] / 20.0, 0.0, 1.0).astype(np.float32).tolist())  # 3

    # Expansion race: board opportunities by player (ignoring costs).
    settle_potential = np.zeros(NUM_PLAYERS, dtype=np.float32)
    road_potential = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for p in range(NUM_PLAYERS):
        sp, rp = _expansion_potential(s, p)
        settle_potential[p] = sp
        road_potential[p] = rp
    settle_rot = _rotate_players(settle_potential, cp)
    road_rot = _rotate_players(road_potential, cp)
    out.extend(np.clip((settle_rot[1:] - settle_rot[0]) / 20.0, -1.0, 1.0).astype(np.float32).tolist())  # 3
    out.extend(np.clip((road_rot[1:] - road_rot[0]) / 30.0, -1.0, 1.0).astype(np.float32).tolist())  # 3

    # Achievement race pressure (opponent minus self).
    lr_rot = _rotate_players(s.longest_road_length, cp).astype(np.float32)
    kn_rot = _rotate_players(s.knights_played, cp).astype(np.float32)
    out.extend(np.clip((lr_rot[1:] - lr_rot[0]) / 10.0, -1.0, 1.0).tolist())  # 3
    out.extend(np.clip((kn_rot[1:] - kn_rot[0]) / 6.0, -1.0, 1.0).tolist())  # 3

    # Trade credibility profile (acceptance/rejection behavior of opponents).
    offers_rot = _rotate_players(s.public_trade_offers, cp)
    accepts_rot = _rotate_players(s.public_trade_accepts, cp)
    rejects_rot = _rotate_players(s.public_trade_rejects, cp)
    opp_offer = offers_rot[1:]
    opp_resp = np.maximum(1e-6, accepts_rot[1:] + rejects_rot[1:])
    opp_accept_ratio = accepts_rot[1:] / opp_resp
    opp_trade_activity = np.clip(opp_offer / 10.0, 0.0, 1.0)
    out.extend(opp_accept_ratio.astype(np.float32).tolist())  # 3
    out.extend(opp_trade_activity.astype(np.float32).tolist())  # 3

    public_vp_rot = _rotate_players(s.public_vp, cp)
    self_vp = float(public_vp_rot[0])
    out.extend((np.clip((public_vp_rot[1:] - self_vp) / 10.0, -1.0, 1.0)).astype(np.float32).tolist())

    # Phase-aware tactical planning helpers (top-3 summaries).
    is_setup_settlement = 1.0 if int(s.phase) == int(Phase.SETUP_SETTLEMENT) else 0.0
    is_main = 1.0 if int(s.phase) == int(Phase.MAIN) else 0.0
    self_prod = _self_production_profile(s, cp)
    global_pips = _global_resource_pips(s)

    setup_scores: list[float] = []
    for v in range(len(s.vertex_owner)):
        if not _is_distance_rule_open(s, v):
            continue
        score = _setup_settlement_score(s, v, self_prod=self_prod, global_pips=global_pips)
        setup_scores.append(float(np.clip(score / 25.0, 0.0, 1.0)))
    out.extend((_top3(setup_scores) * is_setup_settlement).tolist())  # 3

    main_settle_scores: list[float] = []
    frontier_cost = _frontier_costs(s, cp)
    for v in range(len(s.vertex_owner)):
        if not _is_distance_rule_open(s, v):
            continue
        roads_needed = _roads_needed_to_reach_site(s, cp, v, frontier_cost)
        if roads_needed is None:
            continue
        vec = _vertex_resource_pips(s, v, robber_aware=True)
        quality = float(vec.sum()) + 0.7 * float(np.count_nonzero(vec > 0))
        missing_mask = (self_prod <= 1.0).astype(np.float32)
        quality += 0.2 * float((vec * missing_mask).sum())
        contention = _site_contention(s, cp, v)
        score = quality - 1.5 * float(roads_needed) - 0.6 * contention
        main_settle_scores.append(float(np.clip((score + 4.0) / 12.0, 0.0, 1.0)))
    out.extend((_top3(main_settle_scores) * is_main).tolist())  # 3

    city_deltas: list[float] = []
    for v in range(len(s.vertex_owner)):
        if int(s.vertex_owner[v]) != cp or int(s.vertex_building[v]) != int(Building.SETTLEMENT):
            continue
        delta = float(_vertex_resource_pips(s, v, robber_aware=True).sum())
        city_deltas.append(float(np.clip(delta / 15.0, 0.0, 1.0)))
    out.extend(_top3(city_deltas).tolist())  # 3

    road_gain_scores: list[float] = []
    before_settle = _count_connected_settlement_opportunities(s, cp)
    for e in _road_candidates(s, cp):
        after_settle = _count_connected_settlement_opportunities(s, cp, extra_edges={int(e)})
        gain = max(0.0, float(after_settle - before_settle))
        road_gain_scores.append(float(np.clip(gain / 6.0, 0.0, 1.0)))
    out.extend(_top3(road_gain_scores).tolist())  # 3

    return np.asarray(out, dtype=np.float32)


def engineered_feature_summary(state: GameState, current_player: int | None = None) -> dict[str, list[float]]:
    """Return a compact summary of the newest planning-oriented engineered features.

    The summary is extracted from the tail of the engineered feature vector:
    [setup_top3, main_top3, city_delta_top3, road_gain_top3].
    """
    cp = state.current_player if current_player is None else int(current_player)
    feat = _compute_engineered_features(state, cp)
    tail = feat[-12:]
    return {
        "setup_settlement_quality_top3": tail[0:3].astype(np.float32).tolist(),
        "main_settlement_opportunity_top3": tail[3:6].astype(np.float32).tolist(),
        "city_delta_top3": tail[6:9].astype(np.float32).tolist(),
        "road_opportunity_gain_top3": tail[9:12].astype(np.float32).tolist(),
    }


def encode_observation(
    state: GameState,
    full_info: bool = True,
) -> np.ndarray:
    cp = state.current_player
    out: list[np.ndarray] = []

    out.append(np.eye(6, dtype=np.float32)[state.hex_terrain].reshape(-1))
    out.append((state.hex_number / 12.0).astype(np.float32))
    out.append((state.hex_pip_count / 5.0).astype(np.float32))
    robber = np.zeros_like(state.hex_number, dtype=np.float32)
    robber[state.robber_hex] = 1.0
    out.append(robber)

    vb = np.zeros((len(state.vertex_building), 1 + 2 * NUM_PLAYERS), dtype=np.float32)
    for v in range(len(state.vertex_building)):
        b = int(state.vertex_building[v])
        o = int(state.vertex_owner[v])
        if b == Building.EMPTY or o < 0:
            vb[v, 0] = 1.0
        else:
            ro = (o - cp) % NUM_PLAYERS
            vb[v, 1 + 2 * ro + (b - 1)] = 1.0
    out.append(vb.reshape(-1))

    er = np.zeros((len(state.edge_road), 1 + NUM_PLAYERS), dtype=np.float32)
    for e in range(len(state.edge_road)):
        o = int(state.edge_owner[e])
        if o < 0:
            er[e, 0] = 1.0
        else:
            er[e, 1 + ((o - cp) % NUM_PLAYERS)] = 1.0
    out.append(er.reshape(-1))

    out.append(np.eye(6, dtype=np.float32)[state.port_type].reshape(-1))
    out.append(state.topology.port_vertex_mask.astype(np.float32))

    res = _rotate_players(state.resources, cp)
    if full_info:
        out.append(np.clip(res[0] / 20.0, 0.0, 1.0).astype(np.float32))
        out.append(np.clip(res[1:, :].sum(axis=1) / 20.0, 0.0, 1.0).astype(np.float32))
    else:
        out.append(np.clip(res[0] / 20.0, 0.0, 1.0).astype(np.float32))
        totals = _rotate_players(state.resource_total, cp)
        out.append(np.clip(totals[1:] / 20.0, 0.0, 1.0).astype(np.float32))

    dev_hidden = _rotate_players(state.dev_cards_hidden, cp)
    out.append(np.clip(dev_hidden[0] / 5.0, 0.0, 1.0).astype(np.float32))

    knights = _rotate_players(state.knights_played, cp)
    out.append(np.clip(knights[1:] / 8.0, 0.0, 1.0).astype(np.float32))

    out.append(np.clip(_rotate_players(np.stack([state.settlements_left, state.cities_left, state.roads_left], axis=1), cp) / 15.0, 0.0, 1.0).astype(np.float32).reshape(-1))
    out.append(_rotate_players(state.has_port, cp).astype(np.float32).reshape(-1))
    out.append(np.clip(_rotate_players(state.longest_road_length, cp) / 15.0, 0.0, 1.0).astype(np.float32))
    out.append(_rotate_players(np.stack([state.has_longest_road, state.has_largest_army], axis=1), cp).astype(np.float32).reshape(-1))
    out.append(np.clip(_rotate_players(state.public_vp, cp) / 10.0, 0.0, 1.0).astype(np.float32))

    phase_oh = np.zeros(13, dtype=np.float32)
    phase_oh[state.phase] = 1.0
    out.append(phase_oh)

    turn_feat = np.asarray(
        [
            min(state.turn_number, 200) / 200.0,
            float(state.has_rolled),
            float(state.dev_card_played_this_turn),
            state.free_roads_remaining / 2.0,
            state.dice_roll / 12.0,
        ],
        dtype=np.float32,
    )
    out.append(turn_feat)

    if state.trade_proposer < 0:
        proposer_rel = -1.0
    else:
        proposer_rel = float((state.trade_proposer - cp) % NUM_PLAYERS)
    proposer = np.asarray([proposer_rel / 3.0], dtype=np.float32)
    out.append(np.concatenate([state.trade_offer_give, state.trade_offer_want]).astype(np.float32) / 4.0)
    out.append(proposer)
    out.append(_rotate_players(state.trade_responses, cp).astype(np.float32) / 2.0)

    out.append(_compute_engineered_features(state, cp))

    vec = np.concatenate(out, dtype=np.float32)
    return vec
