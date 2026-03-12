"""Observation encoding with relative player indexing."""

from __future__ import annotations

import numpy as np

from .constants import Building, NUM_PLAYERS, NUM_RESOURCES, RESOURCE_COSTS
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

    return np.asarray(out, dtype=np.float32)


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
