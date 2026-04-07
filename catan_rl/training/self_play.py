"""Self-play rollout collection and PPO training loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..actions import Action, CATALOG
from ..constants import Building, DevCard, Phase, Resource, RESOURCE_COSTS
from ..bots import RandomLegalAgent
from ..env import CatanEnv
from ..eval import tournament
from ..strategy_archetypes import strategy_target_vector
from ..strategy_metrics import (
    AUX_LABEL_KEYS,
    bank_trade_value,
    road_frontier_metrics,
    road_port_reach_metrics,
    trade_accept_immediate_build_gain,
    trade_accept_should_avoid_proposer,
    setup_choice_metrics,
    setup_settlement_score,
    strategic_aux_target_weights,
    strategic_aux_targets,
    strategic_evaluator_snapshot,
    trade_accept_value,
    trade_offer_counterparty_value,
    trade_offer_value,
)
from .ppo import PPOConfig, PPOTrainer, PolicyValueNet, compute_gae


@dataclass
class TrainConfig:
    total_updates: int = 50
    rollout_steps: int = 2048
    eval_every: int = 5
    seed: int = 7
    max_episode_steps: int = 2000
    truncation_leader_reward: bool = True
    reward_shaping_vp: float = 0.0
    reward_shaping_resource: float = 0.0
    reward_shaping_robber_block_leader: float = 0.0
    reward_shaping_rob_leader: float = 0.0
    reward_shaping_rob_mistarget: float = 0.0
    reward_shaping_play_knight: float = 0.0
    reward_shaping_play_knight_when_blocked: float = 0.0
    reward_shaping_knight_unblock_penalty: float = 0.0
    reward_shaping_setup_settlement: float = 0.0
    reward_shaping_setup_top6_bonus: float = 0.0
    reward_shaping_setup_one_hex_penalty: float = 0.0
    reward_shaping_setup_round1_floor: float = 0.0
    reward_shaping_setup_road_frontier: float = 0.0
    setup_selection_influence_prob: float = 0.0
    setup_selection_influence_settlement_prob: float = 0.0
    setup_selection_influence_settlement_round2_prob: float = 0.0
    setup_selection_influence_road_prob: float = 0.0
    reward_shaping_ows_actions: float = 0.0
    reward_shaping_main_road_purpose: float = 0.0
    reward_shaping_terminal_table_mean: float = 0.0
    reward_shaping_trade_offer_value: float = 0.0
    reward_shaping_trade_offer_counterparty_value: float = 0.0
    reward_shaping_trade_accept_value: float = 0.0
    reward_shaping_bank_trade_value: float = 0.0
    reward_shaping_trade_reject_bad_offer_value: float = 0.0
    force_trade_bootstrap_prob: float = 0.0
    force_trade_bootstrap_updates: int = 0
    threat_dev_card_weight: float = 0.7


def _truncation_reward_for_player(env: CatanEnv, player: int) -> float:
    top_vp = int(env.state.actual_vp.max(initial=0))
    leaders = set(int(p) for p in np.flatnonzero(env.state.actual_vp == top_vp))
    return 1.0 if player in leaders else -1.0


def terminal_table_mean_shaping(state, player: int, coef: float) -> float:
    if coef == 0.0:
        return 0.0
    centered_vp = float(state.actual_vp[int(player)] - np.mean(state.actual_vp))
    # Normalize by win threshold scale so this stays a small terminal nudge.
    return float(coef) * float(np.clip(centered_vp / 10.0, -1.0, 1.0))


def _opponent_leader_set(state, player: int, threat_dev_card_weight: float) -> set[int]:
    opponents = [p for p in range(len(state.public_vp)) if p != player]
    if not opponents:
        return set()
    dev_counts = (state.dev_cards_hidden + state.dev_cards_bought_this_turn).sum(axis=1).astype(np.float32)
    threats = state.public_vp.astype(np.float32) + float(threat_dev_card_weight) * dev_counts
    top = float(np.max(threats[opponents]))
    return set(p for p in opponents if float(threats[p]) == top)


def _blocked_production_on_hex(state, hex_id: int, player: int) -> float:
    blocked = 0.0
    for v in state.topology.hex_to_vertices[int(hex_id)]:
        if int(state.vertex_owner[v]) != player:
            continue
        b = int(state.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        blocked += float(state.hex_pip_count[int(hex_id)]) * mult
    return blocked


def anti_leader_robber_shaping(
    state_before,
    acting_player: int,
    action_id: int,
    *,
    threat_dev_card_weight: float = 0.7,
    robber_block_leader_coef: float = 0.0,
    rob_leader_coef: float = 0.0,
    rob_mistarget_coef: float = 0.0,
    play_knight_coef: float = 0.0,
) -> float:
    if (
        robber_block_leader_coef == 0.0
        and rob_leader_coef == 0.0
        and rob_mistarget_coef == 0.0
        and play_knight_coef == 0.0
    ):
        return 0.0
    action = CATALOG.decode(int(action_id))
    if action.kind == "PLAY_KNIGHT":
        return float(play_knight_coef)
    leaders = _opponent_leader_set(state_before, int(acting_player), float(threat_dev_card_weight))
    if not leaders:
        return 0.0
    shaped = 0.0
    if action.kind == "MOVE_ROBBER" and robber_block_leader_coef != 0.0:
        (hex_id,) = action.params
        blocked_leader = float(sum(_blocked_production_on_hex(state_before, int(hex_id), p) for p in leaders))
        blocked_self = _blocked_production_on_hex(state_before, int(hex_id), int(acting_player))
        # Encourage blocking opponents in the lead while avoiding self-blocking.
        shaped += float(robber_block_leader_coef) * float(np.clip((blocked_leader - blocked_self) / 10.0, -1.0, 1.0))
    elif action.kind == "ROB_PLAYER":
        (target,) = action.params
        if int(target) in leaders:
            shaped += float(rob_leader_coef)
        else:
            shaped -= float(rob_mistarget_coef)
    return shaped


def _blocked_resource_types_on_robber(state_before, player: int) -> set[int]:
    """Resource types currently blocked for player by robber hex ownership adjacency."""
    h = int(state_before.robber_hex)
    terrain = int(state_before.hex_terrain[h])
    if terrain <= 0:
        return set()
    resource_idx = int(terrain - 1)
    owns_adjacent = False
    for v in state_before.topology.hex_to_vertices[h]:
        if int(state_before.vertex_owner[int(v)]) != int(player):
            continue
        if int(state_before.vertex_building[int(v)]) == int(Building.EMPTY):
            continue
        owns_adjacent = True
        break
    return {resource_idx} if owns_adjacent else set()


def knight_unblock_shaping(
    state_before,
    acting_player: int,
    action_id: int,
    *,
    knight_unblock_penalty_coef: float = 0.0,
    play_knight_when_blocked_coef: float = 0.0,
) -> float:
    """
    Penalize not playing a knight when player is robber-blocked and starved on blocked resource.
    Intended to nudge dev-card conversion into largest-army tempo when it also restores production.
    """
    penalty_coef = float(knight_unblock_penalty_coef)
    blocked_play_bonus = float(play_knight_when_blocked_coef)
    if penalty_coef == 0.0 and blocked_play_bonus == 0.0:
        return 0.0
    player = int(acting_player)
    action = CATALOG.decode(int(action_id))
    blocked_types = _blocked_resource_types_on_robber(state_before, player)
    if not blocked_types:
        return 0.0
    has_blocked_resource_in_hand = any(int(state_before.resources[player, r]) > 0 for r in blocked_types)
    knight_playable = (
        int(state_before.phase) in (int(Phase.PRE_ROLL), int(Phase.MAIN))
        and not bool(state_before.dev_card_played_this_turn)
        and int(state_before.dev_cards_hidden[player, int(DevCard.KNIGHT)]) > 0
    )
    if not knight_playable:
        return 0.0
    if action.kind == "PLAY_KNIGHT":
        # Small positive nudge to convert knight when robber is actively blocking own production.
        return blocked_play_bonus
    if has_blocked_resource_in_hand:
        return 0.0
    return -penalty_coef


def setup_settlement_shaping_details(state_before, action_id: int) -> dict | None:
    action = CATALOG.decode(int(action_id))
    if action.kind != "PLACE_SETTLEMENT":
        return None
    if int(state_before.phase) != int(Phase.SETUP_SETTLEMENT):
        return None
    (vertex,) = action.params
    pips_by_res = np.zeros(5, dtype=np.float32)
    productive_hexes = 0
    for h in state_before.topology.vertex_to_hexes[int(vertex)]:
        if h < 0:
            continue
        terrain = int(state_before.hex_terrain[int(h)])
        if terrain <= 0:
            continue
        productive_hexes += 1
        pips_by_res[terrain - 1] += float(state_before.hex_pip_count[int(h)])

    total_pips = float(pips_by_res.sum())
    diversity = int(np.count_nonzero(pips_by_res > 0.0))
    has_ore_wheat = pips_by_res[4] > 0.0 and pips_by_res[3] > 0.0
    has_wood_brick = pips_by_res[0] > 0.0 and pips_by_res[1] > 0.0
    has_combo = has_ore_wheat or has_wood_brick
    res_weights = np.ones(5, dtype=np.float32)
    # Value ore/wheat more in early setup; city/dev lines depend on these heavily.
    res_weights[3] = 1.25
    res_weights[4] = 1.25
    # If wood+brick co-occur, boost both to reward coherent expansion line.
    if has_wood_brick:
        res_weights[0] = 1.10
        res_weights[1] = 1.10
    weighted_pips = float(np.sum(pips_by_res * res_weights))

    # Complementary pair bonus with resources already represented by current holdings/builds.
    player = int(state_before.current_player)
    choice_metrics = setup_choice_metrics(state_before, player, int(vertex))
    candidate_count = int(choice_metrics.get("candidate_count", 0))
    rank = int(choice_metrics.get("rank", -1))
    top_k = min(6, candidate_count) if candidate_count > 0 else 0
    top6_flag = bool(rank > 0 and rank <= top_k)
    top6_rank_signal = float((top_k - rank + 1) / max(1, top_k)) if top6_flag and top_k > 0 else 0.0
    top6_bonus = (1.0 + 0.5 * top6_rank_signal) if top6_flag else 0.0
    owned_by_res = np.zeros(5, dtype=np.float32)
    for v in range(len(state_before.vertex_owner)):
        if int(state_before.vertex_owner[v]) != player:
            continue
        if int(state_before.vertex_building[v]) == int(Building.EMPTY):
            continue
        for h in state_before.topology.vertex_to_hexes[v]:
            if h < 0:
                continue
            terrain = int(state_before.hex_terrain[int(h)])
            if terrain <= 0:
                continue
            owned_by_res[terrain - 1] += float(state_before.hex_pip_count[int(h)])
    has_wood = owned_by_res[0] > 0.0
    has_brick = owned_by_res[1] > 0.0
    has_sheep = owned_by_res[2] > 0.0
    has_wheat = owned_by_res[3] > 0.0
    has_ore = owned_by_res[4] > 0.0
    cand_has_wood = pips_by_res[0] > 0.0
    cand_has_brick = pips_by_res[1] > 0.0
    cand_has_sheep = pips_by_res[2] > 0.0
    cand_has_wheat = pips_by_res[3] > 0.0
    cand_has_ore = pips_by_res[4] > 0.0
    has_complement_pair = (
        (cand_has_brick and has_wood)
        or (cand_has_wood and has_brick)
        or (cand_has_wheat and has_ore)
        or (cand_has_ore and has_wheat)
    )

    has_diversity_three = diversity >= 3
    has_three_hex_combo = productive_hexes >= 3 and has_combo
    has_high_pip_three_hex = productive_hexes >= 3 and total_pips >= 10.0
    has_high_pip_two_hex_complement = productive_hexes >= 2 and total_pips >= 9.0 and has_complement_pair
    rule_hits: list[str] = []

    second_pick_complement_bonus = 0.0
    second_pick_overlap_penalty = 0.0
    if int(state_before.setup_round) == 2:
        existing_mask = owned_by_res > 0.0
        candidate_mask = pips_by_res > 0.0
        new_resource_count = int(np.count_nonzero(np.logical_and(~existing_mask, candidate_mask)))
        combined_diversity = int(np.count_nonzero(np.logical_or(existing_mask, candidate_mask)))
        overlap_count = int(np.count_nonzero(np.logical_and(existing_mask, candidate_mask)))
        # Allow focused OWS lines to keep overlap if they are coherent.
        ows_coherent = bool(
            has_ore
            and has_wheat
            and (cand_has_ore or cand_has_wheat or cand_has_sheep)
        )
        if new_resource_count >= 2:
            second_pick_complement_bonus = 1.0
            rule_hits.append("setup2_new_resources>=2_bonus")
        elif new_resource_count == 1:
            second_pick_complement_bonus = 0.5
            rule_hits.append("setup2_new_resources=1_bonus")
        if combined_diversity >= 4:
            second_pick_complement_bonus = max(second_pick_complement_bonus, 0.85)
            rule_hits.append("setup2_combined_diversity>=4_bonus")
        if not ows_coherent and new_resource_count == 0:
            second_pick_overlap_penalty = 1.0
            rule_hits.append("setup2_overlap_only_penalty")
        elif not ows_coherent and new_resource_count == 1 and overlap_count >= 2:
            second_pick_overlap_penalty = 0.5
            rule_hits.append("setup2_heavy_overlap_penalty")

    if has_diversity_three:
        rule_hits.append("diversity>=3")
    if has_three_hex_combo:
        rule_hits.append("three_hex_core_combo")
    if has_high_pip_three_hex:
        rule_hits.append("three_hex_pips>=10")
    if has_high_pip_two_hex_complement:
        rule_hits.append("two_hex_pips>=9_complement")

    def _is_setup_open_vertex(v: int) -> bool:
        if int(state_before.vertex_building[int(v)]) != int(Building.EMPTY):
            return False
        for n in state_before.topology.vertex_to_vertices[int(v)]:
            if n >= 0 and int(state_before.vertex_building[int(n)]) != int(Building.EMPTY):
                return False
        return True

    def _vertex_total_pips(v: int) -> float:
        total = 0.0
        for h in state_before.topology.vertex_to_hexes[int(v)]:
            if h < 0:
                continue
            terrain = int(state_before.hex_terrain[int(h)])
            if terrain <= 0:
                continue
            total += float(state_before.hex_pip_count[int(h)])
        return total

    def _vertex_productive_hexes(v: int) -> int:
        count = 0
        for h in state_before.topology.vertex_to_hexes[int(v)]:
            if h < 0:
                continue
            terrain = int(state_before.hex_terrain[int(h)])
            if terrain <= 0:
                continue
            count += 1
        return count

    legal_vertices = [v for v in range(len(state_before.vertex_owner)) if _is_setup_open_vertex(v)]
    legal_totals = [_vertex_total_pips(v) for v in legal_vertices]
    legal_productive_hexes = [_vertex_productive_hexes(v) for v in legal_vertices]
    best_total_pips = float(max(legal_totals)) if legal_totals else float(total_pips)
    pip_gap_to_best = float(max(0.0, best_total_pips - float(total_pips)))
    # Apply pip-gap penalty on first setup-round pick quality (players 0..3 first placements).
    apply_pip_gap_penalty = int(state_before.setup_round) == 1
    # Player 4 (index 3) often needs flexible first pick sequencing because they place twice back-to-back.
    # Relax (but do not remove) this first-pick pip-gap penalty for that seat.
    pip_gap_relax_factor = 0.4 if (int(state_before.current_player) == 3 and int(state_before.setup_round) == 1) else 1.0
    has_large_pip_gap = apply_pip_gap_penalty and pip_gap_to_best >= 4.0
    if has_large_pip_gap:
        rule_hits.append("pip_gap_to_best>=4_penalty")

    pip_quality_raw = float(np.clip(total_pips / 12.0, 0.0, 1.0))
    pip_quality_weighted = float(np.clip(weighted_pips / 14.0, 0.0, 1.0))
    # Blend raw pip quality with resource-prioritized quality.
    pip_quality = 0.5 * pip_quality_raw + 0.5 * pip_quality_weighted
    combo_factor = 1.0 if has_combo else 0.75
    if has_high_pip_three_hex:
        combo_factor = max(combo_factor, 0.95)
    if has_high_pip_two_hex_complement:
        combo_factor = max(combo_factor, 0.95)
    pip_gap_penalty = 0.0
    if has_large_pip_gap:
        # Gap=4 starts meaningful penalty; larger gaps increase sharply.
        pip_gap_penalty = float(np.clip((pip_gap_to_best - 3.0) / 3.0, 0.0, 1.0)) * float(pip_gap_relax_factor)

    round1_floor_penalty = 0.0
    if int(state_before.setup_round) == 1 and diversity < 2 and legal_vertices:
        better_diverse_exists = False
        best_diverse_total = 0.0
        for v in legal_vertices:
            pips = np.zeros(5, dtype=np.float32)
            for h in state_before.topology.vertex_to_hexes[int(v)]:
                if h < 0:
                    continue
                terrain = int(state_before.hex_terrain[int(h)])
                if terrain <= 0:
                    continue
                pips[terrain - 1] += float(state_before.hex_pip_count[int(h)])
            v_total = float(pips.sum())
            v_diversity = int(np.count_nonzero(pips > 0.0))
            if v_diversity >= 2:
                better_diverse_exists = True
                best_diverse_total = max(best_diverse_total, v_total)
        if better_diverse_exists and best_diverse_total >= (float(total_pips) - 1.0):
            # Harder floor for first setup pick: avoid one-resource starts if viable diverse options exist.
            round1_floor_penalty = 1.0
            rule_hits.append("round1_diversity_floor_penalty")

    one_hex_penalty = 0.0
    if productive_hexes <= 1 and any(c >= 2 for c in legal_productive_hexes):
        one_hex_penalty = 1.0
        rule_hits.append("one_hex_setup_penalty")

    return {
        "vertex": int(vertex),
        "productive_hexes": int(productive_hexes),
        "total_pips": total_pips,
        "weighted_pips": float(weighted_pips),
        "diversity": int(diversity),
        "has_combo": bool(has_combo),
        "has_complement_pair": bool(has_complement_pair),
        "pip_quality": pip_quality,
        "pip_quality_raw": float(pip_quality_raw),
        "pip_quality_weighted": float(pip_quality_weighted),
        "combo_factor": float(combo_factor),
        "best_total_pips": float(best_total_pips),
        "pip_gap_to_best": float(pip_gap_to_best),
        "pip_gap_penalty": float(pip_gap_penalty),
        "round1_floor_penalty": float(round1_floor_penalty),
        "one_hex_penalty": float(one_hex_penalty),
        "second_pick_complement_bonus": float(second_pick_complement_bonus),
        "second_pick_overlap_penalty": float(second_pick_overlap_penalty),
        "top6_flag": bool(top6_flag),
        "top6_rank": int(rank),
        "top6_candidate_count": int(candidate_count),
        "top6_rank_signal": float(top6_rank_signal),
        "top6_bonus": float(top6_bonus),
        "pip_gap_relax_factor": float(pip_gap_relax_factor),
        "eligible": bool(len(rule_hits) > 0),
        "rule_hits": rule_hits,
    }


def setup_settlement_shaping(
    state_before,
    action_id: int,
    *,
    setup_settlement_coef: float = 0.0,
    setup_top6_bonus_coef: float = 0.0,
    setup_one_hex_penalty_coef: float = 0.0,
    setup_round1_floor_coef: float = 0.0,
) -> float:
    if (
        setup_settlement_coef == 0.0
        and setup_top6_bonus_coef == 0.0
        and setup_one_hex_penalty_coef == 0.0
        and setup_round1_floor_coef == 0.0
    ):
        return 0.0
    details = setup_settlement_shaping_details(state_before, action_id)
    if details is None:
        return 0.0
    base_bonus = 0.0
    if bool(details["eligible"]) and float(setup_settlement_coef) != 0.0:
        base_bonus = float(setup_settlement_coef) * float(details["pip_quality"]) * float(details["combo_factor"])
    penalty = float(details.get("pip_gap_penalty", 0.0))
    floor_penalty = float(details.get("round1_floor_penalty", 0.0))
    one_hex_penalty = float(details.get("one_hex_penalty", 0.0))
    second_pick_bonus = float(details.get("second_pick_complement_bonus", 0.0))
    second_pick_penalty = float(details.get("second_pick_overlap_penalty", 0.0))
    top6_bonus = float(details.get("top6_bonus", 0.0))
    shaped = (
        base_bonus
        - float(setup_settlement_coef) * penalty
        - float(setup_round1_floor_coef) * floor_penalty
        - float(setup_one_hex_penalty_coef) * one_hex_penalty
    )
    shaped += float(setup_settlement_coef) * (0.60 * second_pick_bonus - 0.70 * second_pick_penalty)
    shaped += float(setup_top6_bonus_coef) * top6_bonus
    return float(shaped)


def setup_road_frontier_shaping(
    state_before,
    acting_player: int,
    action_id: int,
    *,
    setup_road_frontier_coef: float = 0.0,
) -> float:
    if float(setup_road_frontier_coef) == 0.0:
        return 0.0
    if int(state_before.phase) != int(Phase.SETUP_ROAD):
        return 0.0
    action = CATALOG.decode(int(action_id))
    if action.kind != "PLACE_ROAD":
        return 0.0
    (edge,) = action.params
    frontier = road_frontier_metrics(state_before, int(acting_player), int(edge))
    port = road_port_reach_metrics(state_before, int(acting_player), int(edge))
    connected_gain = float(frontier.get("connected_gain", 0.0))
    best_site_delta = float(frontier.get("best_site_roads_delta", 0.0))
    port_delta = float(port.get("best_port_roads_delta", 0.0))
    # Encourage roads that increase immediate settlement opportunities and improve port pathing.
    score = 0.65 * connected_gain + 0.20 * best_site_delta + 0.55 * port_delta
    score = float(np.clip(score / 3.0, -1.0, 1.0))
    return float(setup_road_frontier_coef) * score


def setup_selection_influence_action(
    state_before,
    acting_player: int,
    chosen_action_id: int,
    action_mask: np.ndarray,
    *,
    influence_prob: float = 0.0,
    settlement_prob: float = 0.0,
    settlement_round2_prob: float = 0.0,
    road_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, bool]:
    """
    Training-time setup selector override:
    with probability p, choose best legal setup action by strategic heuristic.
    """
    phase = int(state_before.phase)
    if phase not in (int(Phase.SETUP_SETTLEMENT), int(Phase.SETUP_ROAD)):
        return int(chosen_action_id), False
    if phase == int(Phase.SETUP_SETTLEMENT):
        if int(state_before.setup_round) == 2 and float(settlement_round2_prob) > 0.0:
            p = float(settlement_round2_prob)
        else:
            p = float(settlement_prob if settlement_prob > 0.0 else influence_prob)
    else:
        p = float(road_prob if road_prob > 0.0 else influence_prob)
    if p <= 0.0:
        return int(chosen_action_id), False
    rr = rng if rng is not None else np.random.default_rng()
    if float(rr.random()) >= p:
        return int(chosen_action_id), False

    legal_ids = np.flatnonzero(action_mask > 0)
    if legal_ids.size == 0:
        return int(chosen_action_id), False
    player = int(acting_player)
    best_id = int(chosen_action_id)
    best_score = -1e18
    round2_existing_mask: np.ndarray | None = None
    round2_best_new_resource_count = 0
    round2_best_combined_diversity = 0
    round2_has_ore = False
    round2_has_wheat = False
    if phase == int(Phase.SETUP_SETTLEMENT) and int(state_before.setup_round) == 2:
        round2_owned_by_res = _player_resource_pips(state_before, player)
        round2_existing_mask = round2_owned_by_res > 0.0
        round2_has_ore = bool(round2_existing_mask[int(Resource.ORE)])
        round2_has_wheat = bool(round2_existing_mask[int(Resource.WHEAT)])
        for a_id in legal_ids:
            action = CATALOG.decode(int(a_id))
            if action.kind != "PLACE_SETTLEMENT":
                continue
            (vertex,) = action.params
            cand_pips = np.zeros(5, dtype=np.float32)
            for h in state_before.topology.vertex_to_hexes[int(vertex)]:
                if h < 0:
                    continue
                terrain = int(state_before.hex_terrain[int(h)])
                if terrain <= 0:
                    continue
                cand_pips[terrain - 1] += float(state_before.hex_pip_count[int(h)])
            candidate_mask = cand_pips > 0.0
            round2_best_new_resource_count = max(
                round2_best_new_resource_count,
                int(np.count_nonzero(np.logical_and(~round2_existing_mask, candidate_mask))),
            )
            round2_best_combined_diversity = max(
                round2_best_combined_diversity,
                int(np.count_nonzero(np.logical_or(round2_existing_mask, candidate_mask))),
            )
    for a_id in legal_ids:
        action = CATALOG.decode(int(a_id))
        if phase == int(Phase.SETUP_SETTLEMENT):
            if action.kind != "PLACE_SETTLEMENT":
                continue
            (vertex,) = action.params
            score = float(setup_settlement_score(state_before, player, int(vertex)))
            if round2_existing_mask is not None:
                cand_pips = np.zeros(5, dtype=np.float32)
                for h in state_before.topology.vertex_to_hexes[int(vertex)]:
                    if h < 0:
                        continue
                    terrain = int(state_before.hex_terrain[int(h)])
                    if terrain <= 0:
                        continue
                    cand_pips[terrain - 1] += float(state_before.hex_pip_count[int(h)])
                candidate_mask = cand_pips > 0.0
                cand_new_resource_count = int(np.count_nonzero(np.logical_and(~round2_existing_mask, candidate_mask)))
                cand_combined_diversity = int(np.count_nonzero(np.logical_or(round2_existing_mask, candidate_mask)))
                cand_ows_coherent = bool(
                    round2_has_ore
                    and round2_has_wheat
                    and (
                        bool(candidate_mask[int(Resource.ORE)])
                        or bool(candidate_mask[int(Resource.WHEAT)])
                        or bool(candidate_mask[int(Resource.SHEEP)])
                    )
                )
                # Hard preference: in round-2 setup, avoid overlap-only picks when legal complement
                # options exist. Keep OWS-coherent exceptions.
                if (
                    not cand_ows_coherent
                    and round2_best_new_resource_count > 0
                    and cand_new_resource_count < round2_best_new_resource_count
                ):
                    score -= 10_000.0
                if (
                    not cand_ows_coherent
                    and round2_best_combined_diversity >= 4
                    and cand_combined_diversity < round2_best_combined_diversity
                ):
                    score -= 2_500.0
        else:
            if action.kind != "PLACE_ROAD":
                continue
            (edge,) = action.params
            frontier = road_frontier_metrics(state_before, player, int(edge))
            port = road_port_reach_metrics(state_before, player, int(edge))
            score = (
                0.65 * float(frontier.get("connected_gain", 0.0))
                + 0.20 * float(frontier.get("best_site_roads_delta", 0.0))
                + 0.55 * float(port.get("best_port_roads_delta", 0.0))
            )
        if score > best_score:
            best_score = float(score)
            best_id = int(a_id)
    if best_id == int(chosen_action_id):
        return int(chosen_action_id), False
    return int(best_id), True


def _player_resource_pips(state_before, player: int) -> np.ndarray:
    pips = np.zeros(5, dtype=np.float32)
    for v in range(len(state_before.vertex_owner)):
        if int(state_before.vertex_owner[v]) != int(player):
            continue
        b = int(state_before.vertex_building[v])
        if b == int(Building.EMPTY):
            continue
        mult = 2.0 if b == int(Building.CITY) else 1.0
        for h in state_before.topology.vertex_to_hexes[int(v)]:
            if h < 0:
                continue
            terrain = int(state_before.hex_terrain[int(h)])
            if terrain <= 0:
                continue
            pips[terrain - 1] += mult * float(state_before.hex_pip_count[int(h)])
    return pips


def _legal_kinds_from_mask(action_mask: np.ndarray) -> set[str]:
    kinds: set[str] = set()
    legal_ids = np.flatnonzero(action_mask > 0)
    for a_id in legal_ids:
        kinds.add(CATALOG.decode(int(a_id)).kind)
    return kinds


def ows_specialization_shaping(
    state_before,
    acting_player: int,
    action_id: int,
    *,
    ows_action_coef: float = 0.0,
) -> float:
    if float(ows_action_coef) == 0.0:
        return 0.0
    if int(state_before.phase) != int(Phase.MAIN):
        return 0.0
    player = int(acting_player)
    pips = _player_resource_pips(state_before, player)
    ore = float(pips[int(Resource.ORE)])
    wheat = float(pips[int(Resource.WHEAT)])
    wood = float(pips[int(Resource.WOOD)])
    brick = float(pips[int(Resource.BRICK)])
    ows_strength = min(ore, wheat)
    rb_strength = min(wood, brick)
    if ows_strength < 8.0 or ows_strength < (1.15 * rb_strength):
        return 0.0

    action = CATALOG.decode(int(action_id))
    if action.kind == "PLACE_CITY":
        return float(ows_action_coef) * 1.0
    if action.kind == "BUY_DEV_CARD":
        return float(ows_action_coef) * 0.7
    if action.kind == "PLACE_ROAD":
        return -float(ows_action_coef) * 0.6
    return 0.0


def main_phase_road_purpose_shaping(
    state_before,
    acting_player: int,
    action_id: int,
    *,
    main_road_purpose_coef: float = 0.0,
    action_mask: np.ndarray | None = None,
) -> float:
    if float(main_road_purpose_coef) == 0.0:
        return 0.0
    if int(state_before.phase) != int(Phase.MAIN):
        return 0.0
    action = CATALOG.decode(int(action_id))
    if action.kind != "PLACE_ROAD":
        return 0.0
    (edge,) = action.params
    frontier = road_frontier_metrics(state_before, int(acting_player), int(edge))
    port = road_port_reach_metrics(state_before, int(acting_player), int(edge))
    connected_gain = float(frontier.get("connected_gain", 0.0))
    best_site_delta = float(frontier.get("best_site_roads_delta", 0.0))
    port_delta = float(port.get("best_port_roads_delta", 0.0))
    purposeful = connected_gain > 0.0 or best_site_delta > 0.0 or port_delta > 0.0
    score = 0.65 * connected_gain + 0.20 * best_site_delta + 0.55 * port_delta
    legal_kinds = _legal_kinds_from_mask(action_mask) if action_mask is not None else set()
    has_alt_progress = bool(
        "PLACE_SETTLEMENT" in legal_kinds
        or "PLACE_CITY" in legal_kinds
        or "BUY_DEV_CARD" in legal_kinds
    )
    if score > 0.0:
        bonus = 0.35 * float(np.clip(score / 3.0, 0.0, 1.0))
        # If stronger progression actions are available, require stronger road value.
        if has_alt_progress and connected_gain <= 0.0:
            bonus *= 0.5
        return float(main_road_purpose_coef) * bonus
    penalty = 0.9 if not purposeful else float(np.clip((0.5 - score) / 2.0, 0.0, 0.8))
    if has_alt_progress:
        penalty += 0.35
    player = int(acting_player)
    self_road_len = int(state_before.longest_road_length[player])
    opp_road_len = max(
        (int(state_before.longest_road_length[p]) for p in range(len(state_before.longest_road_length)) if p != player),
        default=0,
    )
    road_race_active = (opp_road_len - self_road_len) <= 1 and opp_road_len >= 4
    if road_race_active:
        penalty -= 0.30
    penalty = float(np.clip(penalty, 0.0, 1.4))
    return -float(main_road_purpose_coef) * penalty


PLAY_KNIGHT_ACTION_ID = CATALOG.encode(Action("PLAY_KNIGHT"))
ACCEPT_TRADE_ACTION_ID = CATALOG.encode(Action("ACCEPT_TRADE"))
TRADE_DRAFT_EDIT_KINDS = {"TRADE_ADD_GIVE", "TRADE_ADD_WANT", "TRADE_REMOVE_GIVE", "TRADE_REMOVE_WANT"}


def _trade_draft_need_profile(state_before, player: int) -> tuple[set[int], bool]:
    resources = state_before.resources[int(player)]
    build_costs = []
    if state_before.roads_left[int(player)] > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["road"], dtype=np.int64))
    if state_before.settlements_left[int(player)] > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["settlement"], dtype=np.int64))
    if state_before.cities_left[int(player)] > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["city"], dtype=np.int64))
    if state_before.dev_deck_remaining > 0:
        build_costs.append(np.asarray(RESOURCE_COSTS["dev"], dtype=np.int64))

    needed: set[int] = set()
    can_build_now = False
    for cost in build_costs:
        deficit = np.maximum(cost - resources, 0)
        miss = int(deficit.sum())
        if miss == 0:
            can_build_now = True
        if miss <= 2:
            for r in np.flatnonzero(deficit > 0):
                needed.add(int(r))
    return needed, can_build_now


def _trade_draft_after_action(state_before, action: Action) -> tuple[np.ndarray, np.ndarray] | None:
    give = np.asarray(state_before.trade_offer_give, dtype=np.int64).copy()
    want = np.asarray(state_before.trade_offer_want, dtype=np.int64).copy()
    if int(state_before.phase) == int(Phase.MAIN):
        give[:] = 0
        want[:] = 0
    if action.kind == "TRADE_ADD_GIVE":
        (give_r,) = action.params
        give[int(give_r)] += 1
    elif action.kind == "TRADE_ADD_WANT":
        (want_r,) = action.params
        want[int(want_r)] += 1
    elif action.kind == "TRADE_REMOVE_GIVE":
        (give_r,) = action.params
        give[int(give_r)] = max(0, int(give[int(give_r)]) - 1)
    elif action.kind == "TRADE_REMOVE_WANT":
        (want_r,) = action.params
        want[int(want_r)] = max(0, int(want[int(want_r)]) - 1)
    else:
        return None
    return give, want


def _trade_draft_bundle_score(state_before, acting_player: int, give: np.ndarray, want: np.ndarray) -> float:
    player = int(acting_player)
    give = np.asarray(give, dtype=np.int64)
    want = np.asarray(want, dtype=np.int64)
    total_give = int(give.sum())
    total_want = int(want.sum())
    if total_give == 0 and total_want == 0:
        return 0.0
    if np.any((give > 0) & (want > 0)):
        return -1.0

    needed_resources, can_build_now = _trade_draft_need_profile(state_before, player)
    total_resources = int(state_before.resource_total[player])
    near_discard = total_resources > 7
    score = 0.0
    for want_r in np.flatnonzero(want > 0).tolist():
        if int(want_r) in needed_resources:
            score += 0.60 * float(want[int(want_r)])
        elif near_discard and int(want_r) in (int(Resource.WHEAT), int(Resource.ORE)):
            score += 0.25 * float(want[int(want_r)])
        else:
            score -= 0.15 * float(want[int(want_r)])
    for give_r in np.flatnonzero(give > 0).tolist():
        have = int(state_before.resources[player, int(give_r)])
        if int(give_r) in needed_resources and have <= int(give[int(give_r)]) + 1:
            score -= 0.60 * float(give[int(give_r)])
        elif near_discard or have >= int(give[int(give_r)]) + 3:
            score += 0.20 * float(give[int(give_r)])
        else:
            score -= 0.10 * float(give[int(give_r)])
    if can_build_now and not near_discard and total_give <= total_want:
        score -= 0.25
    if total_give > 0 and total_want > 0:
        score += float(trade_offer_value(state_before, player, give, want))
        score += 0.50 * float(trade_offer_counterparty_value(state_before, player, give, want))
    return float(np.clip(score, -2.0, 2.0))


def force_knight_bootstrap_action(
    state_before,
    acting_player: int,
    chosen_action_id: int,
    action_mask: np.ndarray,
    *,
    force_prob: float = 0.0,
    threat_dev_card_weight: float = 0.7,
    rng: np.random.Generator | None = None,
) -> tuple[int, bool]:
    """
    Optional training-only override: occasionally force PLAY_KNIGHT in PRE_ROLL when
    player is blocked, can pressure a leader, or should convert knight stock into
    largest-army tempo. This is a temporary exploration bootstrap and should be
    disabled after early training.
    """
    p = float(force_prob)
    if p <= 0.0:
        return int(chosen_action_id), False
    if int(state_before.phase) not in (int(Phase.PRE_ROLL), int(Phase.MAIN)):
        return int(chosen_action_id), False
    if int(action_mask[PLAY_KNIGHT_ACTION_ID]) <= 0:
        return int(chosen_action_id), False
    player = int(acting_player)
    blocked = len(_blocked_resource_types_on_robber(state_before, player)) > 0
    leaders = _opponent_leader_set(state_before, player, float(threat_dev_card_weight))
    leader_has_cards = any(int(state_before.resource_total[int(lp)]) > 0 for lp in leaders)
    self_knights_played = int(state_before.knights_played[player])
    best_opp_knights = max(
        (int(state_before.knights_played[p]) for p in range(len(state_before.knights_played)) if p != player),
        default=0,
    )
    held_knights = int(state_before.dev_cards_hidden[player, int(DevCard.KNIGHT)])
    largest_army_owned = int(state_before.has_largest_army[player]) == 1

    # Strong army-tempo triggers:
    # - convert the 3rd knight,
    # - stay within one knight of the leader,
    # - avoid hoarding 2+ knights before first conversion,
    # - defend an existing largest-army lead when opponents are close.
    army_race_live = self_knights_played <= best_opp_knights + 1
    at_threshold = self_knights_played >= 2
    hoarding = held_knights >= 2 and self_knights_played == 0
    defend_army = largest_army_owned and best_opp_knights >= max(1, self_knights_played - 1)
    should_force = blocked or leader_has_cards or (army_race_live and at_threshold) or hoarding or defend_army
    if not should_force:
        return int(chosen_action_id), False
    rr = rng if rng is not None else np.random.default_rng()
    if float(rr.random()) < p:
        return int(PLAY_KNIGHT_ACTION_ID), True
    return int(chosen_action_id), False


def force_trade_bootstrap_action(
    state_before,
    acting_player: int,
    chosen_action_id: int,
    action_mask: np.ndarray,
    *,
    force_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, bool]:
    """
    Optional training-only trade override to escape reject/no-trade equilibria.
    - In TRADE_PROPOSED: force ACCEPT when the trade immediately unlocks builds or utility is non-negative.
    - In MAIN: force highest-utility legal PROPOSE_TRADE when utility is positive.
    """
    p = float(force_prob)
    if p <= 0.0:
        return int(chosen_action_id), False
    rr = rng if rng is not None else np.random.default_rng()
    if float(rr.random()) >= p:
        return int(chosen_action_id), False

    phase = int(state_before.phase)
    player = int(acting_player)
    if phase == int(Phase.TRADE_PROPOSED):
        if int(action_mask[ACCEPT_TRADE_ACTION_ID]) > 0:
            if trade_accept_should_avoid_proposer(state_before, player):
                return int(chosen_action_id), False
            if trade_accept_immediate_build_gain(state_before, player):
                return int(ACCEPT_TRADE_ACTION_ID), True
            accept_util = float(trade_accept_value(state_before, player))
            # Prefer converting marginally-positive offers instead of drifting to reject loops.
            if accept_util >= 0.05:
                return int(ACCEPT_TRADE_ACTION_ID), True
            if accept_util >= 0.0:
                return int(ACCEPT_TRADE_ACTION_ID), True
        return int(chosen_action_id), False

    if phase == int(Phase.TRADE_DRAFT):
        if int(action_mask[CATALOG.encode(Action("PROPOSE_TRADE"))]) > 0:
            submit_util = float(
                trade_offer_value(state_before, player, state_before.trade_offer_give, state_before.trade_offer_want)
            )
            if submit_util > 0.0:
                return int(CATALOG.encode(Action("PROPOSE_TRADE"))), True
        return int(chosen_action_id), False

    if phase == int(Phase.MAIN):
        legal_ids = np.flatnonzero(np.asarray(action_mask, dtype=np.int64) > 0)
        best_id = -1
        best_util = -1e9
        for aid in legal_ids.tolist():
            action = CATALOG.decode(int(aid))
            if action.kind not in TRADE_DRAFT_EDIT_KINDS:
                continue
            draft_after = _trade_draft_after_action(state_before, action)
            if draft_after is None:
                continue
            util = _trade_draft_bundle_score(state_before, player, draft_after[0], draft_after[1])
            if util > best_util:
                best_util = util
                best_id = int(aid)
        if best_id >= 0 and best_util > 0.0:
            return best_id, True
    return int(chosen_action_id), False


def trade_value_shaping(
    state_before,
    acting_player: int,
    action_id: int,
    *,
    trade_offer_coef: float = 0.0,
    trade_offer_counterparty_coef: float = 0.0,
    trade_accept_coef: float = 0.0,
    bank_trade_coef: float = 0.0,
    trade_reject_bad_offer_coef: float = 0.0,
) -> tuple[float, float]:
    if (
        trade_offer_coef == 0.0
        and trade_offer_counterparty_coef == 0.0
        and trade_accept_coef == 0.0
        and bank_trade_coef == 0.0
        and trade_reject_bad_offer_coef == 0.0
    ):
        return 0.0, 0.0
    action = CATALOG.decode(int(action_id))
    util = 0.0
    shaped = 0.0
    if action.kind in TRADE_DRAFT_EDIT_KINDS and (trade_offer_coef != 0.0 or trade_offer_counterparty_coef != 0.0):
        before_score = _trade_draft_bundle_score(
            state_before,
            int(acting_player),
            state_before.trade_offer_give,
            state_before.trade_offer_want,
        )
        draft_after = _trade_draft_after_action(state_before, action)
        if draft_after is not None:
            after_score = _trade_draft_bundle_score(state_before, int(acting_player), draft_after[0], draft_after[1])
            util = float(after_score - before_score)
            shaped = float(trade_offer_coef) * util
    elif action.kind == "PROPOSE_TRADE" and (trade_offer_coef != 0.0 or trade_offer_counterparty_coef != 0.0):
        util = trade_offer_value(
            state_before,
            int(acting_player),
            state_before.trade_offer_give,
            state_before.trade_offer_want,
        )
        counter_util = trade_offer_counterparty_value(
            state_before,
            int(acting_player),
            state_before.trade_offer_give,
            state_before.trade_offer_want,
        )
        shaped = float(trade_offer_coef) * util + float(trade_offer_counterparty_coef) * counter_util
    elif action.kind == "ACCEPT_TRADE" and trade_accept_coef != 0.0:
        util = trade_accept_value(state_before, int(acting_player))
        shaped = float(trade_accept_coef) * util
    elif action.kind == "BANK_TRADE" and bank_trade_coef != 0.0:
        give_r, give_n, want_r = action.params
        util = bank_trade_value(state_before, int(acting_player), int(give_r), int(give_n), int(want_r))
        shaped = float(bank_trade_coef) * util
    elif action.kind == "REJECT_TRADE" and trade_reject_bad_offer_coef != 0.0:
        # Reward rejecting locally bad trades; no reward for rejecting neutral/positive accepts.
        accept_util = float(trade_accept_value(state_before, int(acting_player)))
        util = float(max(0.0, -accept_util))
        shaped = float(trade_reject_bad_offer_coef) * util
    return float(shaped), float(util)


def collect_rollout(
    env: CatanEnv,
    trainer: PPOTrainer,
    steps: int,
    max_episode_steps: int = 2000,
    truncation_leader_reward: bool = True,
    reward_shaping_vp: float = 0.0,
    reward_shaping_resource: float = 0.0,
    reward_shaping_robber_block_leader: float = 0.0,
    reward_shaping_rob_leader: float = 0.0,
    reward_shaping_rob_mistarget: float = 0.0,
    reward_shaping_play_knight: float = 0.0,
    reward_shaping_play_knight_when_blocked: float = 0.0,
    reward_shaping_knight_unblock_penalty: float = 0.0,
    reward_shaping_setup_settlement: float = 0.0,
    reward_shaping_setup_top6_bonus: float = 0.0,
    reward_shaping_setup_one_hex_penalty: float = 0.0,
    reward_shaping_setup_round1_floor: float = 0.0,
    reward_shaping_setup_road_frontier: float = 0.0,
    setup_selection_influence_prob: float = 0.0,
    setup_selection_influence_settlement_prob: float = 0.0,
    setup_selection_influence_settlement_round2_prob: float = 0.0,
    setup_selection_influence_road_prob: float = 0.0,
    reward_shaping_terminal_table_mean: float = 0.0,
    reward_shaping_trade_offer_value: float = 0.0,
    reward_shaping_trade_offer_counterparty_value: float = 0.0,
    reward_shaping_trade_accept_value: float = 0.0,
    reward_shaping_bank_trade_value: float = 0.0,
    reward_shaping_trade_reject_bad_offer_value: float = 0.0,
    reward_shaping_ows_actions: float = 0.0,
    reward_shaping_main_road_purpose: float = 0.0,
    force_knight_bootstrap_prob: float = 0.0,
    force_trade_bootstrap_prob: float = 0.0,
    threat_dev_card_weight: float = 0.7,
) -> dict:
    obs, info = env.reset()
    mask = info["action_mask"]
    buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones", "masks")}
    aux_accum = {k: 0.0 for k in AUX_LABEL_KEYS}
    aux_weight_accum = {k: 0.0 for k in AUX_LABEL_KEYS}
    aux_targets = {k: [] for k in AUX_LABEL_KEYS}
    aux_target_weights = {k: [] for k in AUX_LABEL_KEYS}
    trade_value_accum = {"count": 0, "utility_sum": 0.0, "shaping_sum": 0.0}
    forced_knight_count = 0
    forced_trade_count = 0
    setup_selection_override_count = 0
    strategy_targets: list[np.ndarray] = []
    episode_steps = 0
    for _ in range(steps):
        player = env.state.current_player
        prev_public_vp = int(env.state.public_vp[player])
        prev_resource_total = int(env.state.resource_total[player])
        state_before = env.state.copy()
        snap = strategic_evaluator_snapshot(state_before, int(player), threat_dev_card_weight=float(threat_dev_card_weight))
        aux_vals = strategic_aux_targets(snap, state_before, int(player))
        aux_weights = strategic_aux_target_weights(state_before, int(player))
        strategy_targets.append(strategy_target_vector(state_before, int(player)))
        for k in AUX_LABEL_KEYS:
            val = float(aux_vals[k])
            weight = float(aux_weights[k])
            aux_accum[k] += weight * val
            aux_weight_accum[k] += weight
            aux_targets[k].append(val)
            aux_target_weights[k].append(weight)
        action, logp, value = trainer.act(obs, mask)
        action, setup_override = setup_selection_influence_action(
            state_before,
            int(player),
            int(action),
            mask,
            influence_prob=float(setup_selection_influence_prob),
            settlement_prob=float(setup_selection_influence_settlement_prob),
            settlement_round2_prob=float(setup_selection_influence_settlement_round2_prob),
            road_prob=float(setup_selection_influence_road_prob),
            rng=env.rng,
        )
        if setup_override:
            setup_selection_override_count += 1
        action, forced_knight = force_knight_bootstrap_action(
            state_before,
            int(player),
            int(action),
            mask,
            force_prob=float(force_knight_bootstrap_prob),
            threat_dev_card_weight=float(threat_dev_card_weight),
            rng=env.rng,
        )
        if forced_knight:
            forced_knight_count += 1
        action, forced_trade = force_trade_bootstrap_action(
            state_before,
            int(player),
            int(action),
            mask,
            force_prob=float(force_trade_bootstrap_prob),
            rng=env.rng,
        )
        if forced_trade:
            forced_trade_count += 1
        res = env.step(action)
        reward = float(res.reward)
        delta_vp = int(env.state.public_vp[player]) - prev_public_vp
        delta_resources = int(env.state.resource_total[player]) - prev_resource_total
        reward += reward_shaping_vp * float(delta_vp)
        reward += reward_shaping_resource * float(delta_resources)
        reward += anti_leader_robber_shaping(
            state_before,
            int(player),
            int(action),
            threat_dev_card_weight=float(threat_dev_card_weight),
            robber_block_leader_coef=float(reward_shaping_robber_block_leader),
            rob_leader_coef=float(reward_shaping_rob_leader),
            rob_mistarget_coef=float(reward_shaping_rob_mistarget),
            play_knight_coef=float(reward_shaping_play_knight),
        )
        reward += knight_unblock_shaping(
            state_before,
            int(player),
            int(action),
            knight_unblock_penalty_coef=float(reward_shaping_knight_unblock_penalty),
            play_knight_when_blocked_coef=float(reward_shaping_play_knight_when_blocked),
        )
        reward += setup_settlement_shaping(
            state_before,
            int(action),
            setup_settlement_coef=float(reward_shaping_setup_settlement),
            setup_top6_bonus_coef=float(reward_shaping_setup_top6_bonus),
            setup_one_hex_penalty_coef=float(reward_shaping_setup_one_hex_penalty),
            setup_round1_floor_coef=float(reward_shaping_setup_round1_floor),
        )
        reward += setup_road_frontier_shaping(
            state_before,
            int(player),
            int(action),
            setup_road_frontier_coef=float(reward_shaping_setup_road_frontier),
        )
        trade_shaped, trade_util = trade_value_shaping(
            state_before,
            int(player),
            int(action),
            trade_offer_coef=float(reward_shaping_trade_offer_value),
            trade_offer_counterparty_coef=float(reward_shaping_trade_offer_counterparty_value),
            trade_accept_coef=float(reward_shaping_trade_accept_value),
            bank_trade_coef=float(reward_shaping_bank_trade_value),
            trade_reject_bad_offer_coef=float(reward_shaping_trade_reject_bad_offer_value),
        )
        reward += trade_shaped
        reward += ows_specialization_shaping(
            state_before,
            int(player),
            int(action),
            ows_action_coef=float(reward_shaping_ows_actions),
        )
        reward += main_phase_road_purpose_shaping(
            state_before,
            int(player),
            int(action),
            main_road_purpose_coef=float(reward_shaping_main_road_purpose),
            action_mask=mask,
        )
        if trade_util != 0.0 or trade_shaped != 0.0:
            trade_value_accum["count"] += 1
            trade_value_accum["utility_sum"] += float(trade_util)
            trade_value_accum["shaping_sum"] += float(trade_shaped)

        done = bool(res.done)
        if done and int(env.state.winner) >= 0:
            reward += terminal_table_mean_shaping(
                env.state,
                int(player),
                float(reward_shaping_terminal_table_mean),
            )
        episode_steps += 1
        if not done and episode_steps >= max_episode_steps:
            done = True
            if truncation_leader_reward:
                reward += _truncation_reward_for_player(env, player)

        buf["obs"].append(obs)
        buf["actions"].append(action)
        buf["logp"].append(logp)
        buf["values"].append(value)
        buf["rewards"].append(reward)
        buf["dones"].append(float(done))
        buf["masks"].append(mask)
        if done:
            obs, info = env.reset()
            mask = info["action_mask"]
            episode_steps = 0
        else:
            obs = res.obs
            mask = res.info["action_mask"]
    for k in ("obs", "masks"):
        buf[k] = np.asarray(buf[k], dtype=np.float32)
    for k in ("actions",):
        buf[k] = np.asarray(buf[k], dtype=np.int64)
    for k in ("logp", "values", "rewards", "dones"):
        buf[k] = np.asarray(buf[k], dtype=np.float32)
    buf["aux_targets"] = {k: np.asarray(v, dtype=np.float32) for k, v in aux_targets.items()}
    buf["aux_target_weights"] = {k: np.asarray(v, dtype=np.float32) for k, v in aux_target_weights.items()}
    buf["strategy_targets"] = np.asarray(strategy_targets, dtype=np.float32)
    adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], trainer.cfg.gamma, trainer.cfg.lam)
    buf["advantages"] = adv
    buf["returns"] = ret
    buf["aux_label_stats"] = {
        k: float(aux_accum[k]) / max(1.0, float(aux_weight_accum[k]))
        for k in AUX_LABEL_KEYS
    }
    buf["aux_label_counts"] = {k: int(round(float(aux_weight_accum[k]))) for k in AUX_LABEL_KEYS}
    tcount = max(1, int(trade_value_accum["count"]))
    buf["trade_value_stats"] = {
        "mean_trade_utility": float(trade_value_accum["utility_sum"]) / tcount,
        "mean_trade_shaping": float(trade_value_accum["shaping_sum"]) / tcount,
        "trade_actions_scored": int(trade_value_accum["count"]),
    }
    buf["forced_knight_bootstrap_count"] = int(forced_knight_count)
    buf["forced_trade_bootstrap_count"] = int(forced_trade_count)
    buf["setup_selection_override_count"] = int(setup_selection_override_count)
    return buf


def train_self_play(config: TrainConfig, ppo_cfg: PPOConfig) -> tuple[PolicyValueNet, list[dict]]:
    rng = np.random.default_rng(config.seed)
    env = CatanEnv(seed=config.seed)
    obs, _ = env.reset(seed=config.seed)
    model = PolicyValueNet(obs_dim=obs.shape[0], action_dim=env.action_mask().shape[0])
    trainer = PPOTrainer(model, ppo_cfg)
    history: list[dict] = []

    for update in range(1, config.total_updates + 1):
        env.seed = int(rng.integers(0, 1_000_000))
        force_trade_bootstrap_prob = (
            float(config.force_trade_bootstrap_prob)
            if int(config.force_trade_bootstrap_updates) <= 0 or int(update) <= int(config.force_trade_bootstrap_updates)
            else 0.0
        )
        batch = collect_rollout(
            env,
            trainer,
            config.rollout_steps,
            max_episode_steps=config.max_episode_steps,
            truncation_leader_reward=config.truncation_leader_reward,
            reward_shaping_vp=config.reward_shaping_vp,
            reward_shaping_resource=config.reward_shaping_resource,
            reward_shaping_robber_block_leader=config.reward_shaping_robber_block_leader,
            reward_shaping_rob_leader=config.reward_shaping_rob_leader,
            reward_shaping_rob_mistarget=config.reward_shaping_rob_mistarget,
            reward_shaping_play_knight=config.reward_shaping_play_knight,
            reward_shaping_play_knight_when_blocked=config.reward_shaping_play_knight_when_blocked,
            reward_shaping_knight_unblock_penalty=config.reward_shaping_knight_unblock_penalty,
            reward_shaping_setup_settlement=config.reward_shaping_setup_settlement,
            reward_shaping_setup_top6_bonus=config.reward_shaping_setup_top6_bonus,
            reward_shaping_setup_one_hex_penalty=config.reward_shaping_setup_one_hex_penalty,
            reward_shaping_setup_round1_floor=config.reward_shaping_setup_round1_floor,
            reward_shaping_setup_road_frontier=config.reward_shaping_setup_road_frontier,
            setup_selection_influence_prob=config.setup_selection_influence_prob,
            setup_selection_influence_settlement_prob=config.setup_selection_influence_settlement_prob,
            setup_selection_influence_settlement_round2_prob=config.setup_selection_influence_settlement_round2_prob,
            setup_selection_influence_road_prob=config.setup_selection_influence_road_prob,
            reward_shaping_terminal_table_mean=config.reward_shaping_terminal_table_mean,
            reward_shaping_trade_offer_value=config.reward_shaping_trade_offer_value,
            reward_shaping_trade_offer_counterparty_value=config.reward_shaping_trade_offer_counterparty_value,
            reward_shaping_trade_accept_value=config.reward_shaping_trade_accept_value,
            reward_shaping_bank_trade_value=config.reward_shaping_bank_trade_value,
            reward_shaping_trade_reject_bad_offer_value=config.reward_shaping_trade_reject_bad_offer_value,
            reward_shaping_ows_actions=config.reward_shaping_ows_actions,
            reward_shaping_main_road_purpose=config.reward_shaping_main_road_purpose,
            force_trade_bootstrap_prob=force_trade_bootstrap_prob,
            threat_dev_card_weight=config.threat_dev_card_weight,
        )
        stats = trainer.update(batch)
        row = {"update": update, **stats}
        if update % config.eval_every == 0:
            # Quick baseline benchmark.
            from .wrappers import PolicyAgent

            policy_agent = PolicyAgent(model)
            opp = RandomLegalAgent(seed=config.seed + update)
            eval_stats = tournament([policy_agent, opp, opp, opp], num_games=20, base_seed=config.seed + update * 100)
            row.update({"eval_win_rate": eval_stats["win_rates"][0], "eval_avg_turns": eval_stats["avg_turns"]})
        history.append(row)
    return model, history


def save_checkpoint(model: PolicyValueNet, path: str) -> None:
    torch.save(model.state_dict(), path)
