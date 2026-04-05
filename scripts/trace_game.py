"""Run a single game and print a human-readable action walkthrough."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from catan_rl.actions import CATALOG, Action
from catan_rl.bots import HeuristicAgent, RandomLegalAgent
from catan_rl.constants import Building, DevCard, INITIAL_DEV_DECK, Phase
from catan_rl.encoding import engineered_feature_summary, trade_offer_readiness_deltas
from catan_rl.env import CatanEnv
from catan_rl.strategy_metrics import (
    bank_trade_value,
    init_dev_timing_tracker,
    is_setup_settlement_phase,
    record_dev_timing_step,
    road_frontier_metrics,
    strategic_evaluator_snapshot,
    setup_choice_metrics,
    summarize_dev_timing,
    trade_accept_value,
    trade_accept_immediate_build_gain,
    trade_accept_should_avoid_proposer,
    trade_offer_counterparty_value,
    trade_offer_value,
)
from catan_rl.strategy_archetypes import archetype_definitions, infer_primary_strategy, infer_strategy_distribution
from catan_rl.training.self_play import setup_settlement_shaping_details
from catan_rl.training.ppo import load_policy_value_net


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


def _policy_choice_with_probs(
    model: torch.nn.Module,
    obs: np.ndarray,
    mask: np.ndarray,
    *,
    sample_policy: bool = False,
    policy_temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, list[tuple[int, float]]]:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(obs_t)
        logits = logits.masked_fill(mask_t <= 0, -1e9)
        if policy_temperature <= 0:
            raise ValueError("--policy-temperature must be > 0.")
        logits = logits / float(policy_temperature)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    if sample_policy:
        if rng is None:
            rng = np.random.default_rng()
        action_id = int(rng.choice(np.arange(len(probs)), p=probs))
    else:
        action_id = int(np.argmax(probs))
    top_ids = np.argsort(probs)[-5:][::-1]
    top = [(int(i), float(probs[i])) for i in top_ids if probs[i] > 0.0]
    return action_id, top


def _policy_strategy_mixture(model: torch.nn.Module, obs: np.ndarray) -> list[tuple[str, float]]:
    if not hasattr(model, "predict_strategy"):
        return []
    defs = archetype_definitions()
    keys = [d["key"] for d in defs]
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        probs = model.predict_strategy(obs_t).squeeze(0).cpu().numpy()
    top_ids = np.argsort(probs)[-3:][::-1]
    out: list[tuple[str, float]] = []
    for i in top_ids:
        if i < 0 or i >= len(keys):
            continue
        out.append((keys[int(i)], float(probs[int(i)])))
    return out


def _print_policy_strategy_mixture(top: list[tuple[str, float]]) -> None:
    if not top:
        return
    defs = {d["key"]: d["title"] for d in archetype_definitions()}
    txt = " | ".join(f"{defs.get(k, k)}:{v:.3f}" for k, v in top)
    print("  policy_strategy_top3:", txt)


def _print_setup(env: CatanEnv) -> None:
    s = env.state
    print("=== Setup ===")
    print(f"seed={env.seed}")
    print(f"robber_hex={s.robber_hex}")
    print(f"hex_terrain={s.hex_terrain.tolist()}")
    print(f"hex_number={s.hex_number.tolist()}")
    print(f"port_type={s.port_type.tolist()}")
    print("=============\n")


def _agent_for_player(player: int, model_players: set[int], model: torch.nn.Module, seed: int):
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


TRADE_KINDS = {"PROPOSE_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "BANK_TRADE"}


def _record_trade_event(
    events: list[dict],
    *,
    step: int,
    player: int,
    role: str,
    phase: str,
    action_id: int,
    state_before,
) -> None:
    action = CATALOG.decode(action_id)
    if action.kind not in TRADE_KINDS:
        return
    rec = {
        "step": int(step),
        "player": int(player),
        "role": role,
        "phase": phase,
        "action_id": int(action_id),
        "action": _format_action(action),
        "kind": action.kind,
    }

    if action.kind == "PROPOSE_TRADE":
        give_r, give_n, want_r, want_n = action.params
        proposer_util = float(
            trade_offer_value(state_before, int(player), int(give_r), int(give_n), int(want_r), int(want_n))
        )
        rec["proposer_utility"] = proposer_util
        rec["best_responder_utility"] = float(
            trade_offer_counterparty_value(state_before, int(player), int(give_r), int(give_n), int(want_r), int(want_n))
        )
    elif action.kind == "ACCEPT_TRADE":
        accepter_util = float(trade_accept_value(state_before, int(player)))
        rec["response_accept_utility"] = accepter_util
        rec["immediate_unlock"] = bool(trade_accept_immediate_build_gain(state_before, int(player)))
        rec["avoid_proposer"] = bool(trade_accept_should_avoid_proposer(state_before, int(player)))
        proposer = int(state_before.trade_proposer)
        offer_give = state_before.trade_offer_give.astype(np.int64)
        offer_want = state_before.trade_offer_want.astype(np.int64)
        proposer_offer_util = 0.0
        if proposer >= 0 and int(np.sum(offer_give)) > 0 and int(np.sum(offer_want)) > 0:
            give_r = int(np.argmax(offer_give))
            want_r = int(np.argmax(offer_want))
            give_n = int(offer_give[give_r])
            want_n = int(offer_want[want_r])
            proposer_offer_util = float(
                trade_offer_value(state_before, proposer, give_r, give_n, want_r, want_n)
            )
        rec["accepter_utility"] = accepter_util
        rec["proposer"] = proposer
        rec["proposer_offer_utility"] = proposer_offer_util
    elif action.kind == "REJECT_TRADE":
        rec["response_accept_utility"] = float(trade_accept_value(state_before, int(player)))
        rec["immediate_unlock"] = bool(trade_accept_immediate_build_gain(state_before, int(player)))
        rec["avoid_proposer"] = bool(trade_accept_should_avoid_proposer(state_before, int(player)))
    elif action.kind == "BANK_TRADE":
        give_r, give_n, want_r = action.params
        rec["bank_trade_utility"] = float(
            bank_trade_value(state_before, int(player), int(give_r), int(give_n), int(want_r))
        )

    events.append(rec)


def _print_trade_summary(events: list[dict]) -> None:
    print("\n=== Trade Summary ===")
    if not events:
        print("No trade actions recorded.")
        print("=====================")
        return

    by_kind = Counter(e["kind"] for e in events)
    by_player: dict[int, Counter] = defaultdict(Counter)
    for e in events:
        by_player[int(e["player"])][str(e["kind"])] += 1

    print(f"total_trade_actions={len(events)}")
    for kind in ("PROPOSE_TRADE", "ACCEPT_TRADE", "REJECT_TRADE", "BANK_TRADE"):
        print(f"{kind}={int(by_kind.get(kind, 0))}")

    print("\nper_player_trade_counts:")
    for p in range(4):
        c = by_player.get(p, Counter())
        print(
            f"  P{p}: propose={int(c.get('PROPOSE_TRADE', 0))} "
            f"accept={int(c.get('ACCEPT_TRADE', 0))} "
            f"reject={int(c.get('REJECT_TRADE', 0))} "
            f"bank={int(c.get('BANK_TRADE', 0))}"
        )

    proposer_utils = [float(e["proposer_utility"]) for e in events if "proposer_utility" in e]
    best_responder_utils = [float(e["best_responder_utility"]) for e in events if "best_responder_utility" in e]
    accepter_utils = [float(e["accepter_utility"]) for e in events if "accepter_utility" in e]
    proposer_offer_utils_on_accept = [float(e["proposer_offer_utility"]) for e in events if "proposer_offer_utility" in e]
    bank_utils = [float(e["bank_trade_utility"]) for e in events if "bank_trade_utility" in e]
    response_events = [e for e in events if e["kind"] in {"ACCEPT_TRADE", "REJECT_TRADE"}]
    positive_accept_util_count = sum(float(e.get("response_accept_utility", -1.0)) > 0.0 for e in response_events)
    immediate_unlock_count = sum(bool(e.get("immediate_unlock", False)) for e in response_events)
    avoid_proposer_count = sum(bool(e.get("avoid_proposer", False)) for e in response_events)
    rejected_positive_util_count = sum(
        e["kind"] == "REJECT_TRADE" and float(e.get("response_accept_utility", -1.0)) > 0.0 for e in response_events
    )
    rejected_unlock_count = sum(
        e["kind"] == "REJECT_TRADE" and bool(e.get("immediate_unlock", False)) for e in response_events
    )
    if proposer_utils or best_responder_utils or accepter_utils or proposer_offer_utils_on_accept or bank_utils:
        print("\nutility_estimates:")
        if proposer_utils:
            print(f"  propose_mean_proposer_utility={float(np.mean(proposer_utils)):.3f}")
        if best_responder_utils:
            print(f"  propose_mean_best_responder_utility={float(np.mean(best_responder_utils)):.3f}")
        if accepter_utils:
            print(f"  accept_mean_accepter_utility={float(np.mean(accepter_utils)):.3f}")
        if proposer_offer_utils_on_accept:
            print(
                "  accept_mean_proposer_offer_utility="
                f"{float(np.mean(proposer_offer_utils_on_accept)):.3f}"
            )
        if bank_utils:
            print(f"  bank_trade_mean_utility={float(np.mean(bank_utils)):.3f}")
    if response_events:
        print("\nresponse_diagnostics:")
        print(
            f"  positive_accept_utility={positive_accept_util_count}/{len(response_events)} "
            f"({positive_accept_util_count / max(1, len(response_events)):.3f})"
        )
        print(
            f"  immediate_unlock_if_accept={immediate_unlock_count}/{len(response_events)} "
            f"({immediate_unlock_count / max(1, len(response_events)):.3f})"
        )
        print(f"  reject_with_positive_accept_utility={rejected_positive_util_count}")
        print(f"  reject_with_immediate_unlock={rejected_unlock_count}")
        print(f"  response_events_avoid_proposer={avoid_proposer_count}")

    print("\nlast_trade_events:")
    for e in events[-12:]:
        util_bits: list[str] = []
        if "proposer_utility" in e:
            util_bits.append(f"prop_u={float(e['proposer_utility']):+.3f}")
        if "best_responder_utility" in e:
            util_bits.append(f"best_resp_u={float(e['best_responder_utility']):+.3f}")
        if "accepter_utility" in e:
            util_bits.append(f"acc_u={float(e['accepter_utility']):+.3f}")
        if "response_accept_utility" in e and "accepter_utility" not in e:
            util_bits.append(f"acc_if_accept_u={float(e['response_accept_utility']):+.3f}")
        if "proposer_offer_utility" in e:
            util_bits.append(f"prop_offer_u={float(e['proposer_offer_utility']):+.3f}")
        if "bank_trade_utility" in e:
            util_bits.append(f"bank_u={float(e['bank_trade_utility']):+.3f}")
        if "immediate_unlock" in e:
            util_bits.append(f"unlock={bool(e['immediate_unlock'])}")
        if "avoid_proposer" in e:
            util_bits.append(f"avoid={bool(e['avoid_proposer'])}")
        util_txt = (" " + " ".join(util_bits)) if util_bits else ""
        print(
            f"  step={int(e['step']):04d} player={int(e['player'])} "
            f"role={str(e['role']):9s} phase={str(e['phase']):15s} action={str(e['action'])}{util_txt}"
        )
    print("=====================")


def _opponent_leader_set(state, player: int, threat_dev_card_weight: float = 0.7) -> set[int]:
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


def _record_robber_knight_event(
    events: list[dict],
    *,
    step: int,
    player: int,
    action: Action,
    state_before,
    threat_dev_card_weight: float = 0.7,
) -> None:
    kind = action.kind
    if kind not in {"PLAY_KNIGHT", "MOVE_ROBBER", "ROB_PLAYER"}:
        return

    rec = {
        "step": int(step),
        "player": int(player),
        "kind": kind,
    }

    leaders = _opponent_leader_set(state_before, int(player), float(threat_dev_card_weight))
    rec["leaders"] = sorted(int(x) for x in leaders)

    if kind == "MOVE_ROBBER":
        (hex_id,) = action.params
        blocked_leader = float(sum(_blocked_production_on_hex(state_before, int(hex_id), p) for p in leaders))
        blocked_self = float(_blocked_production_on_hex(state_before, int(hex_id), int(player)))
        rec["blocked_leader"] = blocked_leader
        rec["blocked_self"] = blocked_self
        rec["block_leader"] = bool(blocked_leader > blocked_self)
    elif kind == "ROB_PLAYER":
        (target,) = action.params
        rec["target"] = int(target)
        rec["robbed_leader"] = bool(int(target) in leaders)

    events.append(rec)


def _print_robber_knight_summary(events: list[dict]) -> None:
    print("\n=== Robber/Knight Summary ===")
    if not events:
        print("No robber/knight actions recorded.")
        print("=============================")
        return

    by_player: dict[int, Counter] = defaultdict(Counter)
    by_kind = Counter()
    for e in events:
        p = int(e["player"])
        k = str(e["kind"])
        by_kind[k] += 1
        by_player[p][k] += 1
        if k == "ROB_PLAYER":
            if bool(e.get("robbed_leader", False)):
                by_player[p]["ROB_LEADER"] += 1
            else:
                by_player[p]["ROB_NON_LEADER"] += 1
        if k == "MOVE_ROBBER" and bool(e.get("block_leader", False)):
            by_player[p]["MOVE_BLOCK_LEADER"] += 1

    print(
        "totals:",
        f"PLAY_KNIGHT={int(by_kind.get('PLAY_KNIGHT', 0))}",
        f"MOVE_ROBBER={int(by_kind.get('MOVE_ROBBER', 0))}",
        f"ROB_PLAYER={int(by_kind.get('ROB_PLAYER', 0))}",
    )
    print("\nper_player:")
    for p in range(4):
        c = by_player.get(p, Counter())
        rob_total = int(c.get("ROB_PLAYER", 0))
        rob_leader = int(c.get("ROB_LEADER", 0))
        rob_rate = (rob_leader / rob_total) if rob_total > 0 else 0.0
        print(
            f"  P{p}: knights={int(c.get('PLAY_KNIGHT', 0))} "
            f"move_robber={int(c.get('MOVE_ROBBER', 0))} "
            f"move_block_leader={int(c.get('MOVE_BLOCK_LEADER', 0))} "
            f"rob_total={rob_total} rob_leader={rob_leader} rob_leader_rate={rob_rate:.3f}"
        )

    print("\nlast_robber_knight_events:")
    for e in events[-12:]:
        kind = str(e["kind"])
        if kind == "ROB_PLAYER":
            extra = f" target={int(e.get('target', -1))} robbed_leader={bool(e.get('robbed_leader', False))}"
        elif kind == "MOVE_ROBBER":
            extra = (
                f" block_leader={bool(e.get('block_leader', False))}"
                f" blocked_leader={float(e.get('blocked_leader', 0.0)):.2f}"
                f" blocked_self={float(e.get('blocked_self', 0.0)):.2f}"
            )
        else:
            extra = ""
        print(f"  step={int(e['step']):04d} player={int(e['player'])} kind={kind}{extra}")
    print("=============================")


def _print_setup_shaping_event(*, step: int, player: int, action_id: int, state_before) -> None:
    details = setup_settlement_shaping_details(state_before, int(action_id))
    if details is None:
        return
    hits = details.get("rule_hits", [])
    hits_txt = ",".join(str(x) for x in hits) if hits else "none"
    print(
        "  setup_shaping:",
        f"vertex={int(details.get('vertex', -1))}",
        f"hits={hits_txt}",
        f"eligible={bool(details.get('eligible', False))}",
        f"pips={float(details.get('total_pips', 0.0)):.2f}",
        f"hexes={int(details.get('productive_hexes', 0))}",
        f"diversity={int(details.get('diversity', 0))}",
        f"has_combo={bool(details.get('has_combo', False))}",
        f"has_complement_pair={bool(details.get('has_complement_pair', False))}",
        f"pip_quality={float(details.get('pip_quality', 0.0)):.3f}",
        f"combo_factor={float(details.get('combo_factor', 0.0)):.3f}",
        f"(step={int(step)} player={int(player)})",
    )


def _record_opening_event(events: list[dict], *, step: int, player: int, action: Action, state_before) -> dict | None:
    if action.kind != "PLACE_SETTLEMENT" or not is_setup_settlement_phase(state_before):
        return None
    (vertex,) = action.params
    metrics = setup_choice_metrics(state_before, int(player), int(vertex))
    rec = {
        "step": int(step),
        "player": int(player),
        "vertex": int(vertex),
        **metrics,
    }
    events.append(rec)
    return rec


def _print_opening_summary(events: list[dict]) -> None:
    print("\n=== Opening Quality Summary ===")
    if not events:
        print("No setup settlement events recorded.")
        print("==============================")
        return
    by_player: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_player[int(e["player"])].append(e)
    for p in range(4):
        ev = by_player.get(p, [])
        if not ev:
            print(f"  P{p}: no setup settlements recorded")
            continue
        mean_rank = float(np.mean([float(x["rank"]) for x in ev]))
        mean_pct = float(np.mean([float(x["percentile"]) for x in ev]))
        mean_gap = float(np.mean([float(x["score_gap_to_best"]) for x in ev]))
        mean_pips = float(np.mean([float(x["total_pips"]) for x in ev]))
        print(
            f"  P{p}: count={len(ev)} mean_rank={mean_rank:.2f} "
            f"mean_percentile={mean_pct:.3f} mean_gap_to_best={mean_gap:.3f} mean_pips={mean_pips:.2f}"
        )
    print("last_opening_events:")
    for e in events[-8:]:
        print(
            f"  step={int(e['step']):04d} P{int(e['player'])} vertex={int(e['vertex'])} "
            f"rank={int(e['rank'])}/{int(e['candidate_count'])} pct={float(e['percentile']):.3f} "
            f"gap={float(e['score_gap_to_best']):.3f} pips={float(e['total_pips']):.2f}"
        )
    print("==============================")


def _record_road_frontier_event(events: list[dict], *, step: int, player: int, phase: str, action: Action, state_before) -> dict | None:
    if action.kind != "PLACE_ROAD":
        return None
    (edge,) = action.params
    metrics = road_frontier_metrics(state_before, int(player), int(edge))
    rec = {
        "step": int(step),
        "player": int(player),
        "phase": str(phase),
        "edge": int(edge),
        **metrics,
    }
    events.append(rec)
    return rec


def _print_road_frontier_summary(events: list[dict]) -> None:
    print("\n=== Road Frontier Summary ===")
    if not events:
        print("No road placement events recorded.")
        print("=============================")
        return
    by_player: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        by_player[int(e["player"])].append(e)
    for p in range(4):
        ev = by_player.get(p, [])
        if not ev:
            print(f"  P{p}: no roads placed")
            continue
        mean_gain = float(np.mean([float(x["connected_gain"]) for x in ev]))
        useful = int(sum(1 for x in ev if int(x["connected_gain"]) > 0 or int(x["best_site_roads_delta"]) > 0))
        mean_delta = float(np.mean([float(x["best_site_roads_delta"]) for x in ev]))
        print(
            f"  P{p}: roads={len(ev)} useful={useful} useful_rate={(useful / len(ev)):.3f} "
            f"mean_connected_gain={mean_gain:.3f} mean_best_site_delta={mean_delta:.3f}"
        )
    print("last_road_events:")
    for e in events[-10:]:
        print(
            f"  step={int(e['step']):04d} P{int(e['player'])} phase={str(e['phase'])} edge={int(e['edge'])} "
            f"conn_gain={int(e['connected_gain'])} best_site_delta={int(e['best_site_roads_delta'])}"
        )
    print("=============================")


def _print_engineered_summary(env: CatanEnv) -> None:
    s = env.state
    summary = engineered_feature_summary(s, current_player=int(s.current_player))
    print("\n=== Engineered Feature Summary ===")
    print(f"current_player={int(s.current_player)} phase={Phase(s.phase).name}")
    for k, v in summary.items():
        vals = ", ".join(f"{float(x):.3f}" for x in v)
        print(f"{k}=[{vals}]")
    print("==================================")


def _print_strategy_metrics_summary(env: CatanEnv) -> None:
    s = env.state
    print("\n=== Strategy Metrics Summary ===")
    for p in range(len(s.public_vp)):
        snap = strategic_evaluator_snapshot(s, int(p))
        race = snap["race_pressure"]
        ready = snap["build_readiness"]
        rbq = snap["robber_block_quality"]
        print(
            f"  P{p}: city_top_delta={float(snap['city_top_delta_score']):.3f} "
            f"lr_pressure={float(race['longest_road_pressure']):.3f} "
            f"army_pressure={float(race['largest_army_pressure']):.3f} "
            f"robber_quality={float(rbq['quality_score']):.3f}"
        )
        print(
            f"      readiness: road={float(ready['road']['estimated_turns']):.2f} "
            f"settle={float(ready['settlement']['estimated_turns']):.2f} "
            f"city={float(ready['city']['estimated_turns']):.2f} "
            f"dev={float(ready['dev']['estimated_turns']):.2f}"
        )
    print("===============================")


def _print_strategy_archetype_summary(env: CatanEnv) -> None:
    s = env.state
    defs = {d["key"]: d["title"] for d in archetype_definitions()}
    print("\n=== Strategy Archetype Heuristic Labels ===")
    for p in range(len(s.public_vp)):
        dist = infer_strategy_distribution(s, int(p))
        primary = infer_primary_strategy(s, int(p))
        ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_txt = " | ".join(f"{defs.get(k, k)}={float(v):.3f}" for k, v in ranked)
        print(f"  P{p}: primary={defs.get(primary, primary)} ; top3={top_txt}")
    print("===========================================")


def _print_dev_card_summary(env: CatanEnv, dev_timing_tracker: dict | None = None) -> None:
    s = env.state
    labels = ["KNIGHT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY", "VP"]
    print("\n=== Dev Card Summary ===")
    for p in range(len(s.public_vp)):
        held = (s.dev_cards_hidden[p] + s.dev_cards_bought_this_turn[p]).astype(int)
        held_txt = ", ".join(f"{labels[i]}={int(held[i])}" for i in range(len(labels)))
        print(
            f"  P{p}: held[{held_txt}] "
            f"played_knights={int(s.knights_played[p])} "
            f"vp_cards_held={int(s.vp_cards_held[p])}"
        )

    # Non-knight dev plays are not tracked per-player historically.
    # Show global played estimates by conservation.
    estimated_played = np.asarray(INITIAL_DEV_DECK, dtype=np.int64) - s.dev_deck_composition - (
        s.dev_cards_hidden + s.dev_cards_bought_this_turn
    ).sum(axis=0)
    est_txt = ", ".join(f"{labels[i]}={int(max(0, estimated_played[i]))}" for i in range(len(labels)))
    print(f"  global_estimated_played[{est_txt}]")
    if dev_timing_tracker is not None:
        timing = summarize_dev_timing(dev_timing_tracker)
        print("\n  timing_per_player:")
        for p in range(4):
            by_card = timing["per_player"][p]
            k = by_card["KNIGHT"]
            y = by_card["YEAR_OF_PLENTY"]
            m = by_card["MONOPOLY"]
            print(
                f"    P{p}: "
                f"KN(opp={int(k['opportunity_turns'])},play={int(k['play_turns'])},first={int(k['first_play_turn'])},"
                f"hold_before={int(k['held_turns_before_first_play'])},rate={float(k['play_rate_when_available']):.3f}) "
                f"YOP(opp={int(y['opportunity_turns'])},play={int(y['play_turns'])},first={int(y['first_play_turn'])},"
                f"rate={float(y['play_rate_when_available']):.3f}) "
                f"MONO(opp={int(m['opportunity_turns'])},play={int(m['play_turns'])},first={int(m['first_play_turn'])},"
                f"rate={float(m['play_rate_when_available']):.3f})"
            )
        t = timing["summary"]
        print(
            "  timing_summary:"
            f" knight_rate={float(t['knight_mean_play_rate_when_available']):.3f}"
            f" knight_hold_before_first={float(t['knight_mean_held_turns_before_first_play']):.3f}"
            f" yop_rate={float(t['yop_mean_play_rate_when_available']):.3f}"
            f" monopoly_rate={float(t['monopoly_mean_play_rate_when_available']):.3f}"
        )
    print("========================")


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
    lr_flags = [int(x) for x in s.has_longest_road.tolist()]
    la_flags = [int(x) for x in s.has_largest_army.tolist()]
    print(
        "awards_summary: "
        f"has_longest_road={lr_flags} "
        f"has_largest_army={la_flags}"
    )
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


def _checkpoint_meta(path: str) -> dict:
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_model_for_trace(
    *,
    checkpoint: str,
    obs_dim: int,
    action_dim: int,
    model_arch: str | None,
    model_hidden: int | None,
    model_residual_blocks: int | None,
):
    meta = _checkpoint_meta(checkpoint)
    arch = model_arch
    hidden = model_hidden
    blocks = model_residual_blocks

    # If arch is provided but shape args are omitted, prefer checkpoint metadata.
    if arch is not None:
        if hidden is None and "model_hidden" in meta:
            hidden = int(meta["model_hidden"])
        if blocks is None and "model_residual_blocks" in meta:
            blocks = int(meta["model_residual_blocks"])

    # If user provided an incomplete manual config and metadata is unavailable, auto-infer.
    if arch is not None and (hidden is None or blocks is None):
        print(
            "warning: partial model config provided (arch without hidden/blocks) and no usable metadata; "
            "falling back to checkpoint auto-inference."
        )
        arch = None

    try:
        return load_policy_value_net(
            checkpoint,
            obs_dim=obs_dim,
            action_dim=action_dim,
            model_arch=arch,
            hidden=256 if hidden is None else int(hidden),
            residual_blocks=4 if blocks is None else int(blocks),
        )
    except Exception:
        if arch is not None:
            print("warning: provided model config did not match checkpoint; retrying with auto-inference.")
            return load_policy_value_net(
                checkpoint,
                obs_dim=obs_dim,
                action_dim=action_dim,
                model_arch=None,
                hidden=256,
                residual_blocks=4,
            )
        raise


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
    parser.add_argument(
        "--sample-policy",
        action="store_true",
        help="Sample policy actions from masked probabilities instead of argmax.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Temperature for policy sampling/argmax probabilities (>0).",
    )
    parser.add_argument(
        "--show-policy-strategy-mixture",
        action="store_true",
        help="For strategy-aware models, print top-3 predicted strategy mixture each policy step.",
    )
    parser.add_argument(
        "--model-arch",
        choices=[
            "mlp",
            "residual_mlp",
            "phase_aware_residual_mlp",
            "strategy_phase_aware_residual_mlp",
            "graph_entity_hybrid",
            "graph_entity_phase_aware_hybrid",
            "graph_entity_only",
        ],
        default=None,
    )
    parser.add_argument("--model-hidden", type=int, default=None)
    parser.add_argument("--model-residual-blocks", type=int, default=None)
    parser.add_argument("--max-main-actions-per-turn", type=int, default=10)
    parser.add_argument("--disable-player-trade", action="store_true")
    parser.add_argument("--trade-action-mode", choices=["guided", "full"], default="guided")
    parser.add_argument("--max-player-trade-proposals-per-turn", type=int, default=None)
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
    parser.add_argument(
        "--show-engineered-summary",
        action="store_true",
        help="Print planning-focused engineered feature top3 values from final state.",
    )
    parser.add_argument(
        "--show-strategy-metrics",
        action="store_true",
        help="Print reusable strategy-evaluator snapshot metrics in final state.",
    )
    parser.add_argument(
        "--show-strategy-archetypes",
        action="store_true",
        help="Print heuristic strategy archetype labels derived from current state.",
    )
    parser.add_argument(
        "--show-dev-card-summary",
        action="store_true",
        help="Print per-player dev-card holding and played summary in final state.",
    )
    parser.add_argument(
        "--show-setup-shaping-events",
        action="store_true",
        help="Print per-setup-settlement shaping rule hits using training reward logic.",
    )
    args = parser.parse_args()

    model_players = set(int(x) for x in args.model_players.split(",") if x != "")
    policy_rng = np.random.default_rng(args.seed + 424242) if args.sample_policy else None
    env = CatanEnv(
        seed=args.seed,
        max_main_actions_per_turn=args.max_main_actions_per_turn,
        allow_player_trade=not args.disable_player_trade,
        trade_action_mode=args.trade_action_mode,
        max_player_trade_proposals_per_turn=args.max_player_trade_proposals_per_turn,
    )
    obs, info = env.reset(seed=args.seed)
    mask = info["action_mask"]

    model = _load_model_for_trace(
        checkpoint=args.checkpoint,
        obs_dim=obs.shape[0],
        action_dim=mask.shape[0],
        model_arch=args.model_arch,
        model_hidden=args.model_hidden,
        model_residual_blocks=args.model_residual_blocks,
    )
    model.eval()

    _print_setup(env)

    out_f = open(args.json_out, "w", encoding="utf-8") if args.json_out else None
    trade_events: list[dict] = []
    robber_knight_events: list[dict] = []
    opening_events: list[dict] = []
    road_frontier_events: list[dict] = []
    dev_timing_tracker = init_dev_timing_tracker()
    try:
        for step in range(1, args.max_steps + 1):
            if env.state.phase == Phase.GAME_OVER:
                break
            player = int(env.state.current_player)
            phase = Phase(env.state.phase).name
            legal_count = int(mask.sum())
            role, bot = _agent_for_player(player, model_players, model, args.seed)

            if role == "policy":
                action_id, top = _policy_choice_with_probs(
                    model,
                    obs,
                    mask,
                    sample_policy=bool(args.sample_policy),
                    policy_temperature=float(args.policy_temperature),
                    rng=policy_rng,
                )
                strat_top = _policy_strategy_mixture(model, obs) if args.show_policy_strategy_mixture else []
            else:
                action_id = int(bot.act(obs, mask))
                top = []
                strat_top = []
            action = CATALOG.decode(action_id)
            state_before = env.state.copy()
            record_dev_timing_step(dev_timing_tracker, state_before, int(player), str(action.kind))

            print(
                f"step={step:03d} player={player} role={role:9s} phase={phase:15s} "
                f"legal={legal_count:3d} action={_format_action(action)}"
            )
            if int(state_before.phase) == int(Phase.TRADE_PROPOSED):
                d = trade_offer_readiness_deltas(state_before, int(player))
                print(
                    "  trade_accept_readiness_delta:"
                    f" road={float(d[0]):+.1f}"
                    f" settle={float(d[1]):+.1f}"
                    f" city={float(d[2]):+.1f}"
                    f" dev={float(d[3]):+.1f}"
                    f" total={float(d[4]):+.2f}"
                )
            if top:
                _print_top_actions(top)
            if strat_top:
                _print_policy_strategy_mixture(strat_top)
            if args.show_setup_shaping_events:
                _print_setup_shaping_event(step=step, player=player, action_id=action_id, state_before=state_before)
            opening_event = _record_opening_event(
                opening_events,
                step=step,
                player=player,
                action=action,
                state_before=state_before,
            )
            road_event = _record_road_frontier_event(
                road_frontier_events,
                step=step,
                player=player,
                phase=phase,
                action=action,
                state_before=state_before,
            )
            _record_trade_event(
                trade_events,
                step=step,
                player=player,
                role=role,
                phase=phase,
                action_id=action_id,
                state_before=state_before,
            )
            _record_robber_knight_event(
                robber_knight_events,
                step=step,
                player=player,
                action=action,
                state_before=state_before,
                threat_dev_card_weight=0.7,
            )

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
                            "policy_strategy_top3": strat_top,
                            "reward": float(result.reward),
                            "winner": int(result.info.get("winner", -1)),
                            "public_vp": env.state.public_vp.tolist(),
                            "actual_vp": env.state.actual_vp.tolist(),
                            "opening_event": opening_event,
                            "road_frontier_event": road_event,
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
        if args.show_engineered_summary:
            _print_engineered_summary(env)
        if args.show_strategy_metrics:
            _print_strategy_metrics_summary(env)
        if args.show_strategy_archetypes:
            _print_strategy_archetype_summary(env)
        if args.show_dev_card_summary:
            _print_dev_card_summary(env, dev_timing_tracker)
        _print_opening_summary(opening_events)
        _print_road_frontier_summary(road_frontier_events)
        _print_trade_summary(trade_events)
        _print_robber_knight_summary(robber_knight_events)
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
