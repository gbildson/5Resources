"""Evaluate a trained checkpoint and write a consolidated baseline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_rl.actions import CATALOG
from catan_rl.bots import HeuristicAgent, RandomLegalAgent
from catan_rl.constants import Phase
from catan_rl.env import CatanEnv
from catan_rl.strategy_metrics import (
    init_dev_timing_tracker,
    is_setup_settlement_phase,
    record_dev_timing_step,
    road_frontier_metrics,
    setup_choice_metrics,
    summarize_dev_timing,
)
from catan_rl.training.ppo import load_policy_value_net
from catan_rl.training.wrappers import PolicyAgent


def _player_resource_presence_mask(state, player: int) -> np.ndarray:
    mask = np.zeros(5, dtype=np.bool_)
    for v in range(len(state.vertex_owner)):
        if int(state.vertex_owner[v]) != int(player):
            continue
        if int(state.vertex_building[v]) <= 0:
            continue
        for h in state.topology.vertex_to_hexes[int(v)]:
            if h < 0:
                continue
            terrain = int(state.hex_terrain[int(h)])
            if terrain <= 0:
                continue
            mask[terrain - 1] = True
    return mask


def _vertex_resource_presence_mask(state, vertex: int) -> np.ndarray:
    mask = np.zeros(5, dtype=np.bool_)
    for h in state.topology.vertex_to_hexes[int(vertex)]:
        if h < 0:
            continue
        terrain = int(state.hex_terrain[int(h)])
        if terrain <= 0:
            continue
        mask[terrain - 1] = True
    return mask


def _load_policy_agent(
    checkpoint: Path,
    *,
    model_arch: str | None = None,
    model_hidden: int = 256,
    model_residual_blocks: int = 4,
) -> PolicyAgent:
    env = CatanEnv(seed=0)
    obs, _ = env.reset(seed=0)
    model = load_policy_value_net(
        checkpoint,
        obs_dim=obs.shape[0],
        action_dim=env.action_mask().shape[0],
        model_arch=model_arch,
        hidden=model_hidden,
        residual_blocks=model_residual_blocks,
    )
    model.eval()
    return PolicyAgent(model)


def _eval_across_seats(
    policy_agent: PolicyAgent,
    games_per_seat: int,
    seed: int,
    opponent: str,
    max_steps: int,
    env_kwargs: dict,
    opponent_policy_agent: PolicyAgent | None = None,
) -> dict:
    seat_stats = []
    for seat in range(4):
        agents = []
        for i in range(4):
            if i == seat:
                agents.append(policy_agent)
            elif opponent == "heuristic":
                agents.append(HeuristicAgent(seed=seed + 1000 + i))
            elif opponent == "checkpoint":
                if opponent_policy_agent is None:
                    raise ValueError("opponent_policy_agent is required when opponent='checkpoint'")
                agents.append(opponent_policy_agent)
            else:
                agents.append(RandomLegalAgent(seed=seed + 1000 + i))
        stats = _tournament_with_metrics(
            agents,
            num_games=games_per_seat,
            base_seed=seed + seat * 10000,
            max_steps=max_steps,
            env_kwargs=env_kwargs,
            focal_player=seat,
        )
        seat_stats.append(
            {
                "seat": seat,
                "win_rate": stats["win_rates"][seat],
                "strict_win_rate": stats["strict_win_rates"][seat],
                "avg_turns": stats["avg_turns"],
                "avg_steps": stats["avg_steps"],
                "truncated_games": stats["truncated_games"],
                "terminal_games": stats["terminal_games"],
                "opening_mean_percentile": stats["opening_mean_percentile"],
                "opening_mean_rank": stats["opening_mean_rank"],
                "road_useful_rate": stats["road_useful_rate"],
                "road_mean_connected_gain": stats["road_mean_connected_gain"],
                "knight_play_rate_when_available": stats["knight_play_rate_when_available"],
                "knight_mean_held_turns_before_first_play": stats["knight_mean_held_turns_before_first_play"],
                "yop_play_rate_when_available": stats["yop_play_rate_when_available"],
                "monopoly_play_rate_when_available": stats["monopoly_play_rate_when_available"],
                "trade_offer_rate_when_main": stats["trade_offer_rate_when_main"],
                "trade_bank_rate_when_main": stats["trade_bank_rate_when_main"],
                "trade_accept_rate_when_prompted": stats["trade_accept_rate_when_prompted"],
                "trade_reject_rate_when_prompted": stats["trade_reject_rate_when_prompted"],
                "trade_response_count": stats["trade_response_count"],
                "setup_second_pick_adds_missing_rate": stats["setup_second_pick_adds_missing_rate"],
                "setup_second_pick_overlap_only_rate": stats["setup_second_pick_overlap_only_rate"],
                "setup_second_pick_count": stats["setup_second_pick_count"],
            }
        )
    strict_values = [float(s["strict_win_rate"]) for s in seat_stats]
    return {
        "seat_stats": seat_stats,
        "mean_win_rate": float(np.mean([s["win_rate"] for s in seat_stats])),
        "mean_strict_win_rate": float(np.mean([s["strict_win_rate"] for s in seat_stats])),
        "worst_seat_strict": float(np.min(strict_values)),
        "seat_strict_stddev": float(np.std(strict_values)),
        "mean_avg_turns": float(np.mean([s["avg_turns"] for s in seat_stats])),
        "mean_avg_steps": float(np.mean([s["avg_steps"] for s in seat_stats])),
        "mean_truncated_games": float(np.mean([s["truncated_games"] for s in seat_stats])),
        "mean_opening_percentile": float(np.mean([s["opening_mean_percentile"] for s in seat_stats])),
        "mean_road_useful_rate": float(np.mean([s["road_useful_rate"] for s in seat_stats])),
        "mean_knight_play_rate_when_available": float(
            np.mean([s["knight_play_rate_when_available"] for s in seat_stats])
        ),
        "mean_knight_held_turns_before_first_play": float(
            np.mean([s["knight_mean_held_turns_before_first_play"] for s in seat_stats])
        ),
        "mean_yop_play_rate_when_available": float(np.mean([s["yop_play_rate_when_available"] for s in seat_stats])),
        "mean_monopoly_play_rate_when_available": float(
            np.mean([s["monopoly_play_rate_when_available"] for s in seat_stats])
        ),
        "mean_trade_offer_rate_when_main": float(np.mean([s["trade_offer_rate_when_main"] for s in seat_stats])),
        "mean_trade_bank_rate_when_main": float(np.mean([s["trade_bank_rate_when_main"] for s in seat_stats])),
        "mean_trade_accept_rate_when_prompted": float(
            np.mean([s["trade_accept_rate_when_prompted"] for s in seat_stats])
        ),
        "mean_trade_reject_rate_when_prompted": float(
            np.mean([s["trade_reject_rate_when_prompted"] for s in seat_stats])
        ),
        "mean_setup_second_pick_adds_missing_rate": float(
            np.mean([s["setup_second_pick_adds_missing_rate"] for s in seat_stats])
        ),
        "mean_setup_second_pick_overlap_only_rate": float(
            np.mean([s["setup_second_pick_overlap_only_rate"] for s in seat_stats])
        ),
        "mean_setup_second_pick_count": float(np.mean([s["setup_second_pick_count"] for s in seat_stats])),
    }


def _play_match_with_metrics(
    agents: list,
    *,
    focal_player: int,
    seed: int | None,
    max_steps: int,
    env_kwargs: dict,
) -> dict:
    env = CatanEnv(seed=seed, **env_kwargs)
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    done = False
    steps = 0
    opening_percentiles: list[float] = []
    opening_ranks: list[float] = []
    road_useful_flags: list[float] = []
    road_connected_gains: list[float] = []
    dev_timing_tracker = init_dev_timing_tracker()
    focal_main_turns = 0
    focal_trade_offer_actions = 0
    focal_trade_bank_actions = 0
    focal_trade_response_opps = 0
    focal_trade_accepts = 0
    focal_trade_rejects = 0
    setup_second_pick_adds_missing_flags: list[float] = []
    setup_second_pick_overlap_only_flags: list[float] = []

    while not done and steps < max_steps:
        if np.flatnonzero(mask).size == 0:
            break
        player = int(env.state.current_player)
        action_id = int(agents[player].act(obs, mask))
        action = CATALOG.decode(action_id)
        state_before = env.state.copy()
        if player == int(focal_player):
            if int(state_before.phase) == int(Phase.MAIN):
                focal_main_turns += 1
                if action.kind == "PROPOSE_TRADE":
                    focal_trade_offer_actions += 1
                elif action.kind == "BANK_TRADE":
                    focal_trade_bank_actions += 1
            elif int(state_before.phase) == int(Phase.TRADE_PROPOSED):
                focal_trade_response_opps += 1
                if action.kind == "ACCEPT_TRADE":
                    focal_trade_accepts += 1
                elif action.kind == "REJECT_TRADE":
                    focal_trade_rejects += 1
            record_dev_timing_step(dev_timing_tracker, state_before, int(player), str(action.kind))
        if player == int(focal_player):
            if action.kind == "PLACE_SETTLEMENT" and is_setup_settlement_phase(state_before):
                (vertex,) = action.params
                opening = setup_choice_metrics(state_before, player, int(vertex))
                opening_percentiles.append(float(opening["percentile"]))
                opening_ranks.append(float(opening["rank"]))
                if int(state_before.setup_round) == 2:
                    existing_mask = _player_resource_presence_mask(state_before, int(player))
                    candidate_mask = _vertex_resource_presence_mask(state_before, int(vertex))
                    new_resource_count = int(np.count_nonzero(np.logical_and(~existing_mask, candidate_mask)))
                    overlap_count = int(np.count_nonzero(np.logical_and(existing_mask, candidate_mask)))
                    setup_second_pick_adds_missing_flags.append(1.0 if new_resource_count > 0 else 0.0)
                    setup_second_pick_overlap_only_flags.append(
                        1.0 if (new_resource_count == 0 and overlap_count > 0) else 0.0
                    )
            elif action.kind == "PLACE_ROAD":
                (edge,) = action.params
                road = road_frontier_metrics(state_before, player, int(edge))
                useful = 1.0 if (int(road["connected_gain"]) > 0 or int(road["best_site_roads_delta"]) > 0) else 0.0
                road_useful_flags.append(useful)
                road_connected_gains.append(float(road["connected_gain"]))
        res = env.step(action_id)
        obs, done = res.obs, bool(res.done)
        mask = res.info["action_mask"]
        steps += 1

    if env.state.winner >= 0:
        winners = [int(env.state.winner)]
        truncated = False
    else:
        top_vp = int(env.state.actual_vp.max(initial=0))
        winners = [int(p) for p in np.flatnonzero(env.state.actual_vp == top_vp)]
        truncated = True
    dev_timing = summarize_dev_timing(dev_timing_tracker)
    focal_dev = dev_timing["per_player"][int(focal_player)]
    return {
        "winners": winners,
        "truncated": truncated,
        "turns": int(env.state.turn_number),
        "steps": int(steps),
        "opening_percentiles": opening_percentiles,
        "opening_ranks": opening_ranks,
        "road_useful_flags": road_useful_flags,
        "road_connected_gains": road_connected_gains,
        "dev_timing_focal": focal_dev,
        "trade_offer_rate_when_main": float(focal_trade_offer_actions / max(1, focal_main_turns)),
        "trade_bank_rate_when_main": float(focal_trade_bank_actions / max(1, focal_main_turns)),
        "trade_accept_rate_when_prompted": float(focal_trade_accepts / max(1, focal_trade_response_opps)),
        "trade_reject_rate_when_prompted": float(focal_trade_rejects / max(1, focal_trade_response_opps)),
        "trade_response_count": int(focal_trade_response_opps),
        "setup_second_pick_adds_missing_flags": setup_second_pick_adds_missing_flags,
        "setup_second_pick_overlap_only_flags": setup_second_pick_overlap_only_flags,
    }


def _tournament_with_metrics(
    agents: list,
    *,
    num_games: int,
    base_seed: int,
    max_steps: int,
    env_kwargs: dict,
    focal_player: int,
) -> dict:
    wins = np.zeros(len(agents), dtype=np.float64)
    strict_wins = np.zeros(len(agents), dtype=np.float64)
    turns: list[int] = []
    steps_list: list[int] = []
    truncated_games = 0
    opening_percentiles: list[float] = []
    opening_ranks: list[float] = []
    road_useful_flags: list[float] = []
    road_connected_gains: list[float] = []
    knight_play_rates: list[float] = []
    knight_hold_before_first: list[float] = []
    yop_play_rates: list[float] = []
    monopoly_play_rates: list[float] = []
    trade_offer_rates: list[float] = []
    trade_bank_rates: list[float] = []
    trade_accept_rates: list[float] = []
    trade_reject_rates: list[float] = []
    trade_response_counts: list[float] = []
    setup_second_pick_adds_missing_flags: list[float] = []
    setup_second_pick_overlap_only_flags: list[float] = []

    for i in range(num_games):
        res = _play_match_with_metrics(
            agents,
            focal_player=int(focal_player),
            seed=base_seed + i,
            max_steps=max_steps,
            env_kwargs=env_kwargs,
        )
        if bool(res["truncated"]):
            truncated_games += 1
        elif len(res["winners"]) == 1:
            strict_wins[int(res["winners"][0])] += 1.0
        share = 1.0 / max(1, len(res["winners"]))
        for winner in res["winners"]:
            wins[int(winner)] += share
        turns.append(int(res["turns"]))
        steps_list.append(int(res["steps"]))
        opening_percentiles.extend(float(x) for x in res["opening_percentiles"])
        opening_ranks.extend(float(x) for x in res["opening_ranks"])
        road_useful_flags.extend(float(x) for x in res["road_useful_flags"])
        road_connected_gains.extend(float(x) for x in res["road_connected_gains"])
        dev = res["dev_timing_focal"]
        knight_play_rates.append(float(dev["KNIGHT"]["play_rate_when_available"]))
        knight_hold_before_first.append(float(dev["KNIGHT"]["held_turns_before_first_play"]))
        yop_play_rates.append(float(dev["YEAR_OF_PLENTY"]["play_rate_when_available"]))
        monopoly_play_rates.append(float(dev["MONOPOLY"]["play_rate_when_available"]))
        trade_offer_rates.append(float(res["trade_offer_rate_when_main"]))
        trade_bank_rates.append(float(res["trade_bank_rate_when_main"]))
        trade_accept_rates.append(float(res["trade_accept_rate_when_prompted"]))
        trade_reject_rates.append(float(res["trade_reject_rate_when_prompted"]))
        trade_response_counts.append(float(res["trade_response_count"]))
        setup_second_pick_adds_missing_flags.extend(float(x) for x in res["setup_second_pick_adds_missing_flags"])
        setup_second_pick_overlap_only_flags.extend(float(x) for x in res["setup_second_pick_overlap_only_flags"])

    return {
        "games": int(num_games),
        "terminal_games": int(num_games - truncated_games),
        "truncated_games": int(truncated_games),
        "wins": [float(x) for x in wins.tolist()],
        "win_rates": (wins / max(1, num_games)).tolist(),
        "strict_wins": [float(x) for x in strict_wins.tolist()],
        "strict_win_rates": (strict_wins / max(1, num_games)).tolist(),
        "avg_turns": float(np.mean(turns)) if turns else 0.0,
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "opening_mean_percentile": float(np.mean(opening_percentiles)) if opening_percentiles else 0.0,
        "opening_mean_rank": float(np.mean(opening_ranks)) if opening_ranks else 0.0,
        "road_useful_rate": float(np.mean(road_useful_flags)) if road_useful_flags else 0.0,
        "road_mean_connected_gain": float(np.mean(road_connected_gains)) if road_connected_gains else 0.0,
        "knight_play_rate_when_available": float(np.mean(knight_play_rates)) if knight_play_rates else 0.0,
        "knight_mean_held_turns_before_first_play": float(np.mean(knight_hold_before_first))
        if knight_hold_before_first
        else 0.0,
        "yop_play_rate_when_available": float(np.mean(yop_play_rates)) if yop_play_rates else 0.0,
        "monopoly_play_rate_when_available": float(np.mean(monopoly_play_rates)) if monopoly_play_rates else 0.0,
        "trade_offer_rate_when_main": float(np.mean(trade_offer_rates)) if trade_offer_rates else 0.0,
        "trade_bank_rate_when_main": float(np.mean(trade_bank_rates)) if trade_bank_rates else 0.0,
        "trade_accept_rate_when_prompted": float(np.mean(trade_accept_rates)) if trade_accept_rates else 0.0,
        "trade_reject_rate_when_prompted": float(np.mean(trade_reject_rates)) if trade_reject_rates else 0.0,
        "trade_response_count": float(np.mean(trade_response_counts)) if trade_response_counts else 0.0,
        "setup_second_pick_adds_missing_rate": float(np.mean(setup_second_pick_adds_missing_flags))
        if setup_second_pick_adds_missing_flags
        else 0.0,
        "setup_second_pick_overlap_only_rate": float(np.mean(setup_second_pick_overlap_only_flags))
        if setup_second_pick_overlap_only_flags
        else 0.0,
        "setup_second_pick_count": int(len(setup_second_pick_adds_missing_flags)),
    }


def _load_json_or_jsonl(path: Path):
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def _default_history_path(checkpoint: Path) -> Path:
    return checkpoint.parent.parent / "progress_reports.jsonl"


def _default_promotion_path(checkpoint: Path) -> Path:
    return checkpoint.parent.parent / "best_checkpoint_meta.json"


def _checkpoint_model_metadata(checkpoint: Path) -> dict:
    meta_path = Path(str(checkpoint) + ".meta.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--training-history", required=False, type=str, default=None)
    parser.add_argument("--promotion-decision", required=False, type=str, default=None)
    parser.add_argument("--games-per-seat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opponent", choices=["random", "heuristic", "both", "checkpoint"], default="random")
    parser.add_argument("--opponent-checkpoint", required=False, type=str, default=None)
    parser.add_argument(
        "--opponent-model-arch",
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
    parser.add_argument("--opponent-model-hidden", type=int, default=256)
    parser.add_argument("--opponent-model-residual-blocks", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--max-main-actions-per-turn", type=int, default=10)
    parser.add_argument("--disable-player-trade", action="store_true")
    parser.add_argument("--trade-action-mode", choices=["guided", "full"], default="guided")
    parser.add_argument("--max-player-trade-proposals-per-turn", type=int, default=None)
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
    parser.add_argument("--model-hidden", type=int, default=256)
    parser.add_argument("--model-residual-blocks", type=int, default=4)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    history_path = Path(args.training_history) if args.training_history else _default_history_path(ckpt_path)
    promotion_path = Path(args.promotion_decision) if args.promotion_decision else _default_promotion_path(ckpt_path)
    history = _load_json_or_jsonl(history_path)
    promotion = _load_json_or_jsonl(promotion_path)
    checkpoint_meta = _checkpoint_model_metadata(ckpt_path)

    policy_agent = _load_policy_agent(
        ckpt_path,
        model_arch=args.model_arch,
        model_hidden=args.model_hidden,
        model_residual_blocks=args.model_residual_blocks,
    )
    opponent_policy_agent: PolicyAgent | None = None
    if args.opponent == "checkpoint":
        if not args.opponent_checkpoint:
            raise ValueError("--opponent-checkpoint is required when --opponent checkpoint")
        opponent_policy_agent = _load_policy_agent(
            Path(args.opponent_checkpoint),
            model_arch=args.opponent_model_arch,
            model_hidden=args.opponent_model_hidden,
            model_residual_blocks=args.opponent_model_residual_blocks,
        )
    env_kwargs = {
        "max_main_actions_per_turn": args.max_main_actions_per_turn,
        "allow_player_trade": not args.disable_player_trade,
        "trade_action_mode": args.trade_action_mode,
        "max_player_trade_proposals_per_turn": args.max_player_trade_proposals_per_turn,
    }
    if args.opponent == "both":
        seat_eval = {
            "random": _eval_across_seats(
                policy_agent,
                args.games_per_seat,
                args.seed,
                "random",
                max_steps=args.max_steps,
                env_kwargs=env_kwargs,
            ),
            "heuristic": _eval_across_seats(
                policy_agent,
                args.games_per_seat,
                args.seed + 1000000,
                "heuristic",
                max_steps=args.max_steps,
                env_kwargs=env_kwargs,
            ),
        }
        strict_score = 0.5 * (
            seat_eval["random"]["mean_strict_win_rate"] + seat_eval["heuristic"]["mean_strict_win_rate"]
        )
        worst_seat_strict = 0.5 * (
            seat_eval["random"]["worst_seat_strict"] + seat_eval["heuristic"]["worst_seat_strict"]
        )
        seat_strict_stddev = 0.5 * (
            seat_eval["random"]["seat_strict_stddev"] + seat_eval["heuristic"]["seat_strict_stddev"]
        )
    else:
        seat_eval = _eval_across_seats(
            policy_agent,
            args.games_per_seat,
            args.seed,
            args.opponent,
            max_steps=args.max_steps,
            env_kwargs=env_kwargs,
            opponent_policy_agent=opponent_policy_agent,
        )
        strict_score = seat_eval["mean_strict_win_rate"]
        worst_seat_strict = seat_eval["worst_seat_strict"]
        seat_strict_stddev = seat_eval["seat_strict_stddev"]

    entropy_values = [h.get("entropy") for h in history if "entropy" in h]
    entropy_trend = {
        "first": float(entropy_values[0]) if entropy_values else 0.0,
        "last": float(entropy_values[-1]) if entropy_values else 0.0,
        "min": float(min(entropy_values)) if entropy_values else 0.0,
        "max": float(max(entropy_values)) if entropy_values else 0.0,
    }

    report = {
        "checkpoint": str(ckpt_path),
        "training_history_path": str(history_path),
        "training_updates": len(history),
        "promotion_decision": promotion if isinstance(promotion, dict) else {},
        "checkpoint_model_metadata": checkpoint_meta,
        "strict_score_eval": strict_score,
        "worst_seat_strict_eval": worst_seat_strict,
        "seat_strict_stddev_eval": seat_strict_stddev,
        "games_per_seat": args.games_per_seat,
        "max_steps": args.max_steps,
        "env_kwargs": env_kwargs,
        "opponent": args.opponent,
        "opponent_checkpoint": args.opponent_checkpoint,
        "seat_permutation_eval": seat_eval,
        "entropy_trend": entropy_trend,
        "invalid_action_rate": 0.0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
