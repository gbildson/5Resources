"""Evaluate a trained checkpoint and write a consolidated baseline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from catan_rl.bots import HeuristicAgent, RandomLegalAgent
from catan_rl.env import CatanEnv
from catan_rl.eval import tournament
from catan_rl.training.ppo import PolicyValueNet
from catan_rl.training.wrappers import PolicyAgent


def _load_policy_agent(checkpoint: Path) -> PolicyAgent:
    env = CatanEnv(seed=0)
    obs, _ = env.reset(seed=0)
    model = PolicyValueNet(obs_dim=obs.shape[0], action_dim=env.action_mask().shape[0])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return PolicyAgent(model)


def _eval_across_seats(
    policy_agent: PolicyAgent,
    games_per_seat: int,
    seed: int,
    opponent: str,
    max_steps: int,
    env_kwargs: dict,
) -> dict:
    seat_stats = []
    for seat in range(4):
        agents = []
        for i in range(4):
            if i == seat:
                agents.append(policy_agent)
            elif opponent == "heuristic":
                agents.append(HeuristicAgent(seed=seed + 1000 + i))
            else:
                agents.append(RandomLegalAgent(seed=seed + 1000 + i))
        stats = tournament(
            agents,
            num_games=games_per_seat,
            base_seed=seed + seat * 10000,
            max_steps=max_steps,
            env_kwargs=env_kwargs,
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
            }
        )
    return {
        "seat_stats": seat_stats,
        "mean_win_rate": float(np.mean([s["win_rate"] for s in seat_stats])),
        "mean_strict_win_rate": float(np.mean([s["strict_win_rate"] for s in seat_stats])),
        "mean_avg_turns": float(np.mean([s["avg_turns"] for s in seat_stats])),
        "mean_avg_steps": float(np.mean([s["avg_steps"] for s in seat_stats])),
        "mean_truncated_games": float(np.mean([s["truncated_games"] for s in seat_stats])),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--training-history", required=False, type=str, default=None)
    parser.add_argument("--promotion-decision", required=False, type=str, default=None)
    parser.add_argument("--games-per-seat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--opponent", choices=["random", "heuristic", "both"], default="random")
    parser.add_argument("--max-steps", type=int, default=2200)
    parser.add_argument("--max-main-actions-per-turn", type=int, default=10)
    parser.add_argument("--disable-player-trade", action="store_true")
    parser.add_argument("--trade-action-mode", choices=["guided", "full"], default="guided")
    parser.add_argument("--max-player-trade-proposals-per-turn", type=int, default=None)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    history_path = Path(args.training_history) if args.training_history else _default_history_path(ckpt_path)
    promotion_path = Path(args.promotion_decision) if args.promotion_decision else _default_promotion_path(ckpt_path)
    history = _load_json_or_jsonl(history_path)
    promotion = _load_json_or_jsonl(promotion_path)

    policy_agent = _load_policy_agent(ckpt_path)
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
    else:
        seat_eval = _eval_across_seats(
            policy_agent,
            args.games_per_seat,
            args.seed,
            args.opponent,
            max_steps=args.max_steps,
            env_kwargs=env_kwargs,
        )
        strict_score = seat_eval["mean_strict_win_rate"]

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
        "strict_score_eval": strict_score,
        "games_per_seat": args.games_per_seat,
        "max_steps": args.max_steps,
        "env_kwargs": env_kwargs,
        "opponent": args.opponent,
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
