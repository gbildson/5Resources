"""Benchmark a candidate checkpoint against a fixed historical checkpoint pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_rl.bots import Agent
from catan_rl.eval import tournament
from catan_rl.training.ppo import load_policy_value_net
from catan_rl.training.wrappers import PolicyAgent
from catan_rl.env import CatanEnv


def _load_policy_agent(
    checkpoint: str,
    *,
    model_arch: str | None,
    model_hidden: int,
    model_residual_blocks: int,
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


def _eval_vs_opponent(
    candidate: PolicyAgent,
    opponent: Agent,
    *,
    games_per_seat: int,
    seed: int,
    max_steps: int,
    env_kwargs: dict,
) -> dict:
    seat_stats = []
    for seat in range(4):
        agents = [opponent, opponent, opponent, opponent]
        agents[seat] = candidate
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
                "win_rate": float(stats["win_rates"][seat]),
                "strict_win_rate": float(stats["strict_win_rates"][seat]),
            }
        )
    return {
        "seat_stats": seat_stats,
        "mean_win_rate": float(np.mean([x["win_rate"] for x in seat_stats])),
        "mean_strict_win_rate": float(np.mean([x["strict_win_rate"] for x in seat_stats])),
        "worst_seat_strict": float(np.min([x["strict_win_rate"] for x in seat_stats])),
        "seat_strict_stddev": float(np.std([x["strict_win_rate"] for x in seat_stats])),
    }


def _load_pool(manifest: str | None, checkpoints_csv: str | None) -> list[dict]:
    rows: list[dict] = []
    if manifest:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Pool manifest must be a JSON list.")
        for item in data:
            if not isinstance(item, dict):
                continue
            if item.get("active", True) is False:
                continue
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            rows.append(
                {
                    "path": path,
                    "name": str(item.get("name", Path(path).stem)),
                    "tags": [str(t) for t in item.get("tags", [])],
                    "weight": float(item.get("weight", 1.0)),
                }
            )
    if checkpoints_csv:
        for i, p in enumerate([x.strip() for x in checkpoints_csv.split(",") if x.strip()]):
            rows.append({"path": p, "name": f"ckpt_{i+1}", "tags": [], "weight": 1.0})
    if not rows:
        raise ValueError("Provide --pool-manifest or --pool-checkpoints.")
    for r in rows:
        if not Path(r["path"]).exists():
            raise FileNotFoundError(f"Pool checkpoint not found: {r['path']}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--pool-manifest", type=str, default=None)
    parser.add_argument("--pool-checkpoints", type=str, default=None)
    parser.add_argument("--games-per-seat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
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

    pool = _load_pool(args.pool_manifest, args.pool_checkpoints)
    candidate = _load_policy_agent(
        args.checkpoint,
        model_arch=args.model_arch,
        model_hidden=args.model_hidden,
        model_residual_blocks=args.model_residual_blocks,
    )
    env_kwargs = {
        "max_main_actions_per_turn": args.max_main_actions_per_turn,
        "allow_player_trade": not args.disable_player_trade,
        "trade_action_mode": args.trade_action_mode,
        "max_player_trade_proposals_per_turn": args.max_player_trade_proposals_per_turn,
    }

    rows = []
    for i, item in enumerate(pool):
        opp = _load_policy_agent(
            item["path"],
            model_arch=None,
            model_hidden=args.model_hidden,
            model_residual_blocks=args.model_residual_blocks,
        )
        stats = _eval_vs_opponent(
            candidate,
            opp,
            games_per_seat=args.games_per_seat,
            seed=args.seed + 100000 * i,
            max_steps=args.max_steps,
            env_kwargs=env_kwargs,
        )
        rows.append(
            {
                "opponent_name": item["name"],
                "opponent_path": item["path"],
                "tags": item["tags"],
                "weight": float(item["weight"]),
                **stats,
            }
        )

    rows.sort(key=lambda r: (r["mean_strict_win_rate"], r["worst_seat_strict"]), reverse=True)
    weights = np.asarray([max(1e-6, float(r["weight"])) for r in rows], dtype=np.float64)
    weights = weights / weights.sum()
    weighted_strict = float(np.sum(weights * np.asarray([float(r["mean_strict_win_rate"]) for r in rows])))
    weighted_worst = float(np.sum(weights * np.asarray([float(r["worst_seat_strict"]) for r in rows])))
    report = {
        "candidate_checkpoint": args.checkpoint,
        "games_per_seat": int(args.games_per_seat),
        "max_steps": int(args.max_steps),
        "env_kwargs": env_kwargs,
        "opponent_count": int(len(rows)),
        "weighted_mean_strict_win_rate": weighted_strict,
        "weighted_mean_worst_seat_strict": weighted_worst,
        "scoreboard": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

