"""Run Stage 8 benchmark suite (historical + board-style) in one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_py(script: str, args: list[str]) -> None:
    cmd = [sys.executable, script, *args]
    subprocess.run(cmd, check=True)


def _float(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--out-dir", type=str, default="artifacts_bench")
    parser.add_argument("--historical-manifest", type=str, default="benchmark_checkpoint_pool.example.json")
    parser.add_argument("--historical-checkpoints", type=str, default=None)
    parser.add_argument("--historical-games-per-seat", type=int, default=8)
    parser.add_argument("--board-opponent", choices=["heuristic", "random", "checkpoint"], default="heuristic")
    parser.add_argument("--board-opponent-checkpoint", type=str, default=None)
    parser.add_argument("--board-seed-start", type=int, default=1000)
    parser.add_argument("--board-scan-count", type=int, default=1500)
    parser.add_argument("--board-seeds-per-group", type=int, default=24)
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    historical_out = out_dir / "historical.json"
    board_out = out_dir / "board_styles.json"
    summary_out = out_dir / "benchmark_suite_summary.json"

    base_eval_flags = [
        "--checkpoint",
        args.checkpoint,
        "--max-steps",
        str(args.max_steps),
        "--max-main-actions-per-turn",
        str(args.max_main_actions_per_turn),
        "--trade-action-mode",
        args.trade_action_mode,
        "--model-hidden",
        str(args.model_hidden),
        "--model-residual-blocks",
        str(args.model_residual_blocks),
    ]
    if args.model_arch is not None:
        base_eval_flags.extend(["--model-arch", args.model_arch])
    if args.disable_player_trade:
        base_eval_flags.append("--disable-player-trade")
    if args.max_player_trade_proposals_per_turn is not None:
        base_eval_flags.extend(
            ["--max-player-trade-proposals-per-turn", str(args.max_player_trade_proposals_per_turn)]
        )

    hist_args = [
        *base_eval_flags,
        "--seed",
        str(args.seed),
        "--games-per-seat",
        str(args.historical_games_per_seat),
        "--out",
        str(historical_out),
    ]
    if args.historical_manifest:
        hist_args.extend(["--pool-manifest", args.historical_manifest])
    if args.historical_checkpoints:
        hist_args.extend(["--pool-checkpoints", args.historical_checkpoints])
    _run_py("scripts/benchmark_historical.py", hist_args)

    board_args = [
        *base_eval_flags,
        "--seed",
        str(args.seed + 10000),
        "--opponent",
        args.board_opponent,
        "--seed-start",
        str(args.board_seed_start),
        "--scan-count",
        str(args.board_scan_count),
        "--seeds-per-group",
        str(args.board_seeds_per_group),
        "--out",
        str(board_out),
    ]
    if args.board_opponent_checkpoint:
        board_args.extend(["--opponent-checkpoint", args.board_opponent_checkpoint])
    _run_py("scripts/benchmark_board_styles.py", board_args)

    historical = json.loads(historical_out.read_text(encoding="utf-8"))
    board = json.loads(board_out.read_text(encoding="utf-8"))
    summary = {
        "candidate_checkpoint": args.checkpoint,
        "historical_report": str(historical_out),
        "board_styles_report": str(board_out),
        "historical_weighted_mean_strict_win_rate": _float(historical, "weighted_mean_strict_win_rate"),
        "historical_weighted_mean_worst_seat_strict": _float(historical, "weighted_mean_worst_seat_strict"),
        "board_group_mean_strict_win_rate": _float(board, "group_mean_strict_win_rate"),
        "board_worst_group_strict_win_rate": float(
            min((_float(v, "mean_strict_win_rate") for v in board.get("groups", {}).values()), default=0.0)
        ),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

