"""Offline search distillation pipeline for setup and robber decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from catan_rl.env import CatanEnv
from catan_rl.training.bc import (
    collect_search_distill_dataset,
    pretrain_policy_with_search_distill,
    save_search_distill_dataset,
)
from catan_rl.training.ppo import build_policy_value_net, load_policy_value_net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-checkpoint", type=str, default=None)
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
        default="mlp",
    )
    parser.add_argument("--model-hidden", type=int, default=256)
    parser.add_argument("--model-residual-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-steps", type=int, default=20000)
    parser.add_argument("--distill-rounds", type=int, default=1)
    parser.add_argument("--refresh-dataset-every", type=int, default=1)
    parser.add_argument("--distill-epochs", type=int, default=2)
    parser.add_argument("--distill-batch-size", type=int, default=256)
    parser.add_argument("--distill-lr", type=float, default=1e-3)
    parser.add_argument("--bc-aux-coef", type=float, default=0.0)
    parser.add_argument("--bc-strategy-coef", type=float, default=0.0)
    parser.add_argument("--max-main-actions-per-turn", type=int, default=12)
    parser.add_argument("--disable-player-trade", action="store_true")
    parser.add_argument("--trade-action-mode", choices=["guided", "full"], default="guided")
    parser.add_argument("--max-player-trade-proposals-per-turn", type=int, default=None)
    parser.add_argument("--disable-setup-search", action="store_true")
    parser.add_argument("--disable-robber-search", action="store_true")
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--setup-search-rollout-steps", type=int, default=0)
    parser.add_argument("--robber-search-rollout-steps", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="artifacts_search_distill")
    parser.add_argument("--out-model", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = {
        "max_main_actions_per_turn": args.max_main_actions_per_turn,
        "allow_player_trade": not args.disable_player_trade,
        "trade_action_mode": args.trade_action_mode,
        "max_player_trade_proposals_per_turn": args.max_player_trade_proposals_per_turn,
    }

    env = CatanEnv(seed=args.seed, **env_kwargs)
    obs, _ = env.reset(seed=args.seed)
    obs_dim = int(obs.shape[0])
    action_dim = int(env.action_mask().shape[0])
    model = (
        load_policy_value_net(
            args.init_checkpoint,
            obs_dim=obs_dim,
            action_dim=action_dim,
            model_arch=args.model_arch,
            hidden=args.model_hidden,
            residual_blocks=args.model_residual_blocks,
        )
        if args.init_checkpoint
        else build_policy_value_net(
            obs_dim=obs_dim,
            action_dim=action_dim,
            model_arch=args.model_arch,
            hidden=args.model_hidden,
            residual_blocks=args.model_residual_blocks,
        )
    )

    logs = []
    dataset = None
    for r in range(1, args.distill_rounds + 1):
        if dataset is None or ((r - 1) % max(1, int(args.refresh_dataset_every)) == 0):
            dataset = collect_search_distill_dataset(
                steps=args.dataset_steps,
                seed=args.seed + r * 1000,
                env_kwargs=env_kwargs,
                teacher_model=model,
                setup_search=not args.disable_setup_search,
                robber_search=not args.disable_robber_search,
                top_k=args.search_top_k,
                setup_rollout_steps=args.setup_search_rollout_steps,
                robber_rollout_steps=args.robber_search_rollout_steps,
                export_decisions_jsonl=str(out_dir / f"search_decisions_r{r}.jsonl"),
            )
            save_search_distill_dataset(out_dir / f"search_dataset_r{r}.npz", dataset)
        stats = pretrain_policy_with_search_distill(
            model=model,
            dataset=dataset,
            epochs=args.distill_epochs,
            batch_size=args.distill_batch_size,
            lr=args.distill_lr,
            aux_coef=args.bc_aux_coef,
            strategy_coef=args.bc_strategy_coef,
        )
        row = {"round": int(r), **stats, **dataset.metadata}
        logs.append(row)
        print(json.dumps(row))

    out_model = Path(args.out_model) if args.out_model else (out_dir / "search_distilled_model.pt")
    torch.save(model.state_dict(), out_model)
    (out_dir / "distill_logs.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(f"out_model: {out_model}")
    print(f"distill_logs: {out_dir / 'distill_logs.json'}")


if __name__ == "__main__":
    main()

