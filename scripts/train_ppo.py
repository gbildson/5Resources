"""Run PPO self-play training and optional league promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catan_rl.training.league import LeagueManager
from catan_rl.training.ppo import PPOConfig
from catan_rl.training.self_play import TrainConfig, train_self_play
from catan_rl.training.wrappers import PolicyAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="artifacts")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        total_updates=args.updates,
        rollout_steps=args.rollout_steps,
        eval_every=args.eval_every,
        seed=args.seed,
    )
    model, history = train_self_play(train_cfg, PPOConfig())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = out_dir / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    league = LeagueManager(root=out_dir / "league")
    ckpt = league.add_candidate(model, name=f"policy_u{args.updates}")
    decision = league.evaluate_candidate(PolicyAgent(model), seed=args.seed + 1000)
    (out_dir / "promotion_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"checkpoint: {ckpt}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
