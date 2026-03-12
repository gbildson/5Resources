"""Run longer PPO training with periodic evaluation reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from catan_rl.bots import HeuristicAgent, MetaStrategyHeuristicAgent, RandomLegalAgent
from catan_rl.constants import NUM_PLAYERS
from catan_rl.env import CatanEnv
from catan_rl.eval import tournament
from catan_rl.training.bc import collect_bc_dataset, pretrain_policy_with_bc
from catan_rl.training.ppo import PPOConfig, PPOTrainer, PolicyValueNet, compute_gae
from catan_rl.training.self_play import collect_rollout, save_checkpoint
from catan_rl.training.wrappers import PolicyAgent


def _seat_eval(
    policy_agent: PolicyAgent,
    opponent_type: str,
    games_per_seat: int,
    seed: int,
    max_steps: int,
    env_kwargs: dict,
) -> dict:
    seat_stats = []
    for seat in range(4):
        agents = []
        for i in range(4):
            if i == seat:
                agents.append(policy_agent)
            elif opponent_type == "heuristic":
                agents.append(HeuristicAgent(seed=seed + 100 + i))
            else:
                agents.append(RandomLegalAgent(seed=seed + 100 + i))
        stats = tournament(
            agents,
            num_games=games_per_seat,
            base_seed=seed + seat * 1000,
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
        "opponent": opponent_type,
        "seat_stats": seat_stats,
        "mean_win_rate": float(np.mean([s["win_rate"] for s in seat_stats])),
        "mean_strict_win_rate": float(np.mean([s["strict_win_rate"] for s in seat_stats])),
        "mean_avg_turns": float(np.mean([s["avg_turns"] for s in seat_stats])),
        "mean_avg_steps": float(np.mean([s["avg_steps"] for s in seat_stats])),
        "mean_truncated_games": float(np.mean([s["truncated_games"] for s in seat_stats])),
    }


def _build_frozen_agents(
    checkpoints_csv: str,
    obs_dim: int,
    action_dim: int,
) -> list[PolicyAgent]:
    checkpoints = [x.strip() for x in checkpoints_csv.split(",") if x.strip()]
    agents: list[PolicyAgent] = []
    for ckpt in checkpoints:
        model = PolicyValueNet(obs_dim=obs_dim, action_dim=action_dim)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        agents.append(PolicyAgent(model))
    return agents


def _sample_opponent_assignments(
    rng: np.random.Generator,
    opponent_seat_count: int,
    heuristic_prob: float,
    random_prob: float,
    meta_prob: float,
    frozen_prob: float,
    frozen_agents: list[PolicyAgent],
) -> tuple[set[int], dict[int, Any]]:
    seat_count = min(max(1, opponent_seat_count), NUM_PLAYERS - 1)
    opponent_seats = set(int(x) for x in rng.choice(np.arange(NUM_PLAYERS), size=seat_count, replace=False))
    assigns: dict[int, Any] = {}
    probs = np.asarray([heuristic_prob, random_prob, meta_prob, frozen_prob], dtype=np.float64)
    if probs.sum() <= 0:
        probs = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    probs = probs / probs.sum()
    for seat in opponent_seats:
        kind = int(rng.choice(np.arange(4), p=probs))
        if kind == 0:
            assigns[seat] = HeuristicAgent(seed=int(rng.integers(0, 1_000_000)))
        elif kind == 1:
            assigns[seat] = RandomLegalAgent(seed=int(rng.integers(0, 1_000_000)))
        elif kind == 2:
            assigns[seat] = MetaStrategyHeuristicAgent(seed=int(rng.integers(0, 1_000_000)))
        else:
            if frozen_agents:
                assigns[seat] = frozen_agents[int(rng.integers(0, len(frozen_agents)))]
            else:
                assigns[seat] = HeuristicAgent(seed=int(rng.integers(0, 1_000_000)))
    return opponent_seats, assigns


def _truncation_reward_for_learner_seats(env: CatanEnv, learner_seats: set[int]) -> float:
    top_vp = int(env.state.actual_vp.max(initial=0))
    leaders = set(int(p) for p in np.flatnonzero(env.state.actual_vp == top_vp))
    learner_leads = len(leaders & learner_seats)
    opponent_leads = len(leaders - learner_seats)
    if learner_leads > 0 and opponent_leads == 0:
        return 1.0
    if opponent_leads > 0 and learner_leads == 0:
        return -1.0
    return 0.0


def _collect_rollout_opponent_mix(
    env: CatanEnv,
    trainer: PPOTrainer,
    steps: int,
    rng: np.random.Generator,
    max_episode_steps: int,
    truncation_leader_reward: bool,
    reward_shaping_vp: float,
    reward_shaping_resource: float,
    opponent_seat_count: int,
    heuristic_prob: float,
    random_prob: float,
    meta_prob: float,
    frozen_prob: float,
    frozen_agents: list[PolicyAgent],
) -> dict:
    obs, info = env.reset()
    mask = info["action_mask"]
    buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones", "masks")}
    episode_steps = 0
    collected = 0
    opponent_seats, opponents = _sample_opponent_assignments(
        rng,
        opponent_seat_count=opponent_seat_count,
        heuristic_prob=heuristic_prob,
        random_prob=random_prob,
        meta_prob=meta_prob,
        frozen_prob=frozen_prob,
        frozen_agents=frozen_agents,
    )
    learner_seats = set(range(NUM_PLAYERS)) - opponent_seats
    last_learner_idx: int | None = None

    while collected < steps:
        player = int(env.state.current_player)
        if player in learner_seats:
            prev_public_vp = int(env.state.public_vp[player])
            prev_resource_total = int(env.state.resource_total[player])
            action, logp, value = trainer.act(obs, mask)
            res = env.step(action)
            reward = float(res.reward)
            delta_vp = int(env.state.public_vp[player]) - prev_public_vp
            delta_resources = int(env.state.resource_total[player]) - prev_resource_total
            reward += reward_shaping_vp * float(delta_vp)
            reward += reward_shaping_resource * float(delta_resources)

            done = bool(res.done)
            episode_steps += 1
            if not done and episode_steps >= max_episode_steps:
                done = True
                if truncation_leader_reward:
                    reward += _truncation_reward_for_learner_seats(env, learner_seats)

            buf["obs"].append(obs)
            buf["actions"].append(action)
            buf["logp"].append(logp)
            buf["values"].append(value)
            buf["rewards"].append(reward)
            buf["dones"].append(float(done))
            buf["masks"].append(mask)
            last_learner_idx = len(buf["dones"]) - 1
            collected += 1
            if done:
                obs, info = env.reset()
                mask = info["action_mask"]
                episode_steps = 0
                opponent_seats, opponents = _sample_opponent_assignments(
                    rng,
                    opponent_seat_count=opponent_seat_count,
                    heuristic_prob=heuristic_prob,
                    random_prob=random_prob,
                    meta_prob=meta_prob,
                    frozen_prob=frozen_prob,
                    frozen_agents=frozen_agents,
                )
                learner_seats = set(range(NUM_PLAYERS)) - opponent_seats
                last_learner_idx = None
            else:
                obs = res.obs
                mask = res.info["action_mask"]
        else:
            opponent = opponents[player]
            action = int(opponent.act(obs, mask))
            res = env.step(action)
            done = bool(res.done)
            episode_steps += 1
            truncated = False
            if not done and episode_steps >= max_episode_steps:
                done = True
                truncated = True

            if done and last_learner_idx is not None:
                if buf["dones"][last_learner_idx] == 0.0:
                    buf["dones"][last_learner_idx] = 1.0
                if env.state.winner >= 0:
                    if int(env.state.winner) in learner_seats:
                        buf["rewards"][last_learner_idx] += 1.0
                    else:
                        buf["rewards"][last_learner_idx] -= 1.0
                elif truncated and truncation_leader_reward:
                    buf["rewards"][last_learner_idx] += _truncation_reward_for_learner_seats(env, learner_seats)

            if done:
                obs, info = env.reset()
                mask = info["action_mask"]
                episode_steps = 0
                opponent_seats, opponents = _sample_opponent_assignments(
                    rng,
                    opponent_seat_count=opponent_seat_count,
                    heuristic_prob=heuristic_prob,
                    random_prob=random_prob,
                    meta_prob=meta_prob,
                    frozen_prob=frozen_prob,
                    frozen_agents=frozen_agents,
                )
                learner_seats = set(range(NUM_PLAYERS)) - opponent_seats
                last_learner_idx = None
            else:
                obs = res.obs
                mask = res.info["action_mask"]

    for k in ("obs", "masks"):
        buf[k] = np.asarray(buf[k], dtype=np.float32)
    for k in ("actions",):
        buf[k] = np.asarray(buf[k], dtype=np.int64)
    for k in ("logp", "values", "rewards", "dones"):
        buf[k] = np.asarray(buf[k], dtype=np.float32)
    adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], trainer.cfg.gamma, trainer.cfg.lam)
    buf["advantages"] = adv
    buf["returns"] = ret
    return buf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--games-per-seat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=str, default="artifacts_schedule")
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--bc-warmstart", action="store_true")
    parser.add_argument("--bc-steps", type=int, default=20000)
    parser.add_argument("--bc-epochs", type=int, default=3)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--eval-max-steps", type=int, default=2000)
    parser.add_argument("--disable-truncation-leader-reward", action="store_true")
    parser.add_argument("--reward-shaping-vp", type=float, default=0.01)
    parser.add_argument("--reward-shaping-resource", type=float, default=0.001)
    parser.add_argument("--max-main-actions-per-turn", type=int, default=12)
    parser.add_argument("--disable-player-trade", action="store_true")
    parser.add_argument("--trade-action-mode", choices=["guided", "full"], default="guided")
    parser.add_argument("--max-player-trade-proposals-per-turn", type=int, default=None)
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-ent-coef-start", type=float, default=None)
    parser.add_argument("--ppo-ent-coef-end", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=64)
    parser.add_argument("--collapse-entropy-threshold", type=float, default=0.25)
    parser.add_argument("--collapse-strict-drop", type=float, default=0.05)
    parser.add_argument("--use-opponent-mixture", action="store_true")
    parser.add_argument("--opponent-seat-count", type=int, default=1)
    parser.add_argument("--mix-heuristic-prob", type=float, default=0.5)
    parser.add_argument("--mix-random-prob", type=float, default=0.2)
    parser.add_argument("--mix-meta-prob", type=float, default=0.0)
    parser.add_argument("--mix-frozen-prob", type=float, default=0.3)
    parser.add_argument("--mix-frozen-checkpoints", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    report_dir = out_dir / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "max_main_actions_per_turn": args.max_main_actions_per_turn,
        "allow_player_trade": not args.disable_player_trade,
        "trade_action_mode": args.trade_action_mode,
        "max_player_trade_proposals_per_turn": args.max_player_trade_proposals_per_turn,
    }

    env = CatanEnv(seed=args.seed, **env_kwargs)
    obs, _ = env.reset(seed=args.seed)
    model = PolicyValueNet(obs_dim=obs.shape[0], action_dim=env.action_mask().shape[0])
    rng = np.random.default_rng(args.seed)

    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location="cpu"))
        print(json.dumps({"loaded_checkpoint": args.init_checkpoint}))

    bc_stats = None
    if args.bc_warmstart and args.init_checkpoint:
        raise ValueError("Use either --bc-warmstart or --init-checkpoint, not both.")
    if args.bc_warmstart:
        dataset = collect_bc_dataset(steps=args.bc_steps, seed=args.seed + 999, env_kwargs=env_kwargs)
        bc_stats = pretrain_policy_with_bc(
            model=model,
            dataset=dataset,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
        )
        print(json.dumps({"bc_warmstart": True, **bc_stats}))

    trainer = PPOTrainer(
        model,
        PPOConfig(
            lr=args.ppo_lr,
            ent_coef=args.ppo_ent_coef,
            epochs=args.ppo_epochs,
            minibatch_size=args.ppo_minibatch_size,
        ),
    )

    report_jsonl = out_dir / "progress_reports.jsonl"
    report_csv = out_dir / "progress_reports.csv"
    csv_header_written = report_csv.exists() and report_csv.stat().st_size > 0
    best_checkpoint_path = out_dir / "best_checkpoint.pt"
    best_meta_path = out_dir / "best_checkpoint_meta.json"
    best_strict_score = -1.0
    best_update = -1
    frozen_agents: list[PolicyAgent] = []
    if args.use_opponent_mixture:
        frozen_agents = _build_frozen_agents(
            args.mix_frozen_checkpoints,
            obs_dim=obs.shape[0],
            action_dim=env.action_mask().shape[0],
        )

    for update in range(1, args.total_updates + 1):
        if args.ppo_ent_coef_start is not None and args.ppo_ent_coef_end is not None:
            progress = (update - 1) / max(1, args.total_updates - 1)
            trainer.cfg.ent_coef = float(
                args.ppo_ent_coef_start + progress * (args.ppo_ent_coef_end - args.ppo_ent_coef_start)
            )

        env.seed = int(rng.integers(0, 1_000_000))
        if args.use_opponent_mixture:
            batch = _collect_rollout_opponent_mix(
                env,
                trainer,
                args.rollout_steps,
                rng=rng,
                max_episode_steps=args.max_episode_steps,
                truncation_leader_reward=not args.disable_truncation_leader_reward,
                reward_shaping_vp=args.reward_shaping_vp,
                reward_shaping_resource=args.reward_shaping_resource,
                opponent_seat_count=args.opponent_seat_count,
                heuristic_prob=args.mix_heuristic_prob,
                random_prob=args.mix_random_prob,
                meta_prob=args.mix_meta_prob,
                frozen_prob=args.mix_frozen_prob,
                frozen_agents=frozen_agents,
            )
        else:
            batch = collect_rollout(
                env,
                trainer,
                args.rollout_steps,
                max_episode_steps=args.max_episode_steps,
                truncation_leader_reward=not args.disable_truncation_leader_reward,
                reward_shaping_vp=args.reward_shaping_vp,
                reward_shaping_resource=args.reward_shaping_resource,
            )
        train_stats = trainer.update(batch)

        if update % args.report_every != 0 and update != args.total_updates:
            continue

        ckpt_path = ckpt_dir / f"policy_u{update}.pt"
        save_checkpoint(model, str(ckpt_path))
        policy_agent = PolicyAgent(model)
        random_eval = _seat_eval(
            policy_agent,
            "random",
            args.games_per_seat,
            args.seed + update * 10,
            max_steps=args.eval_max_steps,
            env_kwargs=env_kwargs,
        )
        heuristic_eval = _seat_eval(
            policy_agent,
            "heuristic",
            max(5, args.games_per_seat // 2),
            args.seed + update * 20,
            max_steps=args.eval_max_steps,
            env_kwargs=env_kwargs,
        )

        report = {
            "update": update,
            "checkpoint": str(ckpt_path),
            "training": train_stats,
            "entropy": float(train_stats.get("entropy", 0.0)),
            "invalid_action_rate": 0.0,
            "max_episode_steps": args.max_episode_steps,
            "eval_max_steps": args.eval_max_steps,
            "truncation_leader_reward": not args.disable_truncation_leader_reward,
            "reward_shaping_vp": args.reward_shaping_vp,
            "reward_shaping_resource": args.reward_shaping_resource,
            "max_main_actions_per_turn": args.max_main_actions_per_turn,
            "allow_player_trade": not args.disable_player_trade,
            "trade_action_mode": args.trade_action_mode,
            "max_player_trade_proposals_per_turn": args.max_player_trade_proposals_per_turn,
            "use_opponent_mixture": bool(args.use_opponent_mixture),
            "opponent_seat_count": args.opponent_seat_count,
            "mix_heuristic_prob": args.mix_heuristic_prob,
            "mix_random_prob": args.mix_random_prob,
            "mix_frozen_prob": args.mix_frozen_prob,
            "mix_meta_prob": args.mix_meta_prob,
            "mix_frozen_checkpoint_count": len(frozen_agents),
            "random_eval": random_eval,
            "heuristic_eval": heuristic_eval,
            "bc_warmstart": bool(args.bc_warmstart),
            "init_checkpoint": args.init_checkpoint,
            "bc_stats": bc_stats,
            "ent_coef": float(trainer.cfg.ent_coef),
        }

        strict_score = 0.5 * (
            float(random_eval["mean_strict_win_rate"]) + float(heuristic_eval["mean_strict_win_rate"])
        )
        report["strict_score"] = strict_score

        if strict_score > best_strict_score:
            best_strict_score = strict_score
            best_update = update
            torch.save(model.state_dict(), best_checkpoint_path)
            best_meta = {
                "update": best_update,
                "strict_score": best_strict_score,
                "source_checkpoint": str(ckpt_path),
            }
            best_meta_path.write_text(json.dumps(best_meta, indent=2), encoding="utf-8")

        collapse_warning = bool(
            float(train_stats.get("entropy", 0.0)) <= args.collapse_entropy_threshold
            and strict_score <= (best_strict_score - args.collapse_strict_drop)
        )
        report["collapse_warning"] = collapse_warning
        with report_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")

        with report_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "update",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "ent_coef",
                    "random_mean_win_rate",
                    "random_mean_strict_win_rate",
                    "random_mean_avg_turns",
                    "random_mean_avg_steps",
                    "random_mean_truncated_games",
                    "heuristic_mean_win_rate",
                    "heuristic_mean_strict_win_rate",
                    "heuristic_mean_avg_turns",
                    "heuristic_mean_avg_steps",
                    "heuristic_mean_truncated_games",
                    "strict_score",
                    "collapse_warning",
                    "checkpoint",
                ],
            )
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(
                {
                    "update": update,
                    "policy_loss": train_stats.get("policy_loss", 0.0),
                    "value_loss": train_stats.get("value_loss", 0.0),
                    "entropy": train_stats.get("entropy", 0.0),
                    "ent_coef": float(trainer.cfg.ent_coef),
                    "random_mean_win_rate": random_eval["mean_win_rate"],
                    "random_mean_strict_win_rate": random_eval["mean_strict_win_rate"],
                    "random_mean_avg_steps": random_eval["mean_avg_steps"],
                    "random_mean_truncated_games": random_eval["mean_truncated_games"],
                    "random_mean_avg_turns": random_eval["mean_avg_turns"],
                    "heuristic_mean_win_rate": heuristic_eval["mean_win_rate"],
                    "heuristic_mean_strict_win_rate": heuristic_eval["mean_strict_win_rate"],
                    "heuristic_mean_avg_steps": heuristic_eval["mean_avg_steps"],
                    "heuristic_mean_truncated_games": heuristic_eval["mean_truncated_games"],
                    "heuristic_mean_avg_turns": heuristic_eval["mean_avg_turns"],
                    "strict_score": strict_score,
                    "collapse_warning": collapse_warning,
                    "checkpoint": str(ckpt_path),
                }
            )

        print(
            json.dumps(
                {
                    "update": update,
                    "random_mean_win_rate": random_eval["mean_win_rate"],
                    "random_mean_strict_win_rate": random_eval["mean_strict_win_rate"],
                    "heuristic_mean_win_rate": heuristic_eval["mean_win_rate"],
                    "heuristic_mean_strict_win_rate": heuristic_eval["mean_strict_win_rate"],
                    "strict_score": strict_score,
                    "entropy": train_stats.get("entropy", 0.0),
                    "ent_coef": float(trainer.cfg.ent_coef),
                    "collapse_warning": collapse_warning,
                }
            )
        )

    final_model = out_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model)
    print(f"final_model: {final_model}")
    print(f"best_checkpoint: {best_checkpoint_path}")
    print(f"best_checkpoint_meta: {best_meta_path}")
    print(f"reports_jsonl: {report_jsonl}")
    print(f"reports_csv: {report_csv}")


if __name__ == "__main__":
    main()
