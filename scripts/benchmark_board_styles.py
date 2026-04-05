"""Board-style robustness benchmark over grouped seed conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from catan_rl.bots import HeuristicAgent, RandomLegalAgent
from catan_rl.env import CatanEnv
from catan_rl.state import new_game_state
from catan_rl.training.ppo import load_policy_value_net
from catan_rl.training.wrappers import PolicyAgent


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


def _board_features(seed: int) -> dict[str, float]:
    s = new_game_state(seed=seed)
    ore_pips = float(np.sum(s.hex_pip_count[s.hex_terrain == 5]))
    sheep_pips = float(np.sum(s.hex_pip_count[s.hex_terrain == 3]))
    road_sites = 0.0
    for v in range(len(s.topology.vertex_to_hexes)):
        wood = 0.0
        brick = 0.0
        total = 0.0
        for h in s.topology.vertex_to_hexes[v]:
            if h < 0:
                continue
            t = int(s.hex_terrain[int(h)])
            pip = float(s.hex_pip_count[int(h)])
            total += pip
            if t == 1:
                wood += pip
            elif t == 2:
                brick += pip
        if wood > 0.0 and brick > 0.0 and total >= 9.0:
            road_sites += 1.0
    port_scores = []
    for p in range(len(s.port_type)):
        a, b = int(s.port_vertices[p, 0]), int(s.port_vertices[p, 1])
        best = 0.0
        for v in (a, b):
            v_sum = 0.0
            for h in s.topology.vertex_to_hexes[v]:
                if h >= 0:
                    v_sum += float(s.hex_pip_count[int(h)])
            best = max(best, v_sum)
        port_scores.append(best)
    strong_port_score = float(np.mean(sorted(port_scores)[-3:])) if port_scores else 0.0
    return {
        "seed": float(seed),
        "ore_pips": ore_pips,
        "sheep_pips": sheep_pips,
        "strong_port_score": strong_port_score,
        "road_site_score": float(road_sites),
    }


def _pick_group_seeds(seed_start: int, scan_count: int, seeds_per_group: int) -> dict[str, list[int]]:
    feats = [_board_features(seed_start + i) for i in range(scan_count)]
    ore_vals = np.asarray([f["ore_pips"] for f in feats], dtype=np.float64)
    sheep_vals = np.asarray([f["sheep_pips"] for f in feats], dtype=np.float64)
    port_vals = np.asarray([f["strong_port_score"] for f in feats], dtype=np.float64)
    road_vals = np.asarray([f["road_site_score"] for f in feats], dtype=np.float64)
    q20_ore = float(np.quantile(ore_vals, 0.20))
    q80_sheep = float(np.quantile(sheep_vals, 0.80))
    q80_port = float(np.quantile(port_vals, 0.80))
    q80_road = float(np.quantile(road_vals, 0.80))
    q40_ore, q60_ore = float(np.quantile(ore_vals, 0.40)), float(np.quantile(ore_vals, 0.60))
    q40_sheep, q60_sheep = float(np.quantile(sheep_vals, 0.40)), float(np.quantile(sheep_vals, 0.60))

    groups = {"low_ore": [], "high_sheep": [], "strong_ports": [], "road_heavy": [], "balanced_mid": []}
    for f in feats:
        seed = int(f["seed"])
        if f["ore_pips"] <= q20_ore:
            groups["low_ore"].append(seed)
        if f["sheep_pips"] >= q80_sheep:
            groups["high_sheep"].append(seed)
        if f["strong_port_score"] >= q80_port:
            groups["strong_ports"].append(seed)
        if f["road_site_score"] >= q80_road:
            groups["road_heavy"].append(seed)
        if q40_ore <= f["ore_pips"] <= q60_ore and q40_sheep <= f["sheep_pips"] <= q60_sheep:
            groups["balanced_mid"].append(seed)

    rng = np.random.default_rng(seed_start + 999)
    out = {}
    for k, vals in groups.items():
        vals = list(dict.fromkeys(vals))
        rng.shuffle(vals)
        out[k] = vals[: max(1, seeds_per_group)]
    return out


def _play_match(agents, *, seed: int, max_steps: int, env_kwargs: dict) -> tuple[list[int], bool]:
    env = CatanEnv(seed=seed, **env_kwargs)
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    done = False
    steps = 0
    while not done and steps < max_steps:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        p = int(env.state.current_player)
        a = int(agents[p].act(obs, mask))
        res = env.step(a)
        obs = res.obs
        mask = res.info["action_mask"]
        done = bool(res.done)
        steps += 1
    if env.state.winner >= 0:
        return [int(env.state.winner)], False
    top = int(env.state.actual_vp.max(initial=0))
    return [int(x) for x in np.flatnonzero(env.state.actual_vp == top)], True


def _eval_group(candidate: PolicyAgent, group_seeds: list[int], opponent_mode: str, opponent_ckpt: str | None, *, seed: int, max_steps: int, env_kwargs: dict, model_hidden: int, model_residual_blocks: int) -> dict:
    opp_policy = None
    if opponent_mode == "checkpoint":
        if not opponent_ckpt:
            raise ValueError("--opponent-checkpoint required for opponent=checkpoint")
        opp_policy = _load_policy_agent(
            opponent_ckpt,
            model_arch=None,
            model_hidden=model_hidden,
            model_residual_blocks=model_residual_blocks,
        )
    seat_strict = []
    seat_win = []
    for seat in range(4):
        strict_wins = 0.0
        wins = 0.0
        for i, s in enumerate(group_seeds):
            if opponent_mode == "heuristic":
                opp = HeuristicAgent(seed=seed + i + seat * 1000)
                agents = [opp, opp, opp, opp]
            elif opponent_mode == "random":
                opp = RandomLegalAgent(seed=seed + i + seat * 1000)
                agents = [opp, opp, opp, opp]
            else:
                agents = [opp_policy, opp_policy, opp_policy, opp_policy]
            agents[seat] = candidate
            winners, truncated = _play_match(agents, seed=int(s), max_steps=max_steps, env_kwargs=env_kwargs)
            if (not truncated) and len(winners) == 1 and winners[0] == seat:
                strict_wins += 1.0
            share = 1.0 / max(1, len(winners))
            if seat in winners:
                wins += share
        n = max(1, len(group_seeds))
        seat_strict.append(strict_wins / n)
        seat_win.append(wins / n)
    return {
        "seed_count": int(len(group_seeds)),
        "seeds": [int(x) for x in group_seeds],
        "mean_win_rate": float(np.mean(seat_win)),
        "mean_strict_win_rate": float(np.mean(seat_strict)),
        "worst_seat_strict": float(np.min(seat_strict)),
        "seat_strict_stddev": float(np.std(seat_strict)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--opponent", choices=["heuristic", "random", "checkpoint"], default="heuristic")
    parser.add_argument("--opponent-checkpoint", type=str, default=None)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--scan-count", type=int, default=1500)
    parser.add_argument("--seeds-per-group", type=int, default=24)
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
    groups = _pick_group_seeds(args.seed_start, args.scan_count, args.seeds_per_group)
    results = {}
    for name, seeds in groups.items():
        name_seed = sum(ord(c) for c in name)
        results[name] = _eval_group(
            candidate,
            seeds,
            args.opponent,
            args.opponent_checkpoint,
            seed=args.seed + name_seed,
            max_steps=args.max_steps,
            env_kwargs=env_kwargs,
            model_hidden=args.model_hidden,
            model_residual_blocks=args.model_residual_blocks,
        )
    mean_strict = float(np.mean([float(v["mean_strict_win_rate"]) for v in results.values()])) if results else 0.0
    report = {
        "candidate_checkpoint": args.checkpoint,
        "opponent": args.opponent,
        "opponent_checkpoint": args.opponent_checkpoint,
        "seed_start": int(args.seed_start),
        "scan_count": int(args.scan_count),
        "seeds_per_group": int(args.seeds_per_group),
        "max_steps": int(args.max_steps),
        "env_kwargs": env_kwargs,
        "group_mean_strict_win_rate": mean_strict,
        "groups": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

