"""Self-play rollout collection and PPO training loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..actions import CATALOG
from ..constants import Building, Phase
from ..bots import RandomLegalAgent
from ..env import CatanEnv
from ..eval import tournament
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
    reward_shaping_setup_settlement: float = 0.0
    reward_shaping_terminal_table_mean: float = 0.0
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

    # Complementary pair bonus with resources already represented by current holdings/builds.
    player = int(state_before.current_player)
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
    has_wheat = owned_by_res[3] > 0.0
    has_ore = owned_by_res[4] > 0.0
    cand_has_wood = pips_by_res[0] > 0.0
    cand_has_brick = pips_by_res[1] > 0.0
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
    if has_diversity_three:
        rule_hits.append("diversity>=3")
    if has_three_hex_combo:
        rule_hits.append("three_hex_core_combo")
    if has_high_pip_three_hex:
        rule_hits.append("three_hex_pips>=10")
    if has_high_pip_two_hex_complement:
        rule_hits.append("two_hex_pips>=9_complement")

    pip_quality = float(np.clip(total_pips / 12.0, 0.0, 1.0))
    combo_factor = 1.0 if has_combo else 0.75
    if has_high_pip_three_hex:
        combo_factor = max(combo_factor, 0.95)
    if has_high_pip_two_hex_complement:
        combo_factor = max(combo_factor, 0.95)

    return {
        "vertex": int(vertex),
        "productive_hexes": int(productive_hexes),
        "total_pips": total_pips,
        "diversity": int(diversity),
        "has_combo": bool(has_combo),
        "has_complement_pair": bool(has_complement_pair),
        "pip_quality": pip_quality,
        "combo_factor": float(combo_factor),
        "eligible": bool(len(rule_hits) > 0),
        "rule_hits": rule_hits,
    }


def setup_settlement_shaping(
    state_before,
    action_id: int,
    *,
    setup_settlement_coef: float = 0.0,
) -> float:
    if setup_settlement_coef == 0.0:
        return 0.0
    details = setup_settlement_shaping_details(state_before, action_id)
    if details is None or not bool(details["eligible"]):
        return 0.0
    return float(setup_settlement_coef) * float(details["pip_quality"]) * float(details["combo_factor"])


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
    reward_shaping_setup_settlement: float = 0.0,
    reward_shaping_terminal_table_mean: float = 0.0,
    threat_dev_card_weight: float = 0.7,
) -> dict:
    obs, info = env.reset()
    mask = info["action_mask"]
    buf = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones", "masks")}
    episode_steps = 0
    for _ in range(steps):
        player = env.state.current_player
        prev_public_vp = int(env.state.public_vp[player])
        prev_resource_total = int(env.state.resource_total[player])
        state_before = env.state.copy()
        action, logp, value = trainer.act(obs, mask)
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
        reward += setup_settlement_shaping(
            state_before,
            int(action),
            setup_settlement_coef=float(reward_shaping_setup_settlement),
        )

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
    adv, ret = compute_gae(buf["rewards"], buf["values"], buf["dones"], trainer.cfg.gamma, trainer.cfg.lam)
    buf["advantages"] = adv
    buf["returns"] = ret
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
            reward_shaping_setup_settlement=config.reward_shaping_setup_settlement,
            reward_shaping_terminal_table_mean=config.reward_shaping_terminal_table_mean,
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
