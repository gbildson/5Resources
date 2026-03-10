"""Tournament and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bots import Agent
from .env import CatanEnv


@dataclass
class MatchResult:
    winners: list[int]
    truncated: bool
    turns: int
    steps: int


def play_match(
    agents: list[Agent],
    seed: int | None = None,
    max_steps: int = 2000,
    env_kwargs: dict | None = None,
) -> MatchResult:
    env = CatanEnv(seed=seed, **(env_kwargs or {}))
    obs, info = env.reset(seed=seed)
    mask = info["action_mask"]
    done = False
    steps = 0
    while not done and steps < max_steps:
        if np.flatnonzero(mask).size == 0:
            break
        player = env.state.current_player
        action_id = agents[player].act(obs, mask)
        res = env.step(action_id)
        obs, done = res.obs, res.done
        mask = res.info["action_mask"]
        steps += 1
    if env.state.winner >= 0:
        winners = [int(env.state.winner)]
        truncated = False
    else:
        top_vp = int(env.state.actual_vp.max(initial=0))
        winners = [int(p) for p in np.flatnonzero(env.state.actual_vp == top_vp)]
        truncated = True
    return MatchResult(winners=winners, truncated=truncated, turns=env.state.turn_number, steps=steps)


def tournament(
    agents: list[Agent],
    num_games: int = 100,
    base_seed: int = 123,
    max_steps: int = 2000,
    env_kwargs: dict | None = None,
) -> dict:
    wins = np.zeros(len(agents), dtype=np.float64)
    strict_wins = np.zeros(len(agents), dtype=np.float64)
    turns = []
    steps = []
    truncated_games = 0
    for i in range(num_games):
        res = play_match(agents, seed=base_seed + i, max_steps=max_steps, env_kwargs=env_kwargs)
        if res.truncated:
            truncated_games += 1
        elif len(res.winners) == 1:
            strict_wins[res.winners[0]] += 1.0
        share = 1.0 / max(1, len(res.winners))
        for winner in res.winners:
            wins[winner] += share
        turns.append(res.turns)
        steps.append(res.steps)
    return {
        "games": num_games,
        "terminal_games": num_games - truncated_games,
        "truncated_games": truncated_games,
        "wins": [float(x) for x in wins.tolist()],
        "win_rates": (wins / max(1, num_games)).tolist(),
        "strict_wins": [float(x) for x in strict_wins.tolist()],
        "strict_win_rates": (strict_wins / max(1, num_games)).tolist(),
        "avg_turns": float(np.mean(turns)) if turns else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
    }
