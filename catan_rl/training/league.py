"""Checkpoint league management for robust self-play."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..bots import RandomLegalAgent
from ..eval import tournament
from .self_play import save_checkpoint
from .wrappers import PolicyAgent


@dataclass
class LeagueConfig:
    keep_last_n: int = 10
    promotion_games: int = 40
    promotion_win_rate: float = 0.55


@dataclass
class LeagueManager:
    root: Path
    cfg: LeagueConfig = field(default_factory=LeagueConfig)
    checkpoints: list[Path] = field(default_factory=list)
    champion: Path | None = None

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def add_candidate(self, model, name: str) -> Path:
        path = self.root / f"{name}.pt"
        save_checkpoint(model, str(path))
        self.checkpoints.append(path)
        self.checkpoints = self.checkpoints[-self.cfg.keep_last_n :]
        if self.champion is None:
            self.champion = path
        return path

    def evaluate_candidate(self, candidate_agent: PolicyAgent, seed: int = 0) -> dict:
        # Cheap online acceptance gate against random baseline.
        opp = RandomLegalAgent(seed=seed + 1)
        stats = tournament([candidate_agent, opp, opp, opp], num_games=self.cfg.promotion_games, base_seed=seed)
        promote = stats["win_rates"][0] >= self.cfg.promotion_win_rate
        return {"promote": promote, **stats}
