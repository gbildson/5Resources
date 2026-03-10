"""Baseline agents for self-play and benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import CATALOG, Action


END_TURN_ACTION_ID = CATALOG.encode(Action("END_TURN"))


class Agent:
    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        raise NotImplementedError


@dataclass
class RandomLegalAgent(Agent):
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return END_TURN_ACTION_ID
        return int(self.rng.choice(legal))


class HeuristicAgent(Agent):
    """Lightweight heuristic: prioritize VP-gaining and progression actions."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.priority = {
            "PLACE_CITY": 10,
            "PLACE_SETTLEMENT": 9,
            "PLAY_MONOPOLY": 8,
            "PLAY_YEAR_OF_PLENTY": 8,
            "PLAY_ROAD_BUILDING": 7,
            "BUY_DEV_CARD": 6,
            "PLACE_ROAD": 5,
            "BANK_TRADE": 4,
            "PROPOSE_TRADE": 3,
            "ROLL_DICE": 2,
            "END_TURN": 1,
        }

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return END_TURN_ACTION_ID
        if legal.size == 1:
            return int(legal[0])
        scores = []
        for a_id in legal:
            kind = CATALOG.decode(int(a_id)).kind
            scores.append(self.priority.get(kind, 0))
        max_score = max(scores)
        best = legal[np.where(np.asarray(scores) == max_score)[0]]
        return int(self.rng.choice(best))


class StrategyHeuristicAgent(Agent):
    """Strategy-profile heuristic approximating plans from strats.md."""

    STRATEGY_PROFILES: dict[str, dict[str, int]] = {
        # OWS / dev-army leaning.
        "full_ows": {
            "PLACE_CITY": 11,
            "BUY_DEV_CARD": 10,
            "PLAY_KNIGHT": 10,
            "PLAY_MONOPOLY": 9,
            "PLAY_YEAR_OF_PLENTY": 9,
            "PLACE_SETTLEMENT": 7,
            "PLACE_ROAD": 5,
            "BANK_TRADE": 7,
            "PROPOSE_TRADE": 7,
            "END_TURN": 1,
        },
        "road_builder": {
            "PLACE_SETTLEMENT": 11,
            "PLACE_ROAD": 10,
            "PLAY_ROAD_BUILDING": 10,
            "PLACE_CITY": 6,
            "BUY_DEV_CARD": 5,
            "BANK_TRADE": 6,
            "PROPOSE_TRADE": 6,
            "END_TURN": 1,
        },
        "five_resources": {
            "PLACE_CITY": 10,
            "PLACE_SETTLEMENT": 10,
            "PLACE_ROAD": 7,
            "BUY_DEV_CARD": 7,
            "PLAY_KNIGHT": 6,
            "BANK_TRADE": 5,
            "PROPOSE_TRADE": 5,
            "END_TURN": 1,
        },
        "city_road": {
            "PLACE_CITY": 11,
            "PLACE_SETTLEMENT": 8,
            "PLACE_ROAD": 8,
            "PLAY_ROAD_BUILDING": 8,
            "BUY_DEV_CARD": 5,
            "BANK_TRADE": 5,
            "PROPOSE_TRADE": 4,
            "END_TURN": 1,
        },
        "port": {
            "BANK_TRADE": 11,
            "PROPOSE_TRADE": 8,
            "BUY_DEV_CARD": 8,
            "PLAY_KNIGHT": 8,
            "PLACE_CITY": 7,
            "PLACE_SETTLEMENT": 7,
            "PLACE_ROAD": 6,
            "END_TURN": 1,
        },
        "hybrid_ows": {
            "PLACE_CITY": 10,
            "BUY_DEV_CARD": 9,
            "PLAY_KNIGHT": 8,
            "PLACE_SETTLEMENT": 8,
            "PLACE_ROAD": 6,
            "BANK_TRADE": 6,
            "PROPOSE_TRADE": 6,
            "END_TURN": 1,
        },
        "city_vp": {
            "PLACE_CITY": 11,
            "BUY_DEV_CARD": 9,
            "PLACE_SETTLEMENT": 8,
            "PLACE_ROAD": 5,
            "PLAY_KNIGHT": 6,
            "BANK_TRADE": 6,
            "PROPOSE_TRADE": 5,
            "END_TURN": 1,
        },
    }

    # Preferred resources by strategy for trade target scoring.
    _PREFERRED_WANT = {
        "full_ows": {2, 3, 4},  # sheep, wheat, ore
        "road_builder": {0, 1},  # wood, brick
        "five_resources": {0, 1, 2, 3, 4},
        "city_road": {3, 4, 0, 1},
        "port": {2, 3, 4},
        "hybrid_ows": {3, 4, 2},
        "city_vp": {3, 4},
    }

    def __init__(self, strategy: str = "five_resources", seed: int | None = None):
        if strategy not in self.STRATEGY_PROFILES:
            raise ValueError(f"Unknown strategy profile: {strategy}")
        self.strategy = strategy
        self.priority = self.STRATEGY_PROFILES[strategy]
        self.rng = np.random.default_rng(seed)

    def _trade_bonus(self, action: Action) -> float:
        if action.kind not in ("PROPOSE_TRADE", "BANK_TRADE"):
            return 0.0
        preferred = self._PREFERRED_WANT[self.strategy]
        if action.kind == "BANK_TRADE":
            _, _, want_r = action.params
            return 1.5 if int(want_r) in preferred else -0.5
        give_r, give_n, want_r, want_n = action.params
        bonus = 0.0
        if int(want_r) in preferred:
            bonus += 1.0
        if int(give_r) in preferred:
            bonus -= 0.5
        # Prefer better deal shapes.
        if int(want_n) >= int(give_n):
            bonus += 0.25
        return bonus

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return END_TURN_ACTION_ID
        if legal.size == 1:
            return int(legal[0])

        legal_actions = [CATALOG.decode(int(a_id)) for a_id in legal]
        legal_kinds = {a.kind for a in legal_actions}
        has_progression = any(k in legal_kinds for k in ("PLACE_SETTLEMENT", "PLACE_CITY", "PLACE_ROAD", "BUY_DEV_CARD"))

        scores = []
        for action in legal_actions:
            score = float(self.priority.get(action.kind, 0))
            score += self._trade_bonus(action)
            # Avoid passing when meaningful progression exists.
            if action.kind == "END_TURN" and has_progression:
                score -= 1.0
            scores.append(score)

        max_score = max(scores)
        best_ids = legal[np.where(np.asarray(scores) == max_score)[0]]
        return int(self.rng.choice(best_ids))


class MetaStrategyHeuristicAgent(Agent):
    """Selects a strategy profile from legal-action context each turn."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._agents = {
            name: StrategyHeuristicAgent(strategy=name, seed=int(self.rng.integers(0, 1_000_000)))
            for name in StrategyHeuristicAgent.STRATEGY_PROFILES
        }

    def _pick_strategy(self, legal_actions: list[Action]) -> str:
        kinds = [a.kind for a in legal_actions]
        has = set(kinds)
        count = {k: kinds.count(k) for k in has}

        if count.get("BANK_TRADE", 0) >= 3:
            return "port"
        if "PLAY_KNIGHT" in has or "PLAY_MONOPOLY" in has or count.get("BUY_DEV_CARD", 0) >= 1:
            return "full_ows"
        if count.get("PLACE_ROAD", 0) >= 2 or "PLAY_ROAD_BUILDING" in has:
            return "road_builder"
        if "PLACE_CITY" in has and "PLACE_SETTLEMENT" in has:
            return "five_resources"
        if "PLACE_CITY" in has:
            return "city_vp"
        return "hybrid_ows"

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return END_TURN_ACTION_ID
        legal_actions = [CATALOG.decode(int(a_id)) for a_id in legal]
        strategy = self._pick_strategy(legal_actions)
        return self._agents[strategy].act(obs, mask)
