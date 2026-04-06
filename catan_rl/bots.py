"""Baseline agents for self-play and benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import CATALOG, Action
from .constants import NUM_PLAYERS, NUM_RESOURCES, Phase, RESOURCE_COSTS, Resource
from .encoding import observation_slices


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
            "TRADE_ADD_WANT": 3,
            "TRADE_ADD_GIVE": 3,
            "PROPOSE_TRADE": 3,
            "TRADE_REMOVE_WANT": 1,
            "TRADE_REMOVE_GIVE": 1,
            "CANCEL_TRADE": 0,
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


class TradeFriendlyAgent(Agent):
    """Heuristic agent that preferentially makes and accepts plausible beneficial trades."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._sl = observation_slices()

    def _phase_id(self, obs: np.ndarray) -> int:
        phase_oh = obs[self._sl["phase_onehot"]]
        return int(np.argmax(phase_oh))

    def _self_resources(self, obs: np.ndarray) -> np.ndarray:
        return np.rint(np.clip(obs[self._sl["self_resources"]], 0.0, 1.0) * 20.0).astype(np.int64)

    def _pieces_left(self, obs: np.ndarray) -> np.ndarray:
        vals = obs[self._sl["pieces_left"]].reshape(NUM_PLAYERS, 3)
        return np.rint(np.clip(vals[0], 0.0, 1.0) * 15.0).astype(np.int64)

    def _build_need_profile(self, obs: np.ndarray) -> tuple[set[int], bool]:
        res = self._self_resources(obs).astype(np.float32)
        pieces = self._pieces_left(obs)
        build_costs: list[np.ndarray] = []
        if int(pieces[2]) > 0:
            build_costs.append(np.asarray(RESOURCE_COSTS["road"], dtype=np.float32))
        if int(pieces[0]) > 0:
            build_costs.append(np.asarray(RESOURCE_COSTS["settlement"], dtype=np.float32))
        if int(pieces[1]) > 0:
            build_costs.append(np.asarray(RESOURCE_COSTS["city"], dtype=np.float32))
        build_costs.append(np.asarray(RESOURCE_COSTS["dev"], dtype=np.float32))
        needed: set[int] = set()
        can_build_now = False
        for cost in build_costs:
            deficit = np.maximum(cost - res, 0.0)
            miss = int(deficit.sum())
            if miss == 0:
                can_build_now = True
            if miss <= 2:
                for r in np.flatnonzero(deficit > 0):
                    needed.add(int(r))
        return needed, can_build_now

    def _estimate_build_value(self, resources: np.ndarray) -> float:
        score = 0.0
        for key in ("road", "settlement", "city", "dev"):
            cost = np.asarray(RESOURCE_COSTS[key], dtype=np.float32)
            deficit = np.maximum(cost - resources.astype(np.float32), 0.0)
            total_deficit = float(deficit.sum())
            ready = 1.0 - min(total_deficit, 6.0) / 6.0
            if key == "city":
                w = 0.85
            elif key == "dev":
                w = 0.55
            else:
                w = 0.65
            score += w * (ready + (1.0 if total_deficit <= 0.0 else 0.0))
        score += float(np.clip(8.0 - float(resources.sum()), -3.0, 3.0)) * 0.03
        return float(score)

    def _score_accept(self, obs: np.ndarray) -> float:
        trade = obs[self._sl["trade_offer_give_want"]]
        give = np.rint(np.clip(trade[:NUM_RESOURCES], 0.0, 1.0) * 4.0).astype(np.int64)
        want = np.rint(np.clip(trade[NUM_RESOURCES:], 0.0, 1.0) * 4.0).astype(np.int64)
        before = self._self_resources(obs).astype(np.float32)
        if np.any(before < want):
            return -1.0
        after = before - want.astype(np.float32) + give.astype(np.float32)
        return float(np.clip(self._estimate_build_value(after) - self._estimate_build_value(before), -1.0, 1.0))

    def _current_trade_bundle(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trade = obs[self._sl["trade_offer_give_want"]]
        give = np.rint(np.clip(trade[:NUM_RESOURCES], 0.0, 1.0) * 4.0).astype(np.int64)
        want = np.rint(np.clip(trade[NUM_RESOURCES:], 0.0, 1.0) * 4.0).astype(np.int64)
        return give, want

    def _draft_bundle_score(self, obs: np.ndarray, give: np.ndarray, want: np.ndarray) -> float:
        resources = self._self_resources(obs)
        give = np.asarray(give, dtype=np.int64)
        want = np.asarray(want, dtype=np.int64)
        if np.any((give > 0) & (want > 0)):
            return -1e9
        needed, can_build_now = self._build_need_profile(obs)
        total = int(resources.sum())
        near_discard = total > 7
        score = 0.0
        for want_r in np.flatnonzero(want > 0).tolist():
            if int(want_r) in needed:
                score += 0.9 * float(want[int(want_r)])
            elif near_discard and int(want_r) in (int(Resource.WHEAT), int(Resource.ORE)):
                score += 0.25 * float(want[int(want_r)])
            else:
                score -= 0.15 * float(want[int(want_r)])
        for give_r in np.flatnonzero(give > 0).tolist():
            give_n = int(give[int(give_r)])
            if resources[int(give_r)] < give_n:
                return -1e9
            if int(give_r) in needed and resources[int(give_r)] <= give_n + 1:
                score -= 0.8 * float(give_n)
            elif near_discard or resources[int(give_r)] >= give_n + 3:
                score += 0.25 * float(give_n)
            else:
                score -= 0.10 * float(give_n)
        give_total = int(give.sum())
        want_total = int(want.sum())
        if can_build_now and not near_discard and give_total <= want_total and give_total > 0:
            score -= 0.6
        if give_total > 0 and want_total > 0:
            before = resources.astype(np.float32)
            after = before - give.astype(np.float32) + want.astype(np.float32)
            score += float(np.clip(self._estimate_build_value(after) - self._estimate_build_value(before), -1.0, 1.0))
            score += 0.20 * float(want_total) / max(1.0, float(give_total))
        return score

    def _score_trade_draft_action(self, obs: np.ndarray, phase: int, action: Action) -> float:
        give, want = self._current_trade_bundle(obs)
        if phase == int(Phase.MAIN):
            give[:] = 0
            want[:] = 0
        if action.kind == "TRADE_ADD_GIVE":
            (give_r,) = action.params
            give[int(give_r)] += 1
            return self._draft_bundle_score(obs, give, want)
        if action.kind == "TRADE_ADD_WANT":
            (want_r,) = action.params
            want[int(want_r)] += 1
            return self._draft_bundle_score(obs, give, want)
        if action.kind == "TRADE_REMOVE_GIVE":
            (give_r,) = action.params
            give[int(give_r)] = max(0, int(give[int(give_r)]) - 1)
            return self._draft_bundle_score(obs, give, want)
        if action.kind == "TRADE_REMOVE_WANT":
            (want_r,) = action.params
            want[int(want_r)] = max(0, int(want[int(want_r)]) - 1)
            return self._draft_bundle_score(obs, give, want)
        if action.kind == "PROPOSE_TRADE":
            if int(give.sum()) <= 0 or int(want.sum()) <= 0:
                return -1e9
            return self._draft_bundle_score(obs, give, want) + 0.4
        if action.kind == "CANCEL_TRADE":
            return -0.5
        return -1e9

    @staticmethod
    def _base_action_score(kind: str) -> float:
        if kind == "PLACE_CITY":
            return 9.0
        if kind == "PLACE_SETTLEMENT":
            return 8.0
        if kind == "BUY_DEV_CARD":
            return 6.0
        if kind == "PLAY_KNIGHT":
            return 5.0
        if kind == "PLACE_ROAD":
            return 4.0
        if kind == "ROLL_DICE":
            return 3.0
        if kind == "BANK_TRADE":
            return 2.0
        if kind in {"TRADE_ADD_GIVE", "TRADE_ADD_WANT"}:
            return 1.8
        if kind in {"TRADE_REMOVE_GIVE", "TRADE_REMOVE_WANT"}:
            return 0.6
        if kind == "CANCEL_TRADE":
            return 0.0
        if kind == "END_TURN":
            return 1.0
        return 0.0

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return END_TURN_ACTION_ID
        if legal.size == 1:
            return int(legal[0])
        phase = self._phase_id(obs)
        legal_actions = [(int(a_id), CATALOG.decode(int(a_id))) for a_id in legal]

        if phase == int(Phase.TRADE_PROPOSED):
            accept_score = self._score_accept(obs)
            accept_id = next((a_id for a_id, action in legal_actions if action.kind == "ACCEPT_TRADE"), None)
            reject_id = next((a_id for a_id, action in legal_actions if action.kind == "REJECT_TRADE"), None)
            if accept_id is not None and accept_score >= 0.05:
                return int(accept_id)
            if reject_id is not None:
                return int(reject_id)

        scored_actions: list[tuple[int, float]] = []
        for a_id, action in legal_actions:
            if action.kind in {
                "TRADE_ADD_GIVE",
                "TRADE_ADD_WANT",
                "TRADE_REMOVE_GIVE",
                "TRADE_REMOVE_WANT",
                "PROPOSE_TRADE",
                "CANCEL_TRADE",
            }:
                score = self._score_trade_draft_action(obs, phase, action)
            else:
                score = self._base_action_score(action.kind)
            scored_actions.append((int(a_id), float(score)))
        if not scored_actions:
            return int(self.rng.choice(legal))
        best_score = max(score for _, score in scored_actions)
        top = [a_id for a_id, score in scored_actions if abs(score - best_score) < 1e-6]
        return int(self.rng.choice(top))
