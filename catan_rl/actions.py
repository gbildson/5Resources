"""Flat action catalog and action encoding/decoding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .constants import NUM_EDGES, NUM_HEXES, NUM_PLAYERS, NUM_RESOURCES, NUM_VERTICES


@dataclass(frozen=True)
class Action:
    kind: str
    params: tuple[int, ...] = ()


class ActionCatalog:
    """Static index mapping for all parameterized action choices."""

    def __init__(self) -> None:
        self.actions: list[Action] = []
        self._index: dict[Action, int] = {}
        self._build()

    def _append(self, action: Action) -> None:
        self._index[action] = len(self.actions)
        self.actions.append(action)

    def _build(self) -> None:
        for v in range(NUM_VERTICES):
            self._append(Action("PLACE_SETTLEMENT", (v,)))
        for v in range(NUM_VERTICES):
            self._append(Action("PLACE_CITY", (v,)))
        for e in range(NUM_EDGES):
            self._append(Action("PLACE_ROAD", (e,)))

        self._append(Action("ROLL_DICE"))
        self._append(Action("BUY_DEV_CARD"))
        self._append(Action("PLAY_KNIGHT"))
        self._append(Action("PLAY_ROAD_BUILDING"))

        for r1, r2 in product(range(NUM_RESOURCES), repeat=2):
            self._append(Action("PLAY_YEAR_OF_PLENTY", (r1, r2)))
        for r in range(NUM_RESOURCES):
            self._append(Action("PLAY_MONOPOLY", (r,)))

        for h in range(NUM_HEXES):
            self._append(Action("MOVE_ROBBER", (h,)))
        for p in range(NUM_PLAYERS):
            self._append(Action("ROB_PLAYER", (p,)))

        # Iterative bargaining: construct a draft offer with bounded edit actions,
        # then submit it once both sides are populated.
        for give in range(NUM_RESOURCES):
            self._append(Action("TRADE_ADD_GIVE", (give,)))
        for want in range(NUM_RESOURCES):
            self._append(Action("TRADE_ADD_WANT", (want,)))
        for give in range(NUM_RESOURCES):
            self._append(Action("TRADE_REMOVE_GIVE", (give,)))
        for want in range(NUM_RESOURCES):
            self._append(Action("TRADE_REMOVE_WANT", (want,)))
        self._append(Action("PROPOSE_TRADE"))
        self._append(Action("CANCEL_TRADE"))

        self._append(Action("ACCEPT_TRADE"))
        self._append(Action("REJECT_TRADE"))

        for give in range(NUM_RESOURCES):
            for want in range(NUM_RESOURCES):
                if give != want:
                    for give_count in (2, 3, 4):
                        self._append(Action("BANK_TRADE", (give, give_count, want)))

        # Discard templates: all vectors summing to 1..7.
        discard_templates = []
        for a in range(8):
            for b in range(8):
                for c in range(8):
                    for d in range(8):
                        for e in range(8):
                            vec = (a, b, c, d, e)
                            s = sum(vec)
                            if 1 <= s <= 7:
                                discard_templates.append(vec)
        for vec in sorted(set(discard_templates)):
            self._append(Action("DISCARD", vec))

        self._append(Action("END_TURN"))

    def size(self) -> int:
        return len(self.actions)

    def decode(self, action_id: int) -> Action:
        return self.actions[action_id]

    def encode(self, action: Action) -> int:
        return self._index[action]

    def legal_ids(self, mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask)


CATALOG = ActionCatalog()
