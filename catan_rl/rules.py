"""Invariant checks and environment diagnostics."""

from __future__ import annotations

import numpy as np

from .constants import NUM_PLAYERS, NUM_RESOURCES, Phase
from .state import GameState


def check_invariants(state: GameState) -> list[str]:
    errs: list[str] = []
    if int(state.resources.sum()) < 0:
        errs.append("negative resource total")
    if not np.array_equal(state.resource_total, state.resources.sum(axis=1)):
        errs.append("resource_total mismatch")
    if np.any(state.settlements_left < 0) or np.any(state.cities_left < 0) or np.any(state.roads_left < 0):
        errs.append("piece inventory below zero")
    if np.any(state.vertex_owner >= NUM_PLAYERS):
        errs.append("invalid vertex owner")
    if np.any(state.edge_owner >= NUM_PLAYERS):
        errs.append("invalid edge owner")
    if state.phase < 0 or state.phase > int(Phase.GAME_OVER):
        errs.append("invalid phase id")
    if state.current_player < 0 or state.current_player >= NUM_PLAYERS:
        errs.append("invalid current player")
    if state.dev_deck_remaining != int(state.dev_deck_composition.sum()):
        errs.append("dev deck count mismatch")
    if np.any(state.resources < 0) or np.any(state.dev_cards_hidden < 0):
        errs.append("negative cards")
    if state.trade_proposer >= NUM_PLAYERS:
        errs.append("invalid trade proposer")
    if state.trade_proposer == -1 and (np.any(state.trade_offer_give) or np.any(state.trade_offer_want)):
        errs.append("trade vectors nonzero without proposer")
    if np.any(state.must_discard < 0) or np.any(state.must_discard > 1):
        errs.append("must_discard non-binary")
    if np.any(state.trade_responses < 0) or np.any(state.trade_responses > 2):
        errs.append("trade responses out of range")
    if np.any(state.public_vp > state.actual_vp):
        errs.append("public_vp exceeds actual_vp")
    return errs


def masked_action_stats(mask: np.ndarray) -> dict:
    legal = int(mask.sum())
    total = int(mask.shape[0])
    return {
        "legal_actions": legal,
        "masked_actions": total - legal,
        "legal_fraction": legal / total if total else 0.0,
    }
