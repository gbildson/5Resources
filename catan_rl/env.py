"""Catan environment with O46-compatible state and legal action masking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import CATALOG, Action
from .constants import (
    Building,
    DevCard,
    NUM_PLAYERS,
    NUM_RESOURCES,
    Phase,
    RESOURCE_COSTS,
    WIN_VP,
)
from .state import GameState, new_game_state


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict


class CatanEnv:
    """Single-environment API: reset/step/observation/action-mask."""

    def __init__(
        self,
        seed: int | None = None,
        full_info_obs: bool = True,
        max_main_actions_per_turn: int | None = None,
        allow_player_trade: bool = True,
        trade_action_mode: str = "guided",
        max_player_trade_proposals_per_turn: int | None = None,
    ):
        self.seed = seed
        self.full_info_obs = full_info_obs
        self.max_main_actions_per_turn = max_main_actions_per_turn
        self.allow_player_trade = allow_player_trade
        self.trade_action_mode = trade_action_mode
        self.max_player_trade_proposals_per_turn = max_player_trade_proposals_per_turn
        self.rng = np.random.default_rng(seed)
        self.state = new_game_state(seed=seed)
        self._last_mask: np.ndarray | None = None
        self._discard_remaining = np.zeros(NUM_PLAYERS, dtype=np.int64)
        self._discard_roller = 0
        self._main_actions_this_turn = 0
        self._trade_proposals_this_turn = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        self.state = new_game_state(seed=self.seed)
        self._discard_remaining[:] = 0
        self._discard_roller = 0
        self._main_actions_this_turn = 0
        self._trade_proposals_this_turn = 0
        self._last_mask = self.action_mask()
        return self.observation(self.full_info_obs), {"action_mask": self._last_mask.copy()}

    def observation(self, full_info: bool | None = None) -> np.ndarray:
        from .encoding import encode_observation

        return encode_observation(
            self.state,
            full_info=self.full_info_obs if full_info is None else full_info,
        )

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(CATALOG.size(), dtype=np.int8)
        for idx, action in enumerate(CATALOG.actions):
            if self._is_action_legal(action):
                mask[idx] = 1
        # Safety fallback to avoid deadlock states with no legal moves.
        if mask.sum() == 0 and self.state.phase == Phase.MAIN:
            mask[CATALOG.encode(Action("END_TURN"))] = 1
        return mask

    def step(self, action_id: int) -> StepResult:
        if self.state.phase == Phase.GAME_OVER:
            return StepResult(self.observation(), 0.0, True, {"already_done": True})

        action = CATALOG.decode(action_id)
        mask = self.action_mask()
        if mask[action_id] == 0:
            return StepResult(
                self.observation(),
                -0.01,
                False,
                {"illegal_action": True, "action": action, "action_mask": mask.copy(), "winner": self.state.winner},
            )

        player = self.state.current_player
        before_phase = self.state.phase
        self._apply_action(player, action)
        if before_phase == Phase.MAIN and action.kind != "END_TURN":
            self._main_actions_this_turn += 1
        self._update_vp_and_achievements()
        winner = self._check_winner()
        reward = 0.0
        done = False
        if winner != -1:
            self.state.phase = Phase.GAME_OVER
            self.state.winner = winner
            done = True
            reward = 1.0 if winner == player else -1.0

        self._last_mask = self.action_mask()
        return StepResult(
            self.observation(),
            reward,
            done,
            {"action_mask": self._last_mask.copy(), "winner": self.state.winner},
        )

    def _is_action_legal(self, action: Action) -> bool:
        s = self.state
        p = s.current_player
        if s.phase == Phase.GAME_OVER:
            return False
        if (
            self.max_main_actions_per_turn is not None
            and s.phase == Phase.MAIN
            and self._main_actions_this_turn >= self.max_main_actions_per_turn
            and action.kind != "END_TURN"
        ):
            return False

        if action.kind == "PLACE_SETTLEMENT":
            (v,) = action.params
            if s.phase not in (Phase.SETUP_SETTLEMENT, Phase.MAIN):
                return False
            if s.vertex_building[v] != Building.EMPTY:
                return False
            for n in s.topology.vertex_to_vertices[v]:
                if n >= 0 and s.vertex_building[n] != Building.EMPTY:
                    return False
            if s.phase == Phase.MAIN:
                if not self._has_resources(p, RESOURCE_COSTS["settlement"]) or s.settlements_left[p] <= 0:
                    return False
                # Must connect to own road in regular phase.
                connected = any(
                    e >= 0 and s.edge_owner[e] == p for e in s.topology.vertex_to_edges[v]
                )
                return connected
            return s.settlements_left[p] > 0

        if action.kind == "PLACE_CITY":
            (v,) = action.params
            return (
                s.phase == Phase.MAIN
                and s.vertex_building[v] == Building.SETTLEMENT
                and s.vertex_owner[v] == p
                and s.cities_left[p] > 0
                and self._has_resources(p, RESOURCE_COSTS["city"])
            )

        if action.kind == "PLACE_ROAD":
            (e,) = action.params
            if s.phase not in (Phase.SETUP_ROAD, Phase.MAIN, Phase.ROAD_BUILDING):
                return False
            if s.edge_road[e] != 0:
                return False
            if s.phase == Phase.MAIN:
                if s.roads_left[p] <= 0 or not self._has_resources(p, RESOURCE_COSTS["road"]):
                    return False
            if s.phase == Phase.ROAD_BUILDING and s.free_roads_remaining <= 0:
                return False
            u, v = s.topology.edge_to_vertices[e]
            return self._can_connect_road(p, int(u), int(v), setup=(s.phase == Phase.SETUP_ROAD))

        if action.kind == "ROLL_DICE":
            return s.phase == Phase.PRE_ROLL and not s.has_rolled

        if action.kind == "BUY_DEV_CARD":
            return s.phase == Phase.MAIN and s.dev_deck_remaining > 0 and self._has_resources(p, RESOURCE_COSTS["dev"])

        if action.kind == "PLAY_KNIGHT":
            return (
                s.phase in (Phase.PRE_ROLL, Phase.MAIN)
                and not s.dev_card_played_this_turn
                and s.dev_cards_hidden[p, DevCard.KNIGHT] > 0
            )

        if action.kind == "PLAY_ROAD_BUILDING":
            return (
                s.phase == Phase.MAIN
                and not s.dev_card_played_this_turn
                and s.dev_cards_hidden[p, DevCard.ROAD_BUILDING] > 0
                and s.roads_left[p] > 0
            )

        if action.kind == "PLAY_YEAR_OF_PLENTY":
            return (
                s.phase == Phase.MAIN
                and not s.dev_card_played_this_turn
                and s.dev_cards_hidden[p, DevCard.YEAR_OF_PLENTY] > 0
            )

        if action.kind == "PLAY_MONOPOLY":
            return (
                s.phase == Phase.MAIN
                and not s.dev_card_played_this_turn
                and s.dev_cards_hidden[p, DevCard.MONOPOLY] > 0
            )

        if action.kind == "MOVE_ROBBER":
            (h,) = action.params
            return s.phase == Phase.MOVE_ROBBER and h != s.robber_hex

        if action.kind == "ROB_PLAYER":
            (target,) = action.params
            if s.phase != Phase.ROB_PLAYER:
                return False
            adj = self._players_adjacent_to_hex(s.robber_hex)
            robbable = [q for q in adj if q != p and s.resource_total[q] > 0]
            return target in robbable

        if action.kind == "PROPOSE_TRADE":
            if not self.allow_player_trade:
                return False
            if s.phase != Phase.MAIN:
                return False
            if (
                self.max_player_trade_proposals_per_turn is not None
                and self._trade_proposals_this_turn >= self.max_player_trade_proposals_per_turn
            ):
                return False
            give_r, give_n, want_r, want_n = action.params
            if give_r == want_r or give_n <= 0 or want_n <= 0:
                return False
            if s.resources[p, give_r] < give_n:
                return False
            # Public-info optimistic feasibility filter (no private-hand leakage):
            # keep offers that at least one responder is plausibly able to satisfy.
            any_plausible_responder = False
            for q in range(NUM_PLAYERS):
                if q == p:
                    continue
                total_cards = int(s.resource_total[q])
                if total_cards < want_n:
                    continue
                recent_gain_r = float(s.public_recent_gain[q, int(want_r)])
                recent_spend_r = float(s.public_recent_spend[q, int(want_r)])
                # Optimistic representative estimate of likely possession for wanted resource.
                likely_have_want = (recent_gain_r - 0.50 * recent_spend_r) >= 0.15
                # If hand is moderately large, allow exploratory proposals for the wanted type.
                hand_overflow_hint = total_cards >= int(want_n + 2)
                if likely_have_want or hand_overflow_hint:
                    any_plausible_responder = True
                    break
            if not any_plausible_responder:
                return False
            if self.trade_action_mode == "full":
                return True
            return self._is_guided_trade_offer_legal(
                player=p,
                give_r=int(give_r),
                give_n=int(give_n),
                want_r=int(want_r),
                want_n=int(want_n),
            )

        if action.kind in ("ACCEPT_TRADE", "REJECT_TRADE"):
            if not self.allow_player_trade:
                return False
            if s.phase != Phase.TRADE_PROPOSED:
                return False
            # Any non-proposer responder can answer while pending.
            if not (s.trade_proposer >= 0 and p != s.trade_proposer and s.trade_responses[p] == 0):
                return False
            if action.kind == "ACCEPT_TRADE":
                proposer = int(s.trade_proposer)
                # Accept is legal only when the pending offer can execute immediately.
                if not np.all(s.resources[proposer] >= s.trade_offer_give):
                    return False
                if not np.all(s.resources[p] >= s.trade_offer_want):
                    return False
            return True

        if action.kind == "BANK_TRADE":
            if s.phase != Phase.MAIN:
                return False
            give_r, give_n, want_r = action.params
            if give_r == want_r:
                return False
            needed = self._bank_rate(p, give_r)
            return give_n == needed and s.resources[p, give_r] >= needed

        if action.kind == "DISCARD":
            if s.phase != Phase.DISCARD or s.must_discard[p] == 0:
                return False
            vec = np.asarray(action.params, dtype=np.int64)
            required = int(min(7, self._discard_remaining[p]))
            if int(vec.sum()) != required:
                return False
            return np.all(s.resources[p] >= vec)

        if action.kind == "END_TURN":
            return s.phase == Phase.MAIN

        return False

    def _can_connect_road(self, player: int, u: int, v: int, setup: bool) -> bool:
        s = self.state
        if setup:
            sv = s.setup_settlement_vertex
            return sv in (u, v)
        # Connected to existing road or own building endpoint, with blocking:
        # an opponent settlement/city on a vertex blocks road continuation
        # through that vertex.
        def _endpoint_connects(vertex: int) -> bool:
            if s.vertex_owner[vertex] == player and s.vertex_building[vertex] != Building.EMPTY:
                return True
            if s.vertex_building[vertex] != Building.EMPTY and s.vertex_owner[vertex] != player:
                return False
            for edge in s.topology.vertex_to_edges[vertex]:
                if edge >= 0 and s.edge_owner[edge] == player:
                    return True
            return False

        return _endpoint_connects(int(u)) or _endpoint_connects(int(v))

    def _apply_action(self, player: int, action: Action) -> None:
        s = self.state
        if action.kind == "PLACE_SETTLEMENT":
            (v,) = action.params
            s.vertex_building[v] = Building.SETTLEMENT
            s.vertex_owner[v] = player
            s.settlements_left[player] -= 1
            if s.phase == Phase.MAIN:
                self._pay_cost(player, RESOURCE_COSTS["settlement"])
            else:
                s.setup_settlement_vertex = v
                if s.setup_round == 2:
                    self._collect_setup_resources(player, v)
            self._update_port_access(player)
            s.phase = Phase.SETUP_ROAD if s.setup_round > 0 else Phase.MAIN
            return

        if action.kind == "PLACE_ROAD":
            (e,) = action.params
            s.edge_road[e] = 1
            s.edge_owner[e] = player
            s.roads_left[player] -= 1
            if s.phase == Phase.MAIN:
                self._pay_cost(player, RESOURCE_COSTS["road"])
            elif s.phase == Phase.ROAD_BUILDING:
                s.free_roads_remaining -= 1
                if s.free_roads_remaining <= 0:
                    s.phase = Phase.MAIN
            elif s.phase == Phase.SETUP_ROAD:
                self._advance_setup_order()
            return

        if action.kind == "PLACE_CITY":
            (v,) = action.params
            s.vertex_building[v] = Building.CITY
            s.cities_left[player] -= 1
            s.settlements_left[player] += 1
            self._pay_cost(player, RESOURCE_COSTS["city"])
            return

        if action.kind == "ROLL_DICE":
            d = int(self.rng.integers(1, 7) + self.rng.integers(1, 7))
            s.dice_roll = d
            s.has_rolled = True
            if d == 7:
                self._discard_roller = int(s.current_player)
                # Catan rule: only players with more than 7 cards discard half.
                self._discard_remaining[:] = np.where(
                    s.resource_total > 7,
                    s.resource_total // 2,
                    0,
                ).astype(np.int64)
                s.must_discard[:] = (self._discard_remaining > 0).astype(np.int64)
                if s.must_discard.any():
                    s.phase = Phase.DISCARD
                    s.current_player = self._next_discard_player(self._discard_roller)
                else:
                    s.phase = Phase.MOVE_ROBBER
            else:
                self._distribute_resources(d)
                s.phase = Phase.MAIN
            return

        if action.kind == "BUY_DEV_CARD":
            self._pay_cost(player, RESOURCE_COSTS["dev"])
            card = self._draw_dev_card()
            s.dev_cards_bought_this_turn[player, card] += 1
            if card == DevCard.VP:
                s.vp_cards_held[player] += 1
            return

        if action.kind == "PLAY_KNIGHT":
            s.dev_cards_hidden[player, DevCard.KNIGHT] -= 1
            s.knights_played[player] += 1
            s.dev_card_played_this_turn = True
            s.phase = Phase.MOVE_ROBBER
            return

        if action.kind == "PLAY_ROAD_BUILDING":
            s.dev_cards_hidden[player, DevCard.ROAD_BUILDING] -= 1
            s.dev_card_played_this_turn = True
            s.free_roads_remaining = min(2, int(s.roads_left[player]))
            s.phase = Phase.ROAD_BUILDING
            return

        if action.kind == "PLAY_YEAR_OF_PLENTY":
            r1, r2 = action.params
            s.dev_cards_hidden[player, DevCard.YEAR_OF_PLENTY] -= 1
            s.dev_card_played_this_turn = True
            self._add_resource(player, r1, 1)
            self._add_resource(player, r2, 1)
            self._record_public_gain(player, r1, 1)
            self._record_public_gain(player, r2, 1)
            s.phase = Phase.MAIN
            return

        if action.kind == "PLAY_MONOPOLY":
            (r,) = action.params
            s.dev_cards_hidden[player, DevCard.MONOPOLY] -= 1
            s.dev_card_played_this_turn = True
            taken = 0
            for q in range(NUM_PLAYERS):
                if q == player:
                    continue
                amt = int(s.resources[q, r])
                if amt:
                    s.resources[q, r] = 0
                    s.resource_total[q] -= amt
                    taken += amt
                    spend = np.zeros(NUM_RESOURCES, dtype=np.int64)
                    spend[r] = amt
                    self._record_public_spend(q, spend)
            self._add_resource(player, r, taken)
            self._record_public_gain(player, r, taken)
            s.phase = Phase.MAIN
            return

        if action.kind == "MOVE_ROBBER":
            (h,) = action.params
            s.robber_hex = h
            if self._has_robbable_opponent_at_hex(player, h):
                s.phase = Phase.ROB_PLAYER
            else:
                s.phase = Phase.MAIN if s.has_rolled else Phase.PRE_ROLL
            return

        if action.kind == "ROB_PLAYER":
            (target,) = action.params
            if target != player:
                self._rob_random_resource(player, target)
            s.phase = Phase.MAIN if s.has_rolled else Phase.PRE_ROLL
            return

        if action.kind == "PROPOSE_TRADE":
            give_r, give_n, want_r, want_n = action.params
            self._trade_proposals_this_turn += 1
            s.public_trade_offers[player] += 1.0
            s.trade_offer_give[:] = 0
            s.trade_offer_want[:] = 0
            s.trade_offer_give[give_r] = give_n
            s.trade_offer_want[want_r] = want_n
            s.trade_proposer = player
            s.trade_responses[:] = 0
            s.phase = Phase.TRADE_PROPOSED
            s.current_player = self._next_trade_responder(player)
            return

        if action.kind in ("ACCEPT_TRADE", "REJECT_TRADE"):
            if action.kind == "REJECT_TRADE":
                s.public_trade_rejects[player] += 1.0
                s.trade_responses[player] = 2
                next_responder = self._next_trade_responder(player)
                if next_responder == s.trade_proposer:
                    proposer = s.trade_proposer
                    self._clear_trade()
                    s.phase = Phase.MAIN
                    s.current_player = proposer
                else:
                    s.current_player = next_responder
            else:
                s.public_trade_accepts[player] += 1.0
                s.trade_responses[player] = 1
                self._execute_trade(s.trade_proposer, player, s.trade_offer_give, s.trade_offer_want)
                proposer = s.trade_proposer
                self._clear_trade()
                s.phase = Phase.MAIN
                s.current_player = proposer
                return
            return

        if action.kind == "BANK_TRADE":
            give_r, give_n, want_r = action.params
            s.resources[player, give_r] -= give_n
            s.resources[player, want_r] += 1
            s.resource_total[player] += 1 - give_n
            spend = np.zeros(NUM_RESOURCES, dtype=np.int64)
            spend[give_r] = give_n
            self._record_public_spend(player, spend)
            self._record_public_gain(player, want_r, 1)
            return

        if action.kind == "DISCARD":
            vec = np.asarray(action.params, dtype=np.int64)
            s.resources[player] -= vec
            s.resource_total[player] -= int(vec.sum())
            self._discard_remaining[player] = max(0, int(self._discard_remaining[player] - int(vec.sum())))
            s.must_discard[player] = 1 if self._discard_remaining[player] > 0 else 0
            if s.must_discard.any():
                s.current_player = self._next_discard_player(player)
            else:
                s.phase = Phase.MOVE_ROBBER
                s.current_player = self._discard_roller
            return

        if action.kind == "END_TURN":
            self._end_turn()
            return

    def _end_turn(self) -> None:
        s = self.state
        self._main_actions_this_turn = 0
        self._trade_proposals_this_turn = 0
        # Exponential decay approximates short-horizon human memory.
        s.public_recent_gain *= 0.90
        s.public_recent_spend *= 0.90
        s.public_recent_rob_unknown_gain *= 0.90
        s.public_recent_rob_unknown_loss *= 0.90
        s.public_trade_offers *= 0.95
        s.public_trade_accepts *= 0.95
        s.public_trade_rejects *= 0.95
        s.current_player = (s.current_player + 1) % NUM_PLAYERS
        s.turn_number += 1
        s.phase = Phase.PRE_ROLL
        s.has_rolled = False
        s.dice_roll = 0
        s.dev_card_played_this_turn = False
        s.dev_cards_hidden += s.dev_cards_bought_this_turn
        s.dev_cards_bought_this_turn[:] = 0

    def _advance_setup_order(self) -> None:
        s = self.state
        if s.setup_round == 1:
            if s.current_player < NUM_PLAYERS - 1:
                s.current_player += 1
                s.phase = Phase.SETUP_SETTLEMENT
            else:
                s.setup_round = 2
                s.setup_forward = False
                s.phase = Phase.SETUP_SETTLEMENT
        elif s.setup_round == 2:
            if s.current_player > 0:
                s.current_player -= 1
                s.phase = Phase.SETUP_SETTLEMENT
            else:
                s.setup_round = 0
                s.setup_forward = True
                s.phase = Phase.PRE_ROLL
                s.has_rolled = False
                s.dice_roll = 0
                s.current_player = 0
                s.turn_number = 1
        s.setup_settlement_vertex = -1

    def _collect_setup_resources(self, player: int, vertex: int) -> None:
        s = self.state
        for h in s.topology.vertex_to_hexes[vertex]:
            if h >= 0:
                terrain = int(s.hex_terrain[h])
                if terrain > 0:
                    self._add_resource(player, terrain - 1, 1)
                    self._record_public_gain(player, terrain - 1, 1)

    def _distribute_resources(self, roll: int) -> None:
        s = self.state
        for h in range(len(s.hex_number)):
            if s.hex_number[h] != roll or h == s.robber_hex:
                continue
            resource = int(s.hex_terrain[h]) - 1
            if resource < 0:
                continue
            for v in s.topology.hex_to_vertices[h]:
                owner = int(s.vertex_owner[v])
                if owner < 0:
                    continue
                amount = 2 if s.vertex_building[v] == Building.CITY else 1
                self._add_resource(owner, resource, amount)
                self._record_public_gain(owner, resource, amount)

    def _has_resources(self, player: int, cost: list[int]) -> bool:
        return bool(np.all(self.state.resources[player] >= np.asarray(cost, dtype=np.int64)))

    def _pay_cost(self, player: int, cost: list[int]) -> None:
        c = np.asarray(cost, dtype=np.int64)
        self.state.resources[player] -= c
        self.state.resource_total[player] -= int(c.sum())
        self._record_public_spend(player, c)

    def _draw_dev_card(self) -> int:
        s = self.state
        probs = s.dev_deck_composition / s.dev_deck_composition.sum()
        card = int(self.rng.choice(np.arange(5), p=probs))
        s.dev_deck_composition[card] -= 1
        s.dev_deck_remaining -= 1
        return card

    def _add_resource(self, player: int, resource: int, amount: int) -> None:
        if amount <= 0:
            return
        self.state.resources[player, resource] += amount
        self.state.resource_total[player] += amount

    def _rob_random_resource(self, thief: int, target: int) -> None:
        s = self.state
        total = int(s.resource_total[target])
        if total <= 0:
            return
        idxs = np.repeat(np.arange(NUM_RESOURCES), s.resources[target])
        r = int(self.rng.choice(idxs))
        s.resources[target, r] -= 1
        s.resource_total[target] -= 1
        s.resources[thief, r] += 1
        s.resource_total[thief] += 1
        # Resource identity is hidden to other players; track as unknown card flow only.
        s.public_recent_rob_unknown_gain[thief] += 1.0
        s.public_recent_rob_unknown_loss[target] += 1.0

    def _next_trade_responder(self, player: int) -> int:
        s = self.state
        for i in range(1, NUM_PLAYERS + 1):
            candidate = (player + i) % NUM_PLAYERS
            if candidate == s.trade_proposer:
                return candidate
            if s.trade_responses[candidate] == 0:
                return candidate
        return s.trade_proposer

    def _next_discard_player(self, start_player: int) -> int:
        s = self.state
        for i in range(NUM_PLAYERS):
            candidate = (start_player + i) % NUM_PLAYERS
            if s.must_discard[candidate] == 1:
                return candidate
        return start_player

    def _players_adjacent_to_hex(self, hex_id: int) -> set[int]:
        s = self.state
        players = set()
        for v in s.topology.hex_to_vertices[hex_id]:
            owner = int(s.vertex_owner[v])
            if owner >= 0:
                players.add(owner)
        return players

    def _has_robbable_opponent_at_hex(self, player: int, hex_id: int) -> bool:
        adj = self._players_adjacent_to_hex(hex_id)
        for q in adj:
            if q != player and self.state.resource_total[q] > 0:
                return True
        return False

    def _clear_trade(self) -> None:
        s = self.state
        s.trade_offer_give[:] = 0
        s.trade_offer_want[:] = 0
        s.trade_proposer = -1
        s.trade_responses[:] = 0

    def _execute_trade(self, a: int, b: int, give: np.ndarray, want: np.ndarray) -> None:
        s = self.state
        if np.all(s.resources[a] >= give) and np.all(s.resources[b] >= want):
            s.resources[a] -= give
            s.resources[b] += give
            s.resources[b] -= want
            s.resources[a] += want
            s.resource_total[a] = int(s.resources[a].sum())
            s.resource_total[b] = int(s.resources[b].sum())
            self._record_public_spend(a, give)
            self._record_public_gain_vec(b, give)
            self._record_public_spend(b, want)
            self._record_public_gain_vec(a, want)

    def _bank_rate(self, player: int, resource: int) -> int:
        if self.state.has_port[player, resource + 1]:
            return 2
        if self.state.has_port[player, 0]:
            return 3
        return 4

    def _trade_need_profile(self, player: int) -> tuple[set[int], bool]:
        resources = self.state.resources[player]
        build_costs = []
        if self.state.roads_left[player] > 0:
            build_costs.append(np.asarray(RESOURCE_COSTS["road"], dtype=np.int64))
        if self.state.settlements_left[player] > 0:
            build_costs.append(np.asarray(RESOURCE_COSTS["settlement"], dtype=np.int64))
        if self.state.cities_left[player] > 0:
            build_costs.append(np.asarray(RESOURCE_COSTS["city"], dtype=np.int64))
        if self.state.dev_deck_remaining > 0:
            build_costs.append(np.asarray(RESOURCE_COSTS["dev"], dtype=np.int64))

        needed: set[int] = set()
        can_build_now = False
        for cost in build_costs:
            deficit = np.maximum(cost - resources, 0)
            miss = int(deficit.sum())
            if miss == 0:
                can_build_now = True
            if miss <= 2:
                for r in np.flatnonzero(deficit > 0):
                    needed.add(int(r))
        return needed, can_build_now

    def _is_guided_trade_offer_legal(
        self,
        player: int,
        give_r: int,
        give_n: int,
        want_r: int,
        want_n: int,
    ) -> bool:
        if want_n != 1:
            return False
        total_resources = int(self.state.resource_total[player])
        near_discard = total_resources > 7
        needed_resources, can_build_now = self._trade_need_profile(player)

        # If we can already execute a build and are not at discard risk, be conservative with 1-for-1.
        # Keep room for selective 2-for-1 overflow/smoothing offers.
        if can_build_now and not near_discard and give_n == 1:
            return False

        # Only ask for near-term useful resources, except allow ore/wheat conversion under discard pressure.
        if want_r not in needed_resources:
            if not (near_discard and want_r in (3, 4)):
                return False

        # Avoid giving away scarce resources that we also need soon.
        if give_r in needed_resources and int(self.state.resources[player, give_r]) <= give_n + 1:
            return False

        if give_n == 2:
            have_give = int(self.state.resources[player, give_r])
            # 2-for-1 is often used to offload redundant cards. Allow more frequently than before.
            if near_discard:
                return True
            # Classic "redundant -> needed" conversion.
            if give_r not in needed_resources and want_r in needed_resources and have_give >= 3:
                return True
            # High-surplus dump even when need profile is noisy.
            if have_give >= 5:
                return True
            return False
        elif give_n != 1:
            return False

        return True

    def _record_public_gain(self, player: int, resource: int, amount: int) -> None:
        if amount <= 0:
            return
        self.state.public_recent_gain[player, resource] += float(amount)

    def _record_public_gain_vec(self, player: int, gain: np.ndarray) -> None:
        if gain.size == 0:
            return
        self.state.public_recent_gain[player] += gain.astype(np.float32)

    def _record_public_spend(self, player: int, cost: np.ndarray) -> None:
        if cost.size == 0:
            return
        self.state.public_recent_spend[player] += cost.astype(np.float32)

    def _update_port_access(self, player: int) -> None:
        s = self.state
        for p in range(len(s.port_vertices)):
            a, b = s.port_vertices[p]
            if s.vertex_owner[a] == player or s.vertex_owner[b] == player:
                s.has_port[player, s.port_type[p]] = 1

    def _update_vp_and_achievements(self) -> None:
        s = self.state
        # Public VP: settlements + cities + open awards (no hidden VP cards).
        settlement_vp = np.zeros(NUM_PLAYERS, dtype=np.int64)
        city_vp = np.zeros(NUM_PLAYERS, dtype=np.int64)
        for v in range(len(s.vertex_owner)):
            o = s.vertex_owner[v]
            if o < 0:
                continue
            if s.vertex_building[v] == Building.SETTLEMENT:
                settlement_vp[o] += 1
            elif s.vertex_building[v] == Building.CITY:
                city_vp[o] += 2

        prev_longest = s.has_longest_road.copy()
        prev_army = s.has_largest_army.copy()

        s.longest_road_length[:] = self._road_lengths()
        s.has_longest_road[:] = self._resolve_award_holder(
            metric=s.longest_road_length,
            previous_holder_flags=prev_longest,
            minimum_required=5,
        )
        s.has_largest_army[:] = self._resolve_award_holder(
            metric=s.knights_played,
            previous_holder_flags=prev_army,
            minimum_required=3,
        )

        s.public_vp[:] = settlement_vp + city_vp + 2 * s.has_longest_road + 2 * s.has_largest_army
        s.actual_vp[:] = s.public_vp + s.vp_cards_held

    def _road_lengths(self) -> np.ndarray:
        s = self.state
        lengths = np.zeros(NUM_PLAYERS, dtype=np.int64)
        for p in range(NUM_PLAYERS):
            owned_edges = np.flatnonzero(s.edge_owner == p)
            if owned_edges.size == 0:
                continue
            best = 0
            # Search longest edge-simple trail over vertices.
            start_vertices = set()
            for e in owned_edges:
                u, v = s.topology.edge_to_vertices[e]
                start_vertices.add(int(u))
                start_vertices.add(int(v))
            for v in start_vertices:
                best = max(best, self._dfs_road_from_vertex(v, p, set(), 0))
            lengths[p] = best
        return lengths

    def _dfs_road_from_vertex(
        self,
        vertex: int,
        player: int,
        used_edges: set[int],
        length: int,
    ) -> int:
        best = length
        if self._vertex_blocks_pass_through(vertex, player) and length > 0:
            return best

        for e in self.state.topology.vertex_to_edges[vertex]:
            if e < 0 or e in used_edges or self.state.edge_owner[e] != player:
                continue
            u, v = self.state.topology.edge_to_vertices[e]
            nxt = int(v if int(u) == vertex else u)
            used_edges.add(int(e))
            best = max(best, self._dfs_road_from_vertex(nxt, player, used_edges, length + 1))
            used_edges.remove(int(e))
        return best

    def _vertex_blocks_pass_through(self, vertex: int, player: int) -> bool:
        owner = int(self.state.vertex_owner[vertex])
        return owner >= 0 and owner != player

    def _check_winner(self) -> int:
        winners = np.flatnonzero(self.state.actual_vp >= WIN_VP)
        return int(winners[0]) if winners.size else -1

    def _resolve_award_holder(
        self,
        metric: np.ndarray,
        previous_holder_flags: np.ndarray,
        minimum_required: int,
    ) -> np.ndarray:
        flags = np.zeros(NUM_PLAYERS, dtype=np.int64)
        top_value = int(metric.max(initial=0))
        if top_value < minimum_required:
            return flags

        contenders = np.flatnonzero(metric == top_value)
        previous = np.flatnonzero(previous_holder_flags == 1)
        previous_holder = int(previous[0]) if previous.size else -1

        if previous_holder >= 0 and previous_holder in contenders:
            flags[previous_holder] = 1
            return flags

        if contenders.size == 1:
            flags[int(contenders[0])] = 1
        return flags
