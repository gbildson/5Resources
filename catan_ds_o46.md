# Catan RL Game State Data Structure

## Design Principles

- **Markov-sufficient**: every field needed to determine legal actions and transition probabilities is present; no external history required.
- **Tensor-friendly**: fixed-size arrays and enums so the state can be flattened into a single vector or structured observation dict for Gym-style environments.
- **Partial observability aware**: clearly separates public vs private information so you can train with full state (self-play) or masked state (opponent modeling).

---

## 1. Board (static after setup)

These are set once during board generation and never change mid-game.

### 1.1 Hex Tiles

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `hex_terrain` | (19,) | int | Terrain enum per hex: 0=desert, 1=wood, 2=brick, 3=sheep, 4=wheat, 5=ore |
| `hex_number` | (19,) | int | Production number (2-12, 0 for desert) |
| `hex_pip_count` | (19,) | int | Pip count for the number token (0-5); precomputed convenience |

### 1.2 Topology (adjacency)

Precomputed and constant; not part of the observation tensor but used by the environment logic.

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `vertex_to_hexes` | (54, 3) | int | Up to 3 hex indices adjacent to each vertex (-1 = none) |
| `vertex_to_edges` | (54, 3) | int | Up to 3 edge indices incident on each vertex (-1 = none) |
| `vertex_to_vertices` | (54, 3) | int | Up to 3 neighboring vertex indices (-1 = none) |
| `edge_to_vertices` | (72, 2) | int | The 2 vertex endpoints of each edge |
| `hex_to_vertices` | (19, 6) | int | The 6 vertices surrounding each hex |

### 1.3 Ports

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `port_type` | (9,) | int | Port resource enum: 0=generic 3:1, 1=wood, 2=brick, 3=sheep, 4=wheat, 5=ore |
| `port_vertices` | (9, 2) | int | The 2 vertex indices that access each port |

---

## 2. Dynamic Board State

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `robber_hex` | (1,) | int | Hex index where the robber currently sits (0-18) |

---

## 3. Per-Vertex State

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `vertex_building` | (54,) | int | 0=empty, 1=settlement, 2=city |
| `vertex_owner` | (54,) | int | Player index (0-3) who owns the building, -1 if empty |

---

## 4. Per-Edge State

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `edge_road` | (72,) | int | 0=empty, 1=road present |
| `edge_owner` | (72,) | int | Player index (0-3) who owns the road, -1 if empty |

---

## 5. Player State (per player, 4 players)

### 5.1 Resources (private to each player)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `resources` | (4, 5) | int | Count of each resource per player [wood, brick, sheep, wheat, ore] |
| `resource_total` | (4,) | int | Total cards in hand (public info, derived but useful) |

### 5.2 Development Cards

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `dev_cards_hidden` | (4, 5) | int | Unplayed dev cards per type: [knight, road_building, year_of_plenty, monopoly, vp] -- private |
| `dev_cards_bought_this_turn` | (4, 5) | int | Dev cards bought this turn (cannot be played until next turn) |
| `knights_played` | (4,) | int | Public count of knight cards played |
| `vp_cards_held` | (4,) | int | Hidden VP dev cards (revealed only at game end) |

### 5.3 Building Inventory (remaining pieces)

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `settlements_left` | (4,) | int | Remaining settlement pieces (start: 5) |
| `cities_left` | (4,) | int | Remaining city pieces (start: 4) |
| `roads_left` | (4,) | int | Remaining road pieces (start: 15) |

### 5.4 Port Access

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `has_port` | (4, 6) | bool | Per player, whether they have access to each port type [generic, wood, brick, sheep, wheat, ore] |

### 5.5 Achievements

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `longest_road_length` | (4,) | int | Current longest contiguous road length per player |
| `has_longest_road` | (4,) | bool | Whether this player holds the Longest Road card |
| `has_largest_army` | (4,) | bool | Whether this player holds the Largest Army card |
| `actual_vp` | (4,) | int | True VP count (private; includes hidden VP cards) |
| `public_vp` | (4,) | int | Publicly visible VP count |

---

## 6. Dev Card Deck

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `dev_deck_remaining` | (1,) | int | Number of dev cards left in the deck (public) |
| `dev_deck_composition` | (5,) | int | Hidden counts remaining per type (for full-info training) |

---

## 7. Game Phase / Turn State

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `current_player` | (1,) | int | Index of the player whose turn it is (0-3) |
| `turn_number` | (1,) | int | Monotonically increasing turn counter |
| `phase` | (1,) | int | Phase enum (see below) |
| `dice_roll` | (1,) | int | Current dice result (2-12, 0 if not yet rolled) |
| `has_rolled` | (1,) | bool | Whether current player has rolled this turn |
| `dev_card_played_this_turn` | (1,) | bool | Whether a dev card has been played this turn (limit 1) |
| `free_roads_remaining` | (1,) | int | Roads left to place from Road Building card (0, 1, or 2) |
| `must_discard` | (4,) | bool | Per player, whether they still need to discard (7 rolled, >7 cards) |
| `setup_round` | (1,) | int | 0=normal play, 1=first settlement round, 2=second settlement round |
| `setup_forward` | (1,) | bool | Direction of setup placement (true=ascending player order) |

### Phase Enum

| Value | Name | Description |
|---|---|---|
| 0 | `SETUP_SETTLEMENT` | Placing initial settlement |
| 1 | `SETUP_ROAD` | Placing initial road |
| 2 | `PRE_ROLL` | Before rolling (can play dev card) |
| 3 | `DICE_ROLLED` | Dice just rolled, resources distributed |
| 4 | `DISCARD` | Players with >7 cards must discard |
| 5 | `MOVE_ROBBER` | Must place robber on a new hex |
| 6 | `ROB_PLAYER` | Must choose a player to steal from |
| 7 | `MAIN` | Main phase: build, trade, buy dev card, play dev card, or end turn |
| 8 | `TRADE_PROPOSED` | A trade offer is on the table |
| 9 | `YEAR_OF_PLENTY` | Choosing 2 free resources |
| 10 | `MONOPOLY` | Choosing a resource to monopolize |
| 11 | `ROAD_BUILDING` | Placing free roads |
| 12 | `GAME_OVER` | Terminal state |

---

## 8. Trade State

| Field | Shape | Dtype | Description |
|---|---|---|---|
| `trade_offer_give` | (5,) | int | Resources the proposer offers [wood, brick, sheep, wheat, ore] |
| `trade_offer_want` | (5,) | int | Resources the proposer wants |
| `trade_proposer` | (1,) | int | Player index who proposed the trade (-1 if no active trade) |
| `trade_responses` | (4,) | int | Per player: 0=pending, 1=accepted, 2=rejected |

---

## 9. Action Space

The action space is a discrete set. Each action type is valid only in the corresponding phase.

| Action Type | Parameters | Valid Phase(s) |
|---|---|---|
| `PLACE_SETTLEMENT` | vertex_id (0-53) | SETUP_SETTLEMENT, MAIN |
| `PLACE_CITY` | vertex_id (0-53) | MAIN |
| `PLACE_ROAD` | edge_id (0-71) | SETUP_ROAD, MAIN, ROAD_BUILDING |
| `ROLL_DICE` | (none) | PRE_ROLL |
| `BUY_DEV_CARD` | (none) | MAIN |
| `PLAY_KNIGHT` | (none) | PRE_ROLL, MAIN |
| `PLAY_ROAD_BUILDING` | (none) | MAIN |
| `PLAY_YEAR_OF_PLENTY` | resource1, resource2 | MAIN -> YEAR_OF_PLENTY |
| `PLAY_MONOPOLY` | resource (0-4) | MAIN -> MONOPOLY |
| `MOVE_ROBBER` | hex_id (0-18) | MOVE_ROBBER |
| `ROB_PLAYER` | player_id (0-3) | ROB_PLAYER |
| `PROPOSE_TRADE` | give[5], want[5] | MAIN |
| `ACCEPT_TRADE` | (none) | TRADE_PROPOSED |
| `REJECT_TRADE` | (none) | TRADE_PROPOSED |
| `BANK_TRADE` | give_resource, give_count, want_resource | MAIN |
| `DISCARD` | resources[5] (summing to half hand) | DISCARD |
| `END_TURN` | (none) | MAIN |

Total discrete action space (flat): ~550 actions when fully enumerated with all parameter combinations.

---

## 10. Observation Encoding for Neural Network

For feeding into a policy/value network, flatten the state into a fixed-length vector. Recommended encoding:

| Segment | Size | Notes |
|---|---|---|
| Hex terrain (one-hot) | 19 x 6 = 114 | One-hot terrain type per hex |
| Hex numbers (normalized) | 19 | Number / 12 |
| Hex pip counts | 19 | Pips / 5 |
| Robber location (one-hot) | 19 | Binary mask |
| Vertex buildings (one-hot) | 54 x 9 | {empty, p0_settle, p0_city, p1_settle, ...} |
| Edge roads (one-hot) | 72 x 5 | {empty, p0, p1, p2, p3} |
| Port types | 9 x 6 | One-hot port type |
| Port vertex mask | 54 | Binary: does this vertex touch a port |
| Current player resources | 5 | Normalized counts |
| Opponent resource totals | 3 | Public info only |
| Current player dev cards | 5 | Counts |
| Opponent knight counts | 3 | Public info |
| Building inventories | 4 x 3 = 12 | Normalized remaining pieces |
| Port access | 4 x 6 = 24 | Boolean per player |
| Longest road lengths | 4 | Normalized |
| Achievement flags | 4 x 2 = 8 | Longest road + largest army booleans |
| Public VPs | 4 | Normalized |
| Game phase (one-hot) | 13 | Phase enum |
| Turn features | 5 | turn_number, has_rolled, dev_played, free_roads, dice_roll |
| Trade state | 15 | give[5] + want[5] + proposer[1] + responses[4] |
| **Total** | **~450-500** | Depends on exact encoding choices |

### Relative Player Indexing

Rotate all player-indexed fields so the current player is always index 0. This gives the network translation invariance across seat positions and reduces the effective state space.

---

## 11. Reward Signal

| Event | Reward | Notes |
|---|---|---|
| Win | +1.0 | Terminal reward |
| Lose | -1.0 | Terminal reward |
| Per-VP gained | +0.1 | Optional shaping; can help early training but may distort late-game play |
| Illegal action attempted | -0.01 | Penalty only if using action masking is infeasible |

Prefer sparse terminal rewards with proper action masking over dense shaping for cleaner convergence.

---

## 12. Key Implementation Notes

- **Action masking**: at every step, compute a boolean mask over the action space marking legal actions. This is critical for sample efficiency in Catan's complex action space.
- **Opponent modeling**: for self-play, all 4 players share the same policy network (with relative indexing). For league training, maintain a pool of past checkpoints.
- **Hidden information**: during training with self-play you can use full state. During evaluation or human play, mask `resources`, `dev_cards_hidden`, `vp_cards_held`, and `dev_deck_composition` for opponents.
- **Graph neural networks**: the board has natural graph structure (vertices, edges, hexes). A GNN over this topology can replace the flat vector encoding and often generalizes better across board layouts.
- **Setup phase**: model as part of the same MDP. The setup placement decisions are among the most impactful in the game.
