# Designing a Catan State Summarizer for LLM Agents

## 1. Feasibility Analysis
It is highly feasible to drive a Catan bot using an LLM. 
*   **State Size:** The complete game state is small (tokens < 2k), fitting easily into modern context windows.
*   **Reasoning vs. Calculation:** LLMs excel at heuristic strategy ("I need ore to upgrade to a city") but struggle with strict probability calculation. The design should offload math to the code and let the LLM handle high-level decision making.

## 2. Architecture: The Hybrid Model
To achieve "optimal" play, we cannot rely on the LLM to know the rules perfectly or calculate odds.

**Components:**
1.  **Game Engine (Python/C++):** 
    *   Maintains the true state of the board.
    *   Enforces rules.
    *   Generates a list of **Legal Moves**.
    *   Calculates **Derived Metrics** (probabilities, resource scarcity).
2.  **LLM Interface:** 
    *   Receives a structured summary of the state.
    *   Receives the list of Legal Moves.
    *   Returns the selected move index and reasoning.

## 3. State Representation (The Context Prompt)
The prompt sent to the LLM should contain three distinct sections:

### A. Global Board State
Describes the static layout AND dynamic pieces (roads/buildings).
```json
{
  "board_layout": {
    "hexes": [
      { "id": "h1", "resource": "Ore", "number": 6, "robber": false },
      { "id": "h2", "resource": "Wheat", "number": 8, "robber": true }
    ],
    "ports": { "n12": "Ore 2:1", "n15": "3:1" }
  },
  "board_state": {
    "occupied_nodes": [
      { "id": "n12", "owner": "Red", "type": "City" },
      { "id": "n15", "owner": "Blue", "type": "Settlement" }
    ],
    "roads": [
      { "from": "n12", "to": "n13", "owner": "Red" }
    ],
    "robber_location": "h2"
  },
  "game_status": {
    "turn_number": 15,
    "active_player": "Bot_Red",
    "phase": "MAIN_PHASE"
  }
}
```

### B. Player Knowledge (Public + Private)
What the bot knows about itself and others.
```json
{
  "me": {
    "color": "Red",
    "victory_points": 7,
    "hand": { "brick": 2, "wood": 1, "ore": 3, "wheat": 0, "sheep": 1 },
    "dev_cards": { "hidden": ["VP_Chapel"], "played": ["Knight"] },
    "production_odds": { "ore": 0.13, "wheat": 0.05 }
  },
  "opponents": [
    {
      "color": "Blue",
      "victory_points": 5,
      "hand_count": 6,
      "known_resources": ["sheep"],
      "longest_road_potential": 5
    }
  ]
}
```

### C. Derived Heuristics & Analysis
Pre-calculated insights to guide the LLM.
```json
"heuristics": {
  "resource_scarcity": { "brick": "High", "ore": "Low" },
  "my_needs": ["wheat"],
  "winning_threats": ["Blue is 1 road away from taking Longest Road"],
  "blocking_opportunities": ["Node n15 cuts Blue's network"]
}
```

### D. The Decision Space (Legal Moves)
Crucially, do not ask the LLM to "generate" a move. Ask it to **select** one.
```json
"legal_moves": [
  { "id": 0, "action": "BUILD_CITY", "location": 12, "cost": {"ore":3, "wheat":2} },
  { "id": 1, "action": "TRADE_BANK", "give": "ore", "get": "wheat" },
  { "id": 2, "action": "PASS_TURN" }
]
```

## 4. Heuristics & "Secret Sauce"
To improve performance, pre-calculate data that humans intuit but LLMs might miss:

1.  **Resource Scarcity:** The engine should tell the LLM: *"Brick is rare globally (only rolled on 2 and 12)."*
2.  **Blocking Opportunities:** Flag critical nodes: *"Node 15 cuts Blue's road network."*
3.  **Win Condition Check:** Explicitly state: *"Blue needs 2 VP to win."*

## 5. Handling Hidden Information
Catan has imperfect information (opponents' hands).
*   **Card Counting:** The engine should track card history (e.g., "Blue picked up 2 Ore, traded 1 away -> Blue has at least 1 Ore").
*   **Uncertainty:** Present this to the LLM as *"Blue likely has Ore (80% confidence)"* rather than definite facts.

