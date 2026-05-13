# Unified Plan: Strategy-Guided Search

## Core Idea

Combine the alpha search plan and the opening-strategy idea into one simpler system:

```text
StrategyState + PolicyModel + Planner
```

The existing trained policy remains the main model. A lightweight strategy layer tracks the bot's current plan, and an inference-time planner uses that plan to rescore the model's best candidate actions.

Instead of building many separate specialist models, use one soft strategy vector plus one search wrapper.

## Strategy Vector

The strategy vector is a soft description of the game plan.

Example:

```text
city_dev:    0.45
expansion:   0.30
road:        0.15
port_trade:  0.10
```

It is not a hard commitment. It gives the agent a strategic bias while still allowing adaptation.

## How The Strategy Gets Set

During setup, score candidate settlements and roads for strategic signals:

- ore/wheat production
- sheep support for dev cards
- wood/brick production
- expansion room
- port access
- resource diversity
- resource scarcity on the board
- likely second-settlement complement
- road race potential

Convert those signals into archetype scores:

```text
city_dev    = ore + wheat + sheep + city potential
expansion   = wood + brick + open settlement sites
road        = wood + brick + longest-road lanes
port_trade  = port access + resource surplus
```

Normalize the scores into the initial strategy vector after opening choices.

## How The Strategy Gets Updated

Periodically recompute a fresh vector from the current state, then blend it with the old vector:

```text
strategy = 0.8 * old_strategy + 0.2 * current_state_strategy
```

Good update points:

- after setup completes
- at the start of each turn
- after building a settlement, city, road, or dev card
- after gaining or losing useful port access
- after robber disruption
- after opponent blocks a key route
- when actual resources diverge from the opening plan

The blend prevents thrashing while still letting the bot adapt.

## How Action Selection Works

Current direct policy:

```text
obs + legal mask -> model logits -> masked argmax -> action
```

Unified planner:

```text
obs + legal mask -> model logits -> top K actions
current env state -> clone/simulate/rescore candidates
policy score + tactical score + strategy fit + optional value estimate -> final action
```

Candidate score:

```text
final_score =
  policy_weight * policy_score
+ tactical_weight * search_score
+ value_weight * value_after_action
+ strategy_weight * strategy_fit
```

## Why This Helps

The neural policy provides broad pattern recognition:

- which actions are plausible
- what type of position this resembles
- which tactics are likely legal/useful

The planner provides precise Catan-specific evaluation:

- whether a settlement supports the chosen game plan
- whether a road opens a real expansion path
- whether the robber blocks the leader without self-harm
- whether a trade supports the current strategy

The strategy vector adds coherence:

- a city/dev opening keeps favoring city/dev-supporting moves
- an expansion opening keeps valuing settlement sites and roads
- a port economy keeps valuing surplus resources and port access
- the plan can shift gradually when the board or dice force adaptation

## Existing Code Integration

The repo already has most of the needed pieces:

- `catan_rl/search.py` has env cloning, setup ranking, robber ranking, and shallow heuristic rollout.
- `catan_rl/training/wrappers.py` has `PolicyAgent`; add a planned/search wrapper next to it.
- `catan_rl/eval.py` can dispatch to env-aware agents when available.
- `scripts/eval_checkpoint.py` can expose CLI flags and run seat-balanced comparisons.
- `scripts/trace_game.py` can log planner decisions and policy overrides.
- `catan_rl/training/bc.py` already uses search as an offline teacher; this plan also uses search online at inference time.
- `catan_rl/strategy_archetypes.py` and `catan_rl/strategy_metrics.py` can seed the first version of strategy-vector scoring.

## Proposed Components

### `StrategyState`

Stores the current strategy vector and update history.

Responsibilities:

- initialize from opening/setup state
- update from current game state
- blend old and new strategy estimates
- expose strategy weights to the planner

### `StrategyEvaluator`

Computes a fresh strategy vector from board and player state.

Inputs may include:

- production profile
- resources and dev cards
- ports
- roads and open settlement sites
- VP path
- opponent pressure
- robber disruption

### `SearchPolicyAgent`

Wraps the existing torch model.

Responsibilities:

- behave like `PolicyAgent` when search is disabled
- get model top-K candidate actions
- invoke planner only for selected phases
- maintain per-player strategy state
- return the final action
- store/debug the latest decision record

### `PlannerScorer`

Scores candidate actions using:

- model policy preference
- existing setup/robber/search heuristics
- strategy-fit score
- optional shallow rollout score
- optional model value estimate after applying action

## Records Why It Chose Each Action

The planner should emit a small debug record whenever it searches, especially when it overrides the policy argmax.

Example:

```json
{
  "phase": "SETUP_SETTLEMENT",
  "policy_argmax": "PLACE_SETTLEMENT(17)",
  "chosen_action": "PLACE_SETTLEMENT(23)",
  "reason": "higher setup score and better city_dev fit",
  "strategy_vector": {
    "city_dev": 0.52,
    "expansion": 0.28,
    "road": 0.12,
    "port_trade": 0.08
  },
  "candidates": [
    {
      "action": "PLACE_SETTLEMENT(23)",
      "policy_score": 0.21,
      "tactical_score": 0.84,
      "strategy_fit": 0.76,
      "final_score": 1.81
    },
    {
      "action": "PLACE_SETTLEMENT(17)",
      "policy_score": 0.28,
      "tactical_score": 0.62,
      "strategy_fit": 0.55,
      "final_score": 1.58
    }
  ]
}
```

This makes the planner tunable. If performance improves, the logs show why. If performance gets worse, the logs reveal whether it is overvaluing roads, ports, robber blocks, dev cards, or another proxy.

## Implementation Steps

1. Add env-aware action dispatch.
   - In eval loops, call `agent.act_with_env(env, obs, mask)` if available.
   - Fall back to `agent.act(obs, mask)` for existing agents.

2. Add `SearchPolicyAgent`.
   - Keep `PolicyAgent` unchanged.
   - Add a wrapper that can use env state, model top-K, and planner scoring.

3. Add `StrategyState` and `StrategyEvaluator`.
   - Start heuristic-based using existing strategy metrics.
   - Keep it soft and blend updates over time.

4. Extend `search.py`.
   - Keep existing setup and robber ranking.
   - Add policy-aware candidate rescoring.
   - Add strategy-fit scoring.

5. Add CLI flags.
   - `--use-search-policy`
   - `--search-setup`
   - `--search-robber`
   - `--search-top-k`
   - `--strategy-guidance`
   - optional rollout/value weights

6. Add trace/debug output.
   - Log policy argmax, chosen action, search candidates, score components, and strategy vector.

7. Evaluate conservatively.
   - Compare baseline `PolicyAgent` vs `SearchPolicyAgent`.
   - Use the same checkpoint and fixed seeds.
   - Run seat-balanced evals against random, heuristic, and checkpoint opponents.

## Recommended First Scope

Start with:

- setup settlement search
- setup road search
- robber hex/victim search
- heuristic strategy vector
- traceable score breakdowns

Do not start with:

- separate specialist policies
- hard strategy routing
- full-game MCTS
- trade planning across all branches

## Success Metrics

Track:

- mean win rate
- strict win rate
- worst-seat strict win rate
- opening settlement percentile
- useful road rate
- robber target quality
- truncation rate
- policy override rate
- action score breakdowns

The planner is successful only if it improves actual game outcomes, not just heuristic sub-metrics.

## Main Risks

- The strategy vector may overcommit to a stale opening plan.
- Heuristic strategy-fit scores may reward proxy behavior that does not win.
- Search may slow evaluation too much.
- More planner weights means more tuning surface.

Mitigations:

- keep everything behind explicit CLI flags
- use soft vectors, not hard strategy labels
- blend updates slowly
- log every searched decision
- compare against baseline on fixed seed suites

## Long-Term Direction

If the heuristic strategy vector works, train a small learned `StrategyEvaluator` later. It can predict strategy vectors or action-value bonuses from game states, using search results, self-play outcomes, and expert heuristics as supervision.

The end state is a compact hierarchy:

```text
policy model proposes actions
strategy state gives coherence
planner handles critical tactical choices
trace logs explain decisions
```
