# Alpha Plan: Policy-Guided Search for Catan

## Core Idea

Use the trained neural policy as a **move generator**, then apply a small Catan-specific planner to choose among the best candidates.

Instead of:

```text
obs + mask -> model logits -> argmax action
```

use:

```text
obs + mask -> model logits -> top K actions
current env state -> clone/simulate/rescore candidates
top K + search score + optional value estimate -> final action
```

This keeps PPO and the existing model architecture intact. The model supplies broad pattern recognition; the planner handles precise tactical decisions.

## Why This Helps

Catan has a few decision points where one choice can strongly shape the whole game:

- initial settlement placement
- initial road direction
- robber hex and victim choice
- city vs settlement vs dev card decisions
- trade acceptance and bank trade choices
- road races and port access

Pure PPO has to learn these from noisy game outcomes. A planner can inject structured Catan knowledge at the moment it matters.

## How It Fits the Existing System

The repo already has most of the pieces:

- `catan_rl/search.py` ranks setup and robber actions and can clone env state for shallow rollout.
- `catan_rl/training/wrappers.py` has `PolicyAgent`, which can be extended with a planned/search policy wrapper.
- `catan_rl/eval.py` and `scripts/eval_checkpoint.py` can pass the live env to agents that support env-aware action selection.
- `scripts/trace_game.py` can show when search overrides the model's direct policy choice.
- `catan_rl/training/bc.py` already uses search as an offline teacher; this plan also uses search online at inference time.

## Proposed First Version

Start with a conservative `SearchPolicyAgent`:

- wraps an existing trained model
- behaves like `PolicyAgent` by default
- only invokes search when enabled by CLI flags
- initially searches only setup and robber phases
- falls back to normal masked argmax when search is not applicable

Suggested flags:

```text
--use-search-policy
--search-setup
--search-robber
--search-top-k 5
--search-setup-rollout-steps 0
--search-robber-rollout-steps 0
```

## Implementation Sketch

1. Add env-aware action dispatch:

```text
if agent has act_with_env:
    action = agent.act_with_env(env, obs, mask)
else:
    action = agent.act(obs, mask)
```

2. Add `SearchPolicyAgent` next to `PolicyAgent`.

3. Extend `search.py` to combine:

- model top-K policy candidates
- existing setup/robber heuristic scores
- optional shallow rollout score
- optional model value estimate after applying the candidate

4. Add evaluation and trace flags so baseline policy vs search policy can be compared with the same checkpoint and seeds.

## Expected Benefits

- better opening settlements
- better initial road direction
- stronger robber targeting
- fewer obvious tactical mistakes
- improved checkpoint strength without retraining
- clearer debugging through traceable search decisions

## Main Risk

The planner can overfit to heuristic proxies. It may improve setup or robber metrics while hurting actual win rate.

Mitigation:

- keep it behind explicit flags
- compare against plain `PolicyAgent`
- use fixed seed suites
- track win rate, strict win rate, opening percentile, road usefulness, robber behavior, and truncation rate

## Recommended Path

Implement setup + robber search first. If that improves seat-balanced evaluation, extend the same wrapper pattern to main-phase build choices and trade decisions.
