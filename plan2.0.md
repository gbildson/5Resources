# Catan Plan 2.0

This document lays out a practical roadmap for building a stronger Catan bot than the current flat-observation PPO setup. The core thesis is:

- Keep the current fast simulator, action catalog, and league training loop.
- Stop expecting a small flat MLP plus scalar shaping to discover all of Catan's long-horizon structure.
- Move toward a hybrid system that combines RL, strategy-aware features, topology-aware encoding, and a small amount of selective search on the highest-leverage decisions.

The goal is not a "pure RL" demo. The goal is the strongest bot that can be realistically built in this repo.

## Current Baseline

Current strengths:

- Stable deterministic simulator and topology.
- Full legal action masking.
- PPO self-play with BC warmstart and league training.
- Rich engineered flat features.
- Useful trace/eval tooling.

Current bottlenecks:

- Policy/value model is still a small flat MLP.
- One monolithic policy must learn setup, main phase, robber logic, trade, dev-card timing, and tactical placement all at once.
- Important spatial patterns are only present indirectly through hand-coded features.
- League training can drift into degenerate local metas.
- Reward shaping helps, but too much shaping can easily distort play.

## Design Goals

The next-generation system should:

- Understand board topology directly.
- Maintain coherent strategic intent over many turns.
- Learn phase-specific behavior without forcing one head to do everything.
- Use coded strategic priors where they add leverage.
- Stay trainable inside the current repo without a full rewrite of the simulator.
- Preserve fast iteration and debuggability.

## Guiding Principles

1. Keep game rules and action legality deterministic and explicit.
2. Use RL for adaptation and long-horizon optimization, not for rediscovering obvious board heuristics from scratch.
3. Use coded strategy signals as soft guidance, not as rigid hard rules.
4. Separate high-impact subproblems instead of solving everything with a single policy head.
5. Add complexity in stages so each gain can be measured cleanly.

## Target Architecture

The best long-term architecture for this repo is:

- A shared encoder with topology awareness.
- Phase-aware policy heads.
- A strategy-latent or mixture-of-experts layer.
- Auxiliary prediction heads for strategic concepts.
- Optional selective search for setup and robber actions.

In plain terms:

- The encoder should understand hexes, vertices, edges, and players as related entities.
- The policy should not treat `PLACE_SETTLEMENT`, `ROB_PLAYER`, `PROPOSE_TRADE`, and `END_TURN` as if they were equally similar choices.
- The model should carry a representation of "what strategy am I pursuing in this game?"
- The training loop should reward winning first, but should use shaped and auxiliary signals to learn faster and more robustly.

## Recommended Roadmap

## Phase 0: Stabilize And Instrument The Current Stack

This phase should happen before major architecture work.

Deliverables:

- Keep the current conservative PPO defaults for league runs.
- Keep BC warmstart available for new observation/model variants.
- Keep setup, knight, and robber shaping small and traceable.
- Add more diagnostics around opening quality, dev-card timing, road spam, and seat robustness.

Concrete work:

- Track opening-quality metrics from the first two placements.
- Track dev-card usage timing, not just final counts.
- Track settlement-site quality of actual chosen placements against top legal alternatives.
- Track road placements that reduce shortest path to next settlement versus roads that do not.
- Track per-seat strict score and worst-seat score as first-class league metrics.

Reason:

- Do not build a more complex model while blind to whether it is actually solving the right problems.

## Phase 1: Replace The Tiny Flat MLP With A Better Flat Baseline

Before building a graph encoder, upgrade the baseline network.

Deliverables:

- Deeper residual MLP baseline.
- Shared trunk plus phase-specific heads.
- Backward-compatible training loop where possible.

Recommended model changes:

- Replace `256 -> 256` with a deeper residual MLP, for example `512 -> 512 -> 512` with residual blocks, layer norm, and dropout only if needed.
- Keep a shared backbone.
- Add separate policy heads for:
- setup placement,
- main-phase action choice,
- robber and steal actions,
- trade actions,
- dev-card play actions.
- Keep one value head initially.

Reason:

- This is the cheapest change that can improve capacity and reduce destructive interference across phases.

Success criteria:

- Better setup choices in trace.
- Fewer degenerate action biases.
- Higher strict score and better worst-seat score without heavier shaping.

## Phase 2: Add Topology-Aware Encoding

This is the most important architectural upgrade.

Representation:

- Hex nodes: terrain, token, robber, blocked production, ownership around the hex.
- Vertex nodes: building type, owner, local pip profile, port access, legality/contestation signals.
- Edge nodes: road owner, road pressure, connectivity value, expansion role.
- Player nodes: resources, dev cards, VP, army/road race status, trade posture.

Graph edges:

- vertex <-> vertex adjacency,
- vertex <-> hex membership,
- edge <-> vertex incidence,
- optional player <-> owned piece links.

Model options:

- Message-passing GNN with typed edges.
- Entity transformer with typed embeddings and attention masks.

Recommendation:

- Start with a typed message-passing encoder because the board topology is fixed and small.

Outputs:

- Pooled global embedding for value.
- Per-vertex embeddings for settlement/city action scoring.
- Per-edge embeddings for road action scoring.
- Per-player and per-hex embeddings for robber/trade/dev-card context.

Why this matters:

- Catan strategy is fundamentally relational.
- Flat features can approximate topology, but graph structure should learn road races, blocking, contestation, and expansion fronts more naturally.

## Phase 3: Introduce Strategy Latents

This is the biggest conceptual upgrade.

The model should maintain an internal notion of strategic direction, such as:

- OWS / city engine,
- wood-brick expansion,
- army/dev pressure,
- port conversion engine,
- hybrid flexible line,
- denial / anti-leader posture.

Implementation options:

- Small latent strategy head predicting a soft mixture over strategy archetypes.
- Mixture-of-experts where different policy experts specialize in different strategic styles.
- A recurrent "plan token" updated once per turn or once per phase.

Recommendation:

- Start with a soft strategy-mixture head because it is simpler than full hierarchical recurrent planning.

Use of the strategy latent:

- Condition policy logits.
- Condition auxiliary heads.
- Use for trace/debug output so strategy drift is visible.

How to supervise it:

- Initially weakly supervise from heuristics and board-derived labels.
- Later let RL fine-tune it.

## Phase 4: Add Auxiliary Strategic Losses

Do not make the network learn everything only through PPO reward.

Useful auxiliary targets:

- Setup settlement quality score.
- Setup road follow-through quality.
- City upgrade gain estimate.
- Settlement opportunity count.
- Roads needed to reach top candidate sites.
- Robber block value.
- Probability that a dev card should be played this turn.
- Largest army race pressure.
- Longest road race pressure.
- Expected turns to settlement/city/dev with current production.
- Trade acceptance value.

How to train:

- Compute labels from state evaluators or heuristic policies.
- Add auxiliary heads off the shared encoder.
- Weight these losses modestly so PPO/value still dominate.

Reason:

- This teaches high-level concepts directly and reduces the burden on sparse policy gradients.

## Phase 5: Add Selective Search For Critical Decisions

Do not attempt full MCTS over the entire 4-player stochastic game.

Instead, apply search only to high-leverage, bounded branches:

- Setup settlement placement.
- Setup road placement.
- Robber move and rob target.
- Perhaps top-k build sequences within a turn.

Possible methods:

- One-step evaluator over legal candidates.
- Limited rollout search with heuristic opponents.
- Beam search over top tactical actions.

Recommendation:

- Start with setup and robber only.

Important pattern:

- Use search to improve decision quality.
- Distill the improved choices back into the fast network through BC-style data.

## Phase 6: Rethink Trade As A Separate Problem

Trade is unlike the rest of the action space and deserves special handling.

Problems with current approach:

- Trade quality is hard to learn from terminal reward.
- Trade spam can dominate behavior.
- Offer construction, offer acceptance, and bank trade logic are qualitatively different.

Recommendation:

- Split trade into a separate subpolicy or submodule.
- Predict trade value explicitly.
- Use acceptance models and template scoring before exposing the full action surface.
- Keep guided trade action mode as the default training surface for now.

## Phase 7: Improve League Training For Diversity And Stability

Keep the league, but reduce meta collapse.

Recommended changes:

- Continue keeping a frozen historical pool.
- Do not always reset weakest branch from current champion forever.
- Periodically reset weakest branch from a strong historical checkpoint instead.
- Maintain a curated anchor set of stylistically different strong checkpoints.
- Promote using strict score plus worst-seat guardrails.
- Run larger eval samples before promotion decisions when possible.

Potential policy:

- 50% reset weakest from champion.
- 50% reset weakest from sampled strong historical anchor.

Reason:

- This keeps improvement pressure while preserving strategic diversity.

## Phase 8: Upgrade Evaluation

The bot should be judged by more than one scalar.

Primary metrics:

- Mean strict score vs random and heuristic.
- Worst-seat strict score.
- Seat variance.
- Opening quality score.
- Dev-card timing quality.
- Trade efficiency.
- Robber anti-leader quality.

Secondary metrics:

- Win rate vs curated historical anchors.
- Win rate vs strategy-specific opponents.
- Trace-based qualitative review.

Needed eval additions:

- Strategy bucket evaluation.
- Board-shape evaluation on low-ore, high-sheep, strong-port, and road-heavy boards.
- Mirror diversity checks under stochastic sampling.

## Recommended Implementation Order

This is the actual order the coding agents should follow.

1. Strengthen instrumentation and evaluation first.
2. Replace the tiny MLP with a stronger flat baseline plus phase-aware heads.
3. Add auxiliary losses using existing engineered evaluators.
4. Add a strategy-mixture latent on top of the improved flat model.
5. Add topology-aware encoder.
6. Add selective search for setup and robber.
7. Upgrade league reset/promotion logic for diversity.
8. Revisit trade as its own module.

## What Not To Do Yet

- Do not jump straight to full MCTS.
- Do not redesign the simulator unless required by architecture.
- Do not add many new reward terms at once.
- Do not remove the current engineered features when moving to topology-aware encoding; keep them as a hybrid input at first.
- Do not overfit to argmax traces; always test with sampled-policy traces too.

## Minimal Viable Next Version

If only one major version can be built next, it should be:

- Residual flat encoder,
- phase-aware heads,
- strategy-mixture latent,
- auxiliary strategic losses,
- current engineered features retained,
- current league training retained with better diagnostics.

Why this version first:

- It is much easier than a full graph model.
- It directly attacks the largest current weakness: one small undifferentiated policy trying to average incompatible strategic behaviors.
- It should produce useful gains even before topology-aware modeling lands.

## Ideal Long-Term Version

The strongest eventual version likely looks like this:

- topology-aware graph/entity encoder,
- phase-aware action heads,
- strategy latent or mixture-of-experts,
- auxiliary strategic targets,
- selective search for setup and robber,
- league training with diverse historical anchors,
- BC distillation from search-enhanced subpolicies.

## Suggested Repository Workstreams

Workstream A: model architecture

- New encoder module.
- New phase-aware policy heads.
- Strategy latent.
- Auxiliary heads.

Workstream B: data and labels

- Strategic evaluator utilities.
- Setup/robber labels.
- Trade value labels.
- Search-distillation datasets.

Workstream C: training

- PPO integration for new heads.
- Auxiliary loss weighting.
- BC pretraining updates.
- League diversity controls.

Workstream D: evaluation

- Opening metrics.
- Seat robustness metrics.
- Historical anchor benchmark suite.
- Better trace summaries.

## Final Recommendation

The best path is not "more PPO alone."

The best path is:

- use RL,
- keep engineered strategy signals,
- introduce phase structure,
- introduce strategic latent structure,
- then add topology-aware encoding,
- and use selective search only where its cost is justified.

That is the most realistic path to a genuinely strong Catan bot in this codebase.
