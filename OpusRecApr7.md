# Catan RL — Evaluation & Recommendations (Apr 7)

## Overall Assessment

This is a well-structured, serious RL project that follows proven patterns from competitive multi-agent game AI (reminiscent of AlphaStar's league training and OpenAI Five's self-play PPO). The engineering is thoughtful, the iteration loop is tight, and the recent `phase_aware_v5` results — real trading, strong openings, coherent knight play, clean game endings — are genuinely impressive for a 4-player stochastic imperfect-information game. The codebase is ~9k lines of focused Python, which is lean for the complexity of the problem.

That said, there are architectural choices that may define the ceiling of the system.

---

## What's Working Well

### 1) League training loop

Rolling frozen pool with champion promotion, branch diversity (A/B/C), sampled frozen subsets to control compute — this is the right structure for multi-agent self-play. It prevents the single-agent "forgetting" problem and maintains strategy diversity. The `strict_score` metric gating checkpoint promotion is a good safeguard.

### 2) BC warmstart + PPO self-play

Cold-start PPO in a game this complex (1113 actions, 1313 obs dims) would waste enormous compute discovering basic legality patterns. Imitation from a heuristic policy is pragmatic and well-justified. The clear separation (`--bc-warmstart` vs `--init-checkpoint`) shows good engineering discipline.

### 3) Iterative trade-draft protocol

Replacing the one-shot trade action with `TRADE_ADD_GIVE / TRADE_ADD_WANT / PROPOSE / CANCEL` is a much better action decomposition for RL. It converts a combinatorial single-step decision into a sequential one the policy can learn incrementally. The budget-exhaustion fix (draft actions consume the turn budget) was critical and correctly identified.

### 4) Phase-aware architecture with per-entity heads

Catan's decision space is fundamentally different across phases — settlement placement, road building, robber targeting, and trade response are structurally distinct problems. Routing through phase-specific heads and using entity-level (vertex/edge/hex) scoring for spatial decisions is architecturally sound. The split settlement heads for setup-round-1 vs setup-round-2 vs main is a good example of encoding game structure directly.

### 5) Graph message-passing layer

Hex-vertex-edge topology is the natural graph for Catan. Propagating information through `GraphMessageLayer` over this structure lets the model reason about adjacency, production neighborhoods, and expansion paths without the flat observation needing to explicitly encode all pairwise relationships.

---

## Concerns and Risks

### 1) Heavy hand-engineered features may cap the ceiling

The 87-dimensional engineered feature tail is essentially a hand-coded strategic evaluator: `army_race_margin`, `city_tempo_proxy`, `dev_synergy_proxy`, `port_conversion_readiness`, `robber_target_value`, build readiness estimates, etc. This is pragmatically valuable — it accelerates learning by giving the policy access to strategic abstractions it would take billions of frames to discover. But it creates two risks:

- **The policy may overfit to the encoded heuristic understanding.** If `dev_timing_pressure` encodes a particular human theory about when to buy dev cards, the policy will learn that theory rather than potentially discovering a better one.
- **Feature maintenance becomes a bottleneck.** Every strategic insight must be manually encoded, tested, and tuned. The `audit_data.md` already identifies "orphan intelligence" — scoring logic that exists in metrics/shaping but isn't in the model's input path.

For truly high-level play, the long-term direction should be reducing reliance on these features. But in the medium term, they're probably the right trade-off.

### 2) Flat action space is very large (1113 discrete actions)

Every parameterized action — all 54 settlement vertices, 72 road edges, 19 robber hexes, all discard 5-tuples, all bank trade variants, all iterative trade edits — lives in a single flat catalog. The policy head emits 1113 logits, masked to legality.

This works, and masking is the standard approach. But the discard templates alone (all 5-tuples summing to 1..7) consume a huge chunk of the action space. In any given state, only a tiny fraction of actions are legal, so most policy capacity is spent on dimensions that are always masked off. A hierarchical action decomposition (choose action type, then choose parameters) could improve sample efficiency, though it adds implementation complexity.

### 3) Reward shaping is extensive and fragile

There are 20+ shaping coefficients being manually tuned: `REWARD_SHAPING_SETUP_SETTLEMENT`, `REWARD_SHAPING_SETUP_TOP6_BONUS`, `REWARD_SHAPING_SETUP_ONE_HEX_PENALTY`, `REWARD_SHAPING_PLAY_KNIGHT`, `REWARD_SHAPING_PLAY_KNIGHT_WHEN_BLOCKED`, `MIX_TRADE_FRIENDLY`, `REWARD_SHAPING_TRADE_OFFER_COUNTERPARTY_VALUE`, `REWARD_SHAPING_TRADE_ACCEPT_VALUE`, `FORCE_TRADE_BOOTSTRAP_PROB`, `REWARD_SHAPING_MAIN_ROAD_PURPOSE`, etc.

This is the classic RL engineering trap: each coefficient fixes one behavioral issue but introduces interaction effects with every other coefficient. The `strict_score` collapsing to 0.0 from the trade-draft loop bug is a good example — well-intentioned shaping can have catastrophic emergent effects. The project manages this carefully (gradual bootstrap fade, trace inspection), but the tuning surface is high-dimensional and manually navigated.

Some of these shaping signals could potentially be replaced by **auxiliary prediction objectives** (predict opponent VP progress, predict resource income, predict time-to-build) which shape the representation without directly distorting the reward signal.

### 4) No opponent modeling or theory of mind

The observation includes opponent public state (VP, knights played, resource counts, road/army achievements) but the model has no mechanism to predict opponent intentions or strategies. For high-level Catan:

- Robber placement should model who is closest to winning and *how* (road race? army race? dev card explosion?).
- Trade decisions should model what the opponent will build with the resources they receive.
- Road/settlement choices should model denial value against opponent expansion paths.

The `strategy_archetypes.py` and strategy prediction head are a step in this direction, but they predict the *learner's* archetype, not the opponents'. An opponent modeling head (predict opponent's next build, or predict opponent's hidden dev cards from observed behavior) could be high-impact.

### 5) No search at play time

The system is pure policy inference — no MCTS, no rollout-based planning. In deterministic two-player games (Go, Chess), search is transformative. In 4-player stochastic imperfect-information Catan, tree search is much harder, but some form of limited lookahead (even 1-ply "what happens if I do X?") could improve decisions at critical moments, especially for trade evaluation and end-game sequencing.

This is likely a later-iteration concern, but worth noting as a ceiling.

### 6) Stale `train_self_play` code path

`train_self_play` in `self_play.py` still hardcodes `PolicyValueNet` (MLP only). All real training goes through `train_schedule.py` + `build_policy_value_net`, but having a stale code path that instantiates the wrong architecture is a maintenance risk.

---

## Strategic Recommendations (Prioritized)

### Short-term

**A) Fix trade responder conservatism.**
The context identifies this as the top behavioral gap. Consider a dedicated trade evaluation auxiliary head that predicts build-unlock probability from accepting vs rejecting. This gives the responder model a direct signal about what the trade enables, rather than relying solely on shaped rewards.

**B) Add opponent strategy prediction.**
An auxiliary head predicting each opponent's likely next build (or estimated turns-to-win) would give the model useful context for robber targeting, trade decisions, and road denial without adding much complexity. Unlike the current strategy archetype head (which predicts the learner's own archetype), this would model what *opponents* are doing.

### Medium-term

**C) Gradually reduce engineered features.**
As the model gets stronger, try ablating engineered features to see which ones the model can learn from raw state. Features the model can replicate from raw input are adding complexity without value. Features it can't replicate are the ones worth keeping. This also reveals which features are actually load-bearing versus noise.

**D) Consider hierarchical action decomposition.**
Even a simple two-level hierarchy (choose action kind → choose parameters) could improve sample efficiency and make the policy more interpretable. The current phase-specific heads are already a partial version of this; formalizing it could help especially for the large discard and trade sub-spaces.

**E) Replace some reward shaping with auxiliary prediction objectives.**
Auxiliary heads that predict game-relevant quantities (opponent VP trajectory, resource income rate, turns-to-next-build) shape the representation without directly distorting the policy gradient. This can achieve similar behavioral effects to reward shaping but with less fragile coefficient tuning. The project already has `AUX_LABEL_KEYS` infrastructure — extending it is straightforward.

### Long-term

**F) Investigate limited search at play time.**
Even a simple 1-ply policy rollout ("simulate my top-3 actions for one turn") at play time could improve critical decisions without requiring full MCTS infrastructure. For Catan, the high branching factor and stochasticity make deep search impractical, but shallow search with the learned value function as evaluation could still help at decision points where the policy is uncertain.

**G) Deeper opponent modeling.**
Beyond predicting opponent builds, consider encoding opponent behavioral patterns over the game history (trade acceptance rates, resource hoarding patterns, robber targeting tendencies) as recurrent or attention-based features. This is what separates strong human Catan players from average ones — reading opponents.

---

## Comparison to State-of-the-Art Game AI

| System | Similarity | Key Difference |
|--------|-----------|----------------|
| AlphaZero (Go/Chess) | Self-play, neural policy+value | AlphaZero uses MCTS; Catan's stochasticity and 4-player structure make tree search much harder |
| AlphaStar (StarCraft) | League training, BC warmstart, population diversity | AlphaStar had massive compute scale and used imitation from human replays rather than heuristic bots |
| OpenAI Five (Dota 2) | PPO self-play, reward shaping, curriculum | OpenAI Five operated at much larger scale with team coordination; similar shaping challenges |
| Pluribus (Poker) | Imperfect information, opponent modeling | Pluribus used CFR (counterfactual regret) rather than policy gradient; different algorithmic family but similar information-theoretic challenges |

The Catan project's approach is most similar to a scaled-down AlphaStar: league-based self-play with BC warmstart, progressive curriculum, and extensive reward shaping. The main gaps relative to SOTA are scale (compute), opponent modeling, and search.

---

---

## Additional Findings from Code Review (Apr 7 addendum)

### H) GAE bootstrapping across players is approximate

`collect_rollout` interleaves transitions from all 4 players into a single buffer, then runs GAE over the whole sequence. At step `t`, `values[t+1]` may belong to a completely different player's observation. Because all 4 seats share the same network and `encode_observation` rotates to the current player's perspective, this approximation works "well enough" — the value function learns "how good is this state for whoever is acting." But it's still conceptually wrong for single-trajectory GAE: the next-step bootstrap is for a different agent's state.

**Fix options (medium-term):**
- Track per-player trajectory indices and compute GAE per-player. This avoids cross-player bootstrap contamination.
- Alternatively, store a player-id per step and zero out `nextnonterminal` when the next step belongs to a different player. Simpler but discards inter-turn temporal structure.

This is a known approximation in multi-agent shared-weight PPO. Fixing it would likely improve value function accuracy and reduce advantage estimation noise.

### I) No gradient clipping in PPO

`PPOTrainer.update` calls `loss.backward()` and `self.optim.step()` with no `torch.nn.utils.clip_grad_norm_`. This is standard practice in PPO and important with the complex multi-source loss (policy + value + entropy + aux + strategy). A single large shaping reward can produce outsized gradients.

**Fix:** Add `torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)` before `self.optim.step()`. This is a one-line fix with significant stability benefits.

### J) `action_mask()` is recomputed from scratch every step

`action_mask()` iterates over all 1113 catalog entries, calling `_is_action_legal` for each. Many of those checks involve inner loops (checking all vertices for settlement distance rule, all edges for road connectivity, etc.). This runs twice per `step()` (once inside `step` for validation, once to build the new mask).

At 2048 rollout steps per update, this is ~4096 full mask computations per rollout. The vast majority of the action space is illegal at any given time; computing legality for all 1113 actions when typically <20 are legal is wasteful.

**Fix options:**
- Phase-dispatch: only check actions relevant to the current phase (most actions are phase-gated as the first check anyway).
- Cache and incrementally update the mask: most actions don't change legality between steps. After a road placement, only road and settlement legality near the new road changes.

### K) Shared backbone for policy and value creates gradient conflicts

All architectures use a single shared backbone for both policy and value heads. This is simple but can lead to conflicting gradient directions — the value function wants representations useful for state evaluation, while the policy wants representations useful for action selection.

**Fix options:**
- Add a separate value-specific residual block (or 2) between the shared backbone and `value_head`. This gives the value function its own representational capacity while still benefiting from shared low-level features.
- Full split (separate backbone for value) is more expensive but eliminates conflicts entirely.

### L) No learning rate schedule

The PPO trainer uses a fixed learning rate throughout training. Linear decay or cosine annealing is standard practice and typically improves final performance, especially in long training runs. The entropy schedule (`--ppo-ent-coef-start/--end`) already exists in `train_schedule.py` and works well; adding the same for learning rate would be straightforward.

### M) Longest road DFS recomputes from scratch every step

`_road_lengths()` runs DFS from every vertex of every player's road network on every call to `_update_vp_and_achievements()` (i.e., every `step()`). In late-game states with 15+ roads per player, this is expensive.

**Fix:** Incrementally update road lengths only when a road is placed. Store the current longest-road length and only recompute for the player who placed a road (and potentially the player whose longest road was broken by a settlement placement on their path).

### N) Winner can technically be detected on the wrong player's turn

`_check_winner` returns the first player with `actual_vp >= WIN_VP` after any action. In real Catan, you can only win on your own turn. VP changes from longest road / largest army shifts could technically trigger a win on another player's turn (e.g., if player A builds a road that steals longest road from player B, and that gives player A 10 VP, player A should win — but if player B had 10 VP from some other route and loses longest road, they shouldn't "lose" until they fail to win on their own turn). This is a minor rule fidelity issue but could matter in tight endgames.

### O) No seat-position awareness in observation

`encode_observation` rotates everything so the current player is always index 0. This is correct for shared-weight self-play. But it means the model has no explicit signal about absolute seat position. In Catan, seat order matters — player 0 has first-mover advantage in setup (but last pick in round 2), and overall turn order affects trading leverage and race dynamics.

**Fix:** Add a small seat-position feature (e.g., one-hot or scalar `absolute_seat / 3`). This lets the policy learn seat-specific strategies (e.g., player 0 should prioritize differently in round-1 setup vs player 3).

### P) `_compute_engineered_features` is expensive and called every step

This function does: BFS for frontier costs (O(V+E)), full vertex iteration for expansion potential (4 players × O(V)), settlement quality scores for all open vertices, road gain scores for all candidate roads, port proximity BFS for each vertex, etc. At 2048 steps per rollout, this is a meaningful chunk of wall-clock time.

**Fix options:**
- Cache board-topology-dependent computations (frontier costs, expansion potential) and invalidate only on board-changing actions (settlement, road, city placement).
- Precompute vertex quality scores once per turn rather than per-step (they only change when the board changes, not when dice are rolled or trades happen).

### Q) Population-based training (PBT) for automated hyperparameter tuning

The league system trains branches A/B/C with fixed hyperparameters. The 20+ shaping coefficients are manually tuned. PBT would:
- Run branches with perturbed hyperparameters
- Periodically copy weights from the best-performing branch
- Mutate hyperparameters of underperforming branches

This automates the shaping coefficient search that is currently done by hand. The league infrastructure (frozen pool, `strict_score` metric) is already 80% of what PBT needs.

### R) Discard action space is needlessly large

The discard templates enumerate all 5-tuples (wood, brick, sheep, wheat, ore) summing to each possible discard count (1 through 7). This is a combinatorial explosion that consumes a large chunk of the 1113-action catalog. In practice, discard decisions are relatively simple (keep what you need, discard surplus).

**Alternative:** Decompose discard into sequential single-card-discard actions. Instead of choosing a full discard template in one step, repeatedly choose "discard 1 card of resource X" until the discard obligation is met. This dramatically reduces the action space for discards and makes the decision more learnable.

### S) Trade response order leaks information

When a trade is proposed, responders are polled sequentially in player order. Each responder sees previous responses before deciding. This is realistic to Catan but creates a strategic asymmetry: later responders benefit from observing earlier responses. The first acceptor wins the trade, so later responders can free-ride on early rejections.

This is probably correct behavior to keep, but worth noting that it creates positional advantage in trade response that the model may or may not learn to exploit. Simultaneous sealed-bid responses would be an alternative design that encourages more independent accept/reject reasoning.

### T) No attention in graph message passing

`GraphMessageLayer` uses fixed mean-pooling for neighbor aggregation. Adding learned attention weights (GAT-style) over neighbors would let the model dynamically focus on the most relevant spatial relationships — for example, attending more to high-pip hexes when evaluating settlement sites, or attending more to opponent-adjacent edges when evaluating road denial.

This is a medium-complexity change with potentially significant representational benefit, especially for robber targeting and road planning where spatial relationships are highly context-dependent.

---

## Summary

The approach is fundamentally sound and well-executed. The main risks are the ones common to all heavily-shaped RL systems: the feature engineering and reward shaping that enable learning in the short term may constrain the ceiling in the long term. The most impactful near-term improvements are likely on the behavioral side (trade acceptance, opponent modeling) rather than architectural, since the graph entity phase-aware hybrid is already a strong inductive bias for this domain.

### Quick-win fixes (high impact, low effort)
- **I**: Add gradient clipping (1 line) — checkpoint compatible
- **L**: Add LR schedule (extend existing entropy schedule pattern) — checkpoint compatible
- **O**: Add seat-position feature to observation (small encoding change) — changes `obs_dim`; workaround: pad input weight matrix with zeros to preserve existing representations, then fine-tune

### Medium-effort improvements (significant impact)
- **A**: Trade responder aux head — checkpoint compatible (new head loads randomly, backbone intact via `strict=False`)
- **B**: Opponent strategy prediction head — checkpoint compatible (same as A)
- **H**: Per-player GAE computation — checkpoint compatible (training loop change only)
- **J**: Phase-dispatched action mask — checkpoint compatible (env optimization only)
- **K**: Separate value head layers — checkpoint compatible; policy unaffected, value re-converges through new layers (short warmup, not full retrain)
- **M**: Incremental longest-road updates — checkpoint compatible (env optimization only)
- **P**: Cached engineered features — checkpoint compatible (env optimization only)

### Larger structural improvements (high ceiling impact)
- **C**: Engineered feature ablation — **REQUIRES RETRAIN** if features are removed (changes `obs_dim`). Workaround: zero features at runtime while keeping `obs_dim` constant to test which features are load-bearing before committing to removal.
- **D**: Hierarchical action decomposition — **REQUIRES RETRAIN** (restructures action space; no checkpoint path)
- **E**: Aux prediction replacing reward shaping — checkpoint compatible (new aux heads initialize randomly; shaping removal is a hyperparameter change)
- **Q**: Population-based training — checkpoint compatible (uses existing checkpoints as seeds)
- **R**: Sequential discard decomposition — **REQUIRES RETRAIN** (changes `action_dim` and phase flow)
- **T**: Attention in graph message passing — **REQUIRES RETRAIN** (replaces core graph layer internals)

### Checkpoint compatibility summary

| Category | Recommendations |
|----------|----------------|
| Fully compatible, continue training | A, B, E, F, H, I, J, L, M, N, P, Q |
| New layers added, short re-convergence | K, O (with weight padding) |
| Requires full retrain from scratch | C (if features removed), D, R, T |
| Depends on implementation | G (aux head = compatible; recurrent/attention over history = retrain) |

---

## External Review (GPT 5.4, Apr 7)

### Corrections to this document

**Checkpoint compatibility is too optimistic in places.** In this project, anything that changes `obs_dim`, `action_dim`, or phase/action semantics has repeatedly turned into a practical retrain boundary, even when a theoretical partial-load path exists. Items like **O** (seat-position feature) and **K** (value-head restructuring) should be treated as fundamental changes requiring explicit approval and fresh validation, not casual drop-ins.

**The document overweights architectural concerns relative to the current bottleneck.** The strongest evidence says `phase_aware_v5` finally has healthy behavior: setup, knights, and trades are all alive. Jumping to hierarchy (D), search (F), or large feature ablations (C) risks disrupting a line that's working. The highest-value next work is still behavioral.

**The flat-action-space critique is directionally right but not the current priority.** The discard space and large mask are real costs, but they are not what's currently suppressing strength. The draft-trade loop bug was far more impactful than raw action-count issues.

### What the review endorsed as strong

- The overall assessment is fair and correctly recognizes that `phase_aware_v5` results are meaningful, not accidental.
- **B** (opponent modeling) is probably the best medium-term idea — the next ceiling is "who is the most dangerous opponent right now?" not "more setup shaping."
- **A** (trade responder improvement) is high leverage now that trading is alive. Direct evidence exists that responder conservatism is leaving value on the table.
- Low-risk engineering fixes (**I**, **J**, **L**, **M**, **P**) are sensible once behavior is stable.
- The addendum on road race, army race, and latent threat (from Context_Apr7.md) is consistent with what traces are showing.

### Revised priority ordering

| Priority | Recommendation | Rationale |
|----------|---------------|-----------|
| **1 — Do now** | **B**: Opponent modeling / predictive threat | Closest to actual current ceiling. Robber, trade, and denial decisions need a better forecast of who is about to win and how. |
| **2 — Do now** | **A**: Trade responder improvement | Trading is alive; direct evidence of responder conservatism leaving value on the table. No longer speculative. |
| **3 — Do soon** | **I, J, L, M, P**: Stability/perf fixes | Good "engineering tax reduction" items. Low risk, checkpoint compatible. |
| **4 — Do later** | **D, F, T**: Hierarchy, search, attention | May raise the ceiling, but exactly the sort of structural changes that can reset progress and make comparisons murky. |

### Items to defer or handle carefully

| Recommendation | Reviewer position |
|----------------|-------------------|
| **C** (engineered-feature ablation) | True in principle, risky in practice right now. Defer. |
| **K** (separate value layers) | Maybe useful, but stabilize the current strong line first. |
| **O** (seat-position feature) | Plausible, but changes the observation contract — treat as fundamental, not a quick win. |
| **H** (per-player GAE) | Intellectually appealing, but need evidence it's actually limiting the current line before touching it. |

### Verdict

OpusRecApr7.md is thoughtful and mostly good. Its strongest recommendations are opponent modeling, responder-trade improvement, and low-risk training/runtime hygiene fixes. The larger architectural refactors should be deferred until the current `phase_aware_v5` line has been exploited more fully.
