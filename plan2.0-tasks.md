# Plan 2.0 Tasks

This file converts `plan2.0.md` into an execution checklist for coding agents.

The intent is:

- keep tasks concrete,
- preserve implementation order,
- make each task independently reviewable,
- define what "done" means.

## Stage 0: Instrumentation And Evaluation

### Task 0.1: Opening quality metrics

Add evaluation metrics for the first two settlement placements and their paired setup roads.

Implement:

- compute placement quality for actual chosen setup settlements,
- compute rank of the chosen site among legal candidates,
- compute road follow-through quality after each setup road,
- emit metrics into trace and eval outputs.

Done when:

- `trace_game.py` can show actual setup quality versus top legal alternatives,
- eval outputs include opening-quality summary fields,
- league comparisons can see whether a policy wins with bad openings or actually opens well.

### Task 0.2: Dev-card timing metrics

Add metrics for when dev cards are played, not just whether they were ever held.

Implement:

- first-turn-played timing for knights,
- count of turns holding playable knight before playing,
- year-of-plenty and monopoly usage counts when available,
- final summary in trace output.

Done when:

- traces show timing-based dev-card usage behavior,
- evaluation can distinguish hoarding from timely use.

### Task 0.3: Road spam and frontier value metrics

Measure whether roads improve settlement reach or are low-value spam.

Implement:

- per-road delta in reachable settlement opportunities,
- roads-to-next-best-site metric,
- contested-site approach metric,
- summary counts in trace and eval.

Done when:

- a road can be labeled as strategically useful or low-value,
- evaluation can detect when "more roads" is not actually helping.

### Task 0.4: First-class seat robustness metrics

Promote worst-seat and seat variance to top-level evaluation outputs.

Implement:

- explicit `worst_seat_strict`,
- `seat_strict_stddev`,
- promotion summary that prints seat spread alongside `strict_score`.

Done when:

- promotion decisions can be inspected with mean and worst-seat signals together.

## Stage 1: Stronger Flat Baseline

### Task 1.1: Residual MLP policy/value model

Replace the tiny two-layer MLP with a deeper residual baseline.

Implement:

- new model class in `catan_rl/training/ppo.py` or a new module,
- width around `512`,
- 3-5 residual blocks,
- layer norm or equivalent stabilization if helpful,
- preserved masked-action PPO API.

Done when:

- training scripts run without interface changes beyond model selection,
- old PPO loop can train the new model,
- parameter count and architecture are documented.

### Task 1.2: Phase-aware policy heads

Keep one shared backbone but split policy output by phase family.

Implement:

- separate heads for setup,
- main-turn actions,
- robber/steal actions,
- trade actions,
- dev-card timing actions,
- routing logic based on phase.

Done when:

- model can score the full action catalog using specialized heads,
- traces confirm phase-correct head routing,
- no regression in legal masking.

### Task 1.3: Backward-compatible checkpoint loading plan

Avoid confusion during architecture migration.

Implement:

- explicit model-version metadata in checkpoints,
- clear failure message on incompatible checkpoint load,
- optional model selector CLI arg.

Done when:

- it is obvious which checkpoints belong to which architecture family.

## Stage 2: Auxiliary Strategic Targets

### Task 2.1: Strategic evaluator utilities

Create reusable state evaluators for strategic concepts.

Implement:

- setup site quality,
- city upgrade quality,
- road frontier gain,
- robber block quality,
- anti-leader rob quality,
- longest-road race pressure,
- largest-army race pressure,
- turns-to-next-build estimates.

Done when:

- these evaluators live in a reusable module and can be called from training, eval, and trace.

### Task 2.2: Auxiliary heads

Attach prediction heads for strategic concepts to the shared backbone.

Implement:

- scalar regression heads for quality estimates,
- small classification heads where appropriate,
- loss weighting config.

Done when:

- training logs include auxiliary losses,
- PPO training remains stable with these losses enabled.

### Task 2.3: BC plus auxiliary warmstart

Make BC teach more than next-action imitation.

Implement:

- dataset fields for auxiliary labels,
- pretraining that optimizes policy plus auxiliary losses,
- reporting of auxiliary warmstart metrics.

Done when:

- BC warmstart can initialize both action behavior and strategic concept predictions.

## Stage 3: Strategy Latent

### Task 3.1: Define strategy archetypes

Translate `strats.md` into model-facing archetypes.

Initial archetypes:

- OWS / city engine,
- road expansion,
- port conversion,
- dev-army pressure,
- hybrid flexible line,
- city plus VP line.

Done when:

- each archetype has a short definition,
- heuristic labeling rules exist for weak supervision.

### Task 3.2: Strategy-mixture head

Add a latent strategy distribution conditioned on state.

Implement:

- softmax over strategy archetypes,
- conditioning of policy logits on strategy embedding,
- optional trace output of current strategy mixture.

Done when:

- traces can show "what strategy the model seems to think it is pursuing",
- the policy can vary behavior based on strategy context.

### Task 3.3: Weak supervision for strategy latent

Bootstrap strategy coherence with heuristic labels.

Implement:

- label generation from openings, resource profile, and board conditions,
- optional loss on strategy head during BC and early RL.

Done when:

- strategy latent is not purely random in early training.

## Stage 4: Topology-Aware Encoder

### Task 4.1: Graph/entity state representation

Build a structured view of the board from current state.

Represent:

- hex entities,
- vertex entities,
- edge entities,
- player entities.

Include:

- typed relations,
- fixed indexing,
- batched tensor construction.

Done when:

- one function can convert current state into entity tensors plus adjacency/edge metadata.

### Task 4.2: Message-passing encoder

Implement the first topology-aware encoder.

Recommended:

- typed message passing over fixed board relations,
- 2-4 layers initially,
- pooled graph embedding plus per-entity embeddings.

Done when:

- encoder outputs:
- global embedding,
- per-vertex embedding,
- per-edge embedding,
- per-hex embedding,
- per-player embedding.

### Task 4.3: Entity-based action scoring

Use entity embeddings directly for action families.

Implement:

- settlement/city action scoring from vertex embeddings,
- road scoring from edge embeddings,
- robber scoring from hex embeddings,
- rob-player and trade scoring from player embeddings plus global context,
- fallback pooled logits for fully global actions.

Done when:

- full action catalog can be scored from entity-aware features.

### Task 4.4: Hybrid encoder mode

Do not immediately discard engineered flat features.

Implement:

- hybrid model that concatenates graph summary with existing engineered/global features,
- ablation flag for graph-only versus hybrid.

Done when:

- performance can be compared cleanly across flat-only, graph-only, and hybrid modes.

## Stage 5: Search Where It Pays

### Task 5.1: Setup search module

Add selective search for setup settlement and road decisions.

Implement:

- evaluate all legal setup placements,
- optional shallow rollout or heuristic opponent response model,
- top-k candidate ranking,
- distillation dataset export.

Done when:

- setup search can produce better-than-policy labels on demand.

### Task 5.2: Robber search module

Add targeted search for robber move plus rob target.

Implement:

- evaluate block value,
- evaluate steal target value,
- optionally model near-term retaliation risk,
- distillation dataset export.

Done when:

- robber choices can be improved independently of the main policy.

### Task 5.3: Search distillation pipeline

Teach the fast policy to imitate search-improved decisions.

Implement:

- offline dataset generation,
- BC-style distillation pass,
- optional periodic refresh from current checkpoints.

Done when:

- search-enhanced data can improve policy without search at inference time.

## Stage 6: Trade As A Specialized Subsystem

### Task 6.1: Trade value model

Make trade quality explicit.

Implement:

- offer value estimator,
- acceptance value estimator,
- bank trade utility estimator.

Done when:

- trade actions can be ranked by predicted utility instead of only terminal reward.

### Task 6.2: Guided trade policy split

Separate trade generation from trade acceptance.

Implement:

- distinct scoring path for proposing trades,
- distinct scoring path for accepting/rejecting,
- compatibility with current guided trade action mode.

Done when:

- trade behavior becomes inspectable and tunable as its own subsystem.

## Stage 7: League Stability And Diversity

### Task 7.1: Diverse reset policy

Reduce meta collapse from always cloning the champion into the weakest slot.

Implement:

- champion reset probability,
- historical-anchor reset probability,
- configurable anchor pool,
- logging of reset source each cycle.

Done when:

- weakest branch resets are no longer always from current champion.

### Task 7.2: Strong historical anchor set

Curate a pool of checkpoints by style and quality.

Implement:

- helper manifest for anchor checkpoints,
- optional manual tags like `road`, `ows`, `port`, `army`, `hybrid`,
- league option to guarantee anchor exposure.

Done when:

- training regularly faces diverse strong opponents, not just recent drifted ones.

### Task 7.3: Promotion gate upgrade

Make promotion less noisy.

Implement:

- require strong `strict_score`,
- check `worst_seat_strict`,
- optionally check opening-quality threshold,
- optionally increase eval sample count for borderline promotions.

Done when:

- promoted models are less likely to be seat-skewed or opening-bad.

## Stage 8: Benchmark Suite

### Task 8.1: Historical checkpoint benchmark

Create a fixed benchmark against strong historical models.

Implement:

- script to evaluate current model against curated checkpoint pool,
- aggregate scoreboard output.

Done when:

- new architectures can be judged against the same reference set every time.

### Task 8.2: Board-style benchmark

Measure robustness across board conditions.

Implement:

- evaluate over sampled seeds grouped by low ore, high sheep, strong ports, road-heavy, etc.,
- report per-group performance.

Done when:

- the model can be tested for strategic adaptability, not just average strength.

### Task 8.3: Sampled-trace regression suite

Avoid over-reading greedy traces.

Implement:

- small repeated `trace_game.py --sample-policy` suite,
- saved summaries for setup, trade, robber, dev-card timing.

Done when:

- behavior regressions can be spotted before long league runs are wasted.

## Immediate Priority Order

If only the first few tasks are started now, do them in this order:

1. Task 0.1 opening quality metrics
2. Task 0.3 road/frontier metrics
3. Task 1.1 residual MLP baseline
4. Task 1.2 phase-aware policy heads
5. Task 2.1 strategic evaluator utilities
6. Task 2.2 auxiliary heads
7. Task 3.1 strategy archetypes
8. Task 3.2 strategy-mixture head
9. Task 4.1 graph/entity representation
10. Task 4.2 message-passing encoder

## Suggested Ownership Split

If multiple coding agents are working in parallel:

- Agent A: instrumentation, eval, trace metrics
- Agent B: model architecture and PPO integration
- Agent C: strategic evaluators, labels, and auxiliary losses
- Agent D: league/eval/benchmark infrastructure

## Acceptance Gates

Use these gates so coding agents can make objective ship/no-ship calls during Plan 2.0.

### Baseline assumption

- Current known best `strict_score` is approximately `0.45`.

### Quantitative gates

- Initial success gate: `strict_score >= 0.46`.
- Stronger success gate: `strict_score >= 0.48`.
- Stability gate: pass the selected gate on at least two independent eval seed batches.
- Robustness gate: no meaningful regression in `worst_seat_strict` relative to current baseline.

### Evaluation protocol

- Do not declare success from one run or one seed.
- Use the same eval settings when comparing baseline versus candidate.
- For borderline improvements, increase eval games-per-seat before promotion.

### Qualitative gates

Even if scalar gates pass, review sampled traces before declaring stage success.

Required checks:

- opening quality does not regress,
- road placements improve reach/contestation rather than spam,
- dev-card timing is reasonable (reduced hoarding),
- robber/rob target behavior remains anti-leader and not self-sabotaging,
- trade behavior remains sane under sampled-policy traces.

### Promotion guidance

- If quantitative gate passes but qualitative checks fail, classify result as provisional and iterate.
- If qualitative checks pass but quantitative gate fails, keep as exploratory and do not promote as new primary baseline.

## Experiment Result Template

Use this block after each experiment so comparisons stay consistent.

```md
### Experiment: <name>

- Date: <YYYY-MM-DD>
- Owner: <agent/person>
- Stage/Task: <e.g., Stage 1 / Task 1.2>
- Code/Config summary: <what changed>
- Checkpoint path: <path>

#### Eval setup
- Games per seat: <n>
- Opponents: <random+heuristic / other>
- Seeds: <list or seed range>
- Eval command: `<command>`

#### Quantitative results
- strict_score: <value>
- worst_seat_strict: <value>
- seat_strict_stddev: <value>
- baseline_strict_score: <value>
- delta_strict_score: <value>
- gate_result: <pass/fail against 0.46 or 0.48>
- stability_result: <pass/fail across seed batches>

#### Qualitative trace review
- opening quality: <ok/regressed + notes>
- roads/frontier behavior: <ok/regressed + notes>
- dev-card timing: <ok/regressed + notes>
- robber targeting: <ok/regressed + notes>
- trade sanity: <ok/regressed + notes>

#### Decision
- Promote as primary baseline: <yes/no>
- If no, classify as: <provisional/exploratory/regression>
- Next action: <one concrete follow-up>
```

## Definition Of Success

Plan 2.0 is working if the new system shows:

- better setup decisions in trace,
- fewer degenerate road and dev-card behaviors,
- higher worst-seat score,
- less regression across league cycles,
- competitive performance against curated historical anchors,
- more interpretable strategic identity from the model.
