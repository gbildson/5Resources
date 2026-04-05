# Data/Input Audit

## Goal
- Ensure high-value decision data is available to the model at the point of action selection, not only in shaping/override/eval.

## Audit Plan (Succinct)
- Inventory observation and engineered features.
- Map each feature to where it is consumed (policy trunk, entity heads, shaping, override, eval).
- Mark phase-critical decisions (`setup1`, `setup2`, `main settlement`, `road`, `trade`) and check if inputs are location-aware.
- Flag "orphan intelligence" (good scoring logic not directly in model input/head path).
- Prioritize fixes by impact and implementation effort.

## Tracking Format
- **Feature/Logic**
- **Where computed**
- **Where consumed**
- **Decision impact**
- **Gap**
- **Recommended fix**
- **Status**

## Input Data Model (Canonical)
- Policy input at decision time is:
  - `observation` (flat encoded vector from `encode_observation`)
  - `action_mask` (legal action mask from env)
- For graph/entity architectures, `observation` is transformed into an entity grid/tensors.

### A) Flat Observation Model (Source of Truth)
- Source: `catan_rl/encoding.py` (`encode_observation`, `observation_slices`)
- Major blocks:
  - Hex-level: terrain one-hot, number, pip count, robber one-hot
  - Vertex-level: building-owner one-hot
  - Edge-level: owner one-hot
  - Ports/topology: port type, port vertex mask
  - Player/public state: resources, dev info, pieces, ports, road/army, VP
  - Turn/phase/trade state
  - Engineered features tail

### B) Entity Grid Model (Derived View for Graph Models)
- Source: `catan_rl/graph_features.py` (`graph_observation_to_entities`)
- Shapes:
  - `hex_feat`: `[B, 19, 9]`
  - `vertex_feat`: `[B, 54, 10]`
  - `edge_feat`: `[B, 72, 5]`
  - `player_feat`: `[B, 4, 68]`
  - `global_feat`: `[B, G]` (phase/turn/trade + engineered tail)

#### Entity feature contents
- Hex (`9`): terrain one-hot (`6`) + number (`1`) + pip (`1`) + robber (`1`)
- Vertex (`10`): building-owner one-hot (`1 + 2*players = 9`) + port-vertex-mask (`1`)
- Edge (`5`): edge-owner one-hot (`1 + players = 5`)
- Player (`68`): rotated player/public summary blocks (flattened then expanded across 4 players)
- Global (`G`): phase/turn/trade blocks + engineered features

### C) Scope Notes
- The entity grid list above is the full model input view for graph architectures, but it is derived from the flat observation model.
- `action_mask` is separate and always required for legal action selection.
- Shaping scores, override heuristics, and eval-only metrics are not direct model inputs unless explicitly encoded.

## Initial Results (Confirmed)
- **Setup settlement ranking (`setup_settlement_score`, `setup_choice_metrics`)**
  - Where computed: `catan_rl/strategy_metrics.py`
  - Where consumed: setup override, shaping, eval/trace metrics
  - Gap: not a direct supervised target for settlement logits
  - Status: Open

- **Engineered `setup_settlement_quality_top3`**
  - Where computed: `catan_rl/encoding.py`
  - Where consumed: observation -> model input
  - Gap: scalar top-3 values only; no vertex identity/location mapping
  - Status: Open

- **Graph entity settlement scoring (current)**
  - Where computed/used: `catan_rl/training/ppo.py` (`settlement_head` over `vertex_h`)
  - Gap: single shared settlement head across setup/main contexts
  - Status: Open

- **Phase-aware graph hybrid (current)**
  - Where used: `graph_entity_phase_aware_hybrid` has phase-specific global heads
  - Gap: settlement entity head remains shared (no setup round-specific settlement heads)
  - Status: Open

- **Setup influence/shaping pressure**
  - Where used: `catan_rl/training/self_play.py`
  - Result: active and firing (`setup_selection_override_count` non-zero in cycle reports)
  - Gap: behavior still weak in traces/eval, indicating representation/routing limitations
  - Status: Confirmed

## Next Audit Steps
- Add full feature-to-consumer table for all major observation blocks.
- Verify per-phase location-aware inputs for settlement decisions.
- Decide first structural fix:
  - per-vertex setup quality channel in graph vertex features, and/or
  - setup round-specific settlement heads.

## Implemented (Current Branch)
- `graph_entity_phase_aware_hybrid` now adds settlement-context vertex channels in-model:
  - `quality`, `top6_flag`, `top6_rank_signal` (per vertex).
- Settlement action scoring for this architecture now uses three pathways:
  - setup round-1 settlement head,
  - setup round-2 settlement head,
  - main settlement head.
- Settlement-context features are phase-conditioned:
  - setup round-1: global setup quality,
  - setup round-2: complement-aware setup quality,
  - main: reachable/connected-biased quality.
- Added trade accept net-readiness delta engineered features (global):
  - `delta_ready_road`, `delta_ready_settlement`, `delta_ready_city`, `delta_ready_dev`,
  - `delta_ready_count` (normalized net change in ready actions).
- Added trace visibility for trade-response turns:
  - prints per-step readiness deltas when phase is `TRADE_PROPOSED`.
- Added road-context edge channels in `graph_entity_phase_aware_hybrid`:
  - `settle_unlock_gain` (new connected settlement-site unlock signal),
  - `best_settle_delta` (improvement in best reachable settlement-quality proxy),
  - `best_port_delta` (newly reached port access proxy),
  - `dead_end_risk` (frontier branch risk proxy),
  - `connected_gain` (network expansion signal).
- These are injected into edge embeddings before graph message passing and then consumed by the road action head.
- Added compact race/threat/tempo engineered proxies (global):
  - `army_race_margin_best_opp`,
  - `army_lock_pressure`,
  - `public_lead_pressure`,
  - `robber_exposure_proxy`,
  - `city_tempo_proxy`,
  - `dev_synergy_proxy`.
- These are intended as guidance features for city vs settlement vs dev prioritization (soft policy context, not hard rules).
- Added explicit in-model settlement port utility integration (`graph_entity_phase_aware_hybrid`):
  - uses candidate reach to each port (0/1/2-road neighborhood),
  - weights port desirability by generic vs specific and production match,
  - applies as a small bonus to setup/main settlement context scoring.
- Added compact robber target fusion inputs (global engineered):
  - per-opponent `robber_target_value` combining blocked-production, card-pressure, and public-threat proxies.
- Exposed in engineered summaries as `robber_target_value_opp3` for trace/debug inspection.
- Added compact port convertibility input (global engineered):
  - `port_conversion_readiness` estimating how actionable current hand + owned port rates are for near-term builds.
- Added compact dev timing urgency input (global engineered):
  - `dev_timing_pressure` blending army-race pressure, robber-blocked self production, OWS profile, dev affordability, and held knight stock.
