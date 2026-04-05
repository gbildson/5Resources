# Catan RL Context Handoff (Mar 20)

## Current Goal
- Improve strategic quality in self-play, especially:
  - setup settlement coordination between first/second picks,
  - purposeful road building (roads should advance settlement/port/strategy goals),
  - maintain strong baseline performance while improving model-vs-model robustness.

## Recent Functional Changes

### 1) Setup shaping improvements (`catan_rl/training/self_play.py`)
- Added stronger setup logic for:
  - pip-gap penalty vs best legal setup spot,
  - resource-weighted pip value (ore/wheat and wood/brick coherence),
  - round-1 diversity floor penalty when only 1-resource starts are chosen despite viable diverse alternatives,
  - second-pick complement logic:
    - bonus for adding missing resources and improving combined diversity,
    - penalty for overlap-only second picks unless coherent OWS specialization.

### 2) Road-purpose shaping improvements (`catan_rl/training/self_play.py`)
- Strengthened `main_phase_road_purpose_shaping(...)`:
  - roads get positive only when they improve frontier/site/port value,
  - stronger penalty for dead-end roads,
  - extra penalty when better progression actions were available (settlement/city/dev),
  - reduced penalty in active longest-road race contexts.
- `scripts/train_schedule.py` now passes action mask into this shaping to capture opportunity cost.

### 3) Setup influence controls
- Split setup override probabilities supported:
  - settlement: `SETUP_SELECTION_INFLUENCE_SETTLEMENT_PROB`
  - road: `SETUP_SELECTION_INFLUENCE_ROAD_PROB`
- Backward-compatible fallback to `SETUP_SELECTION_INFLUENCE_PROB`.

### 4) Eval script capability (`scripts/eval_checkpoint.py`)
- Added optional checkpoint-vs-checkpoint eval mode:
  - `--opponent checkpoint`
  - `--opponent-checkpoint <path>`
  - optional opponent model arch/hidden/residual args.
- Default behavior for random/heuristic/both remains unchanged.

### 5) Rules bug fix (`catan_rl/env.py`)
- Fixed road legality bug: cannot extend road through a vertex occupied by an opponent settlement/city.
- Added regression test: `tests/test_road_network.py::test_cannot_extend_road_through_opponent_settlement_vertex`.

## Runtime Errors Encountered and Fixed
- `UnboundLocalError: rule_hits` in setup shaping:
  - cause: `rule_hits.append(...)` used before initialization.
  - fix: initialize `rule_hits` before second-pick block.
- `NameError: cand_has_sheep`:
  - cause: referenced in OWS coherence check but not defined.
  - fix: define `cand_has_sheep = pips_by_res[2] > 0.0`.

## Current Training/League Notes
- `run_league.sh` currently uses aggressive influence/shaping profile, including:
  - `SETUP_SELECTION_INFLUENCE_SETTLEMENT_PROB=0.38`
  - `SETUP_SELECTION_INFLUENCE_ROAD_PROB=0.20`
  - `REWARD_SHAPING_SETUP_ROUND1_FLOOR=0.10`
  - `REWARD_SHAPING_MAIN_ROAD_PURPOSE=0.08` (recent suggestion: test 0.12 then 0.16 if needed)
  - `REWARD_SHAPING_OWS_ACTIONS=0.07`
  - `REWARD_SHAPING_SETUP_SETTLEMENT=0.21` (may be raised for stronger second-pick complement signal).

## Eval Interpretation Reminder
- `strict_score` in league reports is baseline eval (random/heuristic aggregate), not model-vs-model.
- For checkpoint-vs-checkpoint:
  - use `eval_checkpoint.py --opponent checkpoint ...`,
  - prefer `games_per_seat >= 32` and multiple seeds,
  - compare mean strict + worst-seat strict + seat stddev.

## Operational Caveat
- `league3.sh` default CKPTs still point to old bootstrap paths unless overridden.
- Ensure `CKPT_A/CKPT_B/CKPT_C` in `run_league.sh` reference existing files before starting new cycles.

## Recommended Next Steps
- Run 2-4 cycles with fixed hyperparams except targeted ablation:
  - raise `REWARD_SHAPING_MAIN_ROAD_PURPOSE` first (0.12),
  - if still weak, raise to 0.16,
  - optionally increase `REWARD_SHAPING_SETUP_SETTLEMENT` (0.26-0.30) for stronger second-pick complement effect.
- Validate with:
  - trace-game qualitative review (roads with clear purpose),
  - model-vs-model seat-robustness eval against older anchor checkpoints.

