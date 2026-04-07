# Catan RL Context Handoff (Apr 7)

## Operator Checklist
- Restarting from current league line:
  - `run_league.sh` currently points at `phase_aware_v5/cycle_2/branch_A/best_checkpoint.pt`
  - `phase_aware_v5/cycle_2/branch_B/best_checkpoint.pt`
  - `phase_aware_v5/cycle_2/branch_C/best_checkpoint.pt`
- If bootstrapping a fresh incompatible trade line again:
  - run `bash rbootstrap.sh`
  - verify each `trade_draft_boot_s4101`, `trade_draft_boot_s4102`, `trade_draft_boot_s4103` directory has a reasonable `best_checkpoint.pt`
  - only then point `run_league.sh` at those checkpoints
- Before trusting a checkpoint:
  - check `strict_score`
  - inspect a `trace_game.py` run for setup quality, knight usage, trade accept behavior, and whether games end cleanly
- If league runtime gets too heavy:
  - `league3.sh` now samples a smaller frozen subset per branch
  - adjust `FROZEN_SAMPLE_COUNT` if needed

## Related Files
- Audit/status notes: `audit_data.md`
- Previous context handoff: `Context_Mar20.md`
- Trade-draft plan and completion record: `/Users/gregb/.cursor/plans/iterative_trade_bargaining_70f297c6.plan.md`
- Current handoff: `Context_Apr7.md`

## Current State
- The project is now on a new `phase_aware_v5` line built around the iterative trade-draft protocol.
- Recent behavior improved materially versus the earlier degenerate `phase_aware_v4` / incompatible-checkpoint phase:
  - strong setup traces,
  - real player-to-player trading,
  - meaningful knight usage,
  - cleaner game endings,
  - better `strict_score` alignment with actual trace quality.
- A strong recent example:
  - `phase_aware_v5/cycle_6/branch_C/checkpoints/policy_u10.pt`
  - reported `strict_score = 0.85625`
  - trace looked strategically coherent: good openings, many knight plays, meaningful player trades, and a non-degenerate finish.

## Major Recent Changes

### 1) New iterative player-trade protocol
- Replaced one-shot single-resource `PROPOSE_TRADE(give_r, give_n, want_r, want_n)` with a short bargaining protocol.
- Current trade flow:
  - `TRADE_ADD_GIVE`
  - `TRADE_ADD_WANT`
  - `TRADE_REMOVE_GIVE`
  - `TRADE_REMOVE_WANT`
  - `PROPOSE_TRADE`
  - `CANCEL_TRADE`
  - then `ACCEPT_TRADE` / `REJECT_TRADE` once submitted
- V1 constraints:
  - max 2 total cards per side,
  - no responder counteroffers,
  - public/table-wide submitted offers remain as before,
  - bank trading unchanged.
- Core files changed:
  - `catan_rl/actions.py`
  - `catan_rl/constants.py`
  - `catan_rl/env.py`
  - `catan_rl/encoding.py`
  - `catan_rl/strategy_metrics.py`
  - `catan_rl/training/ppo.py`
  - `catan_rl/training/self_play.py`
  - `catan_rl/bots.py`
  - `scripts/trace_game.py`
  - `scripts/eval_checkpoint.py`
  - `scripts/train_schedule.py`

### 2) Fresh incompatible training line
- This trade-system redesign changed:
  - action catalog,
  - phase layout (`TRADE_DRAFT` added),
  - action dimension,
  - semantics of player-trade actions.
- As a result, old checkpoints are not safe continuation points for the new line.
- A fresh BC warmstart bootstrap path was created in `rbootstrap.sh`.
- `scripts/train_schedule.py` now records:
  - `trade_protocol_version = iterative_bundle_v1`
  - `obs_dim`
  - `action_dim`

### 3) Bootstrap and league script changes
- `rbootstrap.sh`
  - now launches 3 fresh BC warmstart runs for the new trade-draft line,
  - outputs `trade_draft_boot_s4101`, `trade_draft_boot_s4102`, `trade_draft_boot_s4103`.
- `run_league.sh`
  - was updated several times during tuning,
  - currently points to:
    - `phase_aware_v5/cycle_2/branch_A/best_checkpoint.pt`
    - `phase_aware_v5/cycle_2/branch_B/best_checkpoint.pt`
    - `phase_aware_v5/cycle_2/branch_C/best_checkpoint.pt`
  - current output family:
    - `BASE_OUT=phase_aware_v5`
- `league3.sh`
  - still maintains a rolling frozen pool,
  - but now samples only a smaller subset per branch run:
    - `FROZEN_SAMPLE_COUNT=3` by default,
    - `DYNAMIC_FROZEN_MAX=15` retained as the larger master pool cap,
    - if anchors are used and `GUARANTEE_ANCHOR_EXPOSURE=1`, the sampled subset tries to include at least one anchor.

## Important Bug Fixes

### 1) Trade-draft loop bug
- Initial trade-draft implementation allowed policies to spend many steps in `TRADE_DRAFT` without consuming the normal turn-action budget.
- Result:
  - many eval games hit `eval_max_steps`,
  - `strict_score` collapsed to `0.0` despite nonzero raw win-rate.
- Fix:
  - `TRADE_DRAFT` actions now consume the same per-turn action budget as `MAIN`,
  - once that budget is exhausted in `TRADE_DRAFT`, only `PROPOSE_TRADE` or `CANCEL_TRADE` remain legal.
- This was the key fix that stopped trade-draft churn from destroying episode termination.

### 2) Missing import in trade shaping
- `RESOURCE_COSTS` was missing from `catan_rl/training/self_play.py` after the new draft-bundle scoring helpers were added.
- Fixed by importing `RESOURCE_COSTS`.

### 3) Frozen-pool performance fix
- `league3.sh` no longer forces each branch to load the full frozen pool every cycle.
- It now samples a smaller subset to reduce startup cost, memory, and rollout overhead.

## Current Tuning Direction

### Setup
- Setup quality improved substantially and is no longer the dominant failure mode.
- Current `run_league.sh` still has moderate setup guidance:
  - `REWARD_SHAPING_SETUP_SETTLEMENT=0.14`
  - `REWARD_SHAPING_SETUP_TOP6_BONUS=0.18`
  - `REWARD_SHAPING_SETUP_ONE_HEX_PENALTY=0.20`
  - `SETUP_SELECTION_INFLUENCE_SETTLEMENT_PROB=0.18`
  - `SETUP_SELECTION_INFLUENCE_SETTLEMENT_ROUND2_PROB=0.24`
  - `SETUP_SELECTION_INFLUENCE_ROAD_PROB=0.05`
  - `REWARD_SHAPING_SETUP_ROAD_FRONTIER=0.03`

### Trading
- Trading is no longer dead:
  - traces showed meaningful proposals and accepts,
  - a strong trace had `96` proposals and `51` accepts.
- But responder behavior is still somewhat too conservative:
  - positive responder opportunities are still sometimes rejected,
  - accepted trades still tend to favor the proposer more than the accepter.
- Current `run_league.sh` trade encouragement:
  - `MIX_TRADE_FRIENDLY=0.28`
  - `REWARD_SHAPING_TRADE_OFFER_COUNTERPARTY_VALUE=0.14`
  - `REWARD_SHAPING_TRADE_ACCEPT_VALUE=0.06`
  - `FORCE_TRADE_BOOTSTRAP_PROB=0.10`
  - `FORCE_TRADE_BOOTSTRAP_UPDATES=8`

### Knights / Robber
- Knight play improved a lot and is no longer being aggressively forced.
- Current settings:
  - `REWARD_SHAPING_PLAY_KNIGHT=0.05`
  - `REWARD_SHAPING_PLAY_KNIGHT_WHEN_BLOCKED=0.08`
  - `REWARD_SHAPING_KNIGHT_UNBLOCK_PENALTY=0.003`
  - `FORCE_KNIGHT_BOOTSTRAP_PROB=0.08`
  - `FORCE_KNIGHT_BOOTSTRAP_UPDATES=6`
- Robber behavior improved, especially for public-leader targeting.
- Remaining limitation:
  - robber/rob-player logic is still much more explicit about public VP + hidden-dev threat than about road-race or army-race threat.

### Roads
- Road choices improved but remain only moderately purposeful.
- Recent tuning increased:
  - `REWARD_SHAPING_MAIN_ROAD_PURPOSE=0.06`
- Interpretation:
  - roads are often reasonable,
  - but still less sharp than setup/trade/knight behavior,
  - especially for best-site frontier planning and road-race-aware denial.

## Recent Behavioral Conclusions
- The strongest new evidence is that `strict_score` and trace quality started matching again on good `phase_aware_v5` checkpoints.
- The most important positive outcomes:
  - games finish normally,
  - setup is strong,
  - knights are played,
  - player trades happen,
  - bank trades no longer dominate everything.
- Important remaining strategic gaps:
  - responder-side under-acceptance of positive trades,
  - road planning still only moderate,
  - robber target logic is not yet explicitly road-race / army-race aware,
  - latent threat modeling from dev-card possibilities is still fairly coarse.

## Current League / Runtime Notes
- `run_league.sh` is a moving target and was actively tuned during these changes.
- `run_league.sh` currently points to cycle-2 `phase_aware_v5` best checkpoints, not the original bootstrap checkpoints.
- `league3.sh` intentionally:
  - keeps a rolling frozen pool,
  - adds champions over time,
  - and now samples `3` frozen checkpoints per branch run by default.

## Validation Done
- New trade-draft env/tests:
  - `tests/test_trade_discard.py`
  - `tests/test_action_mask.py`
- Focused validations were run repeatedly during the changes:
  - `pytest tests/test_trade_discard.py tests/test_action_mask.py`
  - `python -m py_compile ...` on touched Python files
  - small runtime smoke tests for env + bots

## Pointers To Recent Markdown Context / Status
- Project audit and older structural notes:
  - `audit_data.md`
- Prior context handoff before this trade-draft line:
  - `Context_Mar20.md`
- Recent iterative trade plan and completion status:
  - `/Users/gregb/.cursor/plans/iterative_trade_bargaining_70f297c6.plan.md`
- This handoff:
  - `Context_Apr7.md`

## Recommended Immediate Next Steps
- Continue monitoring `phase_aware_v5` checkpoints by both:
  - `strict_score`
  - trace quality
- Watch especially for:
  - whether trade accept rates remain healthy,
  - whether responder utility stays positive,
  - whether road usefulness continues improving,
  - whether stronger checkpoints continue to end games cleanly.
- If another strategic layer is added next, the most promising future directions are:
  - road-race / army-race-aware robber threat,
  - more predictive opponent threat modeling,
  - explicit fear of latent dev-card explosiveness (especially unplayed monopoly / road building).
