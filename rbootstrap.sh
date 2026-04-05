#!/usr/bin/env bash
set -euo pipefail

# Bootstrap-safe mixture: disable frozen opponents while feature-width mismatch exists.
# (Old frozen checkpoints can fail to load into the new observation shape.)

for SEED in 4101 4102 4103; do
  .venv/bin/python -u scripts/train_schedule.py \
    --out-dir "boot_v3_s${SEED}" \
    --seed "${SEED}" \
    --model-arch graph_entity_phase_aware_hybrid \
    --model-hidden 192 \
    --model-residual-blocks 3 \
    --bc-warmstart \
    --bc-steps 12000 \
    --bc-epochs 2 \
    --bc-batch-size 256 \
    --bc-lr 0.001 \
    --total-updates 16 \
    --report-every 4 \
    --rollout-steps 768 \
    --games-per-seat 16 \
    --max-episode-steps 500 \
    --eval-max-steps 2200 \
    --ppo-lr 5e-5 \
    --ppo-ent-coef 0.02 \
    --ppo-epochs 1 \
    --setup-phase-loss-weight 3.0 \
    --use-opponent-mixture \
    --opponent-seat-count 2 \
    --mix-heuristic-prob 0.75 \
    --mix-random-prob 0.20 \
    --mix-meta-prob 0.05 \
    --mix-frozen-prob 0.00 \
    --trade-action-mode guided \
    --max-player-trade-proposals-per-turn 1 \
    --reward-shaping-setup-settlement 0.35 \
    --reward-shaping-setup-round1-floor 0.20 \
    --reward-shaping-setup-road-frontier 0.03 \
    --setup-selection-influence-settlement-prob 0.85 \
    --setup-selection-influence-settlement-round2-prob 1.00 \
    --setup-selection-influence-road-prob 0.05 \
    --setup-selection-influence-prob 0.00 \
    --reward-shaping-main-road-purpose 0.05 \
    --reward-shaping-ows-actions 0.02 \
    --reward-shaping-rob-leader 0.02 \
    --reward-shaping-play-knight-when-blocked 0.02 \
    --reward-shaping-knight-unblock-penalty 0.003 \
    > "boot_v2_s${SEED}.log" 2>&1 &
done
wait
