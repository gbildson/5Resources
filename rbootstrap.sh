#!/usr/bin/env bash
set -euo pipefail

# Fresh BC bootstrap for the iterative trade-draft action line.
# These runs are intentionally checkpoint-incompatible with older trade-action families.

for SEED in 4101 4102 4103; do
  .venv/bin/python -u scripts/train_schedule.py \
    --out-dir "trade_draft_boot_s${SEED}" \
    --seed "${SEED}" \
    --model-arch graph_entity_phase_aware_hybrid \
    --model-hidden 192 \
    --model-residual-blocks 3 \
    --bc-warmstart \
    --bc-steps 24000 \
    --bc-epochs 3 \
    --bc-batch-size 256 \
    --bc-lr 0.0008 \
    --bc-aux-coef 0.01 \
    --total-updates 20 \
    --report-every 10 \
    --rollout-steps 768 \
    --games-per-seat 12 \
    --max-episode-steps 600 \
    --eval-max-steps 2600 \
    --ppo-lr 5e-5 \
    --ppo-ent-coef 0.02 \
    --ppo-epochs 1 \
    --setup-phase-loss-weight 1.4 \
    --use-opponent-mixture \
    --opponent-seat-count 2 \
    --mix-heuristic-prob 0.55 \
    --mix-random-prob 0.15 \
    --mix-meta-prob 0.10 \
    --mix-trade-friendly-prob 0.20 \
    --mix-frozen-prob 0.00 \
    --trade-action-mode guided \
    --max-player-trade-proposals-per-turn 2 \
    --reward-shaping-setup-settlement 0.10 \
    --reward-shaping-setup-top6-bonus 0.12 \
    --reward-shaping-setup-one-hex-penalty 0.12 \
    --reward-shaping-setup-round1-floor 0.05 \
    --reward-shaping-setup-road-frontier 0.03 \
    --setup-selection-influence-settlement-prob 0.12 \
    --setup-selection-influence-settlement-round2-prob 0.16 \
    --setup-selection-influence-road-prob 0.05 \
    --setup-selection-influence-prob 0.00 \
    --reward-shaping-main-road-purpose 0.03 \
    --reward-shaping-ows-actions 0.01 \
    --reward-shaping-trade-offer-value 0.02 \
    --reward-shaping-trade-offer-counterparty-value 0.10 \
    --reward-shaping-trade-accept-value 0.04 \
    --reward-shaping-rob-leader 0.02 \
    --reward-shaping-play-knight-when-blocked 0.02 \
    --reward-shaping-knight-unblock-penalty 0.003 \
    --force-trade-bootstrap-prob 0.15 \
    --force-trade-bootstrap-updates 8 \
    > "trade_draft_boot_s${SEED}.log" 2>&1 &
done
wait
