#!/bin/bash

.venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint phase_aware_v5/cycle_88/branch_B/checkpoints/policy_u10.pt \
  --opponent checkpoint \
  --opponent-checkpoint phase_aware_v5/cycle_73/branch_C/checkpoints/policy_u10.pt \
  --games-per-seat 32 \
  --seed 128 \
  --max-steps 3000 \
  --max-main-actions-per-turn 12 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 3 \
  --out phase_aware_v5/cycle_88/branch_B/eval_vs_73C.json

