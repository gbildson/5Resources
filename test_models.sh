#!/bin/bash

.venv/bin/python scripts/eval_checkpoint.py \
  --checkpoint phase_aware_v2/cycle_8/branch_A/checkpoints/policy_u10.pt \
  --opponent checkpoint \
  --opponent-checkpoint phase_aware_v2/cycle_13/branch_A/checkpoints/policy_u20.pt \
  --games-per-seat 32 \
  --seed 128 \
  --max-steps 3600 \
  --max-main-actions-per-turn 10 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 1 \
  --out phase_aware_v2/cycle_8/branch_A/eval_vs_cycle13A.json

#  --checkpoint league_runs_hybrid_v1/cycle_101/branch_A/checkpoints/policy_u30.pt \
