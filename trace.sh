
.venv/bin/python scripts/trace_game.py \
  --checkpoint phase_aware_v5/cycle_109/branch_C/checkpoints/policy_u30.pt \
  --seed 151 \
  --model-players 0,1,2,3 \
  --max-main-actions-per-turn 10 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 3 \
  --max-steps 50000 \
  --sample-policy \
  --policy-temperature 1.0 \
  --show-policy-strategy-mixture \
  --show-policy-aux-predictions \
  --show-final-inspect-board \
  --show-dev-card-summary \
  --show-final-board \
  --show-final-inspect-topology

  #--trade-action-mode guided \
  #--max-player-trade-proposals-per-turn 1 \


#  --checkpoint phase_aware_v5/cycle_88/branch_B/checkpoints/policy_u10.pt \
#  --checkpoint phase_aware_v5/cycle_86/branch_B/checkpoints/policy_u10.pt \
#  --checkpoint phase_aware_v5/cycle_81/branch_B/checkpoints/policy_u10.pt \
#  --checkpoint phase_aware_v5/cycle_73/branch_C/checkpoints/policy_u10.pt \
#  --checkpoint phase_aware_v5/cycle_62/branch_B/checkpoints/policy_u30.pt \
#  --checkpoint phase_aware_v5/cycle_63/branch_C/checkpoints/policy_u30.pt \
#  --checkpoint phase_aware_v5/cycle_6/branch_C/checkpoints/policy_u10.pt \
