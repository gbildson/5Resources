
.venv/bin/python scripts/trace_game.py \
  --checkpoint league_runs/cycle_7/branch_B/checkpoints/policy_u30.pt \
  --seed 61 \
  --model-players 0,1,2,3 \
  --max-main-actions-per-turn 10 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 1 \
  --max-steps 50000 \
  --sample-policy \
  --policy-temperature 1.0 \
  --show-final-inspect-board \
  --show-dev-card-summary \
  --show-final-inspect-topology

#  --checkpoint league_runs/cycle_4/branch_C/checkpoints/policy_u10.pt \
#  --checkpoint league_runs/cycle_1/branch_B/checkpoints/policy_u20.pt \
#  --checkpoint league_runs/cycle_22/branch_B/checkpoints/policy_u10.pt \
#  --checkpoint league_runs/cycle_12/branch_C/checkpoints/policy_u10.pt \
#  --checkpoint league_runs/cycle_9/branch_A/checkpoints/policy_u20.pt \
#  --checkpoint league_runs/cycle_8/branch_B/checkpoints/policy_u10.pt \
#  --checkpoint league_runs/cycle_7/branch_B/checkpoints/policy_u10.pt \
#  --checkpoint league_runs/cycle_7/branch_C/checkpoints/policy_u10.pt \
#  --checkpoint league_runs/cycle_6/branch_C/checkpoints/policy_u20.pt \
#  --checkpoint league_runs/cycle_5/branch_B/checkpoints/policy_u20.pt \
#  --checkpoint league_runs/cycle_2/branch_C/checkpoints/policy_u20.pt \
#  --checkpoint league_runs/cycle_7/branch_A/checkpoints/policy_u10.pt \
#  --checkpoint enhm3_seed23107/checkpoints/policy_u20.pt \
#  --checkpoint enh2_seed22103/checkpoints/policy_u20.pt \
#.venv/bin/python scripts/trace_game.py \
#  --checkpoint  artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt \
#  --seed 555 \
#  --model-players 0,1,2,3 \
#  --max-steps 50000 \
#  --show-final-inspect-board \
#  --show-final-inspect-topology
#

#  --disable-player-trade \
#  --checkpoint  artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt \
#  --checkpoint  artifacts_schedule_realboard_v1/best_checkpoint.pt \
#.venv/bin/python scripts/trace_game.py \
#  --checkpoint artifacts_schedule_mix_v1/best_checkpoint.pt \
#  --seed 42 \
#  --model-players 0 \
#  --max-steps 10000 \
#  --disable-player-trade \
#  --show-final-board-pycatan

#source .venv/bin/activate 
#python scripts/trace_game.py \
#  --checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt \
#  --model-players 0,1,2,3 \
#  --seed 42 \
#  --disable-player-trade \
#  --max-steps 10000


#  --checkpoint artifacts_schedule_v9_heuristic_lift/best_checkpoint.pt \
