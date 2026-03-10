.venv/bin/python scripts/trace_game.py \
  --checkpoint  artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt \
  --seed 555 \
  --model-players 0,1,2,3 \
  --max-steps 50000 \
  --show-final-inspect-board \
  --show-final-inspect-topology


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
