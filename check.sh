source .venv/bin/activate

python -u scripts/eval_checkpoint.py \
  --checkpoint artifacts_schedule_realboard_trade_mix_overnight_seed5110/checkpoints/policy_u20.pt \
  --games-per-seat 20 \
  --seed 880110 \
  --opponent both \
  --max-main-actions-per-turn 10 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 1 \
  --max-steps 2200 \
  --out artifacts_eval_top4_seed5110_u20.json

python -u scripts/eval_checkpoint.py \
  --checkpoint artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt \
  --games-per-seat 20 \
  --seed 880109 \
  --opponent both \
  --max-main-actions-per-turn 10 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 1 \
  --max-steps 2200 \
  --out artifacts_eval_top4_seed5109_u10.json

python -u scripts/eval_checkpoint.py \
  --checkpoint artifacts_schedule_realboard_trade_mix_overnight_seed5108/checkpoints/policy_u10.pt \
  --games-per-seat 20 \
  --seed 880108 \
  --opponent both \
  --max-main-actions-per-turn 10 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 1 \
  --max-steps 2200 \
  --out artifacts_eval_top4_seed5108_u10.json

python -u scripts/eval_checkpoint.py \
  --checkpoint artifacts_schedule_realboard_trade_mix_overnight_seed5103/checkpoints/policy_u15.pt \
  --games-per-seat 20 \
  --seed 880103 \
  --opponent both \
  --max-main-actions-per-turn 10 \
  --trade-action-mode guided \
  --max-player-trade-proposals-per-turn 1 \
  --max-steps 2200 \
  --out artifacts_eval_top4_seed5103_u15.json

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_trade_mix_v1/best_checkpoint.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.50 \
#  --mix-random-prob 0.25 \
#  --mix-frozen-prob 0.25 \
#  --mix-frozen-checkpoints "artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt,artifacts_schedule_realboard_trade_guided_probe_seed1005/best_checkpoint.pt,artifacts_schedule_realboard_trade_guided_probe_seed1002/best_checkpoint.pt,artifacts_schedule_realboard_trade_mix_v1/best_checkpoint.pt" \
#  --total-updates 20 \
#  --report-every 5 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 3031 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.000006 \
#  --ppo-ent-coef-start 0.0045 \
#  --ppo-ent-coef-end 0.0025 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_trade_mix_v2_short


#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_trade_guided_probe_seed1005/best_checkpoint.pt \
#  --total-updates 50 \
#  --report-every 10 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 1115 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.000008 \
#  --ppo-ent-coef-start 0.005 \
#  --ppo-ent-coef-end 0.0025 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_trade_guided_cont_seed1005_v1
#
#python -u scripts/eval_checkpoint.py --checkpoint artifacts_schedule_realboard_v1/checkpoints/policy_u75.pt --games-per-seat 8 --max-main-actions-per-turn 10 --disable-player-trade --max-steps 2200 --out artifacts_eval_v1_fast.json
#python -u scripts/eval_checkpoint.py --checkpoint artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt --games-per-seat 8 --max-main-actions-per-turn 10 --disable-player-trade --max-steps 2200 --out artifacts_eval_s456_fast.json
#python -u scripts/eval_checkpoint.py --checkpoint artifacts_schedule_realboard_v3_long_seed457/checkpoints/policy_u275.pt --games-per-seat 8 --max-main-actions-per-turn 10 --disable-player-trade --max-steps 2200 --out artifacts_eval_s457_fast.json
#

## Convert JSONL histories -> JSON arrays (required by eval_checkpoint.py)
#python - <<'PY'
#import json
#from pathlib import Path
#
#runs = [
#    "artifacts_schedule_realboard_v1",
#    "artifacts_schedule_realboard_v3_long_seed456",
#    "artifacts_schedule_realboard_v3_long_seed457",
#]
#for run in runs:
#    p = Path(run) / "progress_reports.jsonl"
#    out = Path(run) / "progress_reports.json"
#    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
##    out.write_text(json.dumps(rows), encoding="utf-8")
#    print(out)
#PY
#
## v1 (u75)
#python -u scripts/eval_checkpoint.py \
#  --checkpoint artifacts_schedule_realboard_v1/checkpoints/policy_u75.pt \
#  --training-history artifacts_schedule_realboard_v1/progress_reports.json \
#  --promotion-decision artifacts_schedule_realboard_v1/best_checkpoint_meta.json \
#  --games-per-seat 60 \
#  --seed 202601 \
#  --opponent random \
#  --out artifacts_eval_bakeoff_v1_u75_random.json
#
#python -u scripts/eval_checkpoint.py \
#  --checkpoint artifacts_schedule_realboard_v1/checkpoints/policy_u75.pt \
#  --training-history artifacts_schedule_realboard_v1/progress_reports.json \
#  --promotion-decision artifacts_schedule_realboard_v1/best_checkpoint_meta.json \
#  --games-per-seat 60 \
#  --seed 202611 \
##  --opponent heuristic \
#  --out artifacts_eval_bakeoff_v1_u75_heuristic.json
#
### seed456 (u1275)
#python -u scripts/eval_checkpoint.py \
#  --checkpoint artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt \
#  --training-history artifacts_schedule_realboard_v3_long_seed456/progress_reports.json \
#  --promotion-decision artifacts_schedule_realboard_v3_long_seed456/best_checkpoint_meta.json \
#  --games-per-seat 60 \
#  --seed 202602 \
##  --opponent random \
#  --out artifacts_eval_bakeoff_seed456_u1275_random.json
#
#python -u scripts/eval_checkpoint.py \
#  --checkpoint artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt \
#  --training-history artifacts_schedule_realboard_v3_long_seed456/progress_reports.json \
#  --promotion-decision artifacts_schedule_realboard_v3_long_seed456/best_checkpoint_meta.json \
#  --games-per-seat 60 \
#  --seed 202612 \
#  --opponent heuristic \
#  --out artifacts_eval_bakeoff_seed456_u1275_heuristic.json
#
## seed457 (u275)
#python -u scripts/eval_checkpoint.py \
#  --checkpoint artifacts_schedule_realboard_v3_long_seed457/checkpoints/policy_u275.pt \
#  --training-history artifacts_schedule_realboard_v3_long_seed457/progress_reports.json \
#  --promotion-decision artifacts_schedule_realboard_v3_long_seed457/best_checkpoint_meta.json \
#  --games-per-seat 60 \
#  --seed 202603 \
#  --opponent random \
#  --out artifacts_eval_bakeoff_seed457_u275_random.json
#
#python -u scripts/eval_checkpoint.py \
#  --checkpoint artifacts_schedule_realboard_v3_long_seed457/checkpoints/policy_u275.pt \
#  --training-history artifacts_schedule_realboard_v3_long_seed457/progress_reports.json \
#  --promotion-decision artifacts_schedule_realboard_v3_long_seed457/best_checkpoint_meta.json \
#  --games-per-seat 60 \
#  --seed 202613 \
#  --opponent heuristic \
#  --out artifacts_eval_bakeoff_seed457_u275_heuristic.json
#
##source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_mix_v1/best_checkpoint.pt \
#  --total-updates 1 \
#  --report-every 1 \
#  --rollout-steps 64 \
#  --games-per-seat 50 \
#  --seed 9494 \
#  --reward-shaping-vp 0.06 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 650 \
#  --eval-max-steps 2400 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.0 \
#  --ppo-ent-coef 0.01 \
#  --ppo-epochs 1 \
#  --out-dir artifacts_schedule_v9_validation

#  --init-checkpoint artifacts_schedule_v9_heuristic_lift/best_checkpoint.pt \
#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt \
#  --total-updates 1 \
#  --report-every 1 \
#  --rollout-steps 64 \
#  --games-per-seat 50 \
#  --seed 9191 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.0 \
#  --ppo-ent-coef 0.011 \
#  --ppo-epochs 1 \
#  --out-dir artifacts_schedule_v8_validation

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v7_notrade_push/best_checkpoint.pt \
#  --total-updates 30 \
#  --report-every 10 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 7272 \
#  --reward-shaping-vp 0.04 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 650 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 12 \
#  --ppo-lr 0.00003 \
#  --ppo-ent-coef-start 0.012 \
#  --ppo-ent-coef-end 0.010 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_v7_trade_probe

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v5_no_trade/best_checkpoint.pt \
#  --total-updates 1 \
#  --report-every 1 \
#  --rollout-steps 64 \
#  --games-per-seat 50 \
#  --seed 2026 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.0 \
#  --ppo-ent-coef 0.012 \
#  --ppo-epochs 1 \
#  --out-dir artifacts_schedule_v5_validation
