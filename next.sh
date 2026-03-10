
for s in 22101 22102 22103 22104 22105 22106 22107 22108 22109 22110 22111 22112; do
  python -u scripts/train_schedule.py \
    --init-checkpoint art_enh3_bootstrap/best_checkpoint.pt \
    --total-updates 20 \
    --report-every 5 \
    --rollout-steps 1024 \
    --games-per-seat 10 \
    --seed "$s" \
    --enhanced-obs-features \
    --reward-shaping-vp 0.05 \
    --reward-shaping-resource 0.001 \
    --max-episode-steps 600 \
    --eval-max-steps 2200 \
    --max-main-actions-per-turn 10 \
    --trade-action-mode guided \
    --max-player-trade-proposals-per-turn 1 \
    --ppo-lr 0.000005 \
    --ppo-ent-coef-start 0.004 \
    --ppo-ent-coef-end 0.0025 \
    --ppo-epochs 2 \
    --out-dir "ext1_seed${s}"
done

#    --init-checkpoint artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt \
#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.25 \
#  --mix-random-prob 0.10 \
#  --mix-meta-prob 0.40 \
#  --mix-frozen-prob 0.25 \
#  --mix-frozen-checkpoints "artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt,artifacts_schedule_realboard_trade_guided_probe_seed1005/best_checkpoint.pt,artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt" \
#  --total-updates 15 \
#  --report-every 5 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 6202 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.000004 \
#  --ppo-ent-coef-start 0.0038 \
#  --ppo-ent-coef-end 0.0025 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_vs_meta_mix_v2

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.10 \
#  --mix-random-prob 0.10 \
#  --mix-meta-prob 0.60 \
#  --mix-frozen-prob 0.20 \
#  --mix-frozen-checkpoints "artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt,artifacts_schedule_realboard_trade_mix_overnight_seed5109/checkpoints/policy_u10.pt" \
#  --total-updates 20 \
#  --report-every 5 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 6201 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.000005 \
#  --ppo-ent-coef-start 0.004 \
#  --ppo-ent-coef-end 0.0025 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_vs_meta_mix_v1

#for s in 5101 5102 5103 5104 5105 5106 5107 5108 5109 5110 5111 5112; do
#  python -u scripts/train_schedule.py \
#    --init-checkpoint artifacts_schedule_realboard_trade_mix_v1/best_checkpoint.pt \
#    --use-opponent-mixture \
#    --opponent-seat-count 1 \
#    --mix-heuristic-prob 0.65 \
#    --mix-random-prob 0.15 \
#    --mix-frozen-prob 0.20 \
#    --mix-frozen-checkpoints "artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt,artifacts_schedule_realboard_trade_guided_probe_seed1005/best_checkpoint.pt,artifacts_schedule_realboard_trade_guided_probe_seed1002/best_checkpoint.pt,artifacts_schedule_realboard_trade_mix_v1/best_checkpoint.pt" \
#    --total-updates 20 \
#    --report-every 5 \
#    --rollout-steps 1024 \
#    --games-per-seat 10 \
#    --seed "$s" \
#    --reward-shaping-vp 0.05 \
#    --reward-shaping-resource 0.001 \
#    --max-episode-steps 600 \
#    --eval-max-steps 2200 \
#    --max-main-actions-per-turn 10 \
#    --trade-action-mode guided \
#    --max-player-trade-proposals-per-turn 1 \
#    --ppo-lr 0.000005 \
#    --ppo-ent-coef-start 0.004 \
#    --ppo-ent-coef-end 0.0025 \
#    --ppo-epochs 2 \
#    --out-dir "artifacts_schedule_realboard_trade_mix_overnight_seed${s}"
#done

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_trade_mix_v1/best_checkpoint.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.65 \
#  --mix-random-prob 0.15 \
#  --mix-frozen-prob 0.20 \
#  --mix-frozen-checkpoints "artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt,artifacts_schedule_realboard_trade_guided_probe_seed1005/best_checkpoint.pt,artifacts_schedule_realboard_trade_guided_probe_seed1002/best_checkpoint.pt,artifacts_schedule_realboard_trade_mix_v1/best_checkpoint.pt" \
#  --total-updates 15 \
#  --report-every 5 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 4041 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.000005 \
#  --ppo-ent-coef-start 0.004 \
#  --ppo-ent-coef-end 0.0025 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_trade_mix_heuristic_lift_micro_v1
#
#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_trade_guided_probe_seed1005/best_checkpoint.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.45 \
#  --mix-random-prob 0.20 \
#  --mix-frozen-prob 0.35 \
#  --mix-frozen-checkpoints "artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt,artifacts_schedule_realboard_trade_guided_probe_seed1005/best_checkpoint.pt,artifacts_schedule_realboard_trade_guided_probe_seed1002/best_checkpoint.pt" \
#  --total-updates 45 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 2027 \
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
#  --out-dir artifacts_schedule_realboard_trade_mix_v1

#source .venv/bin/activate
#for s in 1001 1002 1003 1004 1005; do
#  python -u scripts/train_schedule.py \
#    --init-checkpoint artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt \
#    --total-updates 40 \
#    --report-every 10 \
#    --rollout-steps 1024 \
#    --games-per-seat 10 \
#    --seed "$s" \
#    --reward-shaping-vp 0.05 \
#    --reward-shaping-resource 0.001 \
#    --max-episode-steps 600 \
#    --eval-max-steps 2200 \
#    --max-main-actions-per-turn 10 \
#    --trade-action-mode guided \
#    --max-player-trade-proposals-per-turn 1 \
#    --ppo-lr 0.00001 \
#    --ppo-ent-coef-start 0.006 \
#    --ppo-ent-coef-end 0.003 \
#    --ppo-epochs 2 \
#    --out-dir "artifacts_schedule_realboard_trade_guided_probe_seed${s}"
#done


#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt \
#  --total-updates 40 \
#  --report-every 10 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 910 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.00001 \
#  --ppo-ent-coef-start 0.006 \
#  --ppo-ent-coef-end 0.003 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_trade_guided_v2

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_v3_long_seed456/checkpoints/policy_u1275.pt \
#  --total-updates 80 \
#  --report-every 20 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 909 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --ppo-lr 0.00002 \
#  --ppo-ent-coef-start 0.012 \
#  --ppo-ent-coef-end 0.008 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_trade_guided_v1

#source .venv/bin/activate
##for s in 456 457 458; do
#for s in  411 412 413; do
#  python -u scripts/train_schedule.py \
#    --init-checkpoint artifacts_schedule_realboard_v2/best_checkpoint.pt \
#    --total-updates 300 \
#    --report-every 25 \
#    --rollout-steps 2048 \
#    --games-per-seat 40 \
#    --seed "$s" \
#    --reward-shaping-vp 0.05 \
#    --reward-shaping-resource 0.001 \
#    --max-episode-steps 600 \
#    --eval-max-steps 2200 \
#    --max-main-actions-per-turn 10 \
#    --disable-player-trade \
#    --ppo-lr 0.00002 \
#    --ppo-ent-coef-start 0.020 \
#    --ppo-ent-coef-end 0.004 \
#    --ppo-epochs 4 \
#    --out-dir "artifacts_schedule_realboard_v3_long_seed${s}"
#done

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_realboard_v1/best_checkpoint.pt \
#  --total-updates 60 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 456 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.00002 \
#  --ppo-ent-coef-start 0.010 \
#  --ppo-ent-coef-end 0.008 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_v2

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --bc-warmstart \
#  --bc-steps 20000 \
#  --bc-epochs 3 \
#  --total-updates 125 \
#  --report-every 25 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 123 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.00003 \
#  --ppo-ent-coef-start 0.014 \
#  --ppo-ent-coef-end 0.010 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_realboard_v1

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_mix_v1/best_checkpoint.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.55 \
#  --mix-random-prob 0.25 \
#  --mix-frozen-prob 0.20 \
#  --mix-frozen-checkpoints "artifacts_schedule_mix_v1/best_checkpoint.pt,artifacts_schedule_v8_notrade_long/best_checkpoint.pt" \
#  --total-updates 45 \
#  --report-every 10 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 1212 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.00003 \
#  --ppo-ent-coef-start 0.014 \
#  --ppo-ent-coef-end 0.010 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_mix_v3_stable

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_mix_v1/best_checkpoint.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.45 \
#  --mix-random-prob 0.15 \
#  --mix-frozen-prob 0.40 \
#  --mix-frozen-checkpoints "artifacts_schedule_mix_v1/best_checkpoint.pt,artifacts_schedule_v8_notrade_long/best_checkpoint.pt,artifacts_schedule_v5_no_trade/best_checkpoint.pt" \
#  --total-updates 90 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 1111 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.000035 \
#  --ppo-ent-coef-start 0.015 \
#  --ppo-ent-coef-end 0.010 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_mix_v2

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt \
#  --use-opponent-mixture \
#  --opponent-seat-count 1 \
#  --mix-heuristic-prob 0.5 \
#  --mix-random-prob 0.2 \
#  --mix-frozen-prob 0.3 \
#  --mix-frozen-checkpoints "artifacts_schedule_v5_no_trade/best_checkpoint.pt,artifacts_schedule_v8_notrade_long/best_checkpoint.pt" \
#  --total-updates 90 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 1010 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.00004 \
#  --ppo-ent-coef-start 0.016 \
#  --ppo-ent-coef-end 0.011 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_mix_v1

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt \
#  --total-updates 120 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 9393 \
#  --reward-shaping-vp 0.06 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 650 \
#  --eval-max-steps 2400 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.000035 \
#  --ppo-ent-coef-start 0.014 \
#  --ppo-ent-coef-end 0.010 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_v9_heuristic_lift

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v5_no_trade/best_checkpoint.pt \
#  --total-updates 180 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 8181 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.00004 \
#  --ppo-ent-coef-start 0.016 \
#  --ppo-ent-coef-end 0.011 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_v8_notrade_long

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v5_no_trade/best_checkpoint.pt \
#  --total-updates 90 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 7171 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.00005 \
#  --ppo-ent-coef-start 0.016 \
#  --ppo-ent-coef-end 0.010 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_v7_notrade_push

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_v5_no_trade/best_checkpoint.pt \
#  --total-updates 90 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 6060 \
#  --reward-shaping-vp 0.04 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 650 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 12 \
#  --ppo-lr 0.00005 \
#  --ppo-ent-coef-start 0.014 \
#  --ppo-ent-coef-end 0.01 \
#  --ppo-epochs 2 \
#  --out-dir artifacts_schedule_v6_trade_on

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_resume_from_u60/best_checkpoint.pt \
#  --total-updates 120 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 909 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --disable-player-trade \
#  --ppo-lr 0.00006 \
#  --ppo-ent-coef-start 0.018 \
#  --ppo-ent-coef-end 0.012 \
#  --ppo-epochs 3 \
#  --out-dir artifacts_schedule_v5_no_trade

#source .venv/bin/activate && python -u scripts/train_schedule.py \
#  --init-checkpoint artifacts_schedule_resume_from_u60/best_checkpoint.pt \
#  --total-updates 120 \
#  --report-every 15 \
#  --rollout-steps 1024 \
#  --games-per-seat 10 \
#  --seed 4242 \
#  --reward-shaping-vp 0.04 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 650 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 12 \
#  --ppo-lr 0.00008 \
#  --ppo-ent-coef-start 0.018 \
#  --ppo-ent-coef-end 0.01 \
#  --ppo-epochs 3 \
#  --out-dir artifacts_schedule_stage2_trade_on
