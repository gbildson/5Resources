for s in 22101 22102 22103; do
  python -u scripts/train_schedule.py \
    --enhanced-obs-features \
    --total-updates 40 \
    --report-every 10 \
    --rollout-steps 1024 \
    --games-per-seat 15 \
    --seed "$s" \
    --reward-shaping-vp 0.05 \
    --reward-shaping-resource 0.0 \
    --max-episode-steps 600 \
    --eval-max-steps 2200 \
    --max-main-actions-per-turn 10 \
    --trade-action-mode guided \
    --max-player-trade-proposals-per-turn 1 \
    --ppo-lr 0.00005 \
    --ppo-ent-coef 0.02 \
    --ppo-epochs 1 \
    --bc-warmstart \
    --bc-steps 20000 \
    --bc-epochs 3 \
    --out-dir "ablateB_seed${s}"
done

#python -u scripts/train_schedule.py \
#  --enhanced-obs-features \
#  --total-updates 40 \
#  --report-every 10 \
#  --rollout-steps 1024 \
#  --games-per-seat 15 \
#  --seed 20003 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.00005 \
#  --ppo-ent-coef 0.02 \
#  --ppo-epochs 1 \
#  --bc-warmstart \
#  --bc-steps 20000 \
#  --bc-epochs 3 \
#  --out-dir art_enh3_bootstrap

#python -u scripts/train_schedule.py \
#  --enhanced-obs-features \
#  --total-updates 40 \
#  --report-every 10 \
##  --rollout-steps 1024 \
#  --games-per-seat 15 \
#  --seed 20002 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.0001 \
#  --ppo-ent-coef-start 0.03 \
#  --ppo-ent-coef-end 0.015 \
#  --ppo-epochs 2 \
#  --bc-warmstart \
#  --bc-steps 30000 \
#  --bc-epochs 4 \
#  --out-dir art_enh2_bootstrap

#python -u scripts/train_schedule.py \
#  --enhanced-obs-features \
#  --total-updates 40 \
#  --report-every 10 \
#  --rollout-steps 1024 \
#  --games-per-seat 15 \
#  --seed 20001 \
#  --reward-shaping-vp 0.05 \
#  --reward-shaping-resource 0.001 \
#  --max-episode-steps 600 \
#  --eval-max-steps 2200 \
#  --max-main-actions-per-turn 10 \
#  --trade-action-mode guided \
#  --max-player-trade-proposals-per-turn 1 \
#  --ppo-lr 0.0003 \
#  --ppo-ent-coef-start 0.025 \
#  --ppo-ent-coef-end 0.01 \
#  --ppo-epochs 3 \
#  --bc-warmstart \
#  --bc-steps 20000 \
#  --bc-epochs 3 \
#  --out-dir artifacts_enhanced_bootstrap
