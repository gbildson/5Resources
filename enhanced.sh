for s in 4101 4102 4103; do
  python -u scripts/train_schedule.py \
    --out-dir "bpg_s${s}" \
    --total-updates 30 \
    --report-every 10 \
    --rollout-steps 1024 \
    --games-per-seat 20 \
    --seed "$s" \
    --model-arch graph_entity_phase_aware_hybrid \
    --model-hidden 192 \
    --model-residual-blocks 3 \
    --use-opponent-mixture \
    --opponent-seat-count 3 \
    --mix-frozen-prob 0.75 \
    --mix-random-prob 0.10 \
    --mix-meta-prob 0.05 \
    --mix-heuristic-prob 0.10 \
    --mix-frozen-checkpoints "league_runs_hybrid_v1/cycle_113/branch_A/checkpoints/policy_u20.pt,league_runs_hybrid_v1/cycle_113/branch_C/checkpoints/policy_u10.pt" \
    --reward-shaping-main-road-purpose 0.16 \
    --reward-shaping-setup-settlement 0.30 \
    --setup-selection-influence-settlement-prob 0.46 \
    --setup-selection-influence-settlement-round2-prob 0.90 \
    --setup-selection-influence-road-prob 0.20 \
    --reward-shaping-setup-road-frontier 0.09 \
    --reward-shaping-setup-round1-floor 0.10 \
    --reward-shaping-ows-actions 0.07
done

#for s in 104; do
#  python -u scripts/train_schedule.py \
#    --total-updates 20 \
#    --report-every 5 \
#    --rollout-steps 1024 \
#    --games-per-seat 15 \
#    --seed "$s" \
#    --reward-shaping-vp 0.05 \
#    --reward-shaping-resource 0.0 \
#    --max-episode-steps 600 \
#    --eval-max-steps 2200 \
#    --max-main-actions-per-turn 10 \
#    --trade-action-mode guided \
#    --max-player-trade-proposals-per-turn 1 \
#    --ppo-lr 0.00005 \
##    --ppo-ent-coef 0.02 \
#    --ppo-epochs 1 \
#    --bc-warmstart \
#    --bc-steps 20000 \
#    --bc-epochs 3 \
#    --out-dir "boot_seed${s}"
#done
#
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
