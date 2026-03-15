#!/usr/bin/env bash
set -euo pipefail

# Three-branch league trainer for Catan RL.
# Fill checkpoint placeholders below, then run:
#   bash league3.sh
#
# Optional overrides:
#   PYTHON_BIN=.venv/bin/python CYCLES=4 BASE_OUT=league_runs_enh bash league3.sh

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE_OUT="${BASE_OUT:-league_runs}"
CYCLES="${CYCLES:-12}"
RESET_WEAKEST_FROM_CHAMP="${RESET_WEAKEST_FROM_CHAMP:-1}"
SKIP_COMPLETED_BRANCHES="${SKIP_COMPLETED_BRANCHES:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"

# ---- Fill these before running ----
CKPT_A="${CKPT_A:-old_runs/cycle_5/branch_B/checkpoints/policy_u20.pt}"
CKPT_B="${CKPT_B:-old_runs/cycle_7/branch_C/checkpoints/policy_u10.pt}"
CKPT_C="${CKPT_C:-old_runs/cycle_21/branch_A/checkpoints/policy_u30.pt}"
# -----------------------------------

BRANCHES=("A" "B" "C")
BRANCH_SEED_BASES=(23001 24001 25001)
BRANCH_META_MIX=(0.20 0.30 0.40) # slight per-branch diversity

# Stable training settings (edit as needed).
TOTAL_UPDATES="${TOTAL_UPDATES:-30}"
REPORT_EVERY="${REPORT_EVERY:-10}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-1024}"
GAMES_PER_SEAT="${GAMES_PER_SEAT:-20}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-600}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-2200}"
MAX_MAIN_ACTIONS="${MAX_MAIN_ACTIONS:-10}"
TRADE_MODE="${TRADE_MODE:-guided}"
MAX_TRADE_PROPOSALS="${MAX_TRADE_PROPOSALS:-1}"
PPO_LR="${PPO_LR:-0.00003}"
PPO_ENT_COEF="${PPO_ENT_COEF:-0.02}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
REWARD_SHAPING_VP="${REWARD_SHAPING_VP:-0.05}"
REWARD_SHAPING_RESOURCE="${REWARD_SHAPING_RESOURCE:-0.0}"
REWARD_SHAPING_ROBBER_BLOCK_LEADER="${REWARD_SHAPING_ROBBER_BLOCK_LEADER:-0.0}"
REWARD_SHAPING_ROB_LEADER="${REWARD_SHAPING_ROB_LEADER:-0.0}"
REWARD_SHAPING_ROB_MISTARGET="${REWARD_SHAPING_ROB_MISTARGET:-0.0}"
REWARD_SHAPING_PLAY_KNIGHT="${REWARD_SHAPING_PLAY_KNIGHT:-0.0}"
REWARD_SHAPING_SETUP_SETTLEMENT="${REWARD_SHAPING_SETUP_SETTLEMENT:-0.0}"
REWARD_SHAPING_TERMINAL_TABLE_MEAN="${REWARD_SHAPING_TERMINAL_TABLE_MEAN:-0.0}"
THREAT_DEV_CARD_WEIGHT="${THREAT_DEV_CARD_WEIGHT:-0.7}"

# Mixture defaults. We compute heuristic probability from the others.
MIX_RANDOM="${MIX_RANDOM:-0.10}"
MIX_FROZEN="${MIX_FROZEN:-0.40}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_checkpoint() {
  local label="$1"
  local ckpt="$2"
  if [[ "$ckpt" == REPLACE_ME* ]]; then
    die "Set ${label}. Example: CKPT_A=path/to/best_checkpoint.pt"
  fi
  if [[ ! -f "$ckpt" ]]; then
    die "Checkpoint not found for ${label}: $ckpt"
  fi
}

assert_python() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    die "Python binary not executable: $PYTHON_BIN"
  fi
}

contains_csv_item() {
  local csv="$1"
  local item="$2"
  IFS=',' read -r -a parts <<< "$csv"
  for p in "${parts[@]}"; do
    [[ "$p" == "$item" ]] && return 0
  done
  return 1
}

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -pr | wc -l | tr -d ' ')
    if (( running < PARALLEL_JOBS )); then
      break
    fi
    sleep 1
  done
}

run_branch_cycle() {
  local idx="$1"
  local cycle="$2"
  local cycle_dir="$3"

  local branch init_ckpt seed meta_prob heuristic_prob out_dir best_ckpt eval_out
  branch="${BRANCHES[$idx]}"
  init_ckpt="${CURRENT_CKPTS[$idx]}"
  seed=$(( BRANCH_SEED_BASES[$idx] + cycle ))
  meta_prob="${BRANCH_META_MIX[$idx]}"
  heuristic_prob=$(python3 - <<PY
mix_random = float("$MIX_RANDOM")
mix_frozen = float("$MIX_FROZEN")
mix_meta = float("$meta_prob")
v = 1.0 - (mix_random + mix_frozen + mix_meta)
print(max(0.0, v))
PY
)
  out_dir="$cycle_dir/branch_${branch}"
  mkdir -p "$out_dir"
  best_ckpt="$out_dir/best_checkpoint.pt"
  eval_out="$out_dir/eval_both.json"

  if [[ "$SKIP_COMPLETED_BRANCHES" == "1" && -f "$best_ckpt" ]]; then
    echo "-> Skip train branch ${branch}; found existing $best_ckpt"
  else
    echo "-> Train branch ${branch} seed=${seed} init=${init_ckpt}"
    "$PYTHON_BIN" -u scripts/train_schedule.py \
      --init-checkpoint "$init_ckpt" \
      --use-opponent-mixture \
      --opponent-seat-count 1 \
      --mix-heuristic-prob "$heuristic_prob" \
      --mix-random-prob "$MIX_RANDOM" \
      --mix-meta-prob "$meta_prob" \
      --mix-frozen-prob "$MIX_FROZEN" \
      --mix-frozen-checkpoints "$FROZEN_POOL_CSV" \
      --total-updates "$TOTAL_UPDATES" \
      --report-every "$REPORT_EVERY" \
      --rollout-steps "$ROLLOUT_STEPS" \
      --games-per-seat "$GAMES_PER_SEAT" \
      --seed "$seed" \
      --reward-shaping-vp "$REWARD_SHAPING_VP" \
      --reward-shaping-resource "$REWARD_SHAPING_RESOURCE" \
      --reward-shaping-robber-block-leader "$REWARD_SHAPING_ROBBER_BLOCK_LEADER" \
      --reward-shaping-rob-leader "$REWARD_SHAPING_ROB_LEADER" \
      --reward-shaping-rob-mistarget "$REWARD_SHAPING_ROB_MISTARGET" \
      --reward-shaping-play-knight "$REWARD_SHAPING_PLAY_KNIGHT" \
      --reward-shaping-setup-settlement "$REWARD_SHAPING_SETUP_SETTLEMENT" \
      --reward-shaping-terminal-table-mean "$REWARD_SHAPING_TERMINAL_TABLE_MEAN" \
      --threat-dev-card-weight "$THREAT_DEV_CARD_WEIGHT" \
      --max-episode-steps "$MAX_EPISODE_STEPS" \
      --eval-max-steps "$EVAL_MAX_STEPS" \
      --max-main-actions-per-turn "$MAX_MAIN_ACTIONS" \
      --trade-action-mode "$TRADE_MODE" \
      --max-player-trade-proposals-per-turn "$MAX_TRADE_PROPOSALS" \
      --ppo-lr "$PPO_LR" \
      --ppo-ent-coef "$PPO_ENT_COEF" \
      --ppo-epochs "$PPO_EPOCHS" \
      --out-dir "$out_dir"
  fi

  [[ -f "$best_ckpt" ]] || die "Missing best checkpoint for branch ${branch}: $best_ckpt"

  if [[ "$SKIP_COMPLETED_BRANCHES" == "1" && -f "$eval_out" ]]; then
    echo "-> Skip eval branch ${branch}; found existing $eval_out"
  else
    "$PYTHON_BIN" -u scripts/eval_checkpoint.py \
      --checkpoint "$best_ckpt" \
      --training-history "$out_dir/progress_reports.jsonl" \
      --promotion-decision "$out_dir/best_checkpoint_meta.json" \
      --games-per-seat 12 \
      --seed "$seed" \
      --opponent both \
      --max-steps "$EVAL_MAX_STEPS" \
      --max-main-actions-per-turn "$MAX_MAIN_ACTIONS" \
      --trade-action-mode "$TRADE_MODE" \
      --max-player-trade-proposals-per-turn "$MAX_TRADE_PROPOSALS" \
      --out "$eval_out"
  fi

  [[ -f "$eval_out" ]] || die "Missing eval output for branch ${branch}: $eval_out"
}

assert_python
require_checkpoint "CKPT_A" "$CKPT_A"
require_checkpoint "CKPT_B" "$CKPT_B"
require_checkpoint "CKPT_C" "$CKPT_C"

mkdir -p "$BASE_OUT"

CURRENT_CKPTS=("$CKPT_A" "$CKPT_B" "$CKPT_C")
FROZEN_POOL_CSV="$CKPT_A,$CKPT_B,$CKPT_C"

echo "Starting 3-branch league training"
echo "base_out=$BASE_OUT cycles=$CYCLES python=$PYTHON_BIN"
echo "initial_pool=$FROZEN_POOL_CSV"

for cycle in $(seq 1 "$CYCLES"); do
  cycle_dir="$BASE_OUT/cycle_${cycle}"
  mkdir -p "$cycle_dir"
  echo ""
  echo "===== Cycle $cycle ====="

  NEXT_CKPTS=("" "" "")
  eval_files=()
  branch_pids=()

  for idx in 0 1 2; do
    if (( PARALLEL_JOBS > 1 )); then
      wait_for_slot
      run_branch_cycle "$idx" "$cycle" "$cycle_dir" &
      branch_pids+=("$!")
    else
      run_branch_cycle "$idx" "$cycle" "$cycle_dir"
    fi
  done

  if (( PARALLEL_JOBS > 1 )); then
    for pid in "${branch_pids[@]}"; do
      wait "$pid" || die "A parallel branch job failed in cycle ${cycle}"
    done
  fi

  for idx in 0 1 2; do
    branch="${BRANCHES[$idx]}"
    out_dir="$cycle_dir/branch_${branch}"
    best_ckpt="$out_dir/best_checkpoint.pt"
    eval_out="$out_dir/eval_both.json"
    [[ -f "$best_ckpt" ]] || die "Missing best checkpoint for branch ${branch}: $best_ckpt"
    [[ -f "$eval_out" ]] || die "Missing eval output for branch ${branch}: $eval_out"
    NEXT_CKPTS[$idx]="$best_ckpt"
    eval_files+=("$eval_out")
  done

  ranking_json="$cycle_dir/ranking.json"
  python3 - <<'PY' "$ranking_json" "${eval_files[@]}"
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
rows = []
for p in sys.argv[2:]:
    data = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    random_eval = data["seat_permutation_eval"]["random"]
    heuristic_eval = data["seat_permutation_eval"]["heuristic"]
    by_seat = []
    for seat in range(4):
        rs = random_eval["seat_stats"][seat]["strict_win_rate"]
        hs = heuristic_eval["seat_stats"][seat]["strict_win_rate"]
        by_seat.append(0.5 * (float(rs) + float(hs)))
    rows.append(
        {
            "eval_path": p,
            "strict_score_eval": float(data["strict_score_eval"]),
            "worst_seat_strict": float(min(by_seat)),
            "mean_seat_strict": float(sum(by_seat) / len(by_seat)),
            "seat_strict": by_seat,
        }
    )

rows.sort(
    key=lambda r: (r["strict_score_eval"], r["worst_seat_strict"], r["mean_seat_strict"]),
    reverse=True,
)
out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("rank,strict_score,worst_seat,eval_path")
for i, r in enumerate(rows, start=1):
    print(f"{i},{r['strict_score_eval']:.6f},{r['worst_seat_strict']:.6f},{r['eval_path']}")
PY

  champ_eval=$(python3 - <<'PY' "$ranking_json"
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(rows[0]["eval_path"])
PY
)
  weak_eval=$(python3 - <<'PY' "$ranking_json"
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(rows[-1]["eval_path"])
PY
)

  champ_idx=-1
  weak_idx=-1
  for idx in 0 1 2; do
    branch="${BRANCHES[$idx]}"
    eval_path="$cycle_dir/branch_${branch}/eval_both.json"
    if [[ "$eval_path" == "$champ_eval" ]]; then
      champ_idx=$idx
    fi
    if [[ "$eval_path" == "$weak_eval" ]]; then
      weak_idx=$idx
    fi
  done
  (( champ_idx >= 0 )) || die "Could not map champion eval path to branch"
  (( weak_idx >= 0 )) || die "Could not map weakest eval path to branch"

  champ_ckpt="${NEXT_CKPTS[$champ_idx]}"
  echo "Cycle ${cycle} champion branch=${BRANCHES[$champ_idx]} ckpt=$champ_ckpt"
  echo "Cycle ${cycle} weakest branch=${BRANCHES[$weak_idx]}"

  # Advance each branch to its own best checkpoint.
  for idx in 0 1 2; do
    CURRENT_CKPTS[$idx]="${NEXT_CKPTS[$idx]}"
  done

  # Optional league mechanic: weakest branch restarts from champion.
  if [[ "$RESET_WEAKEST_FROM_CHAMP" == "1" ]]; then
    CURRENT_CKPTS[$weak_idx]="$champ_ckpt"
    echo "Reset weakest branch ${BRANCHES[$weak_idx]} -> champion checkpoint"
  fi

  # Expand frozen pool with champion each cycle.
  if ! contains_csv_item "$FROZEN_POOL_CSV" "$champ_ckpt"; then
    FROZEN_POOL_CSV="${FROZEN_POOL_CSV},${champ_ckpt}"
  fi
  echo "updated_frozen_pool=$FROZEN_POOL_CSV"
done

echo ""
echo "League run complete."
echo "Latest branch checkpoints:"
for idx in 0 1 2; do
  echo "  ${BRANCHES[$idx]} => ${CURRENT_CKPTS[$idx]}"
done
