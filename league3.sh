#!/usr/bin/env bash
set -euo pipefail

# Three-branch league trainer for Catan RL.
# Fill checkpoint placeholders below, then run:
#   bash league3.sh
#
# Optional overrides:
#   PYTHON_BIN=.venv/bin/python CYCLES=4 BASE_OUT=league_runs_enh bash league3.sh
#   ANCHOR_MANIFEST=league_anchor_manifest.example.json ANCHOR_TAG_FILTER=road,ows bash league3.sh

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
BASE_OUT="${BASE_OUT:-league_runs}"
CYCLES="${CYCLES:-12}"
RESET_WEAKEST_FROM_CHAMP="${RESET_WEAKEST_FROM_CHAMP:-1}"
RESET_SOURCE_CHAMP_PROB="${RESET_SOURCE_CHAMP_PROB:-0.70}"
RESET_SOURCE_ANCHOR_PROB="${RESET_SOURCE_ANCHOR_PROB:-0.30}"
SKIP_COMPLETED_BRANCHES="${SKIP_COMPLETED_BRANCHES:-1}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
ANCHOR_MANIFEST="${ANCHOR_MANIFEST:-}"
ANCHOR_TAG_FILTER="${ANCHOR_TAG_FILTER:-}"
GUARANTEE_ANCHOR_EXPOSURE="${GUARANTEE_ANCHOR_EXPOSURE:-1}"
ANCHOR_MIN_FROZEN_PROB="${ANCHOR_MIN_FROZEN_PROB:-0.40}"
DYNAMIC_FROZEN_MAX="${DYNAMIC_FROZEN_MAX:-15}"
FROZEN_SAMPLE_COUNT="${FROZEN_SAMPLE_COUNT:-3}"
PROMOTION_GATE_ENABLE="${PROMOTION_GATE_ENABLE:-1}"
PROMOTION_MIN_STRICT="${PROMOTION_MIN_STRICT:-0.43}"
PROMOTION_MIN_WORST_SEAT="${PROMOTION_MIN_WORST_SEAT:-0.30}"
PROMOTION_MIN_OPENING="${PROMOTION_MIN_OPENING:--1}"
PROMOTION_BORDERLINE_MARGIN="${PROMOTION_BORDERLINE_MARGIN:-0.02}"
PROMOTION_BORDERLINE_GAMES_PER_SEAT="${PROMOTION_BORDERLINE_GAMES_PER_SEAT:-24}"

# ---- Fill these before running ----
CKPT_A="${CKPT_A:-dummy.pt}"
CKPT_B="${CKPT_B:-dummy.pt}"
CKPT_C="${CKPT_C:-dummy.pt}"
# -----------------------------------

BRANCHES=("A" "B" "C")
BRANCH_SEED_BASES=(33001 34001 35001)
BRANCH_META_MIX_A="${BRANCH_META_MIX_A:-0.20}"
BRANCH_META_MIX_B="${BRANCH_META_MIX_B:-0.30}"
BRANCH_META_MIX_C="${BRANCH_META_MIX_C:-0.40}"
BRANCH_META_MIX=("$BRANCH_META_MIX_A" "$BRANCH_META_MIX_B" "$BRANCH_META_MIX_C") # per-branch diversity

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
SETUP_PHASE_LOSS_WEIGHT="${SETUP_PHASE_LOSS_WEIGHT:-1.0}"
MODEL_ARCH="${MODEL_ARCH:-mlp}"
MODEL_HIDDEN="${MODEL_HIDDEN:-256}"
MODEL_RESIDUAL_BLOCKS="${MODEL_RESIDUAL_BLOCKS:-4}"
REWARD_SHAPING_VP="${REWARD_SHAPING_VP:-0.05}"
REWARD_SHAPING_RESOURCE="${REWARD_SHAPING_RESOURCE:-0.0}"
REWARD_SHAPING_ROBBER_BLOCK_LEADER="${REWARD_SHAPING_ROBBER_BLOCK_LEADER:-0.0}"
REWARD_SHAPING_ROB_LEADER="${REWARD_SHAPING_ROB_LEADER:-0.0}"
REWARD_SHAPING_ROB_MISTARGET="${REWARD_SHAPING_ROB_MISTARGET:-0.0}"
REWARD_SHAPING_PLAY_KNIGHT="${REWARD_SHAPING_PLAY_KNIGHT:-0.0}"
REWARD_SHAPING_PLAY_KNIGHT_WHEN_BLOCKED="${REWARD_SHAPING_PLAY_KNIGHT_WHEN_BLOCKED:-0.0}"
REWARD_SHAPING_KNIGHT_UNBLOCK_PENALTY="${REWARD_SHAPING_KNIGHT_UNBLOCK_PENALTY:-0.0}"
FORCE_KNIGHT_BOOTSTRAP_PROB="${FORCE_KNIGHT_BOOTSTRAP_PROB:-0.0}"
FORCE_KNIGHT_BOOTSTRAP_UPDATES="${FORCE_KNIGHT_BOOTSTRAP_UPDATES:-0}"
FORCE_TRADE_BOOTSTRAP_PROB="${FORCE_TRADE_BOOTSTRAP_PROB:-0.0}"
FORCE_TRADE_BOOTSTRAP_UPDATES="${FORCE_TRADE_BOOTSTRAP_UPDATES:-0}"
OPPONENT_SEAT_COUNT="${OPPONENT_SEAT_COUNT:-1}"
REWARD_SHAPING_SETUP_SETTLEMENT="${REWARD_SHAPING_SETUP_SETTLEMENT:-0.0}"
REWARD_SHAPING_SETUP_TOP6_BONUS="${REWARD_SHAPING_SETUP_TOP6_BONUS:-0.0}"
REWARD_SHAPING_SETUP_ONE_HEX_PENALTY="${REWARD_SHAPING_SETUP_ONE_HEX_PENALTY:-0.0}"
REWARD_SHAPING_SETUP_ROUND1_FLOOR="${REWARD_SHAPING_SETUP_ROUND1_FLOOR:-0.0}"
REWARD_SHAPING_SETUP_ROAD_FRONTIER="${REWARD_SHAPING_SETUP_ROAD_FRONTIER:-0.0}"
SETUP_SELECTION_INFLUENCE_PROB="${SETUP_SELECTION_INFLUENCE_PROB:-0.0}"
SETUP_SELECTION_INFLUENCE_SETTLEMENT_PROB="${SETUP_SELECTION_INFLUENCE_SETTLEMENT_PROB:-0.0}"
SETUP_SELECTION_INFLUENCE_SETTLEMENT_ROUND2_PROB="${SETUP_SELECTION_INFLUENCE_SETTLEMENT_ROUND2_PROB:-0.0}"
SETUP_SELECTION_INFLUENCE_ROAD_PROB="${SETUP_SELECTION_INFLUENCE_ROAD_PROB:-0.0}"
REWARD_SHAPING_OWS_ACTIONS="${REWARD_SHAPING_OWS_ACTIONS:-0.0}"
REWARD_SHAPING_MAIN_ROAD_PURPOSE="${REWARD_SHAPING_MAIN_ROAD_PURPOSE:-0.0}"
REWARD_SHAPING_TERMINAL_TABLE_MEAN="${REWARD_SHAPING_TERMINAL_TABLE_MEAN:-0.0}"
REWARD_SHAPING_TRADE_OFFER_VALUE="${REWARD_SHAPING_TRADE_OFFER_VALUE:-0.0}"
REWARD_SHAPING_TRADE_OFFER_COUNTERPARTY_VALUE="${REWARD_SHAPING_TRADE_OFFER_COUNTERPARTY_VALUE:-0.0}"
REWARD_SHAPING_TRADE_ACCEPT_VALUE="${REWARD_SHAPING_TRADE_ACCEPT_VALUE:-0.0}"
REWARD_SHAPING_BANK_TRADE_VALUE="${REWARD_SHAPING_BANK_TRADE_VALUE:-0.0}"
REWARD_SHAPING_TRADE_REJECT_BAD_OFFER_VALUE="${REWARD_SHAPING_TRADE_REJECT_BAD_OFFER_VALUE:-0.0}"
THREAT_DEV_CARD_WEIGHT="${THREAT_DEV_CARD_WEIGHT:-0.7}"

# Mixture defaults. We compute heuristic probability from the others.
MIX_RANDOM="${MIX_RANDOM:-0.10}"
MIX_TRADE_FRIENDLY="${MIX_TRADE_FRIENDLY:-0.0}"
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

csv_append_unique() {
  local csv="$1"
  local item="$2"
  if [[ -z "$csv" ]]; then
    echo "$item"
    return 0
  fi
  if contains_csv_item "$csv" "$item"; then
    echo "$csv"
  else
    echo "${csv},${item}"
  fi
}

csv_tail_n() {
  local csv="$1"
  local n="$2"
  python3 - <<'PY' "$csv" "$n"
import sys

vals = [x for x in sys.argv[1].split(",") if x]
try:
    n = int(sys.argv[2])
except Exception:
    n = 0
if n > 0 and len(vals) > n:
    vals = vals[-n:]
print(",".join(vals))
PY
}

csv_sample_unique() {
  local csv="$1"
  local n="$2"
  local seed="$3"
  local anchor_csv="${4:-}"
  local guarantee_anchor="${5:-0}"
  python3 - <<'PY' "$csv" "$n" "$seed" "$anchor_csv" "$guarantee_anchor"
import sys
import numpy as np

vals = [x for x in sys.argv[1].split(",") if x]
seen = set()
uniq = []
for v in vals:
    if v not in seen:
        uniq.append(v)
        seen.add(v)

try:
    n = int(sys.argv[2])
except Exception:
    n = 0
try:
    seed = int(sys.argv[3])
except Exception:
    seed = 0

anchor_vals = [x for x in sys.argv[4].split(",") if x]
guarantee_anchor = sys.argv[5] == "1"

if n <= 0 or len(uniq) <= n:
    print(",".join(uniq))
    raise SystemExit

rng = np.random.default_rng(seed)
selected = []
if guarantee_anchor and anchor_vals:
    anchor_seen = set()
    anchor_uniq = []
    for v in anchor_vals:
        if v in uniq and v not in anchor_seen:
            anchor_uniq.append(v)
            anchor_seen.add(v)
    if anchor_uniq:
        selected.append(anchor_uniq[int(rng.integers(len(anchor_uniq)))])

remaining = [v for v in uniq if v not in selected]
k = max(0, min(len(remaining), n - len(selected)))
if k > 0:
    idx = sorted(rng.choice(len(remaining), size=k, replace=False).tolist())
    selected.extend(remaining[i] for i in idx)

print(",".join(selected))
PY
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
  local branch_mix_frozen branch_frozen_pool
  branch="${BRANCHES[$idx]}"
  init_ckpt="${CURRENT_CKPTS[$idx]}"
  seed=$(( BRANCH_SEED_BASES[$idx] + cycle ))
  meta_prob="${BRANCH_META_MIX[$idx]}"
  branch_mix_frozen="$MIX_FROZEN"
  if [[ "$GUARANTEE_ANCHOR_EXPOSURE" == "1" && -n "$ANCHOR_POOL_CSV" ]]; then
    branch_mix_frozen=$(python3 - <<PY
mix_frozen = float("$MIX_FROZEN")
anchor_min = float("$ANCHOR_MIN_FROZEN_PROB")
print(max(mix_frozen, anchor_min))
PY
)
  fi
  branch_frozen_pool="$FROZEN_POOL_CSV"
  if [[ "$FROZEN_SAMPLE_COUNT" -gt 0 ]]; then
    branch_frozen_pool=$(csv_sample_unique "$FROZEN_POOL_CSV" "$FROZEN_SAMPLE_COUNT" "$seed" "$ANCHOR_POOL_CSV" "$GUARANTEE_ANCHOR_EXPOSURE")
  fi
  heuristic_prob=$(python3 - <<PY
mix_random = float("$MIX_RANDOM")
mix_trade_friendly = float("$MIX_TRADE_FRIENDLY")
mix_frozen = float("$branch_mix_frozen")
mix_meta = float("$meta_prob")
v = 1.0 - (mix_random + mix_trade_friendly + mix_frozen + mix_meta)
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
    echo "   frozen_subset=${branch_frozen_pool}"
    "$PYTHON_BIN" -u scripts/train_schedule.py \
      --init-checkpoint "$init_ckpt" \
      --use-opponent-mixture \
      --opponent-seat-count "$OPPONENT_SEAT_COUNT" \
      --mix-heuristic-prob "$heuristic_prob" \
      --mix-random-prob "$MIX_RANDOM" \
      --mix-meta-prob "$meta_prob" \
      --mix-trade-friendly-prob "$MIX_TRADE_FRIENDLY" \
      --mix-frozen-prob "$branch_mix_frozen" \
      --mix-frozen-checkpoints "$branch_frozen_pool" \
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
      --reward-shaping-play-knight-when-blocked "$REWARD_SHAPING_PLAY_KNIGHT_WHEN_BLOCKED" \
      --reward-shaping-knight-unblock-penalty "$REWARD_SHAPING_KNIGHT_UNBLOCK_PENALTY" \
      --force-knight-bootstrap-prob "$FORCE_KNIGHT_BOOTSTRAP_PROB" \
      --force-knight-bootstrap-updates "$FORCE_KNIGHT_BOOTSTRAP_UPDATES" \
      --force-trade-bootstrap-prob "$FORCE_TRADE_BOOTSTRAP_PROB" \
      --force-trade-bootstrap-updates "$FORCE_TRADE_BOOTSTRAP_UPDATES" \
      --reward-shaping-setup-settlement "$REWARD_SHAPING_SETUP_SETTLEMENT" \
      --reward-shaping-setup-top6-bonus "$REWARD_SHAPING_SETUP_TOP6_BONUS" \
      --reward-shaping-setup-one-hex-penalty "$REWARD_SHAPING_SETUP_ONE_HEX_PENALTY" \
      --reward-shaping-setup-round1-floor "$REWARD_SHAPING_SETUP_ROUND1_FLOOR" \
      --reward-shaping-setup-road-frontier "$REWARD_SHAPING_SETUP_ROAD_FRONTIER" \
      --setup-selection-influence-prob "$SETUP_SELECTION_INFLUENCE_PROB" \
      --setup-selection-influence-settlement-prob "$SETUP_SELECTION_INFLUENCE_SETTLEMENT_PROB" \
      --setup-selection-influence-settlement-round2-prob "$SETUP_SELECTION_INFLUENCE_SETTLEMENT_ROUND2_PROB" \
      --setup-selection-influence-road-prob "$SETUP_SELECTION_INFLUENCE_ROAD_PROB" \
      --reward-shaping-ows-actions "$REWARD_SHAPING_OWS_ACTIONS" \
      --reward-shaping-main-road-purpose "$REWARD_SHAPING_MAIN_ROAD_PURPOSE" \
      --reward-shaping-terminal-table-mean "$REWARD_SHAPING_TERMINAL_TABLE_MEAN" \
      --reward-shaping-trade-offer-value "$REWARD_SHAPING_TRADE_OFFER_VALUE" \
      --reward-shaping-trade-offer-counterparty-value "$REWARD_SHAPING_TRADE_OFFER_COUNTERPARTY_VALUE" \
      --reward-shaping-trade-accept-value "$REWARD_SHAPING_TRADE_ACCEPT_VALUE" \
      --reward-shaping-bank-trade-value "$REWARD_SHAPING_BANK_TRADE_VALUE" \
      --reward-shaping-trade-reject-bad-offer-value "$REWARD_SHAPING_TRADE_REJECT_BAD_OFFER_VALUE" \
      --threat-dev-card-weight "$THREAT_DEV_CARD_WEIGHT" \
      --max-episode-steps "$MAX_EPISODE_STEPS" \
      --eval-max-steps "$EVAL_MAX_STEPS" \
      --max-main-actions-per-turn "$MAX_MAIN_ACTIONS" \
      --trade-action-mode "$TRADE_MODE" \
      --max-player-trade-proposals-per-turn "$MAX_TRADE_PROPOSALS" \
      --ppo-lr "$PPO_LR" \
      --ppo-ent-coef "$PPO_ENT_COEF" \
      --ppo-epochs "$PPO_EPOCHS" \
      --setup-phase-loss-weight "$SETUP_PHASE_LOSS_WEIGHT" \
      --model-arch "$MODEL_ARCH" \
      --model-hidden "$MODEL_HIDDEN" \
      --model-residual-blocks "$MODEL_RESIDUAL_BLOCKS" \
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
      --model-arch "$MODEL_ARCH" \
      --model-hidden "$MODEL_HIDDEN" \
      --model-residual-blocks "$MODEL_RESIDUAL_BLOCKS" \
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
DYNAMIC_FROZEN_POOL_CSV="$CKPT_A,$CKPT_B,$CKPT_C"
if [[ "$DYNAMIC_FROZEN_MAX" -gt 0 ]]; then
  DYNAMIC_FROZEN_POOL_CSV=$(csv_tail_n "$DYNAMIC_FROZEN_POOL_CSV" "$DYNAMIC_FROZEN_MAX")
fi
ANCHOR_POOL_CSV=""
if [[ -n "$ANCHOR_MANIFEST" ]]; then
  [[ -f "$ANCHOR_MANIFEST" ]] || die "ANCHOR_MANIFEST not found: $ANCHOR_MANIFEST"
  ANCHOR_POOL_CSV=$(python3 - <<'PY' "$ANCHOR_MANIFEST" "$ANCHOR_TAG_FILTER"
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
tag_filter = {x.strip() for x in sys.argv[2].split(",") if x.strip()}
data = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(data, list):
    raise SystemExit("Anchor manifest must be a JSON list.")

rows = []
for item in data:
    if not isinstance(item, dict):
        continue
    if item.get("active", True) is False:
        continue
    p = str(item.get("path", "")).strip()
    if not p:
        continue
    tags = [str(t).strip() for t in item.get("tags", []) if str(t).strip()]
    if tag_filter and not any(t in tag_filter for t in tags):
        continue
    weight = float(item.get("weight", 1.0))
    reps = max(1, min(5, int(round(weight))))
    if pathlib.Path(p).exists():
        rows.extend([p] * reps)

print(",".join(rows))
PY
)
fi
FROZEN_POOL_CSV="$DYNAMIC_FROZEN_POOL_CSV"
if [[ -n "$ANCHOR_POOL_CSV" ]]; then
  FROZEN_POOL_CSV="${ANCHOR_POOL_CSV},${FROZEN_POOL_CSV}"
fi

echo "Starting 3-branch league training"
echo "base_out=$BASE_OUT cycles=$CYCLES python=$PYTHON_BIN"
echo "initial_pool=$FROZEN_POOL_CSV"
if [[ -n "$ANCHOR_POOL_CSV" ]]; then
  echo "anchor_pool=$ANCHOR_POOL_CSV"
fi

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

  if [[ "$PROMOTION_GATE_ENABLE" == "1" && "$PROMOTION_BORDERLINE_GAMES_PER_SEAT" -gt 12 ]]; then
    for idx in 0 1 2; do
      branch="${BRANCHES[$idx]}"
      out_dir="$cycle_dir/branch_${branch}"
      eval_out="$out_dir/eval_both.json"
      best_ckpt="$out_dir/best_checkpoint.pt"
      borderline=$(python3 - <<'PY' "$eval_out" "$PROMOTION_MIN_STRICT" "$PROMOTION_MIN_WORST_SEAT" "$PROMOTION_MIN_OPENING" "$PROMOTION_BORDERLINE_MARGIN"
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
strict_min = float(sys.argv[2])
worst_min = float(sys.argv[3])
opening_min = float(sys.argv[4])
margin = float(sys.argv[5])

random_eval = data["seat_permutation_eval"]["random"]
heuristic_eval = data["seat_permutation_eval"]["heuristic"]
opening_quality = 0.5 * (
    float(random_eval.get("mean_opening_percentile", 0.0))
    + float(heuristic_eval.get("mean_opening_percentile", 0.0))
)
strict = float(data.get("strict_score_eval", 0.0))
worst = float(data.get("worst_seat_strict_eval", 0.0))
is_borderline = (
    (strict_min - margin <= strict < strict_min)
    or (worst_min - margin <= worst < worst_min)
    or (opening_min >= 0 and opening_min - margin <= opening_quality < opening_min)
)
print("1" if is_borderline else "0")
PY
)
      if [[ "$borderline" == "1" ]]; then
        echo "-> Borderline branch ${branch}; re-evaluating with games_per_seat=${PROMOTION_BORDERLINE_GAMES_PER_SEAT}"
        "$PYTHON_BIN" -u scripts/eval_checkpoint.py \
          --checkpoint "$best_ckpt" \
          --training-history "$out_dir/progress_reports.jsonl" \
          --promotion-decision "$out_dir/best_checkpoint_meta.json" \
          --games-per-seat "$PROMOTION_BORDERLINE_GAMES_PER_SEAT" \
          --seed "$(( BRANCH_SEED_BASES[$idx] + cycle + 900000 ))" \
          --opponent both \
          --max-steps "$EVAL_MAX_STEPS" \
          --max-main-actions-per-turn "$MAX_MAIN_ACTIONS" \
          --trade-action-mode "$TRADE_MODE" \
          --max-player-trade-proposals-per-turn "$MAX_TRADE_PROPOSALS" \
          --model-arch "$MODEL_ARCH" \
          --model-hidden "$MODEL_HIDDEN" \
          --model-residual-blocks "$MODEL_RESIDUAL_BLOCKS" \
          --out "$eval_out"
      fi
    done
  fi

  ranking_json="$cycle_dir/ranking.json"
  python3 - <<'PY' "$ranking_json" "$PROMOTION_GATE_ENABLE" "$PROMOTION_MIN_STRICT" "$PROMOTION_MIN_WORST_SEAT" "$PROMOTION_MIN_OPENING" "${eval_files[@]}"
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
promotion_gate = str(sys.argv[2]) == "1"
strict_min = float(sys.argv[3])
worst_min = float(sys.argv[4])
opening_min = float(sys.argv[5])
rows = []
for p in sys.argv[6:]:
    data = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    random_eval = data["seat_permutation_eval"]["random"]
    heuristic_eval = data["seat_permutation_eval"]["heuristic"]
    by_seat = []
    for seat in range(4):
        rs = random_eval["seat_stats"][seat]["strict_win_rate"]
        hs = heuristic_eval["seat_stats"][seat]["strict_win_rate"]
        by_seat.append(0.5 * (float(rs) + float(hs)))
    opening_quality = 0.5 * (
        float(random_eval.get("mean_opening_percentile", 0.0))
        + float(heuristic_eval.get("mean_opening_percentile", 0.0))
    )
    road_useful_rate = 0.5 * (
        float(random_eval.get("mean_road_useful_rate", 0.0))
        + float(heuristic_eval.get("mean_road_useful_rate", 0.0))
    )
    knight_play_rate = 0.5 * (
        float(random_eval.get("mean_knight_play_rate_when_available", 0.0))
        + float(heuristic_eval.get("mean_knight_play_rate_when_available", 0.0))
    )
    knight_hold_before_first = 0.5 * (
        float(random_eval.get("mean_knight_held_turns_before_first_play", 0.0))
        + float(heuristic_eval.get("mean_knight_held_turns_before_first_play", 0.0))
    )
    yop_play_rate = 0.5 * (
        float(random_eval.get("mean_yop_play_rate_when_available", 0.0))
        + float(heuristic_eval.get("mean_yop_play_rate_when_available", 0.0))
    )
    monopoly_play_rate = 0.5 * (
        float(random_eval.get("mean_monopoly_play_rate_when_available", 0.0))
        + float(heuristic_eval.get("mean_monopoly_play_rate_when_available", 0.0))
    )
    trade_offer_rate_main = 0.5 * (
        float(random_eval.get("mean_trade_offer_rate_when_main", 0.0))
        + float(heuristic_eval.get("mean_trade_offer_rate_when_main", 0.0))
    )
    trade_bank_rate_main = 0.5 * (
        float(random_eval.get("mean_trade_bank_rate_when_main", 0.0))
        + float(heuristic_eval.get("mean_trade_bank_rate_when_main", 0.0))
    )
    trade_accept_rate_prompted = 0.5 * (
        float(random_eval.get("mean_trade_accept_rate_when_prompted", 0.0))
        + float(heuristic_eval.get("mean_trade_accept_rate_when_prompted", 0.0))
    )
    trade_reject_rate_prompted = 0.5 * (
        float(random_eval.get("mean_trade_reject_rate_when_prompted", 0.0))
        + float(heuristic_eval.get("mean_trade_reject_rate_when_prompted", 0.0))
    )
    pass_gate = True
    reasons = []
    if promotion_gate:
        if float(data["strict_score_eval"]) < strict_min:
            pass_gate = False
            reasons.append("strict")
        if float(min(by_seat)) < worst_min:
            pass_gate = False
            reasons.append("worst_seat")
        if opening_min >= 0.0 and opening_quality < opening_min:
            pass_gate = False
            reasons.append("opening")
    rows.append(
        {
            "eval_path": p,
            "strict_score_eval": float(data["strict_score_eval"]),
            "worst_seat_strict": float(min(by_seat)),
            "mean_seat_strict": float(sum(by_seat) / len(by_seat)),
            "seat_strict": by_seat,
            "opening_quality": opening_quality,
            "road_useful_rate": road_useful_rate,
            "knight_play_rate": knight_play_rate,
            "knight_hold_before_first": knight_hold_before_first,
            "yop_play_rate": yop_play_rate,
            "monopoly_play_rate": monopoly_play_rate,
            "trade_offer_rate_main": trade_offer_rate_main,
            "trade_bank_rate_main": trade_bank_rate_main,
            "trade_accept_rate_prompted": trade_accept_rate_prompted,
            "trade_reject_rate_prompted": trade_reject_rate_prompted,
            "pass_gate": bool(pass_gate),
            "gate_reasons": reasons,
        }
    )

rows.sort(
    key=lambda r: (
        r["strict_score_eval"],
        r["worst_seat_strict"],
        r["mean_seat_strict"],
        r["opening_quality"],
        r["road_useful_rate"],
    ),
    reverse=True,
)
out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(
    "rank,strict_score,worst_seat,opening_quality,road_useful_rate,"
    "knight_play_rate,knight_hold_before_first,yop_play_rate,monopoly_play_rate,"
    "trade_offer_main,trade_bank_main,trade_accept_prompted,trade_reject_prompted,eval_path"
)
for i, r in enumerate(rows, start=1):
    print(
        f"{i},{r['strict_score_eval']:.6f},{r['worst_seat_strict']:.6f},"
        f"{r['opening_quality']:.6f},{r['road_useful_rate']:.6f},"
        f"{r['knight_play_rate']:.6f},{r['knight_hold_before_first']:.6f},"
        f"{r['yop_play_rate']:.6f},{r['monopoly_play_rate']:.6f},"
        f"{r['trade_offer_rate_main']:.6f},{r['trade_bank_rate_main']:.6f},"
        f"{r['trade_accept_rate_prompted']:.6f},{r['trade_reject_rate_prompted']:.6f},"
        f"{r['eval_path']}"
    )
PY

  champ_eval=$(python3 - <<'PY' "$ranking_json"
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
champ = next((r for r in rows if r.get("pass_gate", True)), rows[0])
print(champ["eval_path"])
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

  # Promotion gate: keep prior checkpoint if candidate did not pass gate.
  for idx in 0 1 2; do
    branch="${BRANCHES[$idx]}"
    eval_path="$cycle_dir/branch_${branch}/eval_both.json"
    pass_gate=$(python3 - <<'PY' "$ranking_json" "$eval_path"
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
row = next((r for r in rows if r["eval_path"] == sys.argv[2]), None)
print("1" if (row is None or row.get("pass_gate", True)) else "0")
PY
)
    if [[ "$PROMOTION_GATE_ENABLE" == "1" && "$pass_gate" != "1" ]]; then
      echo "Gate blocked promotion for branch ${branch}; keeping prior checkpoint"
      NEXT_CKPTS[$idx]="${CURRENT_CKPTS[$idx]}"
    fi
  done

  champ_ckpt="${NEXT_CKPTS[$champ_idx]}"
  echo "Cycle ${cycle} champion branch=${BRANCHES[$champ_idx]} ckpt=$champ_ckpt"
  echo "Cycle ${cycle} weakest branch=${BRANCHES[$weak_idx]}"

  # Advance each branch to its own best checkpoint.
  for idx in 0 1 2; do
    CURRENT_CKPTS[$idx]="${NEXT_CKPTS[$idx]}"
  done

  # Optional league mechanic: weakest branch restarts from champion.
  if [[ "$RESET_WEAKEST_FROM_CHAMP" == "1" ]]; then
    reset_source=$(python3 - <<PY
import numpy as np
champ_p = max(0.0, float("$RESET_SOURCE_CHAMP_PROB"))
anchor_p = max(0.0, float("$RESET_SOURCE_ANCHOR_PROB"))
if "$ANCHOR_POOL_CSV".strip() == "":
    print("champion")
else:
    s = champ_p + anchor_p
    if s <= 0:
        print("champion")
    else:
        p = np.asarray([champ_p / s, anchor_p / s], dtype=np.float64)
        print(np.random.default_rng().choice(["champion", "anchor"], p=p))
PY
)
    if [[ "$reset_source" == "anchor" && -n "$ANCHOR_POOL_CSV" ]]; then
      reset_ckpt=$(python3 - <<'PY' "$ANCHOR_POOL_CSV"
import numpy as np, sys
vals = [x for x in sys.argv[1].split(",") if x]
print(np.random.default_rng().choice(vals) if vals else "")
PY
)
      if [[ -n "$reset_ckpt" ]]; then
        CURRENT_CKPTS[$weak_idx]="$reset_ckpt"
        echo "Reset weakest branch ${BRANCHES[$weak_idx]} -> anchor checkpoint"
      else
        CURRENT_CKPTS[$weak_idx]="$champ_ckpt"
        reset_source="champion_fallback"
        echo "Reset weakest branch ${BRANCHES[$weak_idx]} -> champion checkpoint (anchor fallback)"
      fi
    else
      CURRENT_CKPTS[$weak_idx]="$champ_ckpt"
      echo "Reset weakest branch ${BRANCHES[$weak_idx]} -> champion checkpoint"
    fi
    python3 - <<'PY' "$cycle_dir/reset_decision.json" "$cycle" "${BRANCHES[$weak_idx]}" "$reset_source" "${CURRENT_CKPTS[$weak_idx]}"
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
row = {
    "cycle": int(sys.argv[2]),
    "weak_branch": sys.argv[3],
    "reset_source": sys.argv[4],
    "checkpoint": sys.argv[5],
}
path.write_text(json.dumps(row, indent=2), encoding="utf-8")
PY
  fi

  # Expand frozen pool with champion each cycle.
  if ! contains_csv_item "$DYNAMIC_FROZEN_POOL_CSV" "$champ_ckpt"; then
    DYNAMIC_FROZEN_POOL_CSV="${DYNAMIC_FROZEN_POOL_CSV},${champ_ckpt}"
  fi
  if [[ "$DYNAMIC_FROZEN_MAX" -gt 0 ]]; then
    DYNAMIC_FROZEN_POOL_CSV=$(csv_tail_n "$DYNAMIC_FROZEN_POOL_CSV" "$DYNAMIC_FROZEN_MAX")
  fi
  FROZEN_POOL_CSV="$DYNAMIC_FROZEN_POOL_CSV"
  if [[ -n "$ANCHOR_POOL_CSV" ]]; then
    FROZEN_POOL_CSV="${ANCHOR_POOL_CSV},${FROZEN_POOL_CSV}"
  fi
  echo "updated_frozen_pool=$FROZEN_POOL_CSV"
done

echo ""
echo "League run complete."
echo "Latest branch checkpoints:"
for idx in 0 1 2; do
  echo "  ${BRANCHES[$idx]} => ${CURRENT_CKPTS[$idx]}"
done
