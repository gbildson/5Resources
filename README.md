# Catan RL

Reinforcement learning project scaffold for Catan based on `catan_ds_o46.md`.

## What is included

- O46-aligned state model and deterministic topology tables.
- Full flat action catalog with parameterized actions and legal-action masks.
- Turn/phase environment with setup, robber/discard flow, building, dev cards, and trade hooks.
- Invariant checks and tests for random legal playouts.
- Baseline agents (`RandomLegalAgent`, `HeuristicAgent`) and tournament harness.
- Masked PPO training loop with self-play, checkpoint saving, and league promotion gate.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/run_tournament.py --games 20
python scripts/train_ppo.py --updates 10 --rollout-steps 1024
python scripts/train_schedule.py --total-updates 100 --report-every 25 --rollout-steps 1024 --games-per-seat 10 --out-dir artifacts_schedule
python scripts/train_schedule.py --bc-warmstart --bc-steps 20000 --bc-epochs 3 --total-updates 100 --report-every 25 --rollout-steps 1024 --games-per-seat 10 --out-dir artifacts_schedule_bc
python scripts/train_schedule.py --bc-warmstart --reward-shaping-vp 0.01 --reward-shaping-resource 0.001 --max-episode-steps 2000 --total-updates 100 --report-every 25 --rollout-steps 1024 --games-per-seat 10 --out-dir artifacts_schedule_rewarded
python scripts/train_schedule.py --bc-warmstart --max-main-actions-per-turn 12 --disable-player-trade --total-updates 100 --report-every 25 --rollout-steps 1024 --games-per-seat 10 --out-dir artifacts_schedule_curriculum
python scripts/train_schedule.py --bc-warmstart --ppo-lr 0.0001 --ppo-ent-coef 0.02 --ppo-epochs 3 --out-dir artifacts_schedule_tuned
python scripts/train_schedule.py --bc-warmstart --ppo-ent-coef-start 0.04 --ppo-ent-coef-end 0.005 --max-episode-steps 600 --eval-max-steps 2000 --out-dir artifacts_schedule_v4
python scripts/train_schedule.py --init-checkpoint artifacts_schedule_curriculum_v4/best_checkpoint.pt --total-updates 100 --report-every 20 --out-dir artifacts_schedule_resume
python scripts/train_schedule.py --init-checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt --use-opponent-mixture --opponent-seat-count 1 --mix-heuristic-prob 0.5 --mix-random-prob 0.2 --mix-frozen-prob 0.3 --mix-frozen-checkpoints "artifacts_schedule_v5_no_trade/best_checkpoint.pt,artifacts_schedule_v8_notrade_long/best_checkpoint.pt" --out-dir artifacts_schedule_mix
python scripts/inspect_board.py --seed 123
python scripts/trace_game.py --checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt --seed 42 --model-players 0 --max-steps 300 --show-final-board
python scripts/trace_game.py --checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt --seed 42 --model-players 0 --max-steps 300 --show-final-inspect-board --show-final-inspect-topology
python scripts/trace_game.py --checkpoint artifacts_schedule_v8_notrade_long/best_checkpoint.pt --seed 42 --model-players 0 --max-steps 300 --show-final-board-pycatan
python scripts/eval_checkpoint.py --checkpoint artifacts_schedule_realboard_v1/checkpoints/policy_u75.pt --games-per-seat 8 --max-main-actions-per-turn 10 --disable-player-trade --max-steps 2200 --out artifacts_eval_fast.json
```

## Notes

- The simulator is designed for RL iteration speed and deterministic contracts.
- The topology is fixed-size and deterministic to satisfy tensorized training interfaces.
- For stronger play, extend rule fidelity and trade parameterization in later iterations.
- `train_schedule.py` writes periodic reports to JSONL/CSV so training progress can be monitored without manual evaluation commands.
- Use `--bc-warmstart` to pretrain from heuristic policy trajectories before PPO self-play.
- Training can include early traction signals: truncation leader reward at max episode steps and small dense shaping from VP/resource deltas.
- For anti-stall curriculum, use `--max-main-actions-per-turn` and optionally `--disable-player-trade` in early runs.
- `train_schedule.py` tracks and saves the best checkpoint by mean strict win rate (`best_checkpoint.pt`) and emits collapse warnings when entropy gets too low while strict score drops.
- You can decouple training/eval horizons with `--max-episode-steps` (training) and `--eval-max-steps` (evaluation), and schedule exploration with `--ppo-ent-coef-start/--ppo-ent-coef-end`.
- Use `--init-checkpoint` to resume training from a previously saved model; do not combine with `--bc-warmstart`.
- For robustness training, `--use-opponent-mixture` lets one or more seats be controlled by sampled frozen/heuristic/random opponents while still updating only the current learner policy.
- `trace_game.py --show-final-board-pycatan` is optional and only needs `pycatan` installed in the active virtualenv.
- Use `scripts/inspect_board.py` to visually inspect generated board layouts (hex map, ports, and token distribution). Add `--show-topology` for full vertex/edge tables.
- `trace_game.py` supports `--show-final-inspect-board` to print an inspect-style view of the final traced state; add `--show-final-inspect-topology` to include full topology tables in the same output.
- `eval_checkpoint.py` now supports fast validation defaults (small `--games-per-seat`), strict-win reporting, direct JSONL training-history loading, and environment parity flags (`--max-main-actions-per-turn`, `--disable-player-trade`, `--max-steps`).
- Player-trade masking can use `--trade-action-mode guided` (default) to narrow propose-trade actions to near-term useful offers, or `--trade-action-mode full` to expose all legal trade templates.
- You can throttle player trade spam with `--max-player-trade-proposals-per-turn` (e.g. `1`) as a trade-only curriculum control.
- Compact engineered strategy features (build deficits/readiness, production opportunity, discard pressure, VP pressure, short-horizon public resource-flow memory, robber-target quality proxies, expansion/achievement race pressure, and trade-credibility signals) are always included in the RL input.
