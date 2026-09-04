#!/usr/bin/env bash
set -euo pipefail

# Run LPR under the same protocol as the paper's baseline matrix.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/avalanche}"
ROOT="${ROOT:-$PROJECT_ROOT/results/rbcl/lpr_baseline_pattern_recognition}"
SEEDS="${SEEDS:-218 219 220 221 222}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-1}"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs" "$ROOT/runtime" "$ROOT/analysis"
cd "$PROJECT_ROOT"

run_one() {
  local seed="$1" benchmark="$2"
  local summary="$ROOT/$benchmark/seed_$seed/lpr/summary.json"
  if [[ -f "$summary" ]] && "$PYTHON" scripts/validate_rbcl_summary.py "$summary" --expected-experiences 10 >/dev/null 2>&1; then
    echo "[LPR] skip valid existing: $summary"
    return
  fi
  local log="$ROOT/logs/seed${seed}_${benchmark}_lpr.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" examples/rbcl_run_experiment.py \
    --benchmark "$benchmark" --n_experiences 10 --model slim_resnet18 \
    --cuda 0 --train_epochs 5 --train_mb_size 64 --replay_mb_size 64 \
    --eval_mb_size 128 --mem_size 100 --lr 0.05 --momentum 0.9 \
    --validation_fraction 0 --dataset_root "$DATASET_ROOT" --seed "$seed" \
    --quiet --deterministic --strategies lpr --no_instability \
    --historical_reference test_stream --full_metrics --output_dir "$ROOT" \
    > "$log" 2>&1
  "$PYTHON" scripts/validate_rbcl_summary.py "$summary" --expected-experiences 10
}

for seed in $SEEDS; do
  for benchmark in split_cifar100 equal_exposure_blurry_cifar100; do
    run_one "$seed" "$benchmark"
  done
done

touch "$ROOT/COMPLETE"
echo "[LPR] complete: $ROOT"
