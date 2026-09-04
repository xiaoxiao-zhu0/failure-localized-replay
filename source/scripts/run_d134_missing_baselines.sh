#!/usr/bin/env bash
set -euo pipefail

# Complete the two methods omitted from the original D134 queue.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/avalanche}"
ROOT="${ROOT:-$PROJECT_ROOT/results/rbcl/d134_tinyimagenet_normalization_correction_parallel2}"
SEEDS="${SEEDS:-218 219 220 221 222}"
GPU="${GPU:-1}"
WORKERS="${WORKERS:-2}"

OBC="persistent_srrd_obc_1"
CEACE="causal_er_ace"
BENCHMARKS=(split_tinyimagenet equal_exposure_blurry_tinyimagenet)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT/logs" "$ROOT/runtime" "$ROOT/analysis"

manifest="$ROOT/runtime/execution_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'seed\tbenchmark\tmethod\tstart_iso\tend_iso\tseconds\tstatus\n' > "$manifest"
fi

run_one() {
  local seed="$1" benchmark="$2" method="$3"
  local summary="$ROOT/$benchmark/seed_$seed/$method/summary.json"
  if [[ -f "$summary" ]] && "$PYTHON" scripts/validate_rbcl_summary.py "$summary" >/dev/null 2>&1; then
    echo "[D134-missing] skip valid existing: $summary"
    return 0
  fi

  local log="$ROOT/logs/seed${seed}_${benchmark}_${method}.log"
  local start end start_iso end_iso status="failed"
  start=$(date +%s)
  start_iso=$(date --iso-8601=seconds)
  if CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" examples/rbcl_run_experiment.py \
    --benchmark "$benchmark" --n_experiences 10 --model slim_resnet18 \
    --cuda 0 --train_epochs 5 --train_mb_size 64 --eval_mb_size 128 \
    --mem_size 100 --lr 0.05 --momentum 0.9 --validation_fraction 0 \
    --dataset_root "$DATASET_ROOT" --seed "$seed" --quiet --deterministic \
    --strategies "$method" --no_instability --memory_trace_signature \
    --historical_reference test_stream --full_metrics --output_dir "$ROOT" \
    > "$log" 2>&1 && "$PYTHON" scripts/validate_rbcl_summary.py "$summary"; then
    status="completed"
  fi
  end=$(date +%s)
  end_iso=$(date --iso-8601=seconds)
  printf '%s\n' "$((end-start))" > "$ROOT/runtime/seed${seed}_${benchmark}_${method}.seconds"
  { flock 9; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$benchmark" "$method" "$start_iso" "$end_iso" \
      "$((end-start))" "$status" >&9; } 9>> "$manifest"
  [[ "$status" == "completed" ]] || { echo "[D134-missing] failed: $log" >&2; return 1; }
  echo "[D134-missing] completed seed=$seed benchmark=$benchmark method=$method"
}

run_pair() {
  run_one "$1" "$2" "$OBC"
  run_one "$1" "$2" "$CEACE"
}
export -f run_one run_pair
export PROJECT_ROOT PYTHON DATASET_ROOT ROOT GPU OBC CEACE manifest

jobs="$ROOT/runtime/missing_baseline_jobs.tsv"
: > "$jobs"
for seed in $SEEDS; do
  for benchmark in "${BENCHMARKS[@]}"; do
    printf '%s\t%s\n' "$seed" "$benchmark" >> "$jobs"
  done
done

xargs -P "$WORKERS" -n 2 bash -c 'run_pair "$1" "$2"' _ < "$jobs"
echo "[D134-missing] all missing baseline runs completed"
