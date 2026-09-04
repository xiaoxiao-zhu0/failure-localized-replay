#!/usr/bin/env bash
set -euo pipefail

# Run LPR under the same protocol as the paper's baseline matrix.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$HOME/miniconda3/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/avalanche}"
ROOT="${ROOT:-$PROJECT_ROOT/results/rbcl/lpr_baseline_pattern_recognition}"
SEEDS="${SEEDS:-218 219 220 221 222}"
GPUS="${GPUS:-0 1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs" "$ROOT/runtime" "$ROOT/analysis"
cd "$PROJECT_ROOT"

manifest="$ROOT/runtime/execution_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'seed\tbenchmark\tmethod\tgpu\tstart_iso\tend_iso\tseconds\tstatus\n' > "$manifest"
fi

validate_lpr_summary() {
  local summary="$1"
  "$PYTHON" scripts/validate_rbcl_summary.py "$summary" \
    --expected-experiences 10 >/dev/null 2>&1 && \
  "$PYTHON" - "$summary" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    payload = json.load(stream)
audit = payload.get("lpr_audit") or {}
if audit.get("preconditioner_updates", 0) <= 0:
    raise SystemExit("LPR preconditioner did not update")
if audit.get("preconditioned_updates", 0) <= 0:
    raise SystemExit("LPR gradients were not preconditioned")
PY
}

run_one() {
  local seed="$1" benchmark="$2" gpu="$3"
  local summary="$ROOT/$benchmark/seed_$seed/lpr/summary.json"
  if [[ -f "$summary" ]] && validate_lpr_summary "$summary"; then
    echo "[LPR] skip valid existing: $summary"
    return
  fi
  local log="$ROOT/logs/seed${seed}_${benchmark}_lpr.log"
  local start end start_iso end_iso status="failed"
  start=$(date +%s)
  start_iso=$(date --iso-8601=seconds)
  echo "[LPR] start seed=$seed benchmark=$benchmark gpu=$gpu"
  if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" examples/rbcl_run_experiment.py \
    --benchmark "$benchmark" --n_experiences 10 --model slim_resnet18 \
    --cuda 0 --train_epochs 5 --train_mb_size 64 --replay_mb_size 64 \
    --eval_mb_size 128 --mem_size 100 --lr 0.05 --momentum 0.9 \
    --validation_fraction 0 --dataset_root "$DATASET_ROOT" --seed "$seed" \
    --quiet --deterministic --strategies lpr --no_instability \
    --historical_reference test_stream --full_metrics --output_dir "$ROOT" \
    > "$log" 2>&1 && validate_lpr_summary "$summary"; then
    status="completed"
  fi
  end=$(date +%s)
  end_iso=$(date --iso-8601=seconds)
  { flock 9; printf '%s\t%s\tlpr\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$benchmark" "$gpu" "$start_iso" "$end_iso" \
      "$((end-start))" "$status" >&9; } 9>> "$manifest"
  if [[ "$status" != "completed" ]]; then
    echo "[LPR] failed seed=$seed benchmark=$benchmark log=$log" >&2
    return 1
  fi
  echo "[LPR] done seed=$seed benchmark=$benchmark seconds=$((end-start))"
}

export -f validate_lpr_summary run_one
export PROJECT_ROOT PYTHON DATASET_ROOT ROOT manifest

jobs="$ROOT/runtime/lpr_jobs.tsv"
: > "$jobs"
read -r -a gpu_list <<< "$GPUS"
(( ${#gpu_list[@]} > 0 )) || { echo "GPUS must contain at least one GPU id" >&2; exit 1; }
i=0
for seed in $SEEDS; do
  for benchmark in split_cifar100 equal_exposure_blurry_cifar100; do
    gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    printf '%s\t%s\t%s\n' "$seed" "$benchmark" "$gpu" >> "$jobs"
    i=$((i + 1))
  done
done

workers="${WORKERS:-${#gpu_list[@]}}"
xargs -P "$workers" -n 3 bash -c 'run_one "$1" "$2" "$3"' _ < "$jobs"

completed=$(find "$ROOT" -path '*/lpr/summary.json' -type f | wc -l)
expected=$((2 * $(wc -w <<< "$SEEDS")))
if [[ "$completed" -eq "$expected" ]]; then
  touch "$ROOT/COMPLETE"
  echo "[LPR] complete: $ROOT"
else
  echo "[LPR] incomplete: $completed/$expected summaries found" >&2
  exit 1
fi
