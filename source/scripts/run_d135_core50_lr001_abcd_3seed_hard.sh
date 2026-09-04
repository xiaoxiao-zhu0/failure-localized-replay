#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/avalanche}"
ROOT="${ROOT:-$PROJECT_ROOT/results/rbcl/d135_core50_lr001_abcd_3seed_hard}"
GPU="${GPU:-1}"
SEEDS="${SEEDS:-319 320 321}"

METHODS=(
  causal_er_ace
  semantic_proto_hybrid_75_25
  persistent_srrd_selective_swap_1
  persistent_srrd_prequential_arbitration_1
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT/logs" "$ROOT/runtime" "$ROOT/analysis"
manifest="$ROOT/runtime/execution_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'seed\tmethod\tstart_iso\tend_iso\tseconds\tstatus\n' > "$manifest"
fi

run_one() {
  local seed="$1" method="$2"
  local summary="$ROOT/core50/seed_$seed/$method/summary.json"
  local log="$ROOT/logs/seed${seed}_${method}.log"
  if [[ -f "$summary" ]] && "$PYTHON" scripts/validate_rbcl_summary.py \
      --expected-experiences 9 "$summary" >/dev/null 2>&1; then
    echo "[D135] skip valid existing: $summary"
    return
  fi

  local start end start_iso end_iso status="failed"
  start=$(date +%s)
  start_iso=$(date --iso-8601=seconds)
  if CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" examples/rbcl_run_experiment.py \
      --benchmark core50 --n_experiences 9 --model slim_resnet18 --cuda 0 \
      --train_epochs 5 --train_mb_size 64 --eval_mb_size 128 --mem_size 100 \
      --lr 0.01 --momentum 0.9 --validation_fraction 0 \
      --dataset_root "$DATASET_ROOT" --seed "$seed" --quiet --deterministic \
      --strategies "$method" --no_instability --memory_trace_signature \
      --historical_reference test_stream --full_metrics --output_dir "$ROOT" \
      > "$log" 2>&1 && "$PYTHON" scripts/validate_rbcl_summary.py \
      --expected-experiences 9 "$summary"; then
    status="completed"
  fi
  end=$(date +%s)
  end_iso=$(date --iso-8601=seconds)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$seed" "$method" "$start_iso" \
    "$end_iso" "$((end-start))" "$status" >> "$manifest"
  [[ "$status" == "completed" ]] || { echo "[D135] failed: $log" >&2; return 1; }
}

for seed in $SEEDS; do
  for method in "${METHODS[@]}"; do
    run_one "$seed" "$method"
  done
done

"$PYTHON" scripts/analyze_d135_abcd.py --root "$ROOT" --seeds $SEEDS
touch "$ROOT/COMPLETE"
