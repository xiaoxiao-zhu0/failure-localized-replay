#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/data/avalanche}"
ROOT="${ROOT:-$PROJECT_ROOT/results/rbcl/d136_design_choice_controls_serial}"
SEEDS="${SEEDS:-252 253 254}"
GPU="${GPU:-0}"

METHODS=(
  persistent_srrd_prequential_random
  persistent_srrd_selective_swap_no_wilson
  persistent_srrd_prequential_no_wilson
)
BENCHMARKS=(split_cifar100 equal_exposure_blurry_cifar100)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT/logs" "$ROOT/analysis" "$ROOT/runtime"
manifest="$ROOT/runtime/execution_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'seed\tbenchmark\tmethod\tstart_iso\tend_iso\tseconds\tstatus\n' > "$manifest"
fi

run_one() {
  local seed="$1" benchmark="$2" method="$3"
  local summary="$ROOT/$benchmark/seed_$seed/$method/summary.json"
  local log="$ROOT/logs/seed${seed}_${benchmark}_${method}.log"
  if [[ -f "$summary" ]] && "$PYTHON" scripts/validate_rbcl_summary.py \
      --expected-experiences 10 "$summary" >/dev/null 2>&1; then
    echo "[D136] skip valid existing: $summary"
    return 0
  fi

  local start end start_iso end_iso status="failed"
  start=$(date +%s)
  start_iso=$(date --iso-8601=seconds)
  echo "[D136] start seed=$seed benchmark=$benchmark method=$method"
  if CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" examples/rbcl_run_experiment.py \
      --benchmark "$benchmark" --n_experiences 10 --model slim_resnet18 \
      --cuda 0 --train_epochs 5 --train_mb_size 64 --eval_mb_size 128 \
      --mem_size 100 --lr 0.05 --momentum 0.9 --validation_fraction 0 \
      --dataset_root "$DATASET_ROOT" --seed "$seed" --quiet --deterministic \
      --strategies "$method" --no_instability --memory_trace_signature \
      --historical_reference test_stream --full_metrics --output_dir "$ROOT" \
      > "$log" 2>&1 && "$PYTHON" scripts/validate_rbcl_summary.py \
      --expected-experiences 10 "$summary" >/dev/null 2>&1; then
    status="completed"
  fi
  end=$(date +%s)
  end_iso=$(date --iso-8601=seconds)
  { flock 9; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$seed" "$benchmark" "$method" "$start_iso" "$end_iso" \
      "$((end-start))" "$status" >&9; } 9>> "$manifest"
  if [[ "$status" != "completed" ]]; then
    echo "[D136] failed seed=$seed benchmark=$benchmark method=$method log=$log" >&2
    return 1
  fi
  echo "[D136] done seed=$seed benchmark=$benchmark method=$method seconds=$((end-start))"
}

for seed in $SEEDS; do
  for benchmark in "${BENCHMARKS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_one "$seed" "$benchmark" "$method"
    done
  done
done

cat > "$ROOT/analysis/experiment_manifest.txt" <<EOF
stage=D136 design-choice controls
seeds=$SEEDS
benchmarks=${BENCHMARKS[*]}
methods=${METHODS[*]}
gpu=$GPU
parent_protocol=D119 Split CIFAR-100 hard and equal-exposure blurry streams
resource_protocol=10 experiences; 5 epochs; train_mb=64; eval_mb=128; mem=100; lr=0.05; momentum=0.9
random_alpha=independent fixed-seed Uniform(0,1), cumulative mean for deployment
no_wilson=point-estimate positive-gap control; SRRD/EMA/threshold/one-swap/tie-break retained
EOF
touch "$ROOT/COMPLETE"
echo "[D136] complete: $ROOT"
