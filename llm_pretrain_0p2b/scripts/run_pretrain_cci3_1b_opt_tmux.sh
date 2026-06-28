#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhuyihui/data/wb/kernel_exp-main
PRETRAIN="$ROOT/llm_pretrain_0p2b"
PY=/home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv/bin/python
CONFIG="$PRETRAIN/configs/llm_0p2b_1k_3090.json"
DATA="$PRETRAIN/data/cci3_hq_1b_llama2"
RESULTS="$PRETRAIN/results"
BASE_SAVE="$PRETRAIN/checkpoints/pretrain_0p2b_1k_cci3_1b_opt"
TARGET_TOKENS=1000000000

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PRETRAIN/src"
export TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache
export TOKENIZERS_PARALLELISM=false

mkdir -p "$PRETRAIN/logs" "$RESULTS" "$BASE_SAVE"

prune_checkpoints() {
  local save_dir="$1"
  [ -d "$save_dir" ] || return 0
  mapfile -t old_files < <(
    find "$save_dir" -maxdepth 1 -type f -name 'step_*.pt' -printf '%T@ %p\n' \
      | sort -nr \
      | awk 'NR>3 {first=$1; $1=""; sub(/^ /,""); print}'
  )
  if [ "${#old_files[@]}" -gt 0 ]; then
    printf '%s\n' "${old_files[@]}" | xargs -r rm -f
  fi
}

checkpoint_pruner_loop() {
  local save_dir="$1"
  local pid="$2"
  while kill -0 "$pid" 2>/dev/null; do
    prune_checkpoints "$save_dir"
    sleep 60
  done
  prune_checkpoints "$save_dir"
}

run_candidate() {
  local name="$1"
  local batch="$2"
  local accum="$3"
  local ckpt_flag="$4"
  shift 4

  local global_tokens=$((batch * accum * 1024))
  local steps=$(((TARGET_TOKENS + global_tokens - 1) / global_tokens))
  local save_dir="$BASE_SAVE/$name"
  local run_name="pretrain_0p2b_1k_cci3_1b_opt_${name}"

  mkdir -p "$save_dir"
  echo
  echo "[$(date)] starting candidate=$name batch=$batch grad_accum=$accum global_tokens=$global_tokens steps=$steps checkpointing=$ckpt_flag"
  echo "[$(date)] save_dir=$save_dir"

  set +e
  "$PY" -m llm0.train \
    --config "$CONFIG" \
    --data-dir "$DATA" \
    --steps "$steps" \
    --batch-size "$batch" \
    --grad-accum-steps "$accum" \
    --dtype bfloat16 \
    --warmup-steps 500 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --eval-every 500 \
    --eval-iters 10 \
    --save-dir "$save_dir" \
    --save-every 500 \
    --results-dir "$RESULTS" \
    --run-name "$run_name" \
    "$@" &
  local train_pid=$!
  checkpoint_pruner_loop "$save_dir" "$train_pid" &
  local prune_pid=$!
  wait "$train_pid"
  local status=$?
  kill "$prune_pid" 2>/dev/null || true
  wait "$prune_pid" 2>/dev/null || true
  prune_checkpoints "$save_dir"
  set -e

  echo "[$(date)] candidate=$name finished status=$status"
  return "$status"
}

# ???????????activation checkpointing?????micro batch????????????????# ??? OOM ?????????????????????????????run_candidate bs16_ga2_nockpt 16 2 off && exit 0
run_candidate bs12_ga3_nockpt 12 3 off && exit 0
run_candidate bs8_ga4_nockpt 8 4 off && exit 0
run_candidate bs16_ga2_ckpt 16 2 on --gradient-checkpointing && exit 0
run_candidate bs8_ga4_ckpt 8 4 on --gradient-checkpointing && exit 0

echo "[$(date)] all candidates failed"
exit 1
