#!/usr/bin/env bash
set -euo pipefail

cd /home/zhuyihui/data/wb/kernel_exp-main

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/src
export TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache

py=/home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv/bin/python
config=llm_pretrain_0p2b/configs/llm_0p2b_1k_3090.json
data_dir=llm_pretrain_0p2b/data/cci3_hq_1b_llama2
save_dir=llm_pretrain_0p2b/checkpoints/pretrain_0p2b_1k_cci3_1b
results_dir=llm_pretrain_0p2b/results
run_name=pretrain_0p2b_1k_cci3_1b_bs8_ga4
log_path=llm_pretrain_0p2b/logs/pretrain_0p2b_cci3_1b.log

exec > >(tee -a "$log_path") 2>&1

echo "===== start $(date -Is) ====="
echo "host=$(hostname) user=$(whoami) pid=$$"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader || true
echo "python=$py"
"$py" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
PY

resume_args=()
if [ -f "$save_dir/last.pt" ]; then
  resume_args=(--resume "$save_dir/last.pt")
  echo "resuming from $save_dir/last.pt"
fi

"$py" -m llm0.train \
  --config "$config" \
  --data-dir "$data_dir" \
  --steps 30518 \
  --batch-size 8 \
  --grad-accum-steps 4 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --warmup-steps 500 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --eval-every 500 \
  --eval-iters 10 \
  --save-dir "$save_dir" \
  --save-every 1000 \
  --results-dir "$results_dir" \
  --run-name "$run_name" \
  "${resume_args[@]}"

echo "===== finished $(date -Is) ====="
