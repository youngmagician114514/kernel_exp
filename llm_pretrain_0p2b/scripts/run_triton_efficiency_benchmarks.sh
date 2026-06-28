#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/zhuyihui/data/wb/kernel_exp-main}
PRETRAIN="$ROOT/llm_pretrain_0p2b"
PY=${PY:-/home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv/bin/python}
CONFIG="$PRETRAIN/configs/llm_0p2b_1k_3090.json"
DATA="$PRETRAIN/data/cci3_hq_1b_llama2"
RESULTS="$PRETRAIN/results/triton_efficiency"
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT="$RESULTS/$RUN_ID"
SUMMARY_JSONL="$OUT/summary.jsonl"
SUMMARY_MD="$OUT/summary.md"
STEPS=${STEPS:-220}
WARMUP_ROWS=${WARMUP_ROWS:-30}
BATCH=${BATCH:-12}
GRAD_ACCUM=${GRAD_ACCUM:-3}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$PRETRAIN/src"
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/home/zhuyihui/data/wb/kernel_exp/.triton_cache}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT"
: > "$SUMMARY_JSONL"
cat > "$SUMMARY_MD" <<'MD'
# Triton 预训练效率短跑

| case | 改动 | status | TPS | MFU | step ms | peak allocated MB | last loss |
|---|---|---:|---:|---:|---:|---:|---:|
MD

summarize_case() {
  local case_name="$1"
  local change="$2"
  local status="$3"
  local metrics="$OUT/$case_name/metrics.jsonl"
  "$PY" - "$metrics" "$case_name" "$change" "$status" "$WARMUP_ROWS" "$SUMMARY_JSONL" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

metrics = Path(sys.argv[1])
case_name = sys.argv[2]
change = sys.argv[3]
status = int(sys.argv[4])
warmup_rows = int(sys.argv[5])
summary_jsonl = Path(sys.argv[6])
summary_md = Path(sys.argv[7])

rows = []
if metrics.exists():
    with metrics.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

row = {"case": case_name, "change": change, "status": status, "rows": len(rows)}
if rows:
    usable = rows[min(warmup_rows, max(0, len(rows) - 1)) :]
    window = usable[-100:] if len(usable) > 100 else usable
    global_tokens = int(rows[-1]["global_batch_tokens"])
    step_ms = [float(item["step_time_ms"]) for item in window]
    avg_step_ms = sum(step_ms) / len(step_ms)
    avg_tps = sum(global_tokens / (ms / 1000.0) for ms in step_ms) / len(step_ms)
    params = 203.92e6
    layers = 24
    hidden = 768
    seq_len = 1024
    peak_tflops = 80.0 * (328.0 / 336.0) * (1695.0 / 1860.0)
    flops_per_token = 6.0 * params + 6.0 * layers * hidden * seq_len
    achieved_tflops = flops_per_token * avg_tps / 1e12
    mfu = achieved_tflops / peak_tflops
    row.update(
        {
            "avg_step_ms": avg_step_ms,
            "avg_tokens_per_sec": avg_tps,
            "mfu": mfu,
            "achieved_tflops": achieved_tflops,
            "peak_allocated_mb": rows[-1].get("max_memory_allocated_mb"),
            "last_loss": rows[-1].get("train_loss"),
        }
    )

with summary_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")

if "avg_tokens_per_sec" in row:
    line = (
        f"| `{case_name}` | {change} | {status} | {row['avg_tokens_per_sec']:.0f} | "
        f"{row['mfu']:.2%} | {row['avg_step_ms']:.1f} | "
        f"{row['peak_allocated_mb']:.0f} | {row['last_loss']:.4f} |\n"
    )
else:
    line = f"| `{case_name}` | {change} | {status} | - | - | - | - | - |\n"
with summary_md.open("a", encoding="utf-8") as f:
    f.write(line)
PY
}

run_case() {
  local case_name="$1"
  local change="$2"
  shift 2
  echo
  echo "[$(date)] start case=$case_name change=$change"
  set +e
  "$PY" -m llm0.train \
    --config "$CONFIG" \
    --data-dir "$DATA" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --dtype bfloat16 \
    --warmup-steps 0 \
    --lr 3e-4 \
    --min-lr 0 \
    --eval-every 0 \
    --save-every 0 \
    --results-dir "$OUT" \
    --run-name "$case_name" \
    "$@"
  local status=$?
  set -e
  echo "[$(date)] end case=$case_name status=$status"
  summarize_case "$case_name" "$change" "$status"
  sleep 5
}

run_case baseline "PyTorch RMSNorm + PyTorch SwiGLU"
run_case triton_rmsnorm "Triton RMSNorm forward/backward" --use-triton-rmsnorm
run_case triton_swiglu "Triton SwiGLU elementwise forward/backward" --use-triton-swiglu
run_case triton_both "Triton RMSNorm + Triton SwiGLU" --use-triton-rmsnorm --use-triton-swiglu

"$PY" - "$SUMMARY_JSONL" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
baseline = next((row for row in rows if row["case"] == "baseline" and "avg_tokens_per_sec" in row), None)
if baseline is not None:
    with Path(sys.argv[2]).open("a", encoding="utf-8") as f:
        f.write("\n## \u76f8\u5bf9 baseline\n\n")
        f.write("| case | TPS ratio | MFU delta |\n")
        f.write("|---|---:|---:|\n")
        for row in rows:
            if "avg_tokens_per_sec" not in row:
                continue
            ratio = row["avg_tokens_per_sec"] / baseline["avg_tokens_per_sec"]
            delta = row["mfu"] - baseline["mfu"]
            f.write(f"| `{row['case']}` | {ratio:.3f}x | {delta:+.2%} |\n")
PY

echo
echo "summary: $SUMMARY_MD"
cat "$SUMMARY_MD"
