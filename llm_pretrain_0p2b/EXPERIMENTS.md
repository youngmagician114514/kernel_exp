# 实验记录

## Tiny Shakespeare byte-level 预训练短跑

### 数据

- 数据源: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- 原始大小: `1,115,394` bytes
- tokenizer: UTF-8 byte-level tokenizer
- vocab size: `256`
- train tokens: `1,003,854`
- val tokens: `111,540`
- split: `90% / 10%`

### 模型与训练设置

- config: `llm_pretrain_0p2b/configs/byte_debug.json`
- 参数量: `0.56M`
- sequence length: `128`
- batch size: `16`
- steps: `50`
- total tokens: `102,400`
- optimizer: `AdamW`
- GPU: `NVIDIA GeForce RTX 3090`
- eval: every `10` steps, `5` validation batches each time

### 结果汇总

| run | RMSNorm 实现 | total time | avg tokens/s | last train loss | last val loss | peak allocated | peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|
| `tiny_shakespeare_byte_debug_50` | Triton forward + PyTorch backward | `2.079s` | `49,256` | `3.3239` | `3.2794` | `82.12 MB` | `112.0 MB` |
| `tiny_shakespeare_byte_debug_50_torch` | PyTorch | `0.989s` | `103,529` | `3.3239` | `3.2794` | `86.15 MB` | `116.0 MB` |

### 观察

1. 真实文本预训练链路已经跑通：download -> encode -> train/val split -> train -> validation -> metrics -> summary。
2. 50 step 内训练 loss 从约 `5.55` 降到 `3.32`，validation loss 从约 `4.64` 降到 `3.28`，说明 next-byte prediction 闭环有效。
3. 这个小模型/短训练下，Triton RMSNorm 版本总时间慢于 PyTorch 版本，主要因为首次 JIT 编译和小 shape 下 kernel launch 开销占比更高。
4. Triton 版本 peak allocated 略低，但差异很小；后续要在更大 hidden size、更长序列和 fused op 上看收益。
5. 之后需要把指标拆成 warmup 后 step time、forward/backward time、eval time，避免 JIT 和验证开销混在总吞吐里。

### 复现实验命令

准备数据：

```bash
cd /home/lwb/kernel_exp
source /home/lwb/anaconda3/etc/profile.d/conda.sh
conda activate /home/lwb/kernel_exp/.conda-vllm-srv
export PYTHONPATH=/home/lwb/kernel_exp/llm_pretrain_0p2b/src
export TRITON_CACHE_DIR=/home/lwb/kernel_exp/.triton_cache
python llm_pretrain_0p2b/scripts/prepare_tiny_shakespeare.py
```

PyTorch baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/byte_debug.json \
  --data-dir llm_pretrain_0p2b/data/tiny_shakespeare \
  --steps 50 \
  --batch-size 16 \
  --eval-every 10 \
  --eval-iters 5 \
  --results-dir llm_pretrain_0p2b/results \
  --run-name tiny_shakespeare_byte_debug_50_torch
```

Triton RMSNorm：

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/byte_debug.json \
  --data-dir llm_pretrain_0p2b/data/tiny_shakespeare \
  --steps 50 \
  --batch-size 16 \
  --eval-every 10 \
  --eval-iters 5 \
  --results-dir llm_pretrain_0p2b/results \
  --run-name tiny_shakespeare_byte_debug_50 \
  --use-triton-rmsnorm
```

## 0.2B 随机 token 单步 profile（待重跑）

目标配置已经切换为 `203.92M` 参数，但当前远端 GPU 驱动处于异常状态，`nvidia-smi` 报 `Driver/library version mismatch`，PyTorch CUDA 初始化也返回不可用。因此旧目标配置的 profile 数字不再沿用，0.2B profile 需要在驱动恢复后重跑。

### 待跑设置

- config: `llm_0p2b_1k_3090.json` 和 `llm_0p2b_2k.json`
- 参数量: `203.92M`
- batch size: `1`
- grad accum: `1`
- dtype: `float16` 或 `bfloat16 autocast`
- activation checkpointing: enabled
- data: random tokens, no checkpoint save

### 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/llm_0p2b_1k_3090.json \
  --steps 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --dtype float16 \
  --gradient-checkpointing \
  --results-dir llm_pretrain_0p2b/results \
  --run-name verify_0p2b_random_1k_1step
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/llm_0p2b_2k.json \
  --steps 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --dtype float16 \
  --gradient-checkpointing \
  --results-dir llm_pretrain_0p2b/results \
  --run-name verify_0p2b_random_2k_1step
```

### 观察计划

1. 记录 1K/2K context 下的 step time、tokens/s 和 peak memory。
2. 再跑 `grad_accum=16/32` 的短训练，确认 `1B tokens` 预算下的预计总时长。
3. 如果 2K context 速度太慢，先用 1K context 完成 pretrain + SFT demo。

## SFT response-only 路径验证

### 设置

- config: `byte_debug.json`
- 数据: 人工生成的 24 条中文 toy chat JSONL
- tokenizer: UTF-8 byte-level
- response mask: system/user token label 为 `-100`，assistant answer token 参与 loss
- steps: `2`
- batch size: `4`
- grad accum: `2`

### 结果

| run | train tokens | labeled train tokens | last train loss | last val loss | peak allocated |
|---|---:|---:|---:|---:|---:|
| `verify_sft_masked` | `7,312` | `3,660` | `5.4196` | `5.2819` | `36.5 MB` |

### 观察

1. `prepare_sft_conversations.py` 能生成 response-only masked memmap。
2. `train.py` 能直接读取 SFT memmap 并训练，无需 TRL/PEFT。
3. 已修复随机采样窗口全是 `-100` label 时可能出现 NaN 的风险：SFT dataset 会优先采样包含 assistant label 的窗口。
4. `generate_chat.py` 能从 checkpoint 加载并按 chat template 生成；toy checkpoint 训练太少，因此不能代表真实聊天能力。
