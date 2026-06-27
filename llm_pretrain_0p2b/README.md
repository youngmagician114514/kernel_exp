# 从零训练 0.2B LLM：预训练、Triton Kernel、SFT/DPO

这个目录是 `kernel_exp` 的训练侧主线：从一个可运行的 decoder-only LLM 训练系统开始，逐步扩展到 **0.2B dense decoder-only 模型**、约 `1B tokens` 量级的预训练实验、聊天 SFT、DPO/输入法偏好优化，以及训练热路径 Triton kernel。

当前项目定位不是“直接拿现成模型微调”，而是：

- 自己实现 GPT-style 模型结构和训练 loop；
- 把目标规模控制在约 `204M` 参数，保证单张 RTX 3090 上能更快出结果；
- 预训练数据量级控制在 `1B tokens` 左右，优先做可复现 loss curve、吞吐和显存记录；
- 在训练链路里替换/融合 RMSNorm、attention、loss、optimizer 等 kernel；
- 后续用 SFT/DPO 把 base model 迁移到简单聊天和输入法优化任务。

## 当前状态

已完成：

- GPT-style decoder-only LM：RMSNorm、RoPE、GQA、SwiGLU、tied LM head。
- 0.2B 目标配置：`24 layer / hidden 768 / 12 Q heads / 4 KV heads / FFN 2560 / vocab 32000`。
- 训练 loop：bf16/float16 autocast、梯度累积、warmup/cosine LR、激活检查点、checkpoint save/resume、metrics JSONL。
- 数据 pipeline：byte-level Tiny Shakespeare、通用 text/jsonl 预训练打包、response-only SFT 对话打包。
- Triton RMSNorm forward kernel，backward 先用 PyTorch 公式保证训练闭环。
- 生成脚本：`generate_chat.py`，用于加载 checkpoint 做简单对话验证。

还没完成：

- `1B tokens` 量级的真实语料预训练长跑。
- 真正可用的聊天能力；当前只有训练管线，模型要经过 pretrain + SFT 才能聊天。
- DPO trainer；目前已定义输入法偏好数据 schema 和工程路线。

## 环境激活

```bash
cd /home/lwb/kernel_exp
source /home/lwb/anaconda3/etc/profile.d/conda.sh
conda activate /home/lwb/kernel_exp/.conda-vllm-srv
export PYTHONPATH=/home/lwb/kernel_exp/llm_pretrain_0p2b/src
export TRITON_CACHE_DIR=/home/lwb/kernel_exp/.triton_cache
```

检查环境和参数量：

```bash
python llm_pretrain_0p2b/scripts/check_env.py
```

## 关键配置

| 配置 | 用途 | 参数量 | 说明 |
|---|---:|---:|---|
| `byte_debug.json` | 快速端到端验证 | 0.56M | byte vocab，Tiny Shakespeare 可跑 |
| `medium_debug.json` | 小规模性能测试 | 29.37M | 适合 kernel/吞吐对比 |
| `llm_0p2b_1k_3090.json` | 单张 3090 快速试跑 | 约 204M | seq_len=1024，更快出聊天 demo |
| `llm_0p2b_2k.json` / `llm_0p2b.json` | 0.2B 主训练配置 | 约 204M | seq_len=2048 |
| `llm_0p2b_4k.json` | 长上下文 profile | 约 204M | seq_len=4096，用于 OOM/吞吐边界测试 |

说明：这些 0.2B 配置用于自研 dense decoder-only 模型的规模实验；这里是从零训练自己的模型，不加载任何外部模型权重。

## 训练预算

本项目预训练数据量级控制在 `1B tokens` 左右：

| 阶段 | tokens | 目标 |
|---|---:|---|
| smoke | `1M-10M` | 检查数据、loss、checkpoint、生成脚本 |
| pilot | `50M-100M` | 得到可观察 loss curve 和吞吐/显存统计 |
| main | `500M-1B` | 作为 0.2B base model 主实验规模 |

在单张 RTX 3090 上，`0.2B` 更适合作为这个项目的主线：目标是尽快跑出完整 pretrain + SFT demo，同时保留训练 kernel 优化空间。

## 传统预训练

### 小数据验证

Tiny Shakespeare byte-level 数据：

```bash
python llm_pretrain_0p2b/scripts/prepare_tiny_shakespeare.py
```

训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/byte_debug.json \
  --data-dir llm_pretrain_0p2b/data/tiny_shakespeare \
  --steps 50 \
  --batch-size 16 \
  --eval-every 10 \
  --eval-iters 5 \
  --results-dir llm_pretrain_0p2b/results \
  --run-name tiny_shakespeare_byte_debug
```

### 更大文本语料

把普通 `.txt/.md/.jsonl` 打包成 memmap：

```bash
python llm_pretrain_0p2b/scripts/prepare_text_corpus.py \
  --input /path/to/text_corpus \
  --output-dir llm_pretrain_0p2b/data/pretrain_text \
  --val-ratio 0.01 \
  --append-eos
```

如果使用 HuggingFace tokenizer：

```bash
python llm_pretrain_0p2b/scripts/prepare_text_corpus.py \
  --input /path/to/text_corpus \
  --output-dir llm_pretrain_0p2b/data/pretrain_hf_tok \
  --tokenizer /path/to/tokenizer_or_hf_name \
  --val-ratio 0.01 \
  --append-eos
```

单张 RTX 3090 上建议先从 1K context 试跑 0.2B：

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/llm_0p2b_1k_3090.json \
  --data-dir llm_pretrain_0p2b/data/pretrain_hf_tok \
  --steps 31250 \
  --batch-size 1 \
  --grad-accum-steps 32 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --warmup-steps 500 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --eval-every 500 \
  --eval-iters 10 \
  --save-dir llm_pretrain_0p2b/checkpoints/pretrain_0p2b_1k \
  --save-every 1000 \
  --run-name pretrain_0p2b_1b_tokens
```

这个命令的训练量级约为：`seq_len 1024 * batch 1 * grad_accum 32 * steps 31250 = 1.024B tokens`。实际长跑时可以按数据集大小和中途 loss 曲线调整 steps。

## SFT：让模型能简单聊天

准备对话 JSONL：

```bash
python llm_pretrain_0p2b/scripts/prepare_sft_conversations.py \
  --input /path/to/chat_sft.jsonl \
  --output-dir llm_pretrain_0p2b/data/chat_sft \
  --tokenizer /path/to/tokenizer_or_hf_name \
  --val-ratio 0.02
```

训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m llm0.train \
  --config llm_pretrain_0p2b/configs/llm_0p2b_1k_3090.json \
  --data-dir llm_pretrain_0p2b/data/chat_sft \
  --resume llm_pretrain_0p2b/checkpoints/pretrain_0p2b_1k/last.pt \
  --steps 2000 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --dtype bfloat16 \
  --gradient-checkpointing \
  --lr 2e-5 \
  --warmup-steps 50 \
  --save-dir llm_pretrain_0p2b/checkpoints/chat_sft \
  --run-name chat_sft
```

生成验证：

```bash
CUDA_VISIBLE_DEVICES=0 python llm_pretrain_0p2b/scripts/generate_chat.py \
  --checkpoint llm_pretrain_0p2b/checkpoints/chat_sft/last.pt \
  --tokenizer /path/to/tokenizer_or_hf_name \
  --chat \
  --prompt "请用三句话解释什么是 Triton kernel。" \
  --max-new-tokens 128
```

## DPO / 输入法优化

详细路线见 `POSTTRAINING.md`。建议把输入法任务统一成：

```jsonl
{"prompt":"上下文 + 拼音 + 候选列表","chosen":"用户最终选择/更自然候选","rejected":"错误或不自然候选"}
```

后续 DPO trainer 会围绕 policy/reference 双模型 logprob、reward margin、preference accuracy 来实现。

## 已实现的 Triton Kernel

当前新项目里已经实现：

- `Triton RMSNorm forward`：`src/llm0/kernels/rms_norm.py`

当前 backward 仍然是 PyTorch 张量公式，后续计划继续做：

- RMSNorm backward Triton kernel
- fused SwiGLU forward/backward
- attention backward 路径
- fused cross entropy loss
- fused AdamW / optimizer step

测试：

```bash
CUDA_VISIBLE_DEVICES=0 python llm_pretrain_0p2b/tests/test_rms_norm.py
CUDA_VISIBLE_DEVICES=0 python llm_pretrain_0p2b/scripts/bench_rms_norm.py --rows 4096 --hidden 768
```

## 实验记录

短跑结果见 `EXPERIMENTS.md`。训练会持续记录：

- `train_loss` / `val_loss`
- `step_time_ms`
- `tokens_per_sec`
- `grad_norm`
- `lr`
- CUDA allocated/reserved/peak memory
- micro batch、gradient accumulation、dtype、activation checkpointing
