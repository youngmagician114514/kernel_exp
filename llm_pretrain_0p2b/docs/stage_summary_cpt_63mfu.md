# 0.2B LLM CPT 训练效率优化阶段总结

## 当前结论

本阶段目标从“训出完整模型”调整为“围绕单张 RTX 3090 的 CPT/预训练效率做 AI Infra 优化”。当前已建立可复现 benchmark，并把单卡训练效率从原始配置的约 `11.7K tokens/s` 提升到最高 `33.9K tokens/s`、`63.70% MFU`。

当前继续 CPT 使用 `torch.compile + batch=24 + grad_accum=3` 版本。该版本编译冷启动成本较高，但长训可以摊销，后编译阶段是目前最高吞吐配置。续训从 `step_006500.pt` 加载模型权重，但重新初始化 optimizer；原因是旧 checkpoint 的 optimizer state 会在 compile 冷启动阶段额外占用 GPU 显存，导致 batch 24 OOM。

## 关键实验结果

| 阶段 | 版本 | TPS | MFU | 峰值显存 | 主要变化 |
|---|---:|---:|---:|---:|---|
| 原始长训 | bs8 + ga4 + checkpointing | `~11700` | `~22%` | 低 | 保守配置，activation checkpointing 开启 |
| 配置 baseline | bs12 + ga3 + no checkpointing | `20688` | `38.86%` | `22089 MB` | 增大 micro batch，关闭 checkpointing |
| Triton 算子 | RMSNorm + SwiGLU | `22210` | `41.72%` | `18884 MB` | 自研 Triton RMSNorm/SwiGLU |
| 训练系统优化 | bf16 activation + RoPE buffer + fused AdamW + no grad clip | `26905` | `50.54%` | `20256 MB` | 降低非 GEMM 路径开销 |
| 编译上界 | `torch.compile` + bs24 | `33915` | `63.70%` | `22628 MB` | 编译图优化，更大 batch 提升利用率 |

## 提升来源拆解

- batch / grad accumulation / checkpointing：把低效保守配置改成更适合 3090 的吞吐配置，贡献最大的一段基础提升。
- Triton RMSNorm：替换 PyTorch RMSNorm forward/backward，减少高频小算子开销。
- Triton SwiGLU：融合 `silu(gate) * up` 的 elementwise forward/backward，减少 kernel launch 和中间张量读写。
- bf16 activation：embedding 后显式转成 autocast dtype，避免 residual stream 继续走 fp32 带宽路径。
- RoPE buffer：把每步重建 RoPE cache 改成模型 buffer，减少重复构造和 dtype/device 转换开销。
- fused AdamW：启用 CUDA fused optimizer，减少 optimizer step 的 Python/kernel 开销。
- no grad clip benchmark：短跑 benchmark 中跳过每步全模型 grad norm/clip，避免每步额外扫描所有梯度；真实长训可改成周期性记录。
- `torch.compile`：后编译阶段明显降低 step time，但首步编译开销很大，适合长训摊销。
- batch=24：在 compile 后显存允许的情况下增大 micro batch，提高 GEMM shape 和 GPU 利用率。

## Fused CE 现状

自研 fused classifier CE 已完成 forward 和 Triton backward v1：

- forward 用 tiled `tl.dot` + online softmax stats，避免 materialize 完整 `[B,T,V]` logits。
- forward-only fused CE 将峰值显存降到约 `14.9GB`。
- Triton backward v1 将显存进一步降到约 `13.2GB`，但由于 `atomic_add` 写冲突和分块 logits 重复计算，速度下降到约 `14.7K tokens/s`。

结论：fused CE 当前是显存优化原型，不作为速度最优路径；后续要继续做 two-stage reduction / split-K reduction，减少 atomic 和重复计算。

## 当前 CPT 续训配置

从旧 checkpoint `step_006500.pt` 继续，旧配置已经训练约 `239.6M tokens`。新配置每步 `73728 tokens`，目标跑到 `step=16678`，预计总训练 token 约 `990.0M`。

当前 tmux 已稳定运行，已观察到 `step=6580` 左右的训练输出：loss 约 `1.9`，真实 step TPS 约 `31K-33.6K tokens/s`，峰值显存约 `22.56GB`。由于 `torch.compile` 首步冷启动会拖低运行初期的累计 TPS，长训主要看 warmup 后的 `step_tokens_per_sec` 和 `step_time_ms`。

```bash
tmux session: cpt_0p2b_63mfu_1b

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 \
/home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv/bin/python -m llm0.train \
  --config configs/llm_0p2b_1k_3090.json \
  --data-dir data/cci3_hq_1b_llama2 \
  --steps 16678 \
  --batch-size 24 \
  --grad-accum-steps 3 \
  --dtype bfloat16 \
  --use-triton-rmsnorm \
  --use-triton-swiglu \
  --adamw-fused \
  --grad-clip 0 \
  --compile \
  --resume checkpoints/pretrain_0p2b_1k_cci3_1b_opt/bs12_ga3_nockpt/step_006500.pt \
  --resume-model-only \
  --results-dir results/cpt_0p2b_63mfu_1b \
  --run-name resume6500_compile_bs24_ga3_1b_model_only \
  --save-dir checkpoints/cpt_0p2b_63mfu_1b \
  --save-every 500 \
  --keep-last-checkpoints 3 \
  --log-every 10
```

## 代码同步

- 服务器代码：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b`
- 本地同步代码：`llm_pretrain_0p2b/`
- 本地文档：`outputs/`

已同步内容包括 `src/`、`configs/`、`tests/`、`scripts/`。训练脚本已支持：

- `--adamw-fused`
- `--compile-mode`
- `--keep-last-checkpoints`
- `--resume-model-only`
- compiled model 保存时自动去掉 `_orig_mod` wrapper
- model-only resume 时 checkpoint 先加载到 CPU，避免旧 optimizer state 占用 GPU 显存
- resume 后 metrics 中 `tokens_per_sec` 使用当前 step 吞吐，另记录 `run_tokens` 和 `step_tokens_per_sec`，避免把历史 token 计入当前运行速度

## 简历表达

从零搭建 0.2B decoder-only LLM CPT/预训练系统，在单张 RTX 3090 上构建 1B tokens 数据管线与可复现训练 benchmark；通过 batch/activation checkpointing 配置优化、Triton RMSNorm/SwiGLU、bf16 activation 修正、RoPE cache、fused AdamW 和 `torch.compile`，将吞吐从约 `11.7K` 提升到 `33.9K tokens/s`，MFU 从约 `22%` 提升到 `63.7%`；同时实现 fused classifier CE 原型，将峰值显存最低降至约 `13.2GB`，并定位其 backward atomic/recompute 瓶颈。
