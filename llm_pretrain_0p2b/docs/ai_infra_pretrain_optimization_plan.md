# 0.2B LLM 预训练 AI Infra 优化方案

## 项目主线

这个项目适合写成 AI Infra 训练效率优化项目，而不是模型能力项目：

> 从零搭建 0.2B decoder-only LLM 预训练系统，在单张 RTX 3090 上构建 1B tokens 数据与训练 baseline，并围绕训练热路径做吞吐、MFU 和显存优化。

## 已完成

- 数据：使用 ModelScope CCI3-HQ 构建 `1B tokens` 预训练数据，避免完整复制大规模原始语料。
- 训练：支持 bf16、梯度累积、checkpoint resume、metrics JSONL、MFU 粗估。
- 配置 baseline：将 `batch=8, grad_accum=4, checkpointing=on` 优化为 `batch=12, grad_accum=3, checkpointing=off`，吞吐从约 `11.7K` 提到约 `20.6K tokens/s`。
- Triton 第一阶段：实现 RMSNorm backward 与 SwiGLU elementwise fusion，并接入训练脚本。
- 50% MFU baseline：修正训练系统 dtype 路径，embedding 后激活转 bf16，RoPE cache 常驻 buffer，启用 fused AdamW，并在 benchmark 中关闭每步 grad norm/clip。
- Attention 周边：确认 PyTorch SDPA 实际走 `pytorch_flash` FlashAttention kernel，但不把它作为项目核心贡献。
- Classifier/CE：参考 `code/kernel_exp_project/triton_gemm.py` 的 tiled GEMM，实现 fused classifier CE forward，并完成 Triton backward v1，避免 materialize 完整 `[B,T,V]` logits。

## 当前最佳短跑结果

固定配置：`checkpointing=off, bf16, seq_len=1024`；速度目标最终使用 `batch=16, grad_accum=3`，早期对照使用 `batch=12, grad_accum=3`。

| 版本 | 改动 | TPS | MFU | step time | 峰值显存 | 相对 baseline |
|---|---|---:|---:|---:|---:|---:|
| baseline | PyTorch RMSNorm + PyTorch SwiGLU | `20688` | `38.86%` | `1782.8 ms` | `22089 MB` | `1.000x` |
| triton_both | Triton RMSNorm + Triton SwiGLU | `22210` | `41.72%` | `1662.3 ms` | `18884 MB` | `1.074x` |
| native_gqa_triton_both | Native GQA + Triton RMSNorm + Triton SwiGLU | `22571` | `42.40%` | `1635.0 ms` | `18303 MB` | `1.091x` |
| 50mfu_bs12 | bf16 activation + RoPE buffer + fused AdamW + no grad clip | `25966` | `48.77%` | `1420.5 ms` | `16137 MB` | `1.255x` |
| 50mfu_bs16 | 50mfu_bs12 + micro batch 16 | `26905` | `50.54%` | `1828.6 ms` | `20256 MB` | `1.300x` |
| compile_bs16 | 50mfu_bs16 + `torch.compile`，去掉编译首步 | `32866` | `61.73%` | `1496.7 ms` | `16307 MB` | `1.589x` |
| compile_bs24 | compile_bs16 + micro batch 24，去掉编译首步 | `33915` | `63.70%` | `2174.9 ms` | `22628 MB` | `1.639x` |
| fused_ce_bs12 | 自研 fused classifier CE forward + chunked backward | `20654` | `38.79%` | `1788.0 ms` | `14912 MB` | `0.998x` |
| fused_ce_bs16 | fused CE 后增大 micro batch 到 16 | `21922` | `41.18%` | `2243.6 ms` | `18623 MB` | `1.060x` |
| fused_ce_triton_bwd_bs12 | 自研 fused CE forward + Triton atomic backward | `14738` | `27.68%` | `2503.9 ms` | `13226 MB` | `0.712x` |

结论：速度 baseline 已达到 `50.54% MFU`，低成本继续优化可到 `63.70% MFU`，但尚未达到 `70% MFU`。关键收益来自 bf16 activation 路径、RoPE buffer、fused AdamW、跳过每步 grad norm/clip、`torch.compile`，以及用 `batch=24` 提高 GPU 利用率。`70% MFU` 在当前口径下需要约 `37.3K tokens/s`，当前最好为 `33.9K tokens/s`。fused classifier CE 仍是显存优化主线：forward-only fused CE 已能把显存降到 `14.9GB`，Triton backward v1 进一步降到 `13.2GB`，但速度明显下降。

## 后续优先级

1. 把 `compile_bs24` 作为新的速度上界记录；如果考虑编译冷启动成本，长训前几百 step 需要单独摊销。
2. 将 `grad_clip=0` 的 benchmark 结论和真实长训区分：长训可以周期性记录 grad norm，而不是每步都扫全模型。
3. 继续优化 fused classifier CE backward：减少 `atomic_add`，尝试 two-stage reduction / split-K reduction，并调 `block_m/block_v/block_k/block_d`。
4. 补 CUDA event / profiler 分段计时，重点看 `torch.compile` 后剩余时间集中在 attention、MLP GEMM、CE 还是 optimizer。
5. 若继续冲 `70% MFU`，优先做 graph/kernel 级优化：稳定 CUDA graph、residual+norm fusion、MLP epilogue fusion、真正高效的 fused CE backward。

## 简历表达

- 从零实现 0.2B decoder-only LLM 预训练系统，支持 RMSNorm、RoPE、GQA、SwiGLU、bf16、梯度累积、checkpoint resume 和 metrics logging。
- 基于 ModelScope CCI3-HQ 构建 `1B tokens` 数据管线，并在单张 RTX 3090 上建立真实预训练效率 baseline。
- 通过 batch / gradient accumulation / activation checkpointing 配置优化，将吞吐从约 `11.7K` 提升到约 `20.6K tokens/s`。
- 修正 bf16 activation 路径，缓存 RoPE，启用 fused AdamW，并优化 benchmark 中的 grad norm 路径，将单卡短跑提升到 `26.9K tokens/s`、`50.5% MFU`。
- 实现 Triton RMSNorm backward、SwiGLU fusion 和 fused classifier CE 原型；fused CE 将峰值显存降至约 `13.2GB`，并定位到 backward atomic/recompute 瓶颈。
