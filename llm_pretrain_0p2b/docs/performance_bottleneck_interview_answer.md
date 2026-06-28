# 预训练性能瓶颈分析：面试回答口径

## 简短回答

如果面试官问“你怎么分析性能瓶颈”，可以这样答：

> 我不会先猜某个 kernel 慢，而是先做端到端分解。第一步固定 batch、seq_len、dtype 和数据，记录 tokens/s、step time、MFU、显存峰值。第二步把一个训练 step 拆成 data、forward、backward、grad norm、optimizer、log/eval/checkpoint。第三步用 profiler 看 CUDA kernel 时间、显存分配和是否有大 tensor materialization。确认瓶颈后再做局部优化，并用同一套短跑 benchmark 证明 TPS、MFU、显存和 loss 都正常。

## 我们项目里的例子

当前模型是 0.2B decoder-only LLM，`seq_len=1024`，`vocab=32000`。分析后发现：

- attention 已经走 PyTorch flash kernel，所以不把 SDPA 当成我们的核心贡献；
- RMSNorm/SwiGLU 是高频小算子，适合先做 Triton fusion；
- classifier/CE 会生成 `[B,T,V]` logits，`batch=12` 时是 `[12,1024,32000]`，显存和带宽开销很大；
- optimizer / grad norm 还没有 fusion，是后续候选瓶颈。

## 实际优化闭环

| 阶段 | 做法 | 结果 |
|---|---|---|
| 配置 baseline | 增大 micro batch，关闭 activation checkpointing | `11.7K -> 20.6K TPS` |
| Triton RMSNorm/SwiGLU | 自研 Triton forward/backward/fusion | `20.7K -> 22.2K TPS` |
| 50% MFU baseline | bf16 activation、RoPE buffer、fused AdamW、跳过每步 grad norm、batch 16 | `22.2K -> 26.9K TPS`，`50.5% MFU` |
| 70% MFU 试探 | `torch.compile` + batch 24 | 最好 `33.9K TPS`、`63.7% MFU`，未达到 70% |
| fused classifier CE 原型 | tiled `tl.dot` + online softmax stats，不保存完整 logits | 显存 `18.3GB -> 14.9GB`，但速度暂未提升 |
| fused CE backward v1 | Triton kernel 内计算 grad-hidden / grad-weight | 显存 `14.9GB -> 13.2GB`，但因 atomic/recompute 降到 `14.7K TPS` |

## 被追问时怎么说

如果问“为什么 backward 都 kernel 化了，fused CE 还是没提速”：

> 因为 kernel 化只解决了 Python loop 和大 logits materialization，不自动等于高 MFU。当前 backward v1 在每个 token/vocab tile 里重算 logits，再用 `atomic_add` 累加 grad-hidden 和 grad-weight。这样虽然显存更低，但写冲突、重复 GEMM 和访存压力变大，所以 TPS 反而下降。下一步要做的是减少 atomic，用 two-stage reduction 或 split-K reduction，把 backward 从“正确可跑”优化到“计算密集且写回规整”。

如果问“50% MFU 是怎么达到的”：

> 我先量化目标，50% MFU 大约需要 `26.6K tokens/s`。原来的 `triton_both` 只有 `22.2K`，差约 20%。后面发现瓶颈不只在 GEMM kernel，而是训练系统还有非主计算开销：embedding 后 residual activation 没有稳定保持 bf16、RoPE 每步重建、AdamW 没走 fused 路径、每步 grad norm 会扫全模型梯度。修正这些后，`batch=12` 到 `48.8% MFU`，再把 micro batch 提到 16，达到 `26.9K tokens/s`、`50.5% MFU`。

如果问“为什么没有到 70% MFU”：

> 我实际试过继续增大 batch、提高 grad_accum、改 seq_len、QKV/MLP Linear fusion 和 `torch.compile`。其中 `torch.compile + batch=24` 最好，后编译阶段到 `33.9K tokens/s`、`63.7% MFU`，但 70% 需要约 `37.3K tokens/s`。说明低成本训练系统优化已经接近上限，后续要靠更深的 graph/kernel 优化，比如稳定 CUDA graph、residual+norm fusion、MLP epilogue fusion，以及高效 fused CE backward。

如果问“为什么不直接优化 attention”：

> 我先用 profiler 确认了 attention 已经走 flash kernel，再继续手写 attention 的边际收益不一定最高。相比之下，classifier/CE 端明确存在大 logits materialization，RMSNorm/SwiGLU 也属于高频小算子，所以我优先优化这些热路径。

如果问“怎么保证优化没错”：

> 每个 kernel 都先对齐 PyTorch reference：forward 数值、backward gradient、训练 smoke test。然后只看 warmup 后窗口的 TPS/MFU，并检查 loss 是否 NaN 或异常。性能提升必须和 correctness 一起成立。
