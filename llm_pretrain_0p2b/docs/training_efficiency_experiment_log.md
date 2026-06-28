# 0.2B LLM 预训练效率优化记录

## 当前定位

这个阶段不追求完整训完 `1B tokens`，目标是围绕单张 RTX 3090 的预训练吞吐做可复现 benchmark 和算子优化。旧长训 `pretrain_0p2b_cci3_1b_opt` 已停止，旧结果只作为 baseline 参考。

## 固定短跑配置

| 项目 | 配置 |
|---|---:|
| GPU | 1 张 RTX 3090，`CUDA_VISIBLE_DEVICES=0` |
| 模型 | 0.2B decoder-only LLM，约 `203.92M` 参数 |
| 数据 | ModelScope CCI3-HQ，已打包 `1B tokens` |
| seq_len | `1024` |
| dtype | `bfloat16` |
| micro batch | `12` |
| grad accumulation | `3` |
| tokens / step | `36864` |
| activation checkpointing | 关闭 |
| benchmark steps | `220` |
| warmup | 前 `30` 条 metrics 不计入汇总 |

## Triton 短跑结果

| 版本 | 改动 | TPS | MFU | step time | 峰值显存 | 相对 baseline | loss |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | PyTorch RMSNorm + PyTorch SwiGLU | `20688` | `38.86%` | `1782.8 ms` | `22089 MB` | `1.000x` | `3.9624` |
| triton_rmsnorm | Triton RMSNorm forward/backward | `22005` | `41.33%` | `1676.4 ms` | `20327 MB` | `1.064x` | `3.9757` |
| triton_swiglu | Triton SwiGLU elementwise forward/backward | `21224` | `39.87%` | `1737.8 ms` | `20650 MB` | `1.026x` | `3.9893` |
| triton_both | Triton RMSNorm + Triton SwiGLU | `22210` | `41.72%` | `1662.3 ms` | `18884 MB` | `1.074x` | `4.0007` |
| native_gqa_triton_both | PyTorch SDPA native GQA + Triton RMSNorm + Triton SwiGLU | `22571` | `42.40%` | `1635.0 ms` | `18303 MB` | `1.091x` | `3.9867` |
| fused_ce_bs12 | 自研 Triton tiled classifier CE forward + chunked backward | `20654` | `38.79%` | `1788.0 ms` | `14912 MB` | `0.998x` | `4.7921` |
| fused_ce_bs16 | fused CE 后增大 micro batch 到 16 | `21922` | `41.18%` | `2243.6 ms` | `18623 MB` | `1.060x` | `5.3115` |
| fused_ce_triton_bwd_bs12 | 自研 fused CE forward + Triton atomic backward | `14738` | `27.68%` | `2503.9 ms` | `13226 MB` | `0.712x` | `4.8064` |
| 50mfu_bs12 | bf16 activation + RoPE buffer + fused AdamW + no grad clip | `25966` | `48.77%` | `1420.5 ms` | `16137 MB` | `1.255x` | `5.2426` |
| 50mfu_bs16 | 50mfu_bs12 + micro batch 16 | `26905` | `50.54%` | `1828.6 ms` | `20256 MB` | `1.300x` | `5.5777` |
| compile_bs16 | 50mfu_bs16 + `torch.compile`，去掉编译首步 | `32866` | `61.73%` | `1496.7 ms` | `16307 MB` | `1.589x` | `5.6082` |
| compile_bs24 | compile_bs16 + micro batch 24，去掉编译首步 | `33915` | `63.70%` | `2174.9 ms` | `22628 MB` | `1.639x` | `5.6243` |

## 结论

第一阶段有效：Triton RMSNorm/SwiGLU 将吞吐提升到 `22.21K tokens/s`，峰值显存降到 `18.9GB`。额外使用 PyTorch SDPA native GQA 可到 `22.57K tokens/s`，但这不作为项目核心贡献，只作为基础库能力记录。

第三阶段已达到 `50% MFU` 目标：把 embedding 后的 residual activation 显式转成 autocast dtype，RoPE cache 改为模型 buffer，AdamW 使用 CUDA fused 实现，并在短跑 benchmark 中关闭每步全模型 grad norm/clip。`batch=12` 时达到 `48.77% MFU`，继续把 micro batch 增大到 `16` 后达到 `26905 tokens/s`、`50.54% MFU`，峰值显存 `20.3GB`。这说明此前 MFU 上不去的主要原因之一是非 GEMM 路径和 fp32 activation 带宽开销，而不只是矩阵乘本身。

70% MFU 试验暂未达标：在相同 MFU 口径下，`seq_len=1024` 达到 `70% MFU` 需要约 `37267 tokens/s`。已尝试 `batch=18`、更高 `grad_accum`、`seq_len=2048`、QKV/MLP Linear fusion、`torch.compile` 和 `compile batch=24`。其中最好的是 `torch.compile + batch=24`，后编译阶段达到 `33915 tokens/s`、`63.70% MFU`。`batch=18` eager OOM，Linear fusion 反而下降，`reduce-overhead` 编译模式和当前梯度累积写法不兼容。

第二阶段 fused classifier/CE 原型已跑通：forward 用参考 `code/kernel_exp_project/triton_gemm.py` 的 tiled `tl.dot` 结构，融合 classifier matmul 与 CE online-softmax stats，不再 materialize 完整 `[B,T,V]` logits。forward-only 融合能把峰值显存从 `18.3GB` 降到 `14.9GB`。

新增的 backward kernel 化也已完成：用 Triton kernel recompute 分块 logits，并在 kernel 内计算 grad-hidden / grad-weight。正确性测试通过，显存进一步降到 `13.2GB`，但吞吐降到 `14.7K tokens/s`。原因是当前 v1 backward 使用大量 `atomic_add` 累加 `grad_hidden` / `grad_weight`，同时每个 vocab tile 都要重复重算 logits，新的瓶颈从“大 logits materialization”转移到了“atomic 写冲突 + 重复 GEMM”。因此这个版本是正确的 kernel 化原型，不作为最终速度最优版本。

## 验证

- RMSNorm：forward 对齐 PyTorch reference，backward 对齐 PyTorch autograd。
- SwiGLU：forward 对齐 `F.silu(gate) * up`，backward 对齐 PyTorch autograd。
- native GQA：forward/backward 与手动 K/V repeat 对齐，测试误差为 `0`。
- fused classifier/CE：loss 和 hidden/weight grad 对齐 PyTorch `F.linear + F.cross_entropy`，forward + Triton backward 在小规模 fp16/bf16 测试通过。
- 四组训练短跑均 `status=0`。

## 远端路径

- benchmark summary：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/triton_efficiency/20260628_095812/summary.md`
- native GQA metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/native_gqa_efficiency/20260628_115118/native_gqa_triton_both/metrics.jsonl`
- 50% MFU metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/target_50mfu_efficiency/20260628_50mfu_v1/bf16_activation_ropebuf_fusedadamw_noclip_bs16/metrics.jsonl`
- 70% MFU trial metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/target_70mfu_efficiency/`
- fused CE bs12 metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/fused_ce_efficiency/20260628_122648/fused_ce_triton_both/metrics.jsonl`
- fused CE bs16 metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/fused_ce_efficiency/20260628_123223_bs16/fused_ce_bs16/metrics.jsonl`
- fused CE Triton backward metrics：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/results/fused_ce_triton_bwd_efficiency/20260628_125542/fused_ce_triton_bwd_bs12/metrics.jsonl`
- benchmark log：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/logs/triton_efficiency_bench.log`
- benchmark script：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/scripts/run_triton_efficiency_benchmarks.sh`
- Triton RMSNorm：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/src/llm0/kernels/rms_norm.py`
- Triton SwiGLU：`/home/zhuyihui/data/wb/kernel_exp-main/llm_pretrain_0p2b/src/llm0/kernels/swiglu.py`
