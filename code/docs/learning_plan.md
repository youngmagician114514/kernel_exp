# 学习计划

目标：把 `kernel_exp` 的 12 周 Markdown 计划落成一个可运行、可复盘、可写进简历的大模型推理算子优化项目。

## 阶段 0：环境与基线

时间：0.5 天

产出：
- 固定使用 `kernel_exp/.conda-vllm-srv` 或原 `vllm_srv`。
- 记录 GPU 型号、PyTorch/Triton/CUDA 版本。
- 跑通 `run_smoke_tests.py`。

理解重点：
- 这台服务器是 RTX 3090，SM 8.6，适合做 Ampere 上的 Triton/PyTorch 实验。
- Blackwell 相关内容只作为代码阅读和设计分析，不写成实测。

## 阶段 1：GEMM 性能阶梯

时间：2-3 天

产出：
- 跑 `benchmark_gemm.py`，记录 512/1024/2048 的 `torch.matmul` 与教学版 Triton GEMM 性能。
- 调整 `block_m/block_n/block_k/num_warps/num_stages`，观察性能变化。
- 写一页实验记录：tile size、TFLOPS、误差、瓶颈判断。

理解重点：
- GEMM 的核心是数据复用：global memory -> shared/L2 -> register。
- Triton 的 `tl.dot` 会映射到 Tensor Core，但性能还取决于 tile、访存、occupancy。

## 阶段 2：Online Softmax 与 FlashAttention 前置知识

时间：2 天

产出：
- 跑 `run_online_softmax.py`，验证 blockwise online softmax 与标准 softmax 的数值误差。
- 写清楚 running max 和 running sum 的更新公式。
- 下一步可把 softmax 从 PyTorch 版本改成 Triton row-wise kernel。

理解重点：
- FlashAttention 不是魔法，核心是避免显式写出 NxN attention matrix。
- Online softmax 允许分块扫描 K/V，同时保持数值稳定。

## 阶段 3：Sparse MoE Routing

时间：2-3 天

产出：
- 跑 `run_moe_routing.py`，记录 expert route counts。
- 分析 token 分布不均衡时的 load imbalance。
- 下一步实现 Triton 版 permute/unpermute 或按 expert 分组 GEMM。

理解重点：
- MoE 推理瓶颈不只有 GEMM，还有 Top-K gating、token permute/unpermute、负载均衡。
- 简化 identity expert 验证能保证 routing/unrouting 的数学闭环。

## 阶段 4：简化 FlashAttention

时间：4-7 天

产出：
- 固定 `S=64/128, D=64`，实现单 head 的 Triton attention。
- 对比 PyTorch baseline：正确性、延迟、显存占用。
- 加入 causal mask 和 blockwise online softmax。

理解重点：
- Prefill 阶段偏 compute-bound，decode 阶段偏 memory-bound。
- KV cache 布局会直接影响 decode attention 的吞吐。

## 阶段 5：简历项目整理

时间：1 天

产出：
- README：项目背景、环境、实验命令、结果表格。
- results：保存 benchmark 输出和图表。
- 简历 bullet：突出 Triton kernel、FlashAttention 原理、MoE routing、profiling。

推荐简历标题：

基于 PyTorch/Triton 的 LLM 推理算子优化实验：GEMM、FlashAttention 与 Sparse MoE 原型实现及性能分析

