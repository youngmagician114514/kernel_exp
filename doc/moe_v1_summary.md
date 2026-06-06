# Sparse MoE V1：Routing + Expert FFN + Grouped Forward 总结

## 1. V1 优化摘要

这次 MoE 没有停在 routing demo，而是直接推进到一个完整可运行的 forward 版本。

本版本完成了三层关键改进：

1. **从纯 routing prototype 推进到完整 MoE forward**
   - 不再只有 `Top-K gating`、`permute/unpermute`
   - 增加了真正的 expert FFN 计算路径

2. **同时保留 naive 与 grouped-by-expert 两种执行方式**
   - `naive`：按 expert 顺序逐个执行 FFN
   - `grouped`：先按 expert 分组 token，再按组执行 compute，形成更合理的推理基线

3. **建立 correctness + routing distribution + latency 的验证链路**
   - 验证 `permute -> identity expert -> unpermute`
   - 验证 `grouped forward` 与 `naive forward` 数值一致
   - 统计 expert route counts，观察 load imbalance

一句话概括：

> MoE V1 基于 Top-K routing、token permute/unpermute 和 expert FFN，完成了一个完整的 Sparse MoE forward baseline，并提供 grouped-by-expert 的执行路径作为后续优化基线。

代码入口：

```text
kernel_exp/code/kernel_exp_project/moe_routing.py
```

---

## 2. 对应文档内容

本版本主要依据：

- `kernel_exp/week10_sparse_moe_basics.md`
  - `10.2 Top-K Gating`
  - `10.3 Token Routing (Permute/Unpermute)`
  - `10.4 Expert GEMM`
- `kernel_exp/week11_sparse_moe_advanced.md`
  - `11.1 Fused MoE Kernel 设计`
  - `11.5 SiLU/GELU Activation Fusion`

当前版本仍然主要落在 `week10` 的范围内，但已经把：

```text
Router → Top-K → Permute → Expert Compute → Unpermute
```

这条主链路做完整了。

---

## 3. 当前实现结构

### 3.1 Routing 相关

当前已有并保留：

- `topk_gating(...)`
- `permute_tokens(...)`
- `unpermute_tokens(...)`
- `identity_moe_reference(...)`

它们对应：

- `Softmax -> TopK -> Renormalize`
- token 按 expert 分组
- expert 输出按原始 token 顺序还原并加权

### 3.2 Expert 权重与 FFN

新增：

- `ExpertWeights`
- `build_random_experts(...)`
- `apply_expert_ffn(...)`

其中 expert FFN 目前采用标准两层结构：

```text
x -> W1 -> activation -> W2
```

激活函数支持：

- `gelu`
- `silu`

### 3.3 Forward 路径

新增两条主路径：

- `moe_forward_naive(...)`
- `moe_forward_grouped(...)`

它们的共同逻辑是：

1. `topk_gating`
2. `permute_tokens`
3. expert FFN compute
4. `unpermute_tokens`

区别在于：

- `naive` 更接近直接按 expert 顺序执行的基线
- `grouped` 更强调“先把同 expert 的 token 聚到一起，再做分组计算”

### 3.4 correctness 对照

新增：

- `moe_reference_error(...)`

它用于比较：

```text
moe_forward_naive vs moe_forward_grouped
```

验证 grouped 版本没有改变数学结果。

---

## 4. 正确性验证

`run_smoke_tests.py` 已新增：

- `smoke_identity_moe_max_error`
- `smoke_moe_forward_grouped_max_error`

其中：

- `identity_moe_max_error` 验证 routing/unrouting 数学闭环
- `moe_forward_grouped_max_error` 验证 grouped forward 与 naive forward 一致

这说明：

- routing 路径是正确的
- 完整 expert compute 路径也是闭环正确的

---

## 5. 实验脚本

MoE 的主实验脚本现在是：

```text
kernel_exp/code/scripts/run_moe_routing.py
```

它支持：

- `tokens`
- `hidden`
- `ffn_hidden`
- `experts`
- `top_k`
- `activation`

并输出：

- routing latency
- naive forward latency
- grouped forward latency
- expert route counts
- identity error
- grouped-vs-naive error

---

## 6. 当前版本的边界

当前已经实现：

- Top-K gating
- token permute / unpermute
- expert FFN compute
- grouped-by-expert forward baseline
- correctness 与 basic benchmark

当前还没有实现：

- Triton 版 permute/unpermute kernel
- cuBLASLt Grouped GEMM
- fused MoE kernel
- persistent grouped GEMM
- communication / compute overlap
- expert parallel
- SwiGLU fused 路径

所以当前更准确的定位是：

> 完整的 Sparse MoE forward baseline

而不是：

> 工业级 fused MoE kernel

---

## 7. 当前最准确的项目表述

可以说：

> 基于 Top-K gating、token permute/unpermute 与 expert FFN，完成了一个完整的 Sparse MoE forward baseline，并实现 grouped-by-expert 的执行路径，用于分析路由分布、数值一致性和推理延迟。

不建议说：

> 实现了 fused MoE / persistent grouped GEMM / expert parallel 推理系统

因为这些还没有真正落地。

---

## 8. 下一步方向

如果继续推进 MoE，最自然的下一步是：

1. 把 grouped forward 推到更接近 `Grouped GEMM`
2. 加入更系统的 route distribution / load balance 实验
3. 再往后才是 fused MoE、persistent kernel 和 overlap
