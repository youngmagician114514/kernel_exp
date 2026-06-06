# Sparse MoE V3：Grouped Triton Kernel 总结

## 1. V3 优化摘要

V3 的目标是把 MoE 从：

> 结构上接近 grouped GEMM

推进到：

> 真正的 grouped problem descriptor + Triton grouped kernel

本版本完成了三层关键改进：

1. **不再使用 padding batched 近似 Grouped GEMM**
   - 从 V2 的 `padding + torch.bmm` 路线切换到真正的 problem descriptor
   - 每个 expert 的 token 数可以不同

2. **补齐 GroupedMatmulProblems 描述层**
   - `expert_ids`
   - `token_starts`
   - `token_counts`
   - `tile_offsets`

3. **引入真正的 grouped Triton matmul**
   - `grouped_matmul(...)`
   - `grouped_expert_ffn_triton_grouped(...)`
   - `moe_forward_triton_grouped(...)`

一句话概括：

> V3 把 MoE 从 padding batched 的近似 grouped 路线，推进成了真正基于 grouped problem descriptor 的 Triton grouped kernel。

---

## 2. 当前实现结构

V3 相关新增核心包括：

- `GroupedMatmulProblems`
- `build_grouped_matmul_problems(...)`
- `grouped_matmul(...)`
- `grouped_expert_ffn_triton_grouped(...)`
- `moe_forward_triton_grouped(...)`
- `moe_triton_grouped_error(...)`

其核心思想是：

1. routing 后先得到按 expert 排序的 token
2. 根据每个 expert 的 token 数构造 grouped problem descriptor
3. Triton kernel 按 tile 映射到各个 problem
4. 完成两次 grouped GEMM：
   - `tokens -> W1`
   - `activation`
   - `hidden -> W2`

---

## 3. 正确性验证

当前 smoke test 已包含：

- `smoke_moe_forward_triton_grouped_max_error`

结果：

```text
smoke_moe_forward_triton_grouped_max_error 0.0
```

说明：

- grouped Triton kernel 数值正确
- 和 `grouped_v2` 基线对齐

---

## 4. 实验结果

### 4.1 参数搜索

参数搜索结果合并如下：

| tokens | hidden | ffn_hidden | 配置 | grouped_v2 ms | triton_grouped ms | grouped_v2 / triton_grouped | max abs err |
|---:|---:|---:|---|---:|---:|---:|---:|
| 8192 | 2048 | 4096 | `BM=64, BN=128, BK=64, W=8, S=4` | 18.6246 | 16.4158 | 113.45% | 0.002930 |

这个结果说明：

> 在 `8192 / 2048 / 4096 / 8 experts` 这档，调参后的 `triton_grouped` 已经可以比 `grouped_v2` 快约 `1.13x`

### 4.2 Route imbalance 实验

同一规模下的 route imbalance 结果如下：

| mode | route_counts | grouped ms | grouped_v2 ms | triton_grouped ms | grouped_v2 / grouped | grouped_v2 / triton_grouped | triton err |
|---|---|---:|---:|---:|---:|---:|---:|
| uniform | `[2052, 2035, 2025, 2051, 2102, 2002, 2031, 2086]` | 16.3085 | 18.9638 | 35.8634 | 86.00% | 52.88% | 0.002930 |
| mild | `[5922, 2598, 1345, 1319, 1342, 1250, 1278, 1330]` | 54.7130 | 76.0949 | 47.6602 | 71.90% | 159.66% | 0.000000 |
| strong | `[8144, 5970, 377, 374, 394, 371, 376, 378]` | 13.7368 | 79.0244 | 42.8393 | 17.38% | 184.47% | 0.000000 |

### 4.3 总结

V3 的收益区间并不统一：

- **uniform**
  - `triton_grouped` 没有优势
- **mild / strong imbalance**
  - `triton_grouped` 明显优于 `grouped_v2`

所以更准确的说法是：

> V3 的 grouped Triton kernel 在不均衡 routing 场景下开始体现出真实价值，但在均匀分布场景下还不是最优路径。

---

## 5. 当前版本的边界

当前已经实现：

- 真正 grouped problem descriptor
- grouped Triton kernel
- grouped Triton autotune
- imbalance 实验

当前还没有实现：

- persistent grouped GEMM
- dynamic work queue
- expert parallel / overlap
- fused MoE

因此当前 V3 更准确的定位是：

> 有真实收益区间的 grouped Triton kernel 版本

---

## 6. 当前最准确的项目表述

可以说：

> 基于 grouped problem descriptor 实现了 Sparse MoE 的 Triton grouped kernel，在 `8192 / 2048 / 4096 / 8 experts` 设置下经调参后相对 `grouped_v2` 获得约 `1.13x` 提升；在 mild / strong route imbalance 场景下，相对 `grouped_v2` 的收益扩大到约 `1.60x / 1.84x`。

不建议说：

> V3 在所有场景下都优于 grouped_v2

因为 uniform 场景并不成立。
