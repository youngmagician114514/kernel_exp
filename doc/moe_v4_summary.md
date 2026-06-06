# Sparse MoE V4：Persistent Grouped GEMM 总结

## 1. V4 优化摘要

V4 的目标是把 `MoE` 从静态 grouped Triton kernel 再推进一步，到更贴近 `week11` 的：

> Persistent Grouped GEMM

本版本完成了三层关键改进：

1. **在 Triton grouped kernel 上加入 persistent 执行模型**
   - 固定 program 数
   - 通过动态 work queue 获取 tile

2. **加入 problem packing / token_count 排序**
   - 按 problem 大小排序
   - 减少极端不均衡时的低效调度

3. **针对 V4 做独立调参与 16k token imbalance 实验**
   - 不再只沿用 V3 的经验参数
   - 单独评估 persistent 在大规模与 skewed routing 下的价值

一句话概括：

> V4 把 MoE 的 grouped Triton 路径推进到了 dynamic work-queue 的 persistent grouped GEMM 原型，并在 16k token 大规模实验中形成了更清晰的收益区间。

---

## 2. 当前实现结构

V4 相关新增核心包括：

- `_persistent_grouped_matmul_kernel(...)`
- `persistent_grouped_matmul(...)`
- `grouped_expert_ffn_triton_persistent(...)`
- `moe_forward_triton_persistent(...)`
- `moe_triton_persistent_error(...)`

相比 V3：

- V3 是静态 grouped descriptor + static tile launch
- V4 是 persistent-style grouped launch + dynamic work queue

当前 work queue 已经从简单 stride-loop 改成了真正的 `atomic_add` 取 tile。

---

## 3. 正确性验证

当前 smoke test 已包含：

- `smoke_moe_forward_triton_persistent_max_error`

结果：

```text
smoke_moe_forward_triton_persistent_max_error 0.0
```

说明：

- persistent grouped kernel 数值正确
- 与 `grouped_v2` 保持一致

---

## 4. 基础实验结果

基础实验（多条路径同场对比）中，V4 的基线结果为：

| tokens | hidden | ffn_hidden | grouped ms | grouped_v2 ms | triton_grouped ms | triton_persistent ms | grouped_v2 / triton_grouped | grouped_v2 / triton_persistent |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1024 | 2048 | 16.8076 | 6.8255 | 14.3960 | 5.3562 | 47.41% | 127.43% |
| 8192 | 2048 | 4096 | 15.0874 | 15.8480 | 28.1573 | 40.6461 | 56.28% | 38.99% |
| 16384 | 4096 | 8192 | 82.2105 | 84.6721 | 172.0130 | 199.2841 | 49.22% | 42.49% |

这个结果告诉我们：

- `4096 / 1024 / 2048`
  - V4 已经能比 `grouped_v2` 快
- `8192 / 2048 / 4096`
  - V4 还没有稳定收益
- `16384 / 4096 / 8192`
  - 原始默认配置下 V4 也还不占优

所以：

> V4 是否有效，强依赖参数和分布，不是默认就稳的版本。

---

## 5. V4 autotune 结果

V4 的专门调参结果如下：

| 配置 | grouped_v2 ms | triton_grouped ms | persistent ms | grouped_v2 / persistent | triton_grouped / persistent | err |
|---|---:|---:|---:|---:|---:|---:|
| `BM=64, BN=128, BK=64, W=8, S=4, P=64` | 172.0244 | 353.4617 | 142.7886 | 120.47% | 247.54% | 0.000000 |

这说明：

> 在 `16384 / 4096 / 8192 / 8 experts` 这一档，经过调参后的 `persistent` 路径可以相对 `grouped_v2` 获得约 `1.20x` 提升。

也就是说：

- V4 不是“完全没收益”
- 而是必须经过针对性 autotune 才能站住

---

## 6. 16k token imbalance 结果

在 `16384 / 4096 / 8192 / 8 experts / top_k=2` 下：

| mode | route_counts | grouped ms | grouped_v2 ms | triton_grouped ms | triton_persistent ms | grouped_v2 / grouped | grouped_v2 / triton_grouped | grouped_v2 / triton_persistent |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | `[4095, 4063, 4140, 4114, 4122, 4050, 4061, 4123]` | 222.0734 | 106.2656 | 379.1210 | 409.9902 | 208.98% | 28.03% | 25.92% |
| mild | `[11869, 5148, 2560, 2702, 2570, 2637, 2677, 2605]` | 205.5491 | 448.6166 | 432.4437 | 384.3441 | 45.82% | 103.74% | 116.72% |
| strong | `[16276, 12032, 763, 773, 728, 748, 750, 698]` | 226.6751 | 735.0149 | 384.2401 | 374.5462 | 30.84% | 191.29% | 196.24% |

结论很清楚：

- **uniform**
  - `grouped_v2` 最优
  - persistent 没价值

- **mild / strong imbalance**
  - `grouped_v2` 会明显退化
  - `persistent` 开始优于 `grouped_v2`
  - 在 `strong` 场景下，收益接近 `2x`

所以：

> V4 的核心价值不在均匀分布，而在大规模 + skewed routing 场景。

---

## 7. 当前版本的边界

当前 V4 已经实现：

- dynamic work-queue grouped kernel
- persistent grouped GEMM 原型
- problem packing / token_count 排序
- 16k 规模调参与 imbalance 实验

当前还没有实现：

- 更大范围的 autotune
- 更细粒度的 route imbalance 曲线
- expert parallel / overlap
- fused activation / fused MoE

因此当前 V4 的准确定位是：

> 在单卡环境下，已经形成真实收益区间的 persistent grouped GEMM 原型

---

## 8. 当前最准确的项目表述

可以说：

> 实现了基于 grouped problem descriptor 和 dynamic work queue 的 persistent Triton grouped GEMM 原型；在 `16384 × 4096 × 8192 × 8 experts` 设置下，经 autotune 后相对 `grouped_v2` 获得约 `1.20x` 提升；在 mild / strong route imbalance 场景下，相对 `grouped_v2` 的收益扩大到约 `1.17x / 1.96x`。

不建议说：

> V4 在所有 MoE 场景下都优于 grouped_v2

因为 uniform 分布并不成立。
