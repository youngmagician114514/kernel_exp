# Sparse MoE V2：Grouped GEMM 结构化 + Triton Expert Compute

## 1. V2 优化摘要

V2 的目标不是继续停留在 “按 expert 分组后逐段做 FFN” 这一层，而是把当前 grouped forward 明确推进成更接近 `Grouped GEMM` 的结构。

本版本完成了三层关键改进：

1. **把 grouped forward 拆成显式的两次 grouped GEMM**
   - `grouped_gemm_1`
   - `activation`
   - `grouped_gemm_2`

2. **新增 Triton 粒度的 expert compute 后端**
   - 在 grouped token 基础上，使用 `triton_gemm.matmul(...)` 代替纯 PyTorch matmul
   - 让 expert FFN 的矩阵乘真正进入 Triton 路径

3. **建立四条路径的统一对比**
   - `naive`
   - `grouped`
   - `grouped_v2`
   - `triton`

一句话概括：

> MoE V2 把 grouped forward 从逻辑分组推进成显式的 grouped-GEMM 风格结构，并引入 Triton expert compute 路径，为后续真正的 grouped GEMM / fused MoE 优化提供更接近底层实现的基线。

代码入口：

```text
kernel_exp/code/kernel_exp_project/moe_routing.py
```

---

## 2. 当前新增结构

V2 在 V1 基础上新增：

- `GroupedExpertTensors`
- `grouped_expert_gemm_1(...)`
- `grouped_expert_gemm_2(...)`
- `grouped_expert_ffn(...)`
- `moe_forward_grouped_v2(...)`
- `moe_forward_triton(...)`
- `moe_triton_error(...)`

其中：

- `moe_forward_grouped_v2(...)`
  - 是结构化后的 grouped GEMM 风格 PyTorch baseline
- `moe_forward_triton(...)`
  - 是使用 Triton matmul 的 expert compute 路径

---

## 3. V1 的提升来自哪里

V1 相比常规 PyTorch 风格的 naive 实现变快，主要不是因为换了数学公式，而是因为三点：

1. **先做 routing，再按 expert 聚合 token**
   - token 不再散着计算
   - 后续 expert FFN 的访问模式更规整

2. **把 expert compute 组织成分组执行**
   - 不是面向单个 token 或零碎切片
   - 而是面向一个 expert 对应的一整段 token

3. **减少了 forward 路径的调度碎片**
   - gating / permute / compute / unpermute 的边界更清晰
   - compute 阶段更像真实推理系统里的 grouped expert work

所以 V1 的本质提升来自：

> 把 MoE 从 routing demo 推进成了完整的 grouped-by-expert forward baseline

---

## 4. V2 的改进点

V2 在 V1 的基础上进一步推进了两件事：

### 4.1 结构上更接近 Grouped GEMM

V1 的 `grouped` 本质还是：

```text
for each expert:
    tokens -> W1 -> activation -> W2
```

V2 则显式拆成：

```text
grouped_gemm_1
activation
grouped_gemm_2
```

这样后面不论要接：

- cuBLASLt Grouped GEMM
- Triton grouped kernel
- Fused MoE

都更容易继续演进。

### 4.2 引入 Triton 粒度 expert compute

V2 新增了：

```text
moe_forward_triton(...)
```

它在 grouped token 基础上，用：

```text
kernel_exp_project.triton_gemm.matmul(...)
```

作为 expert FFN 的 GEMM 后端。

这意味着：

> MoE 不再只是“routing 是我们写的，compute 全靠普通 PyTorch”，而是开始和 Triton kernel 体系打通

---

## 5. 正确性验证

`run_smoke_tests.py` 已新增：

- `smoke_moe_forward_grouped_max_error`
- `smoke_moe_forward_triton_max_error`

其中：

- `grouped_max_error`
  - 验证 grouped 路径与 naive 路径一致
- `triton_max_error`
  - 验证 Triton expert compute 与 grouped_v2 路径一致

因此当前 V2 至少满足：

- routing 正确
- grouped forward 正确
- Triton 路径和 PyTorch grouped baseline 数值一致

---

## 6. 实验脚本

MoE 主实验脚本仍是：

```text
kernel_exp/code/scripts/run_moe_routing.py
```

当前会输出：

- `moe_route`
- `moe_forward_naive`
- `moe_forward_grouped`
- `moe_forward_grouped_v2`
- `moe_forward_triton`

以及：

- `expert route counts`
- `identity moe max error`
- `moe forward grouped-vs-naive max error`
- `moe forward triton-vs-grouped_v2 max error`

---

## 7. 实验结果

### 7.1 常规配置

在：

- `tokens=4096`
- `hidden=1024`
- `ffn_hidden=2048`
- `experts=8`
- `top_k=2`
- `activation=gelu`

下，实验结果为：

```text
moe_route median               = 0.9032 ms
moe_forward_naive median       = 4.6508 ms
moe_forward_grouped median     = 6.4953 ms
moe_forward_grouped_v2 median  = 7.3649 ms
```

当前在这组中小规模配置下，`grouped_v2` 还没有带来速度收益。

### 7.2 大尺寸实验

大尺寸实验结果合并如下：

| tokens | hidden | ffn_hidden | grouped ms | grouped_v2 ms | grouped / grouped_v2 | grouped_v2 err |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1024 | 2048 | 5.2863 | 5.5204 | 95.76% | 0.000000 |
| 8192 | 2048 | 4096 | 20.1912 | 24.4987 | 82.42% | 0.000000 |
| 16384 | 4096 | 8192 | 123.3067 | 86.7419 | 142.15% | 0.000000 |

从结果可以看出：

- `4096 / 1024 / 2048`
  - `grouped_v2` 更慢
- `8192 / 2048 / 4096`
  - `grouped_v2` 更慢
- `16384 / 4096 / 8192`
  - `grouped_v2` 开始优于 `grouped`

也就是说，V2 的 grouped-GEMM 风格结构在更大规模上开始体现价值，但目前收益区间还不稳定。

---

## 8. 当前版本的边界

当前已经实现：

- Top-K gating
- token permute / unpermute
- expert FFN compute
- grouped-by-expert forward baseline
- grouped GEMM 风格结构化 forward
- Triton expert compute 路径

当前还没有实现：

- 真正的 cuBLASLt Grouped GEMM
- Triton 原生 grouped GEMM kernel
- fused MoE kernel
- persistent grouped GEMM
- expert parallel / overlap

因此当前 V2 更准确的定位是：

> 接近 grouped GEMM 结构的 MoE forward baseline + Triton expert compute

而不是：

> 工业级 fused / persistent MoE 实现

---

## 9. 下一步方向

如果继续推进，最自然的下一步是：

1. 把 Triton expert compute 从“逐 expert 调 Triton matmul”推进到更真正的 grouped GEMM 形式
2. 系统分析不同 routing 分布下的 load balance 对性能的影响
3. 再往后才是 fused MoE / persistent kernel / overlap
