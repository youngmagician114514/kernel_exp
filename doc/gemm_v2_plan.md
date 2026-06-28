# GEMM Version 2：Tile 参数搜索计划

## 1. 目标

Version 2 的目标是从固定参数 baseline 进入可调优 GEMM：

> 对 `block_m/block_n/block_k/num_warps/num_stages/group_m` 进行小规模搜索，找出 RTX 3090 上更适合当前 Triton GEMM 的参数组合。

## 2. 为什么先做参数搜索

Version 1 使用固定配置：

```text
block_m = 32
block_n = 64
block_k = 32
num_warps = 4
num_stages = 4
group_m = 8
```

这个配置可以跑通，但不一定适合所有矩阵尺寸。PyTorch/cuBLAS 的优势之一就是会根据矩阵大小和硬件自动选择算法。

因此 Version 2 的重点不是改数学逻辑，而是把 GEMM 从“单一配置”变成“可实验、可比较、可选择”的版本。

## 3. 计划实现

### 3.1 修改 `matmul` 参数

让 `num_warps`、`num_stages`、`group_m` 也可以从外部传入。

当前：

```python
matmul(a, b, block_m=32, block_n=64, block_k=32)
```

目标：

```python
matmul(
    a,
    b,
    block_m=32,
    block_n=64,
    block_k=32,
    group_m=8,
    num_warps=4,
    num_stages=4,
)
```

### 3.2 新增搜索脚本

新增脚本：

```text
kernel_exp/code/scripts/tune_gemm.py
```

搜索配置示例：

```python
configs = [
    {"block_m": 32, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 4, "group_m": 8},
    {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 4, "group_m": 8},
    {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 4, "group_m": 8},
    {"block_m": 128, "block_n": 64, "block_k": 32, "num_warps": 4, "num_stages": 4, "group_m": 8},
    {"block_m": 128, "block_n": 128, "block_k": 32, "num_warps": 4, "num_stages": 4, "group_m": 8},
]
```

### 3.3 输出结果

每组配置记录：

- `M=N=K`
- `block_m`
- `block_n`
- `block_k`
- `group_m`
- `num_warps`
- `num_stages`
- median latency
- TFLOPS
- max absolute error
- max relative error

## 4. 预期产出

Version 2 应该产出：

- 一个可运行的 `tune_gemm.py`
- 一份结果文档：`kernel_exp/doc/gemm_v2_tile_tuning.md`
- 一张参数对比表
- 一个阶段结论：哪组参数在 RTX 3090 上表现最好

## 5. 简历表述

完成 Version 2 后可以写：

> 基于 Triton 实现 FP16 GEMM baseline，并设计多组 tile size / warp / pipeline 参数搜索实验；在 RTX 3090 上对比不同配置的 latency 与 TFLOPS，分析 block size 对 Tensor Core 利用率和 occupancy 的影响。

