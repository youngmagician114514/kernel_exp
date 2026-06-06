# GEMM Version 1：Triton Baseline

## 1. 版本定位

本版本是 GEMM 优化路线的第一版 baseline，目标不是超过 PyTorch/cuBLAS，而是建立一个可以运行、可以验证、可以继续优化的 Triton FP16 GEMM 实现。

一句话概括：

> 使用 Triton 实现一个基于 block tiling 的 FP16 GEMM baseline，并与 PyTorch `torch.matmul` 进行正确性验证和 microbenchmark 对比。

## 2. 参考文档

主要参考：

- `kernel_exp/week04_gemm_tiling.md`
  - `4.1 GEMM 优化路线图`
  - `4.2 三级 Tiling 体系`
  - `4.3 从 Naive 到 Tiled GEMM`
  - `4.7 GEMM 性能阶梯 / Tile Size 实验`

辅助参考：

- `kernel_exp/week03_tensor_core.md`
  - `3.2 WMMA API`
  - `3.3 PTX MMA 指令`

说明：

`week04` 是本版本的主要来源，核心是把输出矩阵 `C` 切成 tile，并沿 K 维度分块累加。`week03` 提供 Tensor Core / MMA 的背景知识；代码没有直接写 WMMA，而是通过 Triton 的 `tl.dot` 间接使用 Tensor Core 路径。

## 3. 当前实现范围

当前实现文件：

- `kernel_exp/code/kernel_exp_project/triton_gemm.py`
- `kernel_exp/code/scripts/benchmark_gemm.py`
- `kernel_exp/code/kernel_exp_project/utils.py`

已经实现：

- FP16 输入矩阵乘：`C = A @ B`
- FP32 accumulator：中间累加使用 FP32
- block tiling：将 `C` 切成 `block_m x block_n` 的 tile
- K 维度分块：每次读取 `block_k` 宽度的 A/B tile
- Triton `tl.dot`：执行 tile 级矩阵乘
- mask 边界处理：支持尺寸不能被 tile 整除的情况
- 简单 `group_m` 调度：提高相邻 program 的 L2 cache 复用机会
- 与 PyTorch `torch.matmul` 对比正确性和性能

尚未实现：

- tile size 自动搜索
- `num_warps` / `num_stages` 自动调优
- 显式 double buffering 设计
- 更精细的 register tiling / warp tiling 控制
- split-K
- epilogue fusion
- cuBLASLt 风格 heuristic / exhaustive search

## 4. 核心代码结构

### 4.1 Benchmark 入口

文件：

```text
kernel_exp/code/scripts/benchmark_gemm.py
```

核心流程：

```python
a = torch.randn((size, size), device="cuda", dtype=torch.float16)
b = torch.randn((size, size), device="cuda", dtype=torch.float16)

torch_result = benchmark_cuda("torch.matmul", lambda: a @ b, warmup=5, repeat=repeat)
triton_result = benchmark_cuda("triton_gemm", lambda: matmul(a, b), warmup=5, repeat=repeat)

ref = a @ b
out = matmul(a, b)
max_err = float((ref - out).abs().max())
```

含义：

- `ref`：PyTorch 官方矩阵乘结果，作为参考答案
- `out`：我们自己的 Triton GEMM 结果
- `max_err`：两者最大绝对误差
- `torch_result`：PyTorch/cuBLAS 的性能
- `triton_result`：我们自己的 Triton kernel 性能

### 4.2 CUDA 计时逻辑

文件：

```text
kernel_exp/code/kernel_exp_project/utils.py
```

关键点：

```python
for _ in range(warmup):
    fn()
torch.cuda.synchronize()

for _ in range(repeat):
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1000.0)
```

为什么需要这样写：

- `warmup`：让 GPU、Triton JIT、cuBLAS 状态稳定
- `torch.cuda.synchronize()`：CUDA kernel 是异步执行的，必须同步后才能得到真实耗时
- 使用 median 而不是只看 mean：减少偶发抖动影响

### 4.3 Python 包装函数

文件：

```text
kernel_exp/code/kernel_exp_project/triton_gemm.py
```

函数：

```python
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
```

作用：

- 检查输入是二维矩阵
- 检查 `A.shape[1] == B.shape[0]`
- 检查 dtype 是 FP16
- 检查 tensor 在 CUDA 上
- 分配输出矩阵 `C`
- 设置 Triton grid
- 调用 `_matmul_kernel`

默认 tile 参数：

```text
block_m = 32
block_n = 64
block_k = 32
```

含义：

```text
每个 Triton program 负责计算 C 的一个 32 x 64 tile。
沿 K 方向每次取 32 个元素做一次 tile 级矩阵乘。
```

### 4.4 Triton Kernel

函数：

```python
@triton.jit
def _matmul_kernel(...):
```

参数分为五类。

#### 第 1 类：矩阵指针

```python
a_ptr
b_ptr
c_ptr
```

含义：

- `a_ptr`：A 矩阵在 GPU 显存中的起始地址
- `b_ptr`：B 矩阵在 GPU 显存中的起始地址
- `c_ptr`：C 矩阵在 GPU 显存中的起始地址

#### 第 2 类：矩阵形状

```python
m
n
k
```

矩阵乘关系：

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

含义：

- `m`：A 和 C 的行数
- `n`：B 和 C 的列数
- `k`：A 的列数 / B 的行数

#### 第 3 类：stride

```python
stride_am
stride_ak
stride_bk
stride_bn
stride_cm
stride_cn
```

`m/n/k` 只描述矩阵形状，stride 描述矩阵在显存里如何排列。

对于连续的 row-major 矩阵：

```text
A[row, col] = a_ptr + row * K + col
```

所以：

```text
A: [M, K]
stride_am = K
stride_ak = 1

B: [K, N]
stride_bk = N
stride_bn = 1

C: [M, N]
stride_cm = N
stride_cn = 1
```

但 PyTorch tensor 不一定连续，例如转置后的 tensor stride 会变化。因此 kernel 不能只依赖 `m/n/k`，还需要真实 stride 才能正确寻址。

#### 第 4 类：tile 参数

```python
block_m
block_n
block_k
```

含义：

- `block_m`：C tile 在 M 方向的高度
- `block_n`：C tile 在 N 方向的宽度
- `block_k`：每次沿 K 方向加载的 tile 宽度

当前默认：

```text
block_m = 32
block_n = 64
block_k = 32
```

如果 `C` 是 `128 x 128`，那么 tile 切分为：

```text
M 方向：128 / 32 = 4 块
N 方向：128 / 64 = 2 块
总计：4 x 2 = 8 个 Triton programs
```

#### 第 5 类：group_m

```python
group_m
```

`group_m` 是 program 调度优化参数，不改变数学结果。

它的作用是把 M 方向的多个 block 组成一组，让相邻 Triton programs 更可能复用相同的 B tile，从而提高 L2 cache 命中率。

普通二维 tile 顺序可能是：

```text
(0,0), (0,1), (0,2), (0,3),
(1,0), (1,1), ...
```

使用 `group_m` 后更像：

```text
(0,0), (1,0), (2,0), (3,0),
(0,1), (1,1), (2,1), (3,1),
(0,2), ...
```

这样对于同一个 `pid_n`，多个不同的 `pid_m` 会使用相同的 B tile。连续执行这些 programs 时，B tile 更可能还在 L2 cache 中。

## 5. 核心计算流程

核心代码：

```python
for k0 in range(0, k, block_k):
    a_tile = tl.load(a, mask=(offs_m[:, None] < m) & (k0 + offs_k[None, :] < k), other=0.0)
    b_tile = tl.load(b, mask=(k0 + offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
    acc += tl.dot(a_tile, b_tile)
    a += block_k * stride_ak
    b += block_k * stride_bk
```

解释：

1. 当前 Triton program 负责输出矩阵 C 的一个 tile。
2. 沿 K 维度分块循环。
3. 每次读取一块 A tile 和一块 B tile。
4. 使用 `tl.dot(a_tile, b_tile)` 做 tile 级矩阵乘。
5. 将结果累加到 FP32 accumulator `acc`。
6. K 维度循环结束后，将 `acc` 转成 FP16 写回 C。

一句话总结：

> 每个 Triton program 负责 C 的一个 `block_m x block_n` tile；它沿 K 方向不断加载 A/B tile，用 `tl.dot` 累加，最后把结果写回 C。

## 6. 实验环境

日期：2026-06-04

GPU：

```text
NVIDIA GeForce RTX 3090
```

环境：

```text
kernel_exp/.conda-vllm-srv
```

关键版本：

```text
torch 2.9.1+cu128
triton 3.5.1
CUDA 12.8 runtime
```

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zhuyihui/data/wb/kernel_exp/code \
TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache \
conda run -p /home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv \
python kernel_exp/code/scripts/benchmark_gemm.py --sizes 512 1024 2048 --repeat 20
```

## 7. 实验结果

| M=N=K | torch 中位延迟 ms | Triton 中位延迟 ms | torch TFLOPS | Triton TFLOPS | 最大误差 |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.0706 | 0.1344 | 3.80 | 2.00 | 0.000000 |
| 1024 | 0.1122 | 0.1930 | 19.15 | 11.13 | 0.000000 |
| 2048 | 0.3457 | 0.4938 | 49.70 | 34.79 | 0.125000 |

## 8. 结果分析

### 8.1 正确性

`512` 和 `1024` 尺寸下最大误差为 `0`，说明当前 Triton GEMM 与 PyTorch reference 完全一致。

`2048` 尺寸下最大误差为 `0.125`。这是 FP16 GEMM 中可以接受的第一版 baseline 误差，因为每个输出元素需要累加 2048 个乘积，最后又写回 FP16。

后续更规范的误差报告应该同时记录：

- 最大绝对误差
- 平均绝对误差
- 最大相对误差
- `torch.allclose` 结果

### 8.2 性能

当前 Triton GEMM 在 `2048` 方阵上达到约 `34.79 TFLOPS`，约为 PyTorch/cuBLAS baseline 的 `70%`。

这个结果说明：

- 当前实现已经能正确利用 Triton `tl.dot` 做 FP16 tile GEMM。
- 当前实现还没有达到工业库性能。
- 性能差距主要来自调优不足，而不是算法方向错误。

### 8.3 为什么不如 PyTorch

`torch.matmul` 背后通常调用 cuBLAS/cuBLASLt，这是 NVIDIA 长期优化的 GEMM 库。它会根据 GPU、矩阵大小、dtype 自动选择高性能 kernel。

当前 Triton baseline 不如 PyTorch，主要原因包括：

1. tile 参数固定，没有针对不同矩阵大小搜索最优配置。
2. 没有实现 `num_warps` / `num_stages` 自动调优。
3. 没有显式 double buffering。
4. 没有更精细的 register tiling / warp tiling 控制。
5. 没有 split-K、swizzle、epilogue fusion 等高级优化。
6. 小矩阵下 kernel launch overhead 占比明显。

## 9. 阶段结论

本版本完成了 GEMM 优化路线的第一步：

> 实现 Triton FP16 GEMM baseline，并与 PyTorch/cuBLAS 做 microbenchmark 对比；在 RTX 3090 上，2048 方阵达到约 34.79 TFLOPS，约为 PyTorch baseline 的 70%。

当前版本的价值不是超过 PyTorch，而是建立可运行、可验证、可继续优化的 GEMM baseline。

## 10. Version 2 改进方向

Version 2 建议聚焦一个明确目标：

> 引入 tile size / kernel 参数搜索，让 GEMM 从固定参数 baseline 变成可调优实现。

优先级从高到低：

1. 增加多组 tile 配置实验
   - `block_m/block_n/block_k`
   - `num_warps`
   - `num_stages`
   - `group_m`

2. 增加相对误差报告
   - 最大绝对误差
   - 平均绝对误差
   - 最大相对误差
   - `torch.allclose`

3. 增加 benchmark CSV/Markdown 自动保存
   - 每次实验自动记录命令、GPU、环境、参数、结果
   - 方便后续写简历和复盘

4. 对比不同矩阵尺寸
   - `512`
   - `1024`
   - `2048`
   - `4096`

5. 分析最佳参数趋势
   - 小矩阵适合什么 tile
   - 大矩阵适合什么 tile
   - `num_warps` 增大是否有效
   - `block_k` 增大是否提升 Tensor Core 利用率

Version 2 不建议一上来做 double buffering 或 split-K，因为这两个复杂度更高。先做参数搜索，最快能产出新的实验结果，也最适合当前暑期实习时间紧的目标。

