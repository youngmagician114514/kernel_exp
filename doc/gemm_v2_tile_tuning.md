# GEMM Version 2：Tile 参数搜索、尺寸扩展与大尺寸专项搜索

## 1. 版本定位

Version 2 的目标不是重新实现一个完整 cuBLAS，而是在 Version 1 的 Triton GEMM baseline 上做可解释、可实验、可写进简历的优化闭环：

> 基于 Triton 实现 FP16 GEMM kernel，通过 tile 参数搜索和尺寸扩展实验，分析自定义 Triton GEMM 与 PyTorch/cuBLAS 在不同矩阵尺寸下的性能差异。

当前实现更准确的定位是：

> Triton Tiled GEMM + `tl.dot` Tensor Core baseline + 参数搜索。

它已经不是 naive GEMM，但也还不是 CUTLASS/cuBLAS 级别的完整高性能 GEMM。

## 2. 当前实现的核心组件

代码入口：

```text
kernel_exp/code/kernel_exp_project/triton_gemm.py
```

### 2.1 Block / Tile 分块

每个 Triton program 负责计算输出矩阵 `C` 中一个 `BM x BN` 的二维 tile。

例如配置：

```text
BM=64, BN=128, BK=32
```

含义是：

- 一个 program 负责 `C` 的 `64 x 128` 个输出元素。
- 每次沿 K 维度取 `32` 个元素做一次局部矩阵乘。
- 多次 K 分块累加后得到这个 `C` tile 的最终结果。

### 2.2 K 维度分块累加

GEMM 计算是：

```text
C[M, N] = A[M, K] @ B[K, N]
```

代码没有一次性读完整个 K 维度，而是按 `BK` 分块循环：

```python
for k0 in range(0, k, block_k):
    a_tile = tl.load(...)
    b_tile = tl.load(...)
    acc += tl.dot(a_tile, b_tile)
```

这样可以控制每个 program 一次处理的数据量，避免寄存器和片上资源压力过大。

### 2.3 `tl.dot` 与 Tensor Core 路径

核心计算使用：

```python
acc += tl.dot(a_tile, b_tile)
```

输入是 FP16 时，Triton 通常会把 `tl.dot` 编译到 GPU Tensor Core 相关指令路径。也就是说，我们没有手写 `mma.sync`，而是通过 Triton 的高级抽象使用 Tensor Core。

这也是当前实现能明显快于朴素 CUDA GEMM 的主要原因之一。

### 2.4 FP32 accumulator

累加器使用 FP32：

```python
acc = tl.zeros((block_m, block_n), tl.float32)
```

这是一种常见 GEMM 策略：

- 输入：FP16
- 中间累加：FP32
- 输出：FP16

这样比全程 FP16 累加更稳定，也更接近 PyTorch/cuBLAS 的常见行为。

### 2.5 Mask 边界处理

`tl.load` 和 `tl.store` 都带有 mask：

```python
mask=(offs_m[:, None] < m) & (k0 + offs_k[None, :] < k)
```

这样即使矩阵尺寸不能整除 `BM/BN/BK`，也不会越界访问。当前实验主要是方阵尺寸，但这个组件让 kernel 更通用。

### 2.6 Stride 地址计算

代码传入了：

```text
stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
```

这些不是为了可读性，而是为了根据 PyTorch Tensor 的真实内存布局计算地址。`M/N/K` 描述的是矩阵形状，`stride` 描述的是元素在内存里怎么排列。

对连续矩阵来说 stride 比较简单；但有了 stride 参数，kernel 的表达方式更接近真实框架里的矩阵算子。

### 2.7 Grouped scheduling / `group_m`

`group_m` 控制 Triton program 的调度顺序。它不是改变数学结果，而是改变先算哪些 tile。

直观理解：

- 如果 program 按普通行优先顺序跑，可能会导致某些 B tile 刚加载完就很快被换出 cache。
- grouped scheduling 会让一组相邻 M 方向的 tile 连续处理同一段 N 方向区域。
- 这样多个 program 更可能复用 B 的数据，提高 L2 cache 利用率。

当前代码中对应的是：

```python
num_pid_in_group = group_m * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * group_m
```

### 2.8 可搜索参数

Version 2 搜索以下参数：

| 参数 | 含义 | 主要影响 |
|---|---|---|
| `BM` | C tile 的 M 方向大小 | 并行粒度、寄存器压力、访存复用 |
| `BN` | C tile 的 N 方向大小 | 并行粒度、寄存器压力、B tile 复用 |
| `BK` | K 维度每次加载宽度 | Tensor Core 利用、shared memory/寄存器压力 |
| `GM` | `group_m` | L2 cache 复用和 program 调度顺序 |
| `W` | `num_warps` | 一个 program 内部使用多少 warp |
| `S` | `num_stages` | Triton pipeline stage 数 |

## 3. 当前还没有实现的高级组件

当前 GEMM 没有完整实现以下组件，所以不能说已经达到 Week04 文档里的 85% 或 95% peak 档次：

- 手写 shared memory layout、bank conflict 优化和 swizzle。
- 手写 register tiling，精确控制每个 thread / warp 负责的输出元素。
- 显式 double buffering / `cp.async` pipeline。
- 手写 `mma.sync` 或 warp-level MMA 数据布局。
- warp specialization。
- split-K。
- persistent kernel。
- fused epilogue，例如 GEMM 后直接融合 bias、activation、quantization。
- cuBLASLt 级别的算法选择器和大规模 autotune。

因此当前项目最稳妥的表述是：

> 实现并调优了一个基于 Triton 的 FP16 GEMM kernel，包含 tile 分块、K 维度分块、`tl.dot` Tensor Core 路径、FP32 accumulator、mask 边界处理、stride 地址计算和 grouped scheduling，并通过参数搜索分析不同矩阵尺寸下相对 PyTorch/cuBLAS 的性能。

## 4. 实验环境

实验环境来自 `kernel_exp/.conda-vllm-srv`：

| 组件 | 版本 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| PyTorch | `2.9.1+cu128` |
| Torch CUDA | `12.8` |
| Triton | `3.5.1` |
| dtype | FP16 input, FP32 accumulate, FP16 output |
| baseline | `torch.matmul`，底层通常走 cuBLAS/cuBLASLt |

注意：服务器是共享环境，GPU 温度、频率和其他进程会影响绝对 TFLOPS。文档里更优先看同一次实验内的 `Triton / torch` 比例。

## 5. 实验一：启发式固定配置尺寸扩展

这个实验来自早期尺寸扩展实验，目的不是找每个尺寸最优配置，而是用一组启发式配置观察尺寸扩大时 Triton 是否自然接近 PyTorch。

使用配置：

- `size <= 1024`：`BM=64, BN=64, BK=32, GM=8, W=4, S=4`
- `size > 1024`：`BM=64, BN=128, BK=32, GM=8, W=8, S=4`

### 5.1 结果

| size | repeat | 最低显存 GB | torch ms | Triton ms | torch TFLOPS | Triton TFLOPS | Triton / torch | max abs err | allclose |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 512 | 30 | 0.00 | 0.1083 | 0.1903 | 2.48 | 1.41 | 56.90% | 0.000000 | True |
| 1024 | 30 | 0.01 | 0.0857 | 0.1223 | 25.05 | 17.56 | 70.10% | 0.000000 | True |
| 2048 | 20 | 0.02 | 0.3521 | 0.4403 | 48.79 | 39.02 | 79.97% | 0.125000 | True |
| 4096 | 10 | 0.09 | 2.2599 | 2.1546 | 60.82 | 63.79 | 104.89% | 0.000000 | True |
| 8192 | 5 | 0.38 | 15.8787 | 22.7061 | 69.24 | 48.42 | 69.93% | 0.000000 | True |
| 16384 | 3 | 1.50 | 120.5823 | 146.1835 | 72.95 | 60.17 | 82.49% | 0.000000 | True |
| 32768 | 2 | 6.00 | 1173.4710 | 1656.9105 | 59.97 | 42.47 | 70.82% | 0.000000 | True |

### 5.2 观察

固定启发式配置下，`512 -> 4096` 的比例整体变好，`4096` 达到 `104.89%`。但是 `8192/16384/32768` 没有继续稳定上升。

这个实验说明：

> 当前实现不能简单声称“矩阵越大越有优势”。大尺寸需要单独搜索参数。

## 6. 实验二：每个尺寸独立搜索

这个实验针对每个矩阵尺寸单独搜索 13 组候选配置，记录每个尺寸的最佳 Triton 配置，并与同场 `torch.matmul` 对比。

运行脚本：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/zhuyihui/data/wb/kernel_exp/code \
TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache \
conda run -p /home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv \
python kernel_exp/code/scripts/tune_gemm_v2_per_size.py \
  --sizes 512 1024 2048 4096 8192 16384 32768 \
  --output-md kernel_exp/code/results/gemm_v2_per_size_raw.md
```

### 6.1 每个尺寸的最优结果

| size | repeat | 最低显存 GB | 最优配置 | torch ms | Triton ms | torch TFLOPS | Triton TFLOPS | Triton / torch | max abs err | mean abs err | allclose |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 512 | 20 | 0.00 | `BM=32, BN=64, BK=32, GM=8, W=4, S=4` | 0.0742 | 0.1050 | 3.62 | 2.56 | 70.69% | 0.000000 | 0.000000 | True |
| 1024 | 20 | 0.01 | `BM=64, BN=128, BK=32, GM=4, W=4, S=4` | 0.1019 | 0.1443 | 21.08 | 14.88 | 70.60% | 0.000000 | 0.000000 | True |
| 2048 | 12 | 0.02 | `BM=64, BN=64, BK=32, GM=8, W=4, S=4` | 0.3597 | 0.3509 | 47.76 | 48.95 | 102.50% | 0.125000 | 0.004513 | True |
| 4096 | 6 | 0.09 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 2.3333 | 2.1049 | 58.90 | 65.30 | 110.85% | 0.000000 | 0.000000 | True |
| 8192 | 3 | 0.38 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 16.3247 | 16.5186 | 67.35 | 66.56 | 98.83% | 0.000000 | 0.000000 | True |
| 16384 | 1 | 1.50 | `BM=128, BN=128, BK=64, GM=8, W=8, S=4` | 126.6901 | 145.8740 | 69.43 | 60.30 | 86.85% | 0.000000 | 0.000000 | True |
| 32768 | 1 | 6.00 | `BM=128, BN=128, BK=32, GM=8, W=8, S=4` | 1321.2913 | 2389.9009 | 53.26 | 29.44 | 55.29% | 0.000000 | 0.000000 | True |

### 6.2 各尺寸 Top-5 搜索记录

#### size = 512

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=32, BN=64, BK=32, GM=8, W=4, S=4` | 0.1050 | 2.56 |
| 2 | `BM=32, BN=128, BK=32, GM=8, W=4, S=4` | 0.1199 | 2.24 |
| 3 | `BM=64, BN=128, BK=32, GM=4, W=4, S=4` | 0.1280 | 2.10 |
| 4 | `BM=64, BN=128, BK=32, GM=8, W=8, S=4` | 0.1312 | 2.05 |
| 5 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 0.1319 | 2.04 |

#### size = 1024

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=64, BN=128, BK=32, GM=4, W=4, S=4` | 0.1443 | 14.88 |
| 2 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 0.1461 | 14.69 |
| 3 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 0.1464 | 14.67 |
| 4 | `BM=64, BN=128, BK=32, GM=8, W=8, S=4` | 0.1472 | 14.59 |
| 5 | `BM=128, BN=64, BK=32, GM=8, W=4, S=4` | 0.1481 | 14.50 |

#### size = 2048

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=64, BN=64, BK=32, GM=8, W=4, S=4` | 0.3509 | 48.95 |
| 2 | `BM=128, BN=64, BK=32, GM=8, W=4, S=4` | 0.3694 | 46.51 |
| 3 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 0.3891 | 44.15 |
| 4 | `BM=64, BN=64, BK=64, GM=8, W=4, S=4` | 0.4079 | 42.12 |
| 5 | `BM=32, BN=128, BK=32, GM=8, W=4, S=4` | 0.4147 | 41.43 |

#### size = 4096

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 2.1049 | 65.30 |
| 2 | `BM=128, BN=64, BK=32, GM=8, W=4, S=4` | 2.1218 | 64.77 |
| 3 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 2.1414 | 64.18 |
| 4 | `BM=64, BN=128, BK=32, GM=4, W=4, S=4` | 2.1433 | 64.13 |
| 5 | `BM=64, BN=128, BK=32, GM=8, W=8, S=4` | 2.1614 | 63.59 |

#### size = 8192

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 16.5186 | 66.56 |
| 2 | `BM=128, BN=128, BK=64, GM=8, W=8, S=4` | 16.8473 | 65.26 |
| 3 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 17.1915 | 63.96 |
| 4 | `BM=64, BN=128, BK=32, GM=8, W=8, S=4` | 17.2037 | 63.91 |
| 5 | `BM=64, BN=128, BK=64, GM=8, W=4, S=4` | 18.3801 | 59.82 |

#### size = 16384

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=128, BN=128, BK=64, GM=8, W=8, S=4` | 145.8740 | 60.30 |
| 2 | `BM=128, BN=128, BK=32, GM=8, W=8, S=4` | 147.2749 | 59.73 |
| 3 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 148.7777 | 59.12 |
| 4 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 149.8536 | 58.70 |
| 5 | `BM=64, BN=128, BK=32, GM=4, W=4, S=4` | 156.1499 | 56.33 |

#### size = 32768

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=128, BN=128, BK=32, GM=8, W=8, S=4` | 2389.9009 | 29.44 |
| 2 | `BM=128, BN=128, BK=64, GM=8, W=8, S=4` | 2495.0892 | 28.20 |
| 3 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 2560.7513 | 27.48 |
| 4 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 2652.8343 | 26.53 |
| 5 | `BM=32, BN=128, BK=32, GM=8, W=4, S=4` | 2781.2839 | 25.30 |

### 6.3 观察

每个尺寸的最佳配置并不完全相同：

| size | 最优 tile |
|---:|---|
| 512 | `BM=32, BN=64, BK=32` |
| 1024 | `BM=64, BN=128, BK=32, GM=4` |
| 2048 | `BM=64, BN=64, BK=32` |
| 4096 | `BM=64, BN=128, BK=32` |
| 8192 | `BM=64, BN=128, BK=32` |
| 16384 | `BM=128, BN=128, BK=64` |
| 32768 | `BM=128, BN=128, BK=32` |

这说明 GEMM 参数需要随矩阵尺寸调整，固定套一个配置不可靠。

## 7. 实验三：大尺寸专用参数搜索

实验二里 `16384/32768` 明显落后，所以继续扩大大尺寸候选配置，加入：

- 更大的 `128 x 128` tile。
- 更宽的 `64 x 256` / `128 x 256` / `256 x 128` tile。
- `BK=64/128` 候选。
- `num_warps=4/8` 和 `num_stages=3/4` 的组合。

运行脚本：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zhuyihui/data/wb/kernel_exp/code \
TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache \
conda run -p /home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv \
python kernel_exp/code/scripts/tune_gemm_v2_large_sizes.py \
  --sizes 16384 32768 \
  --repeat 1 \
  --warmup 1 \
  --output-md kernel_exp/code/results/gemm_v2_large_size_tuning_raw.md
```

### 7.1 大尺寸专项最优结果

| size | 最优配置 | torch ms | Triton ms | torch TFLOPS | Triton TFLOPS | Triton / torch | max abs err | allclose |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 16384 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 131.8621 | 130.0738 | 66.71 | 67.62 | 101.37% | 0.000000 | True |
| 32768 | `BM=128, BN=128, BK=64, GM=8, W=4, S=4` | 1071.0085 | 1258.4409 | 65.70 | 55.92 | 85.11% | 0.000000 | True |

### 7.2 size = 16384 完整搜索记录

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 130.0738 | 67.62 |
| 2 | `BM=128, BN=256, BK=32, GM=8, W=8, S=4` | 131.0644 | 67.11 |
| 3 | `BM=128, BN=128, BK=32, GM=4, W=8, S=4` | 132.4061 | 66.43 |
| 4 | `BM=128, BN=128, BK=64, GM=8, W=8, S=4` | 137.0256 | 64.19 |
| 5 | `BM=128, BN=128, BK=32, GM=8, W=8, S=4` | 138.3898 | 63.56 |
| 6 | `BM=128, BN=128, BK=64, GM=4, W=8, S=4` | 140.5810 | 62.57 |
| 7 | `BM=128, BN=128, BK=64, GM=8, W=8, S=3` | 140.6699 | 62.53 |
| 8 | `BM=64, BN=256, BK=32, GM=8, W=8, S=4` | 141.2830 | 62.26 |
| 9 | `BM=128, BN=128, BK=64, GM=8, W=4, S=4` | 142.4606 | 61.74 |
| 10 | `BM=64, BN=128, BK=64, GM=8, W=4, S=4` | 148.9719 | 59.05 |
| 11 | `BM=256, BN=128, BK=32, GM=8, W=8, S=4` | 177.7676 | 49.48 |
| 12 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 190.7671 | 46.11 |
| 13 | `BM=256, BN=64, BK=32, GM=8, W=8, S=4` | 284.6951 | 30.90 |

### 7.3 size = 32768 完整搜索记录

| rank | 配置 | median ms | TFLOPS |
|---:|---|---:|---:|
| 1 | `BM=128, BN=128, BK=64, GM=8, W=4, S=4` | 1258.4409 | 55.92 |
| 2 | `BM=128, BN=256, BK=32, GM=8, W=8, S=4` | 1317.1888 | 53.42 |
| 3 | `BM=64, BN=128, BK=64, GM=8, W=4, S=4` | 1351.3170 | 52.07 |
| 4 | `BM=128, BN=128, BK=32, GM=8, W=8, S=4` | 1365.8897 | 51.52 |
| 5 | `BM=128, BN=128, BK=64, GM=8, W=8, S=3` | 1369.0905 | 51.40 |
| 6 | `BM=128, BN=128, BK=32, GM=4, W=8, S=4` | 1387.7112 | 50.71 |
| 7 | `BM=128, BN=128, BK=64, GM=8, W=8, S=4` | 1394.1356 | 50.47 |
| 8 | `BM=256, BN=128, BK=32, GM=8, W=8, S=4` | 1421.6809 | 49.50 |
| 9 | `BM=128, BN=128, BK=64, GM=4, W=8, S=4` | 1476.9194 | 47.65 |
| 10 | `BM=64, BN=256, BK=32, GM=8, W=8, S=4` | 1594.6963 | 44.13 |
| 11 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 1643.6705 | 42.81 |
| 12 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 1667.0561 | 42.21 |
| 13 | `BM=256, BN=64, BK=32, GM=8, W=8, S=4` | 2417.7558 | 29.10 |

### 7.4 跳过的配置

以下配置在 RTX 3090 上因为 shared memory 需求超过硬件限制，被 Triton 报 `OutOfResources` 并跳过：

| 配置 | 原因 |
|---|---|
| `BM=128, BN=128, BK=128, GM=8, W=8, S=4` | shared memory 需求约 `196608` bytes，超过硬件限制 `101376` bytes |
| `BM=64, BN=256, BK=64, GM=8, W=8, S=4` | shared memory 需求约 `122880` bytes，超过硬件限制 `101376` bytes |
| `BM=256, BN=64, BK=64, GM=8, W=8, S=4` | shared memory 需求约 `122880` bytes，超过硬件限制 `101376` bytes |

这说明 tile 不能无限放大。更大的 tile 可能提高复用，但也会增加 shared memory、寄存器和调度压力。

## 8. 当前 Version 2 最好结果汇总

下面表格使用当前已有实验中的最佳结果：

- `512-8192` 使用每尺寸独立搜索结果。
- `16384/32768` 使用大尺寸专项搜索结果。

| size | 当前最好配置 | 结果来源 | torch TFLOPS | Triton TFLOPS | Triton / torch | 判断 |
|---:|---|---|---:|---:|---:|---|
| 512 | `BM=32, BN=64, BK=32, GM=8, W=4, S=4` | 每尺寸搜索 | 3.62 | 2.56 | 70.69% | 明显落后 |
| 1024 | `BM=64, BN=128, BK=32, GM=4, W=4, S=4` | 每尺寸搜索 | 21.08 | 14.88 | 70.60% | 明显落后 |
| 2048 | `BM=64, BN=64, BK=32, GM=8, W=4, S=4` | 每尺寸搜索 | 47.76 | 48.95 | 102.50% | 接近或略超 |
| 4096 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 每尺寸搜索 | 58.90 | 65.30 | 110.85% | 当前最佳区间 |
| 8192 | `BM=64, BN=128, BK=32, GM=8, W=4, S=4` | 每尺寸搜索 | 67.35 | 66.56 | 98.83% | 基本接近 |
| 16384 | `BM=64, BN=128, BK=32, GM=8, W=4, S=3` | 大尺寸专项搜索 | 66.71 | 67.62 | 101.37% | 接近或略超 |
| 32768 | `BM=128, BN=128, BK=64, GM=8, W=4, S=4` | 大尺寸专项搜索 | 65.70 | 55.92 | 85.11% | 仍然落后 |

## 9. 结果分析

### 9.1 参数搜索确实有效

相比固定启发式配置，每尺寸搜索和大尺寸专项搜索能明显改善部分尺寸：

| size | 早期结果 | 搜索后结果 | 变化 |
|---:|---:|---:|---:|
| 2048 | 79.97% | 102.50% | +22.53 percentage points |
| 8192 | 69.93% | 98.83% | +28.90 percentage points |
| 16384 | 86.85% | 101.37% | +14.52 percentage points |
| 32768 | 55.29% | 85.11% | +29.82 percentage points |

其中 `16384/32768` 的改善来自大尺寸专项搜索，说明超大尺寸并不是完全没有优化空间，之前落后很大主要有参数没有搜充分的原因。

### 9.2 不能简单说矩阵越大越有优势

当前最好结果比例是：

```text
512:    70.69%
1024:   70.60%
2048:  102.50%
4096:  110.85%
8192:   98.83%
16384: 101.37%
32768:  85.11%
```

它不是单调上升的。更准确的说法是：

> 当前 Triton GEMM V2 在 `2048-16384` 区间已经具备较强竞争力，其中 `2048/4096/8192/16384` 能接近或超过同场 PyTorch/cuBLAS；但 `32768` 仍明显落后，说明超大尺寸还需要更深层的 kernel 结构优化。

### 9.3 为什么 `32768` 仍然落后

可能原因：

- 当前只是 tile 参数搜索，没有改变 kernel 的整体执行结构。
- PyTorch/cuBLASLt 在超大 GEMM 上有更成熟的算法选择、pipeline 和调度策略。
- 当前没有 split-K 或 persistent kernel，超大矩阵下 SM 级别调度和数据复用可能不够充分。
- 当前没有手写 double buffering 和 warp specialization，访存与计算重叠程度有限。
- `repeat=1` 的大尺寸结果受共享服务器状态影响较大，需要后续空闲时复测。

### 9.4 为什么某些大 tile 反而更慢

大 tile 不一定更快，因为它同时带来三种压力：

- 每个 program 的 accumulator 更大，寄存器压力更高。
- A/B tile 更大，shared memory 或内部缓存压力更高。
- 单个 program 更重，可能降低并发度，导致 SM occupancy 下降。

例如 `BM=256, BN=64, BK=32` 在两个大尺寸上都很慢，说明 M 方向过宽的 tile 对当前实现并不合适。

## 10. 简历表述建议

可以写：

> 基于 Triton 实现 FP16 GEMM kernel，支持 tile blocking、K 维度分块、`tl.dot` Tensor Core 计算、FP32 accumulator、mask 边界处理、stride 地址计算和 grouped scheduling；进一步构建参数搜索脚本，对 `512-32768` 方阵进行尺寸扩展实验，在 RTX 3090 上分析与 PyTorch/cuBLAS 的性能差异，当前在 `2048-16384` 区间达到 PyTorch/cuBLAS 约 `98%-111%` 的吞吐。

不建议写：

> 完整实现 cuBLAS 级 GEMM。

也不建议写：

> 矩阵越大 Triton 越有优势。

## 11. 下一步方向

如果继续推进 Version 2，建议先做稳定性复测：

- 对 `2048/4096/8192/16384` 重复运行 3 次，确认超过或接近 PyTorch 的结果是否稳定。
- 对 `32768` 继续搜索更适合的参数，但要控制服务器负载。
- 把搜索结果保存为 CSV，方便画图和写项目报告。

如果进入后续 Version 3，可以考虑：

- fused epilogue：在 GEMM 后融合 bias / activation。
- split-K：改善超大尺寸或特殊形状下的并行度。
- persistent kernel：减少超大 GEMM 的调度开销。
- 更系统的 autotune：按 shape 自动选择配置。
- Nsight Systems / Nsight Compute 分析：定位瓶颈是 Tensor Core 利用率、访存、occupancy 还是调度。
