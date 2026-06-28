# FlashAttention Version 1：Triton Forward Baseline

## 1. 版本目标

本版本开始把 week07 的 FlashAttention 原理落成可运行代码。目标不是一上来超过 PyTorch 内置 SDPA，而是先实现一个能讲清楚、能验证正确性、能做 benchmark 的 Triton FlashAttention forward baseline。

一句话概括：

> 基于 Triton 实现 FP16 FlashAttention forward kernel，通过分块计算和 online softmax 避免显式落地 `S x S` attention score/probability 矩阵，并与显式 PyTorch attention 和 PyTorch SDPA 做对比。

代码入口：

```text
kernel_exp/code/kernel_exp_project/flash_attention.py
```

实验脚本：

```text
kernel_exp/code/scripts/benchmark_flash_attention.py
```

## 2. 参考的文档内容

本版本主要参考：

- `week07_flash_attention.md` 的 7.1：标准 Attention 的 `S x S` 中间矩阵问题。
- `week07_flash_attention.md` 的 7.2：FlashAttention 的 tiling 思想。
- `week07_flash_attention.md` 的 7.3：online softmax 的 running max 和 running sum。
- `week07_flash_attention.md` 的 7.4：分块 FlashAttention 伪代码。
- `week07_flash_attention.md` 的 7.6：causal mask 的基本思想。
- `online_softmax.py`：此前已经验证过 online softmax 与标准 softmax 的数值等价性。

## 3. 当前实现了哪些组件

### 3.1 输入形状

当前 kernel 支持：

```text
q, k, v: [batch, heads, seq_len, head_dim]
dtype: float16
head_dim <= 128
```

实验默认使用：

```text
batch = 1
heads = 4
head_dim = 64
```

这是一个教学版 forward kernel，暂时没有实现 backward。

### 3.2 Q block 和 K/V block 分块

每个 Triton program 负责一块 Q：

```text
block_m = 16
```

然后沿着 K/V 的序列维度分块扫描：

```text
block_n = 64
```

也就是：

```text
一个 program 负责 16 个 query token
每次读取 64 个 key/value token
循环扫描完整 K/V 序列
```

标准 Attention 会显式生成：

```text
scores = Q @ K^T    # [S, S]
probs = softmax(scores)
out = probs @ V
```

当前 Triton V1 在 kernel 内部直接做：

```text
Q_block @ K_block^T
online softmax 更新
累加 softmax_block @ V_block
```

因此不会把完整 `S x S` 的 `scores` 和 `probs` 写回 HBM。

### 3.3 Online softmax

每一行 query 维护两个统计量：

```text
m_i: 当前已经扫描过的 score 最大值
l_i: 基于 m_i 的 exp 分母和
```

处理一个新的 K block 时：

```text
m_new = max(m_i, rowmax(scores_block))
p = exp(scores_block - m_new)
alpha = exp(m_i - m_new)
l_new = l_i * alpha + rowsum(p)
```

这里 `alpha` 很关键。它负责在新的最大值 `m_new` 出现时，修正之前已经累加的结果。

### 3.4 输出 rescale

当前实现维护的是未归一化的输出累加器：

```text
acc = sum(exp(score - running_max) * V)
```

当 running max 更新时，旧的 `acc` 也要乘上：

```text
alpha = exp(m_i - m_new)
```

代码中的核心逻辑是：

```python
acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
```

最后再做：

```python
out = acc / l_i[:, None]
```

这就是 FlashAttention 能在分块情况下保持 softmax 正确性的关键。

### 3.5 Causal mask

当前支持：

```python
flash_attention(q, k, v, causal=True)
```

causal 逻辑是：

```text
第 i 个 query token 只能 attend 到位置 <= i 的 key token
```

代码中通过 mask 把未来 token 的 score 设为 `-inf`。

注意：当前 V1 还没有做真正的“对角线上方 K/V block 跳过”，所以 causal 版本虽然数值正确，但还会对未来 block 做一些被 mask 掉的无效计算。这是 V2 可以改进的地方。

## 4. 当前没有实现的组件

当前 V1 还不是工业级 FlashAttention，缺少：

- backward kernel。
- dropout。
- variable length / padding compact。
- 更高效的 causal block skip。
- 针对不同 `seq_len/head_dim` 的 autotune。
- `exp2` 优化和更精细的数值缩放。
- 更成熟的 block pointer / memory layout 优化。
- KV cache / decode attention。
- PagedAttention。
- FlashAttention-2/3 中的 warp-level work partition 和更高级 pipeline。

所以本版本适合写成：

> 实现 FlashAttention forward baseline 并验证核心 online softmax 机制。

不适合写成：

> 复现完整 FlashAttention-2 或超过 PyTorch SDPA。

## 5. 正确性验证

smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zhuyihui/data/wb/kernel_exp/code \
TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache \
conda run -p /home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv \
python kernel_exp/code/scripts/run_smoke_tests.py
```

结果：

```text
smoke_flash_attention_max_error 0.0009765625
smoke_ok
```

这个误差量级符合 FP16 teaching kernel 的预期。

## 6. 实验设置

实验环境：

| 组件 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| PyTorch | `2.9.1+cu128` |
| Triton | `3.5.1` |
| dtype | FP16 input, FP32 online softmax statistics |
| shape | `[batch=1, heads=4, seq_len, head_dim=64]` |

对比对象：

| 名称 | 含义 |
|---|---|
| 显式 PyTorch attention | 手动执行 `QK^T -> softmax -> PV`，会显式落地 `S x S` 中间矩阵 |
| PyTorch SDPA | `torch.nn.functional.scaled_dot_product_attention`，通常会调用 PyTorch 内置 fused/flash kernel |
| Triton FlashAttention V1 | 本项目实现的 Triton forward baseline |

## 7. Causal Attention 实验结果

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zhuyihui/data/wb/kernel_exp/code \
TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache \
conda run -p /home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv \
python kernel_exp/code/scripts/benchmark_flash_attention.py \
  --sizes 128 256 512 1024 \
  --causal \
  --include-sdpa \
  --output-md kernel_exp/code/results/flash_attention_v1_raw.md
```

| seq_len | repeat | 显式 PyTorch ms | PyTorch SDPA ms | Triton ms | Triton / 显式 PyTorch | Triton / SDPA | max abs err | allclose |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 30 | 0.3435 | 0.1086 | 0.1477 | 232.59% | 73.55% | 0.001953 | True |
| 256 | 30 | 0.4078 | 0.0833 | 0.1458 | 279.72% | 57.18% | 0.000977 | True |
| 512 | 20 | 0.5024 | 0.0991 | 0.1430 | 351.35% | 69.28% | 0.000977 | True |
| 1024 | 10 | 0.6346 | 0.1286 | 0.2180 | 291.14% | 59.01% | 0.001953 | True |

## 8. Non-causal Attention 实验结果

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/zhuyihui/data/wb/kernel_exp/code \
TRITON_CACHE_DIR=/home/zhuyihui/data/wb/kernel_exp/.triton_cache \
conda run -p /home/zhuyihui/data/wb/kernel_exp/.conda-vllm-srv \
python kernel_exp/code/scripts/benchmark_flash_attention.py \
  --sizes 128 256 512 \
  --include-sdpa \
  --repeat 10 \
  --output-md kernel_exp/code/results/flash_attention_v1_noncausal_raw.md
```

| seq_len | repeat | 显式 PyTorch ms | PyTorch SDPA ms | Triton ms | Triton / 显式 PyTorch | Triton / SDPA | max abs err | allclose |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 128 | 10 | 0.2598 | 0.0633 | 0.1041 | 249.70% | 60.83% | 0.000488 | True |
| 256 | 10 | 0.2957 | 0.0791 | 0.1390 | 212.75% | 56.89% | 0.000244 | True |
| 512 | 10 | 0.4357 | 0.0723 | 0.1934 | 225.29% | 37.36% | 0.000244 | True |

## 8.5 原始结果文件

当前 V1 的原始结果已合并进本主文档，原始记录来源于：

- `kernel_exp/code/results/flash_attention_v1_raw.md`
- `kernel_exp/code/results/flash_attention_v1_noncausal_raw.md`

## 9. 结果分析

### 9.1 为什么比显式 PyTorch attention 快

显式 PyTorch attention 会产生两个大的中间矩阵：

```text
scores: [B, H, S, S]
probs:  [B, H, S, S]
```

这些中间结果会写入和读出 HBM。当前 Triton V1 把 `QK^T`、softmax 和 `PV` 融合在一个 kernel 中，用 online softmax 在 block 内完成统计量更新，因此减少了大量中间矩阵读写。

所以 V1 在本次实验中达到显式 PyTorch attention 的 `212%-351%`。

### 9.2 为什么不如 PyTorch SDPA

PyTorch SDPA 不是普通 PyTorch 写法，它通常会调用高度优化的 fused attention kernel。它可能已经使用了 FlashAttention 或 memory-efficient attention 后端。

当前 V1 不如 SDPA，原因包括：

- 没有 causal block skip，causal 模式还会计算一部分未来 block 后再 mask。
- 没有针对不同 `seq_len/head_dim` 做 autotune。
- `block_m/block_n/num_warps/num_stages` 仍是固定默认值。
- 没有 FlashAttention-2 那样的更细粒度 work partition。
- 没有对小尺寸 kernel launch overhead 做特殊处理。
- `p.to(tl.float16)` 是教学实现中的简化，工业 kernel 会更精细地处理精度和 Tensor Core 路径。

因此目前更适合把 SDPA 当成“成熟 fused baseline”，而不是期待 V1 立刻超过它。

### 9.3 当前最稳妥的项目结论

可以说：

> FlashAttention V1 正确实现了分块 attention 和 online softmax rescaling，不显式落地 `S x S` score/probability 矩阵；在 RTX 3090 上，针对 `[1, 4, S, 64]` 的 causal attention，相比显式 PyTorch attention 达到约 `2.3x-3.5x` 的吞吐。

不建议说：

> 当前实现已经超过 PyTorch FlashAttention/SDPA。

## 10. 下一步方向

Version 2 建议优先做：

- 搜索 `block_m/block_n/num_warps/num_stages`。
- 在 causal 模式下跳过对角线上方的 K/V block。
- 对 `seq_len = 2048/4096` 做更长序列实验，但要先确认 GPU 空闲。
- 分别测试 `head_dim = 32/64/128`。
- 把结果汇总成表格，观察 V1 在长序列下是否更有优势。

Version 3 可以考虑：

- LLM prefill 常见 shape benchmark。
- KV cache decode attention。
- PagedAttention 原型。
- 对比 FlashInfer 或 vLLM 中的 attention kernel。
