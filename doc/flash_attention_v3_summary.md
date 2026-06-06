# FlashAttention V3：Varlen / cuSeqlens Triton Kernel 总结

## 1. V3 优化摘要

V3 相比前面的版本，核心不是再做一版固定长度 attention，而是把 attention 推进到 **变长序列 / 无 padding 浪费** 的实现路径。

本版本完成了四层关键改进：

1. **从 dense/padded 输入推进到 packed + `cu_seqlens` 表示**
   - 用累积长度数组描述每条序列的真实 token 区间。
   - 避免 batch 内所有序列统一 pad 到最大长度。

2. **从逐序列 Python 循环推进到单 Triton kernel 读取 `cu_seqlens`**
   - 旧 prototype 是逐条序列切片，再调用 V2。
   - 新 V3 是单 kernel 处理整批 packed token。

3. **针对 varlen 路径重新组织 packed memory layout**
   - 将 packed `Q/K/V` 的物理布局改成更适合 kernel 的 head-major 形式：
     `[heads, total_tokens, head_dim]`
   - 让固定一个 head 时，沿 token 维的访问更连续。

4. **为 varlen 路径单独做参数搜索与默认配置选择**
   - 不再直接沿用 V2 dense 路径的配置。
   - 根据 `max_seq_len` 为 V3 选择更合适的 `block_m / block_n / warps / stages`。

一句话概括：

> V3 把 FlashAttention 从固定长度、dense 输入的教学版 forward，推进成了一个支持 packed token 与 `cu_seqlens` 的 varlen Triton kernel，并在长序列变长 batch 上开始体现出相对 masked SDPA 的优势。

代码入口：

```text
kernel_exp/code/kernel_exp_project/flash_attention.py
```

---

## 2. 对应文档内容

本版本主要依据：

- `kernel_exp/week08_trt_fmha.md`
  - `8.5 Variable Sequence Length (cuSeqlens)`
- `kernel_exp/week07_flash_attention.md`
  - `7.2 FlashAttention 的核心思想`
  - `7.3 Online Softmax`
  - `7.6 Causal Masking`

其中 `week08` 给出了 V3 的核心方向：

```text
传统 padding：
  所有序列 pad 到最大长度，浪费计算

cuSeqlens：
  packed_tokens + cumulative lengths
  → 无 padding，零浪费
```

---

## 3. V3 的实现结构

### 3.1 输入表示

V3 使用：

```text
q, k, v: [heads, total_tokens, head_dim]
cu_seqlens: [batch + 1]
```

其中：

- `cu_seqlens[i]` 是前 `i` 条序列累计的 token 数
- 第 `i` 条序列的真实区间为：
  - `start = cu_seqlens[i]`
  - `end = cu_seqlens[i+1]`

### 3.2 关键辅助函数

V3 相关接口包括：

- `pack_padded_qkv(...)`
- `unpack_padded_output(...)`
- `flash_attention_v3(...)`
- `flash_attention_v3_prototype(...)`
- `attention_reference_v3(...)`
- `max_abs_error_v3(...)`

这里保留 `flash_attention_v3_prototype(...)` 的目的，是让我们始终能对照“旧的逐序列版本”和“新的单 kernel 版本”。

### 3.3 单 kernel varlen 路径

当前主实现是：

```text
_flash_attention_varlen_fwd_kernel(...)
```

它的核心思路是：

1. kernel 根据 `block_batch_ids` 和 `block_q_offsets` 只给真实存在的 `Q block` 发 program。
2. 每个 program 再通过 `cu_seqlens[batch_id]` / `cu_seqlens[batch_id+1]` 取出当前序列的 `seq_start` 和 `seq_len`。
3. `K/V block` 扫描、online softmax、causal block skip、diagonal block mask 都在单 kernel 内完成。

这比最早的 prototype 更符合 `cuSeqlens` 的“零 padding 浪费”目标。

---

## 4. 正确性验证

`run_smoke_tests.py` 已包含：

- `smoke_flash_attention_v3_prototype_max_error`
- `smoke_flash_attention_v3_max_error`

当前结果：

```text
smoke_flash_attention_v3_prototype_max_error 0.0009765625
smoke_flash_attention_v3_max_error 0.0009765625
```

说明：

- 旧 prototype 正确
- 新的单 kernel `cu_seqlens` 路径也正确
- 二者都和 reference 保持在当前 FP16 教学版的误差范围内

---

## 5. V3 相对旧 Prototype 的收益

实验脚本：

```text
kernel_exp/code/scripts/benchmark_flash_attention_v3.py
```

### 5.1 中短序列

结果记录：

```text
kernel_exp/code/results/flash_attention_v3_varlen.md
```

| max_seq_len | lengths | total_tokens | prototype ms | v3 ms | prototype / v3 | allclose |
|---:|---|---:|---:|---:|---:|:---:|
| 128 | `[128, 96, 64, 32]` | 320 | 1.4453 | 0.5170 | 279.54% | True |
| 256 | `[256, 192, 128, 64]` | 640 | 1.5618 | 0.5423 | 288.02% | True |
| 512 | `[512, 384, 256, 128]` | 1280 | 1.5711 | 0.7788 | 201.73% | True |

### 5.2 长序列

结果记录：

```text
kernel_exp/code/results/flash_attention_v3_varlen_long.md
```

| max_seq_len | lengths | total_tokens | prototype ms | v3 ms | prototype / v3 | allclose |
|---:|---|---:|---:|---:|---:|:---:|
| 2048 | `[2048, 1536, 1024, 512]` | 5120 | 1.5231 | 0.8111 | 187.80% | True |
| 4096 | `[4096, 3072, 2048, 1024]` | 10240 | 2.0002 | 1.5444 | 129.52% | True |

### 5.3 小结

V3 相对旧 prototype 的收益来自两点：

- 不再逐序列 launch
- 直接在 kernel 内消费 `cu_seqlens`

在 `128-512` 的中短序列上收益约 `2.0x-2.9x`，到 `2048-4096` 时仍然保持正收益。

---

## 6. V3 专用参数搜索

当前得到的较优配置为：

| max_seq_len | total_tokens | 最优配置 | median ms |
|---:|---:|---|---:|
| 128 | 320 | `BM=16, BN=64, W=8, S=4` | 0.5209 |
| 256 | 640 | `BM=32, BN=64, W=8, S=4` | 0.4421 |
| 512 | 1280 | `BM=16, BN=64, W=4, S=3` | 0.4609 |

当前已经把这些结果回填到：

```text
_select_flash_attention_v3_config(...)
```

因此主实现不再直接沿用 V2 dense 路径的默认配置。

---

## 7. 与 Padded Baseline / Masked SDPA 的公平实验

公平实验脚本：

```text
kernel_exp/code/scripts/compare_flash_attention_v3_with_sdpa.py
```

这里的对比口径是：

- `V3 varlen Triton kernel`
- `padded explicit baseline`
- `masked PyTorch SDPA`

对于超长序列，为避免 `S x S` reference / explicit OOM，也支持：

- 只比较 `V3` 和 `masked SDPA`

### 7.1 中短序列到 1k

| max_seq_len | total_tokens | V3 config | explicit ms | masked SDPA ms | V3 ms | V3 / explicit | V3 / SDPA |
|---:|---:|---|---:|---:|---:|---:|---:|
| 128 | 320 | `BM=16, BN=64, W=8, S=4` | 1.1144 | 0.1362 | 0.5044 | 220.92% | 27.00% |
| 256 | 640 | `BM=32, BN=64, W=8, S=4` | 0.6939 | 0.1393 | 0.4935 | 140.61% | 28.23% |
| 512 | 1280 | `BM=16, BN=64, W=4, S=3` | 0.7588 | 0.1552 | 0.7071 | 107.32% | 21.95% |
| 1024 | 2560 | `BM=16, BN=64, W=4, S=3` | 1.6333 | 0.2539 | 0.7153 | 228.35% | 35.49% |

### 7.2 2k 到 4k

| max_seq_len | total_tokens | V3 config | explicit ms | masked SDPA ms | V3 ms | V3 / explicit | V3 / SDPA |
|---:|---:|---|---:|---:|---:|---:|---:|
| 2048 | 5120 | `BM=16, BN=64, W=4, S=3` | 9.4356 | 0.6498 | 0.9295 | 1015.15% | 69.91% |
| 4096 | 10240 | `BM=16, BN=64, W=4, S=3` | 22.1104 | 1.9985 | 1.3282 | 1664.72% | 150.47% |

### 7.3 8k / 16k / 32k

这组实验只比较 `V3` 和 `masked SDPA`，不再分配 `explicit/reference` 的 `S x S` 中间矩阵。

| max_seq_len | total_tokens | V3 config | masked SDPA ms | V3 ms | V3 / SDPA |
|---:|---:|---|---:|---:|---:|
| 8192 | 20480 | `BM=16, BN=64, W=4, S=3` | 7.5531 | 5.9506 | 126.93% |
| 16384 | 40960 | `BM=16, BN=64, W=4, S=3` | 29.7945 | 10.2792 | 289.85% |
| 32000 | 80000 | `BM=16, BN=64, W=4, S=3` | 120.3813 | 65.9414 | 182.56% |

### 7.4 总结

从公平实验可以看到：

- 在短序列上，V3 明显慢于 masked SDPA。
- 在 `2048` 左右，V3 已经接近 SDPA。
- 在 `4096` 及以上的真实 varlen 长序列 batch 上，V3 开始超过 masked SDPA。

这说明：

> V3 不是一个“全局替代 V2”的版本，而是一个在长序列、变长 batch 场景下开始体现优势的 varlen/cuSeqlens 路径。

### 7.5 原始结果文件

当前 V3 的调参与实验结果已合并进本主文档，原始记录来源于：

- `kernel_exp/code/results/flash_attention_v3_varlen.md`
- `kernel_exp/code/results/flash_attention_v3_varlen_long.md`
- `kernel_exp/doc/flash_attention_v3_fair_compare.md`
- `kernel_exp/doc/flash_attention_v3_fair_compare_long.md`
- `kernel_exp/doc/flash_attention_v3_fair_compare_32k.md`

---

## 8. 当前最准确的项目结论

V3 现在可以准确描述为：

> 实现了基于 packed token 与 `cu_seqlens` 的 variable-length FlashAttention Triton kernel，支持单 kernel 处理整批变长序列；相对旧的逐序列 prototype 在 `128-4096` 序列实验中获得约 `1.3x-2.9x` 提升；在长序列变长 batch 场景下，相对 masked PyTorch SDPA 在 `8k/16k/32k` 实验中达到约 `1.27x / 2.90x / 1.83x` 的性能提升。

不建议写成：

> 全面超过 PyTorch SDPA

因为这只在长序列 varlen 场景成立，在短序列场景并不成立。

---

## 9. 与 V2 的关系

- `V2` 的强项是：
  - dense / 固定长度
  - causal forward
  - 更接近原生 SDPA 的固定长度路径

- `V3` 的强项是：
  - varlen / `cu_seqlens`
  - packed token
  - 长序列变长 batch

所以更合理的理解不是：

> V3 全面替代 V2

而是：

> V3 补上了 V2 不具备的 varlen 能力，并在长序列变长场景开始体现出独立价值。

---

## 10. 后续方向

如果继续推进 V3，最值得做的是：

1. 扩展 `head_dim=128`、更大 batch 的 varlen 调参与实验。
2. 把长序列最佳配置进一步细化，避免 `4096+` 继续沿用单一配置。
3. 分析 `block map` 构建与 launch 开销，继续压缩 host-side 成本。
4. 再往后才是 decode / KV cache / paged attention。
