# FlashAttention V2：Causal Block Skip + Shape-Aware Config 总结

## 1. V2 优化摘要

V2 的目标不是重写整个 kernel，而是在 V1 的 forward baseline 上补上最有收益的工程化改进。

本版本完成了三层关键改进：

1. **加入 causal block skip**
   - 对角线上方的 `K/V block` 直接跳过
   - 不再像 V1 一样扫描完整序列后再逐元素 mask

2. **区分 full-valid blocks 与 diagonal blocks**
   - 对角线以下整块不再逐元素做 causal mask
   - 只对真正的 diagonal block 保留逐元素 mask

3. **引入 shape-aware 默认配置**
   - 基于 `seq_len` 选择更合适的 `block_m / block_n / num_warps / num_stages`
   - 不再固定使用 V1 的默认值

一句话概括：

> V2 在 V1 的基础上补齐了更合理的 causal block 处理和按 shape 选择配置的能力，使 kernel 更贴近真实的 causal FlashAttention 路径。

---

## 2. 参考文档内容

本版本主要依据：

- `kernel_exp/week07_flash_attention.md`
  - `7.6 Causal Masking 的高效实现`
- `kernel_exp/doc/flash_attention_v1_baseline.md`
  - V1 已明确指出缺失：
    - `causal block skip`
    - `shape-aware autotune`

---

## 3. 当前实现结构

V2 仍然复用：

```text
_flash_attention_fwd_kernel(...)
flash_attention(...)
```

但主路径新增了：

- `causal_block_skip`
- full-valid vs diagonal block 分支
- `_select_flash_attention_v2_config(...)`
- `flash_attention_v2(...)`

也就是说，V2 的主改动是：

- **更合理的控制流**
- **更合理的默认配置**

而不是重新定义 attention 数学逻辑。

---

## 4. 正确性验证

`run_smoke_tests.py` 已新增：

- `smoke_flash_attention_v2_max_error`

结果：

```text
smoke_flash_attention_v2_max_error 0.0009765625
```

说明 V2 与 reference 在当前 FP16 教学版误差范围内一致。

---

## 5. 参数搜索结果

V2 的专门调参结果如下：

| seq_len | repeat | 最优配置 | median ms | 相对默认配置 |
|---:|---:|---|---:|---:|
| 512 | 12 | `BM=16, BN=64, W=8, S=4` | 0.1443 | 102.82% |
| 1024 | 8 | `BM=32, BN=128, W=4, S=3` | 0.1461 | 106.20% |
| 2048 | 4 | `BM=32, BN=64, W=4, S=3` | 0.1969 | 116.10% |

当前这些结果已经回填到：

```text
_select_flash_attention_v2_config(...)
```

因此 V2 主实现不再沿用单一默认配置。

---

## 6. V1 vs V2 对比结果

V1/V2 同场结果如下：

| seq_len | PyTorch SDPA ms | Triton V1 ms | Triton V2 ms | Triton V1 / SDPA | Triton V2 / SDPA |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.1199 | 0.1855 | 0.1902 | 64.65% | 63.03% |
| 256 | 0.1202 | 0.2005 | 0.1995 | 59.97% | 60.27% |
| 512 | 0.0714 | 0.1477 | 0.1138 | 48.35% | 62.75% |
| 1024 | 0.0962 | 0.2328 | 0.1597 | 41.34% | 60.27% |

可以看出：

- 短序列上 V2 收益不明显
- 长一些的 causal 序列上，V2 明显优于 V1

这与 `causal block skip` 的收益特征一致。

---

## 7. 当前版本的边界

当前 V2 已经实现：

- causal block skip
- diagonal/full-valid block 区分
- shape-aware 默认配置

当前还没有实现：

- backward
- dropout
- varlen / `cu_seqlens`
- KV cache / decode
- paged attention
- FA2/FA3 级别 work partition

因此当前 V2 更准确的定位是：

> 更完整的 causal forward 优化版

而不是：

> 完整工业级 FlashAttention-2/3 实现

---

## 8. 当前最准确的项目表述

可以说：

> 在 Triton FlashAttention forward baseline 基础上，实现了 causal block skip 和 shape-aware 配置选择；在 `[B=1,H=4,D=64]` 的 causal attention 设置下，相比 V1 在中长序列上获得进一步吞吐提升，并将性能提升到原生 SDPA 的约 `60%+` 水平。

不建议说：

> V2 已经超过原生 PyTorch SDPA

因为当前实验并不支持这个说法。
