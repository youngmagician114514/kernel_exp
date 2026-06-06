# kernel_exp

`kernel_exp` 是一个基于 PyTorch / Triton 的大模型推理算子实验目录，包含：

- 学习型周文档：
  - `week01` 到 `week12`
- 可运行源码：
  - `code/kernel_exp_project`
- 实验脚本：
  - `code/scripts`
- 阶段性总结文档：
  - `doc`

当前主要覆盖三条主线：

1. `GEMM`
2. `FlashAttention`
3. `Sparse MoE`

## 目录结构

```text
kernel_exp/
├── 00_master_plan.md
├── week01_*.md ... week12_*.md
├── code/
│   ├── README.md
│   ├── kernel_exp_project/
│   └── scripts/
├── doc/
│   ├── gemm_*.md
│   ├── flash_attention_*.md
│   └── moe_*.md
└── .gitignore
```

## 当前保留的核心文档

### GEMM

- `doc/gemm_v1_baseline.md`
- `doc/gemm_v2_plan.md`
- `doc/gemm_v2_tile_tuning.md`

### FlashAttention

- `doc/flash_attention_v1_baseline.md`
- `doc/flash_attention_v2_summary.md`
- `doc/flash_attention_v3_summary.md`
- `doc/flash_attention_resume_packaging.md`

### Sparse MoE

- `doc/moe_v1_summary.md`
- `doc/moe_v2_summary.md`
- `doc/moe_v3_summary.md`
- `doc/moe_v4_summary.md`

## 运行环境

本项目实验基于：

- Python 3.10+
- PyTorch
- Triton
- CUDA GPU

依赖列表见：

```text
requirements.txt
```

注意：

- `torch` 和 `triton` 的安装方式需要与你本机 CUDA 版本匹配。
- 如果你使用 NVIDIA GPU，建议优先按 PyTorch 官方方式安装 `torch`，再安装 `triton`。

## 快速开始

建议在项目根目录外单独创建虚拟环境，然后安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r kernel_exp/requirements.txt
```

检查环境：

```bash
PYTHONPATH=kernel_exp/code python kernel_exp/code/scripts/check_env.py
```

运行 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=kernel_exp/code \
TRITON_CACHE_DIR=kernel_exp/.triton_cache \
python kernel_exp/code/scripts/run_smoke_tests.py
```

## 常用实验脚本

### GEMM

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=kernel_exp/code \
TRITON_CACHE_DIR=kernel_exp/.triton_cache \
python kernel_exp/code/scripts/benchmark_gemm.py --sizes 512 1024 --repeat 20
```

### FlashAttention

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=kernel_exp/code \
TRITON_CACHE_DIR=kernel_exp/.triton_cache \
python kernel_exp/code/scripts/benchmark_flash_attention.py --sizes 128 256 512 --causal --include-sdpa
```

### Sparse MoE

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=kernel_exp/code \
TRITON_CACHE_DIR=kernel_exp/.triton_cache \
python kernel_exp/code/scripts/run_moe_routing.py --tokens 4096 --hidden 1024 --ffn-hidden 2048 --experts 8 --top-k 2
```

## 打包建议

如果你只想打包源码和文档，不带本地环境、缓存和临时目录，建议排除这些路径：

- `.conda-vllm-srv/`
- `.conda_tmp/`
- `.conda_cache/`
- `.triton_cache/`
- `code/**/__pycache__/`
- `__MACOSX/`

当前 `.gitignore` 已经排除了大部分本地环境与缓存目录。

一个简单的打包示例：

```bash
tar --exclude='kernel_exp/.conda-vllm-srv' \
    --exclude='kernel_exp/.conda_tmp' \
    --exclude='kernel_exp/.conda_cache' \
    --exclude='kernel_exp/.triton_cache' \
    --exclude='kernel_exp/__MACOSX' \
    -czf kernel_exp_source_docs.tar.gz kernel_exp
```

## 说明

这个目录包含两类内容：

1. **学习/设计文档**
   - 用来解释原理、版本演进和实验结论
2. **可运行实验代码**
   - 用来验证 correctness、性能和实现边界

它更适合作为：

- 算子优化学习项目
- 实验型代码仓
- 项目材料与简历支撑目录

而不是一个已经工程化发布的生产库。
