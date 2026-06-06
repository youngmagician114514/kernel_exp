# kernel_exp 实验代码

这个目录把 `kernel_exp` 里的 Markdown 学习计划落成可运行的 PyTorch/Triton 实验代码。

## 环境

推荐使用已经克隆到项目内的环境：

```bash
conda run -p kernel_exp/.conda-vllm-srv python kernel_exp/code/scripts/check_env.py
```

如果克隆环境不可用，也可以临时复用原来的 `vllm_srv` 环境：

```bash
conda run -n vllm_srv python kernel_exp/code/scripts/check_env.py
```

在共享服务器上实验时，建议显式指定一张空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -p kernel_exp/.conda-vllm-srv python kernel_exp/code/scripts/run_smoke_tests.py
```

## 实验命令

```bash
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=kernel_exp/.triton_cache \
  conda run -p kernel_exp/.conda-vllm-srv python kernel_exp/code/scripts/benchmark_gemm.py --sizes 512 1024 --repeat 20

CUDA_VISIBLE_DEVICES=0 \
  conda run -p kernel_exp/.conda-vllm-srv python kernel_exp/code/scripts/run_online_softmax.py --rows 1024 --cols 4096

CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=kernel_exp/.triton_cache \
  conda run -p kernel_exp/.conda-vllm-srv python kernel_exp/code/scripts/benchmark_flash_attention.py --sizes 128 256 512 --causal --include-sdpa

CUDA_VISIBLE_DEVICES=0 \
  conda run -p kernel_exp/.conda-vllm-srv python kernel_exp/code/scripts/run_moe_routing.py --tokens 4096 --hidden 1024
```

## 模块说明

- `kernel_exp_project/triton_gemm.py`：教学版 Triton GEMM kernel，用于理解矩阵分块和 `tl.dot`。
- `kernel_exp_project/online_softmax.py`：FlashAttention 需要的分块 online softmax。
- `kernel_exp_project/flash_attention.py`：教学版 Triton FlashAttention forward kernel，用于理解分块 attention 和 online softmax rescaling。
- `kernel_exp_project/moe_routing.py`：Sparse MoE 的 Top-K gating 和 token permute/unpermute。
- `scripts/benchmark_gemm.py`：对比教学版 Triton GEMM 和 `torch.matmul`。
- `scripts/benchmark_flash_attention.py`：对比教学版 Triton FlashAttention、显式 PyTorch attention 和 PyTorch SDPA。
- `scripts/run_smoke_tests.py`：快速检查所有核心模块的正确性。
