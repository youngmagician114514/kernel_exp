from __future__ import annotations

import torch

from kernel_exp_project.triton_gemm import matmul
from kernel_exp_project.utils import benchmark_cuda, require_cuda, tflops


BEST_CONFIGS = {
    1024: {"block_m": 64, "block_n": 64, "block_k": 32, "group_m": 8, "num_warps": 4, "num_stages": 4},
    2048: {"block_m": 64, "block_n": 128, "block_k": 32, "group_m": 8, "num_warps": 8, "num_stages": 4},
}


def main() -> None:
    require_cuda()
    torch.manual_seed(0)
    repeat = 20

    print("| size | torch TFLOPS | V2 Triton TFLOPS | V2 / torch | torch ms | V2 ms | max abs err |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for size, config in BEST_CONFIGS.items():
        a = torch.randn((size, size), device="cuda", dtype=torch.float16)
        b = torch.randn((size, size), device="cuda", dtype=torch.float16)

        torch_result = benchmark_cuda("torch.matmul", lambda: a @ b, warmup=5, repeat=repeat)
        triton_result = benchmark_cuda("triton_gemm_v2", lambda: matmul(a, b, **config), warmup=5, repeat=repeat)

        ref = a @ b
        out = matmul(a, b, **config)
        torch.cuda.synchronize()
        max_err = float((ref - out).abs().max())

        torch_tflops = tflops(size, size, size, torch_result.median_ms)
        triton_tflops = tflops(size, size, size, triton_result.median_ms)
        ratio = triton_tflops / torch_tflops
        print(
            f"| {size} | {torch_tflops:.2f} | {triton_tflops:.2f} | {ratio:.2%} | "
            f"{torch_result.median_ms:.4f} | {triton_result.median_ms:.4f} | {max_err:.6f} |"
        )


if __name__ == "__main__":
    main()

