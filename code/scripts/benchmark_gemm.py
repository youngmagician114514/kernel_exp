from __future__ import annotations

import argparse

import torch

from kernel_exp_project.triton_gemm import matmul
from kernel_exp_project.utils import benchmark_cuda, print_result, require_cuda, tflops


def run_case(size: int, repeat: int) -> None:
    require_cuda()
    torch.manual_seed(0)
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)

    torch_result = benchmark_cuda("torch.matmul", lambda: a @ b, warmup=5, repeat=repeat)
    triton_result = benchmark_cuda("triton_gemm", lambda: matmul(a, b), warmup=5, repeat=repeat)

    ref = a @ b
    out = matmul(a, b)
    torch.cuda.synchronize()
    max_err = float((ref - out).abs().max())

    print(f"\nM=N=K={size}, dtype=fp16")
    print_result(torch_result)
    print_result(triton_result)
    print(f"torch TFLOPS  {tflops(size, size, size, torch_result.median_ms):.2f}")
    print(f"triton TFLOPS {tflops(size, size, size, triton_result.median_ms):.2f}")
    print(f"max error     {max_err:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    for size in args.sizes:
        run_case(size, args.repeat)


if __name__ == "__main__":
    main()

