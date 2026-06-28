from __future__ import annotations

import argparse

import torch

from llm0.kernels.rms_norm import rms_norm_reference, rms_norm_triton


def bench_cuda(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Triton benchmark")

    dtype = getattr(torch, args.dtype)
    x = torch.randn((args.rows, args.hidden), device="cuda", dtype=dtype)
    weight = torch.randn((args.hidden,), device="cuda", dtype=dtype)

    ref = rms_norm_reference(x, weight)
    tri = rms_norm_triton(x, weight)
    torch.cuda.synchronize()
    max_err = float((ref - tri).abs().max())
    mean_err = float((ref - tri).abs().mean())

    torch_ms = bench_cuda(lambda: rms_norm_reference(x, weight), warmup=10, repeat=args.repeat)
    triton_ms = bench_cuda(lambda: rms_norm_triton(x, weight), warmup=10, repeat=args.repeat)
    num_bytes = x.numel() * x.element_size() * 2 + weight.numel() * weight.element_size()
    torch_gbs = num_bytes / (torch_ms * 1e6)
    triton_gbs = num_bytes / (triton_ms * 1e6)

    print(f"shape=({args.rows}, {args.hidden}), dtype={args.dtype}")
    print(f"max_err={max_err:.6e} mean_err={mean_err:.6e}")
    print(f"torch_ms={torch_ms:.4f} torch_GBps={torch_gbs:.2f}")
    print(f"triton_ms={triton_ms:.4f} triton_GBps={triton_gbs:.2f}")


if __name__ == "__main__":
    main()
