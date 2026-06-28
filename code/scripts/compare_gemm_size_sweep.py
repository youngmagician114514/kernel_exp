from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from kernel_exp_project.triton_gemm import matmul
from kernel_exp_project.utils import benchmark_cuda, require_cuda, tflops


@dataclass(frozen=True)
class GemmConfig:
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int


@dataclass
class SizeResult:
    size: int
    repeat: int
    memory_gb_min: float
    torch_ms: float
    triton_ms: float
    torch_tflops: float
    triton_tflops: float
    ratio: float
    max_abs_error: float
    allclose: bool
    config: GemmConfig


SMALL_CONFIG = GemmConfig(64, 64, 32, 8, 4, 4)
LARGE_CONFIG = GemmConfig(64, 128, 32, 8, 8, 4)


def config_for_size(size: int) -> GemmConfig:
    if size <= 1024:
        return SMALL_CONFIG
    return LARGE_CONFIG


def auto_repeat(size: int) -> int:
    if size <= 1024:
        return 30
    if size <= 2048:
        return 20
    if size <= 4096:
        return 10
    if size <= 8192:
        return 5
    if size <= 16384:
        return 3
    return 2


def estimate_min_gb(size: int) -> float:
    # A, B, C are fp16. This is a lower bound; framework caches/workspaces need extra memory.
    bytes_total = 3 * size * size * 2
    return bytes_total / 1024**3


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_size(size: int, repeat: int) -> SizeResult:
    config = config_for_size(size)
    kwargs = config.__dict__

    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    torch_result = benchmark_cuda("torch.matmul", lambda: a @ b, warmup=3, repeat=repeat)
    clear_cuda_cache()

    triton_result = benchmark_cuda("triton_gemm", lambda: matmul(a, b, **kwargs), warmup=3, repeat=repeat)
    clear_cuda_cache()

    ref = a @ b
    out = matmul(a, b, **kwargs)
    torch.cuda.synchronize()
    max_abs_error = float((ref - out).abs().max())
    allclose = bool(torch.allclose(ref, out, rtol=1e-2, atol=1e-1))

    torch_tflops = tflops(size, size, size, torch_result.median_ms)
    triton_tflops = tflops(size, size, size, triton_result.median_ms)

    del ref, out, a, b
    clear_cuda_cache()

    return SizeResult(
        size=size,
        repeat=repeat,
        memory_gb_min=estimate_min_gb(size),
        torch_ms=torch_result.median_ms,
        triton_ms=triton_result.median_ms,
        torch_tflops=torch_tflops,
        triton_tflops=triton_tflops,
        ratio=triton_tflops / torch_tflops,
        max_abs_error=max_abs_error,
        allclose=allclose,
        config=config,
    )


def print_result(result: SizeResult) -> None:
    print(
        f"{result.size:6d} repeat={result.repeat:2d} min_mem={result.memory_gb_min:6.2f}GB "
        f"torch={result.torch_tflops:7.2f}TF triton={result.triton_tflops:7.2f}TF "
        f"ratio={result.ratio:6.2%} torch_ms={result.torch_ms:8.3f} triton_ms={result.triton_ms:8.3f} "
        f"err={result.max_abs_error:8.4f} allclose={result.allclose}"
    )


def write_markdown(path: Path, results: list[SizeResult]) -> None:
    lines = [
        "# GEMM Version 2：尺寸扩展实验",
        "",
        "## 1. 实验目标",
        "",
        "比较不同矩阵尺寸下 PyTorch/cuBLAS 与当前 Triton GEMM V2 的性能比例，观察 Triton 实现是否随着矩阵变大更接近 PyTorch。",
        "",
        "## 2. Triton 配置",
        "",
        "- `size <= 1024`：使用 `BM=64, BN=64, BK=32, GM=8, warps=4, stages=4`。",
        "- `size > 1024`：使用 `BM=64, BN=128, BK=32, GM=8, warps=8, stages=4`。",
        "",
        "说明：这是基于 Version 2 参数搜索得到的启发式配置；大尺寸并未重新做完整 autotune。",
        "",
        "## 3. 实验结果",
        "",
        "| size | repeat | 最低显存 GB | torch ms | Triton ms | torch TFLOPS | Triton TFLOPS | Triton / torch | max abs err | allclose |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.size} | {result.repeat} | {result.memory_gb_min:.2f} | "
            f"{result.torch_ms:.4f} | {result.triton_ms:.4f} | "
            f"{result.torch_tflops:.2f} | {result.triton_tflops:.2f} | "
            f"{result.ratio:.2%} | {result.max_abs_error:.6f} | {result.allclose} |"
        )

    lines.extend(
        [
            "",
            "## 4. 初步分析",
            "",
            "- 小尺寸下 kernel launch、调度和框架开销占比更高，Triton baseline 更难接近 cuBLAS。",
            "- 中大尺寸下计算量变大，固定开销被摊薄，Triton V2 通常更接近 PyTorch/cuBLAS。",
            "- 如果比例在大尺寸继续上升，说明当前方法主要受小尺寸 overhead 和参数调优不足影响。",
            "- 如果比例在大尺寸下降，说明当前 tile/pipeline 配置没有很好扩展到更大矩阵，需要继续 autotune 或引入更高级优化。",
            "",
            "## 5. 注意事项",
            "",
            "- `32768` 方阵的最低 A/B/C 显存约 6GB，实际运行还会受到 PyTorch caching allocator 和 cuBLAS workspace 影响。",
            "- 共享服务器上 GPU 频率、温度、其他进程会影响绝对 TFLOPS，因此趋势分析优先看同场 `Triton / torch` 比例。",
            "- 当前大尺寸配置来自 2048 搜索结果，并不代表 8192/16384/32768 的最优 Triton 配置。",
            "",
        ]
    )

    if results:
        best = max(results, key=lambda item: item.ratio)
        worst = min(results, key=lambda item: item.ratio)
        lines.extend(
            [
                "## 6. 阶段结论",
                "",
                f"- 当前实验中 Triton 相对 PyTorch 的最高比例出现在 `size={best.size}`，达到 `{best.ratio:.2%}`。",
                f"- 当前实验中 Triton 相对 PyTorch 的最低比例出现在 `size={worst.size}`，为 `{worst.ratio:.2%}`。",
                "- 是否具有“大矩阵优势”需要看比例是否随 size 单调或总体上升。",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--max-size", type=int, default=32768)
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/code/results/gemm_v2_size_sweep_raw.md"))
    args = parser.parse_args()

    require_cuda()
    results: list[SizeResult] = []
    for size in args.sizes:
        if size > args.max_size:
            continue
        repeat = auto_repeat(size)
        print(f"\nrun size={size}, repeat={repeat}, estimated_min_memory={estimate_min_gb(size):.2f}GB", flush=True)
        try:
            result = run_size(size, repeat)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"OOM at size={size}: {exc}")
            clear_cuda_cache()
            break
        results.append(result)
        print_result(result)

    write_markdown(args.output_md, results)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
