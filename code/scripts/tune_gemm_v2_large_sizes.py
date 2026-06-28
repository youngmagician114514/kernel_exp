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
class ConfigResult:
    size: int
    config: GemmConfig
    median_ms: float
    tflops: float


@dataclass
class LargeSizeResult:
    size: int
    torch_ms: float
    torch_tflops: float
    best: ConfigResult
    ratio: float
    max_abs_error: float
    allclose: bool
    records: list[ConfigResult]


CONFIGS = [
    # Previous V2 winners / near-winners.
    GemmConfig(64, 128, 32, 8, 4, 4),
    GemmConfig(64, 128, 32, 8, 4, 3),
    GemmConfig(64, 128, 64, 8, 4, 4),
    GemmConfig(128, 128, 32, 8, 8, 4),
    GemmConfig(128, 128, 64, 8, 8, 4),
    # Larger K tile / alternative stage settings.
    GemmConfig(128, 128, 128, 8, 8, 4),
    GemmConfig(128, 128, 64, 8, 4, 4),
    GemmConfig(128, 128, 64, 8, 8, 3),
    GemmConfig(128, 128, 32, 4, 8, 4),
    GemmConfig(128, 128, 64, 4, 8, 4),
    # Wider rectangular tiles. Some may be too register-heavy and will be skipped.
    GemmConfig(64, 256, 32, 8, 8, 4),
    GemmConfig(256, 64, 32, 8, 8, 4),
    GemmConfig(64, 256, 64, 8, 8, 4),
    GemmConfig(256, 64, 64, 8, 8, 4),
    GemmConfig(128, 256, 32, 8, 8, 4),
    GemmConfig(256, 128, 32, 8, 8, 4),
]


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def estimate_min_gb(size: int) -> float:
    return 3 * size * size * 2 / 1024**3


def config_text(config: GemmConfig) -> str:
    return (
        f"BM={config.block_m}, BN={config.block_n}, BK={config.block_k}, "
        f"GM={config.group_m}, W={config.num_warps}, S={config.num_stages}"
    )


def run_size(size: int, repeat: int, warmup: int) -> LargeSizeResult:
    print(f"\nsize={size}, repeat={repeat}, warmup={warmup}, min_memory={estimate_min_gb(size):.2f}GB")
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    torch_bench = benchmark_cuda("torch.matmul", lambda: a @ b, warmup=warmup, repeat=repeat)
    torch_tf = tflops(size, size, size, torch_bench.median_ms)
    print(f"torch.matmul median={torch_bench.median_ms:.4f} ms, {torch_tf:.2f} TFLOPS")
    clear_cuda_cache()

    records: list[ConfigResult] = []
    for config in CONFIGS:
        try:
            bench = benchmark_cuda(
                "triton_gemm",
                lambda cfg=config: matmul(a, b, **cfg.__dict__),
                warmup=warmup,
                repeat=repeat,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {config_text(config)}: {type(exc).__name__}: {exc}", flush=True)
            clear_cuda_cache()
            continue

        result = ConfigResult(
            size=size,
            config=config,
            median_ms=bench.median_ms,
            tflops=tflops(size, size, size, bench.median_ms),
        )
        records.append(result)
        print(f"  {config_text(config)} median={result.median_ms:.4f} ms TFLOPS={result.tflops:.2f}", flush=True)
        clear_cuda_cache()

    best = max(records, key=lambda item: item.tflops)
    ref = a @ b
    out = matmul(a, b, **best.config.__dict__)
    torch.cuda.synchronize()
    max_abs_error = float((ref - out).abs().max())
    allclose = bool(torch.allclose(ref, out, rtol=1e-2, atol=1e-1))
    del ref, out, a, b
    clear_cuda_cache()

    return LargeSizeResult(
        size=size,
        torch_ms=torch_bench.median_ms,
        torch_tflops=torch_tf,
        best=best,
        ratio=best.tflops / torch_tf,
        max_abs_error=max_abs_error,
        allclose=allclose,
        records=sorted(records, key=lambda item: item.tflops, reverse=True),
    )


def write_markdown(path: Path, results: list[LargeSizeResult]) -> None:
    lines = [
        "# GEMM Version 2：大尺寸专用参数搜索补充结果",
        "",
        "## 1. 实验目标",
        "",
        "针对 `16384/32768` 大尺寸 GEMM 扩大 tile 搜索空间，验证是否能改善 Version 2 每尺寸搜索中超大尺寸落后的问题。",
        "",
        "## 2. 最优结果",
        "",
        "| size | 最优配置 | torch ms | Triton ms | torch TFLOPS | Triton TFLOPS | Triton / torch | max abs err | allclose |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.size} | `{config_text(result.best.config)}` | "
            f"{result.torch_ms:.4f} | {result.best.median_ms:.4f} | "
            f"{result.torch_tflops:.2f} | {result.best.tflops:.2f} | "
            f"{result.ratio:.2%} | {result.max_abs_error:.6f} | {result.allclose} |"
        )

    lines.extend(["", "## 3. 完整搜索记录", ""])
    for result in results:
        lines.extend(
            [
                f"### size = {result.size}",
                "",
                "| rank | 配置 | median ms | TFLOPS |",
                "|---:|---|---:|---:|",
            ]
        )
        for rank, record in enumerate(result.records, start=1):
            lines.append(
                f"| {rank} | `{config_text(record.config)}` | {record.median_ms:.4f} | {record.tflops:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 结论",
            "",
            "- 本实验只针对超大尺寸做专用搜索，应合并回 `gemm_v2_tile_tuning.md` 作为 Version 2 的后续深化。",
            "- 如果新配置提升有限，说明当前 baseline 结构在超大矩阵上需要更深层优化，而不只是 tile 参数搜索。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/code/results/gemm_v2_large_size_tuning_raw.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    results = [run_size(size, args.repeat, args.warmup) for size in args.sizes]
    write_markdown(args.output_md, results)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
