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
    mean_ms: float
    tflops: float


@dataclass
class BestResult:
    size: int
    repeat: int
    min_memory_gb: float
    torch_median_ms: float
    torch_tflops: float
    best: ConfigResult
    ratio: float
    max_abs_error: float
    mean_abs_error: float
    allclose: bool


CONFIGS = [
    GemmConfig(32, 64, 32, 8, 4, 4),
    GemmConfig(32, 128, 32, 8, 4, 4),
    GemmConfig(64, 64, 32, 8, 4, 4),
    GemmConfig(64, 128, 32, 8, 4, 4),
    GemmConfig(128, 64, 32, 8, 4, 4),
    GemmConfig(64, 64, 64, 8, 4, 4),
    GemmConfig(64, 128, 64, 8, 4, 4),
    GemmConfig(128, 64, 64, 8, 4, 4),
    GemmConfig(64, 128, 32, 4, 4, 4),
    GemmConfig(64, 128, 32, 8, 8, 4),
    GemmConfig(64, 128, 32, 8, 4, 3),
    GemmConfig(128, 128, 32, 8, 8, 4),
    GemmConfig(128, 128, 64, 8, 8, 4),
]


def auto_repeat(size: int) -> int:
    if size <= 1024:
        return 20
    if size <= 2048:
        return 12
    if size <= 4096:
        return 6
    if size <= 8192:
        return 3
    return 1


def auto_warmup(size: int) -> int:
    if size <= 2048:
        return 4
    if size <= 8192:
        return 2
    return 1


def estimate_min_gb(size: int) -> float:
    return 3 * size * size * 2 / 1024**3


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_config(a: torch.Tensor, b: torch.Tensor, config: GemmConfig, repeat: int, warmup: int) -> ConfigResult:
    kwargs = config.__dict__
    bench = benchmark_cuda(
        "triton_gemm",
        lambda: matmul(a, b, **kwargs),
        warmup=warmup,
        repeat=repeat,
    )
    size = a.shape[0]
    return ConfigResult(
        size=size,
        config=config,
        median_ms=bench.median_ms,
        mean_ms=bench.mean_ms,
        tflops=tflops(size, size, size, bench.median_ms),
    )


def check_best(a: torch.Tensor, b: torch.Tensor, config: GemmConfig) -> tuple[float, float, bool]:
    ref = a @ b
    out = matmul(a, b, **config.__dict__)
    torch.cuda.synchronize()
    abs_err = (ref - out).abs()
    max_abs_error = float(abs_err.max())
    mean_abs_error = float(abs_err.mean())
    allclose = bool(torch.allclose(ref, out, rtol=1e-2, atol=1e-1))
    del ref, out, abs_err
    clear_cuda_cache()
    return max_abs_error, mean_abs_error, allclose


def run_size(size: int) -> tuple[BestResult, list[ConfigResult]]:
    repeat = auto_repeat(size)
    warmup = auto_warmup(size)
    print(f"\nsize={size}, repeat={repeat}, warmup={warmup}, min_memory={estimate_min_gb(size):.2f}GB")

    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    torch_bench = benchmark_cuda("torch.matmul", lambda: a @ b, warmup=warmup, repeat=repeat)
    torch_tf = tflops(size, size, size, torch_bench.median_ms)
    print(f"torch.matmul median={torch_bench.median_ms:.4f} ms, {torch_tf:.2f} TFLOPS")

    config_results: list[ConfigResult] = []
    for config in CONFIGS:
        try:
            result = run_config(a, b, config, repeat, warmup)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {config}: {type(exc).__name__}: {exc}", flush=True)
            clear_cuda_cache()
            continue
        config_results.append(result)
        cfg = result.config
        print(
            f"  BM={cfg.block_m:3d} BN={cfg.block_n:3d} BK={cfg.block_k:3d} "
            f"GM={cfg.group_m:2d} W={cfg.num_warps:1d} S={cfg.num_stages:1d} "
            f"median={result.median_ms:9.4f} ms TFLOPS={result.tflops:8.2f}",
            flush=True,
        )

    best = max(config_results, key=lambda item: item.tflops)
    max_abs, mean_abs, allclose = check_best(a, b, best.config)

    best_result = BestResult(
        size=size,
        repeat=repeat,
        min_memory_gb=estimate_min_gb(size),
        torch_median_ms=torch_bench.median_ms,
        torch_tflops=torch_tf,
        best=best,
        ratio=best.tflops / torch_tf,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        allclose=allclose,
    )

    del a, b
    clear_cuda_cache()
    return best_result, sorted(config_results, key=lambda item: item.tflops, reverse=True)


def config_text(config: GemmConfig) -> str:
    return (
        f"BM={config.block_m}, BN={config.block_n}, BK={config.block_k}, "
        f"GM={config.group_m}, W={config.num_warps}, S={config.num_stages}"
    )


def write_markdown(path: Path, best_results: list[BestResult], all_results: dict[int, list[ConfigResult]]) -> None:
    lines = [
        "# GEMM Version 2：每个尺寸独立参数搜索",
        "",
        "## 1. 实验目标",
        "",
        "针对每个矩阵尺寸单独搜索 Triton GEMM 参数，记录每个尺寸的最优配置，并与 PyTorch/cuBLAS 同场对比。",
        "",
        "## 2. 候选参数",
        "",
        "本实验搜索以下参数：",
        "",
        "- `BM / BN / BK`：tile 大小",
        "- `GM`：`group_m`，影响 L2 cache 复用",
        "- `W`：`num_warps`，每个 Triton program 使用的 warp 数",
        "- `S`：`num_stages`，pipeline stage 数",
        "",
        "候选配置数量：`13` 组。",
        "",
        "## 3. 每个尺寸的最优结果",
        "",
        "| size | repeat | 最低显存 GB | 最优配置 | torch ms | Triton ms | torch TFLOPS | Triton TFLOPS | Triton / torch | max abs err | mean abs err | allclose |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for result in best_results:
        best = result.best
        lines.append(
            f"| {result.size} | {result.repeat} | {result.min_memory_gb:.2f} | `{config_text(best.config)}` | "
            f"{result.torch_median_ms:.4f} | {best.median_ms:.4f} | "
            f"{result.torch_tflops:.2f} | {best.tflops:.2f} | {result.ratio:.2%} | "
            f"{result.max_abs_error:.6f} | {result.mean_abs_error:.6f} | {result.allclose} |"
        )

    lines.extend(["", "## 4. 各尺寸完整搜索记录", ""])
    for size in sorted(all_results):
        lines.extend(
            [
                f"### size = {size}",
                "",
                "| rank | 配置 | median ms | TFLOPS |",
                "|---:|---|---:|---:|",
            ]
        )
        for rank, result in enumerate(all_results[size], start=1):
            lines.append(
                f"| {rank} | `{config_text(result.config)}` | {result.median_ms:.4f} | {result.tflops:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 5. 初步结论",
            "",
            "- 每个尺寸的最优配置并不相同，说明固定使用某一个 tile 配置并不合理。",
            "- 如果某些尺寸上 Triton 接近或超过 PyTorch，应该优先确认该结果是否稳定，再作为项目亮点记录。",
            "- 大尺寸是否有优势，需要看每个尺寸独立搜索后的 `Triton / torch` 比例，而不是沿用某个小尺寸最优配置。",
            "",
        ]
    )

    if best_results:
        best_ratio = max(best_results, key=lambda item: item.ratio)
        lines.extend(
            [
                "## 6. 阶段总结",
                "",
                f"- 当前最高相对比例出现在 `size={best_ratio.size}`，Triton 达到 PyTorch 的 `{best_ratio.ratio:.2%}`。",
                "- 后续可以围绕高比例尺寸做复测和 Nsight 分析，确认性能来源。",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/code/results/gemm_v2_per_size_raw.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)

    best_results: list[BestResult] = []
    all_results: dict[int, list[ConfigResult]] = {}
    for size in args.sizes:
        best, records = run_size(size)
        best_results.append(best)
        all_results[size] = records
        print(
            f"best size={size}: {config_text(best.best.config)}, "
            f"{best.best.tflops:.2f} TFLOPS, ratio={best.ratio:.2%}, allclose={best.allclose}",
            flush=True,
        )

    write_markdown(args.output_md, best_results, all_results)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
