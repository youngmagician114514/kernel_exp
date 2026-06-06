from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
class TuneResult:
    size: int
    config: GemmConfig
    median_ms: float
    mean_ms: float
    tflops: float
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
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
]


def run_one(size: int, config: GemmConfig, repeat: int) -> TuneResult:
    a = torch.randn((size, size), device="cuda", dtype=torch.float16)
    b = torch.randn((size, size), device="cuda", dtype=torch.float16)

    kwargs = asdict(config)

    ref = a @ b
    out = matmul(a, b, **kwargs)
    torch.cuda.synchronize()
    abs_err = (ref - out).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-3)

    bench = benchmark_cuda(
        "triton_gemm",
        lambda: matmul(a, b, **kwargs),
        warmup=5,
        repeat=repeat,
    )

    return TuneResult(
        size=size,
        config=config,
        median_ms=bench.median_ms,
        mean_ms=bench.mean_ms,
        tflops=tflops(size, size, size, bench.median_ms),
        max_abs_error=float(abs_err.max()),
        mean_abs_error=float(abs_err.mean()),
        max_rel_error=float(rel_err.max()),
        allclose=bool(torch.allclose(ref, out, rtol=1e-2, atol=1e-1)),
    )


def format_row(result: TuneResult) -> str:
    cfg = result.config
    return (
        f"{result.size:5d} "
        f"{cfg.block_m:4d} {cfg.block_n:4d} {cfg.block_k:4d} "
        f"{cfg.group_m:4d} {cfg.num_warps:5d} {cfg.num_stages:6d} "
        f"{result.median_ms:9.4f} {result.tflops:8.2f} "
        f"{result.max_abs_error:10.6f} {result.max_rel_error:10.4f} {str(result.allclose):>8s}"
    )


def print_results(results: list[TuneResult]) -> None:
    header = (
        " size   BM   BN   BK   GM warps stages median_ms   TFLOPS "
        "max_abs_err max_rel_err allclose"
    )
    print(header)
    print("-" * len(header))
    for result in sorted(results, key=lambda item: (item.size, -item.tflops)):
        print(format_row(result))


def write_markdown(path: Path, results: list[TuneResult], repeat: int) -> None:
    lines = [
        "# GEMM Version 2：Tile 参数搜索结果",
        "",
        "## 1. 实验目标",
        "",
        "对 Version 1 的 Triton GEMM baseline 进行小规模参数搜索，比较不同 tile 配置在 RTX 3090 上的延迟和 TFLOPS。",
        "",
        "## 2. 搜索参数",
        "",
        "- `block_m / block_n / block_k`：控制每个 Triton program 计算的 tile 大小。",
        "- `group_m`：控制 program 调度顺序，影响 L2 cache 复用。",
        "- `num_warps`：控制每个 Triton program 使用多少 warp。",
        "- `num_stages`：控制 Triton pipeline stage 数量。",
        "",
        "## 3. 实验设置",
        "",
        f"- repeat: `{repeat}`",
        "- dtype: `float16`",
        "- baseline: Triton GEMM，不包含 PyTorch/cuBLAS 搜索结果",
        "",
        "## 4. 实验结果",
        "",
        "| size | BM | BN | BK | GM | warps | stages | median ms | TFLOPS | max abs err | max rel err | allclose |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for result in sorted(results, key=lambda item: (item.size, -item.tflops)):
        cfg = result.config
        lines.append(
            f"| {result.size} | {cfg.block_m} | {cfg.block_n} | {cfg.block_k} | {cfg.group_m} | "
            f"{cfg.num_warps} | {cfg.num_stages} | {result.median_ms:.4f} | {result.tflops:.2f} | "
            f"{result.max_abs_error:.6f} | {result.max_rel_error:.4f} | {result.allclose} |"
        )

    best_by_size: dict[int, TuneResult] = {}
    for result in results:
        current = best_by_size.get(result.size)
        if current is None or result.tflops > current.tflops:
            best_by_size[result.size] = result

    lines.extend(["", "## 5. 阶段结论", ""])
    for size in sorted(best_by_size):
        result = best_by_size[size]
        cfg = result.config
        lines.append(
            f"- `M=N=K={size}` 的最佳配置为 `BM={cfg.block_m}, BN={cfg.block_n}, BK={cfg.block_k}, "
            f"GM={cfg.group_m}, warps={cfg.num_warps}, stages={cfg.num_stages}`，"
            f"达到 `{result.tflops:.2f}` TFLOPS。"
        )

    lines.extend(
        [
            "",
            "## 6. 下一步",
            "",
            "- 扩大搜索空间，加入 `4096` 尺寸。",
            "- 与 `torch.matmul` 放在同一张结果表中。",
            "- 对最佳配置继续分析 `block_k`、`num_warps`、`num_stages` 的影响。",
            "- 如果参数搜索收益有限，再进入 double buffering 或 split-K。",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    configs = CONFIGS[: args.limit_configs] if args.limit_configs > 0 else CONFIGS

    results: list[TuneResult] = []
    for size in args.sizes:
        for config in configs:
            try:
                result = run_one(size, config, args.repeat)
            except Exception as exc:  # noqa: BLE001
                print(f"skip size={size} config={config}: {type(exc).__name__}: {exc}")
                continue
            results.append(result)
            print(format_row(result), flush=True)

    print()
    print_results(results)

    if args.output_md is not None:
        write_markdown(args.output_md, results, args.repeat)
        print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()

