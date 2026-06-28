from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from kernel_exp_project.flash_attention import attention_reference, flash_attention_v2
from kernel_exp_project.utils import benchmark_cuda


@dataclass(frozen=True)
class FlashConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


@dataclass
class FlashTuneResult:
    seq_len: int
    repeat: int
    config: FlashConfig
    median_ms: float
    max_abs_error: float
    mean_abs_error: float
    allclose: bool


CONFIGS = [
    FlashConfig(16, 64, 4, 4),
    FlashConfig(16, 64, 4, 3),
    FlashConfig(16, 64, 8, 4),
    FlashConfig(16, 64, 8, 3),
    FlashConfig(16, 128, 4, 4),
    FlashConfig(16, 128, 4, 3),
    FlashConfig(16, 128, 8, 4),
    FlashConfig(16, 128, 8, 3),
    FlashConfig(32, 64, 4, 4),
    FlashConfig(32, 64, 4, 3),
    FlashConfig(32, 64, 8, 4),
    FlashConfig(32, 64, 8, 3),
    FlashConfig(32, 128, 4, 4),
    FlashConfig(32, 128, 4, 3),
    FlashConfig(32, 128, 8, 4),
    FlashConfig(32, 128, 8, 3),
]


def auto_repeat(seq_len: int) -> int:
    if seq_len <= 512:
        return 12
    if seq_len <= 1024:
        return 8
    return 4


def clear_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def probe_cuda() -> None:
    try:
        x = torch.empty((1,), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        del x
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("CUDA probe failed in the local kernel_exp environment") from exc


def run_one(seq_len: int, head_dim: int, causal: bool, config: FlashConfig, repeat: int) -> FlashTuneResult:
    q = torch.randn((1, 4, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    ref = attention_reference(q, k, v, causal=causal)
    out = flash_attention_v2(
        q,
        k,
        v,
        causal=causal,
        block_m=config.block_m,
        block_n=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    torch.cuda.synchronize()
    abs_err = (ref - out).abs()

    bench = benchmark_cuda(
        "flash_attention_v2",
        lambda: flash_attention_v2(
            q,
            k,
            v,
            causal=causal,
            block_m=config.block_m,
            block_n=config.block_n,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        ),
        warmup=2,
        repeat=repeat,
    )

    result = FlashTuneResult(
        seq_len=seq_len,
        repeat=repeat,
        config=config,
        median_ms=bench.median_ms,
        max_abs_error=float(abs_err.max()),
        mean_abs_error=float(abs_err.mean()),
        allclose=bool(torch.allclose(ref, out, rtol=2e-2, atol=2e-2)),
    )
    del ref, out, abs_err, q, k, v
    clear_cuda_cache()
    return result


def config_text(config: FlashConfig) -> str:
    return (
        f"BM={config.block_m}, BN={config.block_n}, "
        f"W={config.num_warps}, S={config.num_stages}"
    )


def write_markdown(path: Path, results_by_size: dict[int, list[FlashTuneResult]], head_dim: int, causal: bool) -> None:
    best_rows = []
    for seq_len, records in sorted(results_by_size.items()):
        best_rows.append(min(records, key=lambda item: item.median_ms))

    default_config = CONFIGS[0]
    lines = [
        "# FlashAttention V2：Causal Block Skip 参数搜索",
        "",
        "## 1. 目标",
        "",
        "在 FlashAttention V2 的 causal block skip 基础上，搜索 `block_m`、`block_n`、`num_warps`、`num_stages`，",
        "找出在当前 RTX 3090 环境下更适合 causal forward 的配置。",
        "",
        "## 2. 实验设置",
        "",
        f"- batch: `1`",
        f"- heads: `4`",
        f"- head_dim: `{head_dim}`",
        f"- causal: `{causal}`",
        f"- 搜索配置数: `{len(CONFIGS)}`",
        f"- 默认 V2 配置: `{config_text(default_config)}`",
        "",
        "## 3. 每个序列长度的最优配置",
        "",
        "| seq_len | repeat | 最优配置 | median ms | 相对默认配置 | max abs err | mean abs err | allclose |",
        "|---:|---:|---|---:|---:|---:|---:|:---:|",
    ]

    for best in best_rows:
        default = next(item for item in results_by_size[best.seq_len] if item.config == default_config)
        ratio = default.median_ms / best.median_ms
        lines.append(
            f"| {best.seq_len} | {best.repeat} | `{config_text(best.config)}` | "
            f"{best.median_ms:.4f} | {ratio:.2%} | "
            f"{best.max_abs_error:.6f} | {best.mean_abs_error:.6f} | {best.allclose} |"
        )

    lines.extend(["", "## 4. 完整搜索记录", ""])
    for seq_len, records in sorted(results_by_size.items()):
        default = next(item for item in records if item.config == default_config)
        lines.extend(
            [
                f"### seq_len = {seq_len}",
                "",
                "| rank | 配置 | median ms | 相对默认配置 | max abs err | mean abs err | allclose |",
                "|---:|---|---:|---:|---:|---:|:---:|",
            ]
        )
        for rank, result in enumerate(sorted(records, key=lambda item: item.median_ms), start=1):
            ratio = default.median_ms / result.median_ms
            lines.append(
                f"| {rank} | `{config_text(result.config)}` | {result.median_ms:.4f} | "
                f"{ratio:.2%} | {result.max_abs_error:.6f} | {result.mean_abs_error:.6f} | {result.allclose} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 5. 结论",
            "",
            "- `causal block skip` 已经解决了无效块计算，下一步性能差异主要来自 tile 形状与 kernel 并行参数。",
            "- 如果最优配置随 `seq_len` 改变，说明 V2 更适合做 shape-aware 配置选择，而不是固定一个参数组合。",
            "- 如果多个 `seq_len` 的最佳配置接近，可以把它们收敛成统一默认值，减少实现复杂度。",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/doc/flash_attention_v2_tuning.md"))
    args = parser.parse_args()

    probe_cuda()
    torch.manual_seed(0)

    results_by_size: dict[int, list[FlashTuneResult]] = {}
    for seq_len in args.sizes:
        repeat = auto_repeat(seq_len)
        records: list[FlashTuneResult] = []
        print(f"\nseq_len={seq_len}, repeat={repeat}", flush=True)
        for config in CONFIGS:
            try:
                result = run_one(seq_len, args.head_dim, True, config, repeat)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"skip seq_len={seq_len} config={config_text(config)} "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                clear_cuda_cache()
                continue
            records.append(result)
            print(
                f"{seq_len:5d} {config.block_m:3d} {config.block_n:4d} "
                f"{config.num_warps:2d} {config.num_stages:2d} "
                f"{result.median_ms:8.4f} {result.max_abs_error:10.6f} "
                f"{result.mean_abs_error:10.6f} {result.allclose}",
                flush=True,
            )
        if not records:
            raise RuntimeError(f"no valid configs for seq_len={seq_len}")
        results_by_size[seq_len] = records

    write_markdown(args.output_md, results_by_size, args.head_dim, True)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
