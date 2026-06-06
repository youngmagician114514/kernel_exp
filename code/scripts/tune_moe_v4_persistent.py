from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from kernel_exp_project.moe_routing import (
    build_random_experts,
    moe_forward_grouped_v2,
    moe_forward_triton_grouped,
    moe_forward_triton_persistent,
    moe_triton_persistent_error,
)
from kernel_exp_project.utils import benchmark_cuda, require_cuda


@dataclass(frozen=True)
class PersistentConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    num_programs: int


@dataclass
class PersistentResult:
    config: PersistentConfig
    grouped_v2_ms: float
    triton_grouped_ms: float
    persistent_ms: float
    persistent_vs_grouped_v2: float
    persistent_vs_triton_grouped: float
    persistent_err: float


CONFIGS = [
    PersistentConfig(16, 64, 32, 4, 4, 16),
    PersistentConfig(16, 64, 32, 4, 4, 32),
    PersistentConfig(16, 64, 32, 4, 4, 64),
    PersistentConfig(32, 64, 32, 4, 4, 16),
    PersistentConfig(32, 64, 32, 4, 4, 32),
    PersistentConfig(32, 64, 32, 4, 4, 64),
    PersistentConfig(32, 64, 64, 4, 4, 32),
    PersistentConfig(32, 64, 64, 4, 4, 64),
    PersistentConfig(64, 64, 32, 4, 4, 32),
    PersistentConfig(64, 64, 32, 4, 4, 64),
    PersistentConfig(64, 128, 64, 8, 4, 32),
    PersistentConfig(64, 128, 64, 8, 4, 64),
]


def auto_repeat(tokens: int, hidden: int, ffn_hidden: int) -> int:
    problem_size = tokens * hidden * ffn_hidden
    if problem_size <= 16384 * 4096 * 8192:
        return 2
    return 1


def config_text(config: PersistentConfig) -> str:
    return (
        f"BM={config.block_m}, BN={config.block_n}, BK={config.block_k}, "
        f"W={config.num_warps}, S={config.num_stages}, P={config.num_programs}"
    )


def run_case(tokens: int, hidden: int, ffn_hidden: int, experts: int, top_k: int, activation: str) -> list[PersistentResult]:
    device = torch.device("cuda")
    x = torch.randn((tokens, hidden), device=device, dtype=torch.float16)
    logits = torch.randn((tokens, experts), device=device, dtype=torch.float16)
    weights = build_random_experts(experts, hidden, ffn_hidden, device=device, dtype=x.dtype, activation=activation)
    repeat = auto_repeat(tokens, hidden, ffn_hidden)

    grouped_v2 = benchmark_cuda(
        "grouped_v2",
        lambda: moe_forward_grouped_v2(x, logits, weights, k=top_k),
        warmup=2,
        repeat=repeat,
    )
    triton_grouped = benchmark_cuda(
        "triton_grouped",
        lambda: moe_forward_triton_grouped(x, logits, weights, k=top_k),
        warmup=2,
        repeat=repeat,
    )

    results: list[PersistentResult] = []
    for config in CONFIGS:
        try:
            persistent = benchmark_cuda(
                "triton_persistent",
                lambda cfg=config: moe_forward_triton_persistent(
                    x,
                    logits,
                    weights,
                    k=top_k,
                    block_m=cfg.block_m,
                    block_n=cfg.block_n,
                    block_k=cfg.block_k,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.num_stages,
                    num_programs=cfg.num_programs,
                ),
                warmup=2,
                repeat=repeat,
            )
            err = moe_triton_persistent_error(x, logits, weights, k=top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"skip config={config_text(config)} {type(exc).__name__}: {exc}")
            continue
        results.append(
            PersistentResult(
                config=config,
                grouped_v2_ms=grouped_v2.median_ms,
                triton_grouped_ms=triton_grouped.median_ms,
                persistent_ms=persistent.median_ms,
                persistent_vs_grouped_v2=grouped_v2.median_ms / persistent.median_ms,
                persistent_vs_triton_grouped=triton_grouped.median_ms / persistent.median_ms,
                persistent_err=err,
            )
        )
    return results


def write_markdown(path: Path, tokens: int, hidden: int, ffn_hidden: int, records: list[PersistentResult]) -> None:
    lines = [
        "# Sparse MoE V4：Persistent Kernel 参数搜索",
        "",
        f"- tokens: `{tokens}`",
        f"- hidden: `{hidden}`",
        f"- ffn_hidden: `{ffn_hidden}`",
        "",
        "| 配置 | grouped_v2 ms | triton_grouped ms | persistent ms | grouped_v2 / persistent | triton_grouped / persistent | err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in records:
        lines.append(
            f"| `{config_text(result.config)}` | {result.grouped_v2_ms:.4f} | {result.triton_grouped_ms:.4f} | "
            f"{result.persistent_ms:.4f} | {result.persistent_vs_grouped_v2:.2%} | {result.persistent_vs_triton_grouped:.2%} | "
            f"{result.persistent_err:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ffn-hidden", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/doc/moe_v4_tuning.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    records = run_case(args.tokens, args.hidden, args.ffn_hidden, args.experts, args.top_k, args.activation)
    for result in records:
        print(
            f"{config_text(result.config)} grouped_v2={result.grouped_v2_ms:.4f} "
            f"triton_grouped={result.triton_grouped_ms:.4f} persistent={result.persistent_ms:.4f} "
            f"grouped_v2/persistent={result.persistent_vs_grouped_v2:.2%} "
            f"triton_grouped/persistent={result.persistent_vs_triton_grouped:.2%} "
            f"err={result.persistent_err:.6f}"
        )
    write_markdown(args.output_md, args.tokens, args.hidden, args.ffn_hidden, records)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
