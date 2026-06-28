from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from kernel_exp_project.moe_routing import (
    build_random_experts,
    moe_forward_grouped_v2,
    moe_forward_triton_grouped,
    moe_triton_grouped_error,
)
from kernel_exp_project.utils import benchmark_cuda, require_cuda


@dataclass(frozen=True)
class TritonMoEConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


@dataclass
class TritonMoEResult:
    tokens: int
    hidden: int
    ffn_hidden: int
    config: TritonMoEConfig
    grouped_v2_ms: float
    triton_grouped_ms: float
    grouped_v2_vs_triton: float
    max_abs_error: float


CONFIGS = [
    TritonMoEConfig(16, 64, 32, 4, 4),
    TritonMoEConfig(16, 64, 32, 8, 4),
    TritonMoEConfig(32, 64, 32, 4, 4),
    TritonMoEConfig(32, 64, 32, 8, 4),
    TritonMoEConfig(32, 64, 64, 4, 4),
    TritonMoEConfig(32, 64, 64, 8, 4),
    TritonMoEConfig(64, 64, 32, 4, 4),
    TritonMoEConfig(64, 64, 32, 8, 4),
    TritonMoEConfig(64, 128, 32, 8, 4),
    TritonMoEConfig(64, 128, 64, 8, 4),
]


def auto_repeat(tokens: int, hidden: int, ffn_hidden: int) -> int:
    problem_size = tokens * hidden * ffn_hidden
    if problem_size <= 8192 * 2048 * 4096:
        return 4
    return 2


def config_text(config: TritonMoEConfig) -> str:
    return f"BM={config.block_m}, BN={config.block_n}, BK={config.block_k}, W={config.num_warps}, S={config.num_stages}"


def run_case(tokens: int, hidden: int, ffn_hidden: int, experts: int, top_k: int, activation: str) -> list[TritonMoEResult]:
    device = torch.device("cuda")
    x = torch.randn((tokens, hidden), device=device, dtype=torch.float16)
    logits = torch.randn((tokens, experts), device=device, dtype=torch.float16)
    weights = build_random_experts(experts, hidden, ffn_hidden, device=device, dtype=x.dtype, activation=activation)
    repeat = auto_repeat(tokens, hidden, ffn_hidden)

    grouped_v2 = benchmark_cuda(
        "moe_forward_grouped_v2",
        lambda: moe_forward_grouped_v2(x, logits, weights, k=top_k),
        warmup=2,
        repeat=repeat,
    )

    results: list[TritonMoEResult] = []
    for config in CONFIGS:
        try:
            triton_grouped = benchmark_cuda(
                "moe_forward_triton_grouped",
                lambda cfg=config: moe_forward_triton_grouped(
                    x,
                    logits,
                    weights,
                    k=top_k,
                    block_m=cfg.block_m,
                    block_n=cfg.block_n,
                    block_k=cfg.block_k,
                    num_warps=cfg.num_warps,
                    num_stages=cfg.num_stages,
                ),
                warmup=2,
                repeat=repeat,
            )
            max_abs_error = moe_triton_grouped_error(
                x,
                logits,
                weights,
                k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"skip config={config_text(config)} {type(exc).__name__}: {exc}")
            continue
        results.append(
            TritonMoEResult(
                tokens=tokens,
                hidden=hidden,
                ffn_hidden=ffn_hidden,
                config=config,
                grouped_v2_ms=grouped_v2.median_ms,
                triton_grouped_ms=triton_grouped.median_ms,
                grouped_v2_vs_triton=grouped_v2.median_ms / triton_grouped.median_ms,
                max_abs_error=max_abs_error,
            )
        )
    return results


def write_markdown(path: Path, records: list[TritonMoEResult]) -> None:
    lines = [
        "# Sparse MoE V3：Grouped Triton Kernel 参数搜索",
        "",
        "| tokens | hidden | ffn_hidden | 配置 | grouped_v2 ms | triton_grouped ms | grouped_v2 / triton_grouped | max abs err |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for result in records:
        lines.append(
            f"| {result.tokens} | {result.hidden} | {result.ffn_hidden} | `{config_text(result.config)}` | "
            f"{result.grouped_v2_ms:.4f} | {result.triton_grouped_ms:.4f} | "
            f"{result.grouped_v2_vs_triton:.2%} | {result.max_abs_error:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--ffn-hidden", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/doc/moe_v3_tuning.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    records = run_case(args.tokens, args.hidden, args.ffn_hidden, args.experts, args.top_k, args.activation)
    for result in records:
        print(
            f"{config_text(result.config)} grouped_v2={result.grouped_v2_ms:.4f} "
            f"triton_grouped={result.triton_grouped_ms:.4f} "
            f"grouped_v2/triton_grouped={result.grouped_v2_vs_triton:.2%} "
            f"err={result.max_abs_error:.6f}"
        )
    write_markdown(args.output_md, records)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
