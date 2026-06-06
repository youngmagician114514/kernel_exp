from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from kernel_exp_project.moe_routing import (
    build_random_experts,
    moe_forward_grouped,
    moe_forward_grouped_v2,
    moe_forward_triton_grouped,
    moe_forward_triton_persistent,
    moe_triton_grouped_error,
    moe_triton_persistent_error,
)
from kernel_exp_project.utils import benchmark_cuda, require_cuda


@dataclass
class PersistentResult:
    tokens: int
    hidden: int
    ffn_hidden: int
    grouped_ms: float
    grouped_v2_ms: float
    triton_grouped_ms: float
    triton_persistent_ms: float
    triton_grouped_vs_v2: float
    triton_persistent_vs_v2: float
    triton_err: float
    persistent_err: float


def auto_repeat(tokens: int, hidden: int, ffn_hidden: int) -> int:
    problem_size = tokens * hidden * ffn_hidden
    if problem_size <= 8192 * 2048 * 4096:
        return 4
    return 2


def run_case(tokens: int, hidden: int, ffn_hidden: int, experts: int, top_k: int, activation: str) -> PersistentResult:
    device = torch.device("cuda")
    x = torch.randn((tokens, hidden), device=device, dtype=torch.float16)
    logits = torch.randn((tokens, experts), device=device, dtype=torch.float16)
    weights = build_random_experts(experts, hidden, ffn_hidden, device=device, dtype=x.dtype, activation=activation)
    repeat = auto_repeat(tokens, hidden, ffn_hidden)

    grouped = benchmark_cuda("grouped", lambda: moe_forward_grouped(x, logits, weights, k=top_k), warmup=2, repeat=repeat)
    grouped_v2 = benchmark_cuda("grouped_v2", lambda: moe_forward_grouped_v2(x, logits, weights, k=top_k), warmup=2, repeat=repeat)
    triton_grouped = benchmark_cuda(
        "triton_grouped",
        lambda: moe_forward_triton_grouped(x, logits, weights, k=top_k),
        warmup=2,
        repeat=repeat,
    )
    triton_persistent = benchmark_cuda(
        "triton_persistent",
        lambda: moe_forward_triton_persistent(x, logits, weights, k=top_k),
        warmup=2,
        repeat=repeat,
    )

    return PersistentResult(
        tokens=tokens,
        hidden=hidden,
        ffn_hidden=ffn_hidden,
        grouped_ms=grouped.median_ms,
        grouped_v2_ms=grouped_v2.median_ms,
        triton_grouped_ms=triton_grouped.median_ms,
        triton_persistent_ms=triton_persistent.median_ms,
        triton_grouped_vs_v2=grouped_v2.median_ms / triton_grouped.median_ms,
        triton_persistent_vs_v2=grouped_v2.median_ms / triton_persistent.median_ms,
        triton_err=moe_triton_grouped_error(x, logits, weights, k=top_k),
        persistent_err=moe_triton_persistent_error(x, logits, weights, k=top_k),
    )


def write_markdown(path: Path, results: list[PersistentResult], experts: int, top_k: int, activation: str) -> None:
    lines = [
        "# Sparse MoE V4：Persistent Grouped GEMM 实验",
        "",
        f"- experts: `{experts}`",
        f"- top_k: `{top_k}`",
        f"- activation: `{activation}`",
        "",
        "| tokens | hidden | ffn_hidden | grouped ms | grouped_v2 ms | triton_grouped ms | triton_persistent ms | grouped_v2 / triton_grouped | grouped_v2 / triton_persistent | triton err | persistent err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.tokens} | {result.hidden} | {result.ffn_hidden} | "
            f"{result.grouped_ms:.4f} | {result.grouped_v2_ms:.4f} | {result.triton_grouped_ms:.4f} | {result.triton_persistent_ms:.4f} | "
            f"{result.triton_grouped_vs_v2:.2%} | {result.triton_persistent_vs_v2:.2%} | "
            f"{result.triton_err:.6f} | {result.persistent_err:.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=[4096, 1024, 2048, 8192, 2048, 4096, 16384, 4096, 8192],
        help="Triples of tokens hidden ffn_hidden",
    )
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/doc/moe_v4_persistent.md"))
    args = parser.parse_args()

    if len(args.cases) % 3 != 0:
        raise ValueError("--cases must be triples of tokens hidden ffn_hidden")

    require_cuda()
    torch.manual_seed(0)
    triples = [tuple(args.cases[i : i + 3]) for i in range(0, len(args.cases), 3)]
    results = []
    for tokens, hidden, ffn_hidden in triples:
        result = run_case(tokens, hidden, ffn_hidden, args.experts, args.top_k, args.activation)
        results.append(result)
        print(
            f"tokens={tokens} hidden={hidden} ffn_hidden={ffn_hidden} "
            f"grouped={result.grouped_ms:.4f} grouped_v2={result.grouped_v2_ms:.4f} "
            f"triton_grouped={result.triton_grouped_ms:.4f} triton_persistent={result.triton_persistent_ms:.4f} "
            f"grouped_v2/triton_grouped={result.triton_grouped_vs_v2:.2%} "
            f"grouped_v2/triton_persistent={result.triton_persistent_vs_v2:.2%} "
            f"triton_err={result.triton_err:.6f} persistent_err={result.persistent_err:.6f}"
        )
    write_markdown(args.output_md, results, args.experts, args.top_k, args.activation)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
