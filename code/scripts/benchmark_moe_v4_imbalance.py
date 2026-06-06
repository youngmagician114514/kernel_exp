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
class V4ImbalanceResult:
    mode: str
    route_counts: list[int]
    grouped_ms: float
    grouped_v2_ms: float
    triton_grouped_ms: float
    triton_persistent_ms: float
    grouped_v2_vs_grouped: float
    triton_grouped_vs_grouped_v2: float
    triton_persistent_vs_grouped_v2: float
    triton_err: float
    persistent_err: float


def build_logits(num_tokens: int, experts: int, mode: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    logits = torch.randn((num_tokens, experts), device=device, dtype=dtype)
    if mode == "uniform":
        return logits
    if mode == "mild":
        logits[:, 0] += 1.5
        logits[:, 1] += 0.5
        return logits
    if mode == "strong":
        logits[:, 0] += 4.0
        logits[:, 1] += 2.0
        return logits
    raise ValueError("mode must be uniform, mild, or strong")


def route_counts_from_logits(logits: torch.Tensor, top_k: int) -> list[int]:
    weights = torch.softmax(logits.float(), dim=-1)
    topk_indices = torch.topk(weights, k=top_k, dim=-1).indices
    return torch.bincount(topk_indices.reshape(-1), minlength=logits.shape[1]).detach().cpu().tolist()


def run_case(tokens: int, hidden: int, ffn_hidden: int, experts: int, top_k: int, activation: str, mode: str) -> V4ImbalanceResult:
    device = torch.device("cuda")
    x = torch.randn((tokens, hidden), device=device, dtype=torch.float16)
    logits = build_logits(tokens, experts, mode, device, x.dtype)
    weights = build_random_experts(experts, hidden, ffn_hidden, device=device, dtype=x.dtype, activation=activation)

    grouped = benchmark_cuda("grouped", lambda: moe_forward_grouped(x, logits, weights, k=top_k), warmup=2, repeat=2)
    grouped_v2 = benchmark_cuda("grouped_v2", lambda: moe_forward_grouped_v2(x, logits, weights, k=top_k), warmup=2, repeat=2)
    triton_grouped = benchmark_cuda(
        "triton_grouped",
        lambda: moe_forward_triton_grouped(x, logits, weights, k=top_k),
        warmup=2,
        repeat=2,
    )
    triton_persistent = benchmark_cuda(
        "triton_persistent",
        lambda: moe_forward_triton_persistent(x, logits, weights, k=top_k),
        warmup=2,
        repeat=2,
    )

    return V4ImbalanceResult(
        mode=mode,
        route_counts=route_counts_from_logits(logits, top_k),
        grouped_ms=grouped.median_ms,
        grouped_v2_ms=grouped_v2.median_ms,
        triton_grouped_ms=triton_grouped.median_ms,
        triton_persistent_ms=triton_persistent.median_ms,
        grouped_v2_vs_grouped=grouped.median_ms / grouped_v2.median_ms,
        triton_grouped_vs_grouped_v2=grouped_v2.median_ms / triton_grouped.median_ms,
        triton_persistent_vs_grouped_v2=grouped_v2.median_ms / triton_persistent.median_ms,
        triton_err=moe_triton_grouped_error(x, logits, weights, k=top_k),
        persistent_err=moe_triton_persistent_error(x, logits, weights, k=top_k),
    )


def write_markdown(path: Path, tokens: int, hidden: int, ffn_hidden: int, results: list[V4ImbalanceResult]) -> None:
    lines = [
        "# Sparse MoE V4：16k Token Imbalance 实验",
        "",
        f"- tokens: `{tokens}`",
        f"- hidden: `{hidden}`",
        f"- ffn_hidden: `{ffn_hidden}`",
        "",
        "| mode | route_counts | grouped ms | grouped_v2 ms | triton_grouped ms | triton_persistent ms | grouped_v2 / grouped | grouped_v2 / triton_grouped | grouped_v2 / triton_persistent | triton err | persistent err |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.mode} | `{result.route_counts}` | {result.grouped_ms:.4f} | {result.grouped_v2_ms:.4f} | "
            f"{result.triton_grouped_ms:.4f} | {result.triton_persistent_ms:.4f} | {result.grouped_v2_vs_grouped:.2%} | "
            f"{result.triton_grouped_vs_grouped_v2:.2%} | {result.triton_persistent_vs_grouped_v2:.2%} | "
            f"{result.triton_err:.6f} | {result.persistent_err:.6f} |"
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
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/doc/moe_v4_imbalance.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    results = [
        run_case(args.tokens, args.hidden, args.ffn_hidden, args.experts, args.top_k, args.activation, mode)
        for mode in ["uniform", "mild", "strong"]
    ]
    for result in results:
        print(
            f"mode={result.mode} route_counts={result.route_counts} "
            f"grouped={result.grouped_ms:.4f} grouped_v2={result.grouped_v2_ms:.4f} "
            f"triton_grouped={result.triton_grouped_ms:.4f} triton_persistent={result.triton_persistent_ms:.4f} "
            f"grouped_v2/grouped={result.grouped_v2_vs_grouped:.2%} "
            f"grouped_v2/triton_grouped={result.triton_grouped_vs_grouped_v2:.2%} "
            f"grouped_v2/triton_persistent={result.triton_persistent_vs_grouped_v2:.2%} "
            f"triton_err={result.triton_err:.6f} persistent_err={result.persistent_err:.6f}"
        )
    write_markdown(args.output_md, args.tokens, args.hidden, args.ffn_hidden, results)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
