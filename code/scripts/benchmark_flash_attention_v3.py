from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from kernel_exp_project.flash_attention import (
    attention_reference_v3,
    flash_attention_v3,
    flash_attention_v3_prototype,
    pack_padded_qkv,
)
from kernel_exp_project.utils import benchmark_cuda, require_cuda


@dataclass
class VarlenResult:
    lengths: list[int]
    total_tokens: int
    prototype_ms: float
    v3_ms: float
    v3_vs_prototype: float
    max_abs_error: float
    mean_abs_error: float
    allclose: bool


def build_lengths(batch: int, max_seq_len: int) -> torch.Tensor:
    base = [max_seq_len, max_seq_len - max_seq_len // 4, max_seq_len // 2, max_seq_len // 4]
    values = [max(1, base[i % len(base)]) for i in range(batch)]
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def run_one(batch: int, heads: int, max_seq_len: int, head_dim: int, causal: bool, repeat: int) -> VarlenResult:
    q = torch.randn((batch, heads, max_seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    lengths = build_lengths(batch, max_seq_len)
    q_packed, k_packed, v_packed, cu_seqlens = pack_padded_qkv(q, k, v, lengths)

    ref = attention_reference_v3(q_packed, k_packed, v_packed, cu_seqlens, causal=causal)
    out = flash_attention_v3(q_packed, k_packed, v_packed, cu_seqlens, causal=causal)
    torch.cuda.synchronize()
    abs_err = (ref - out).abs()
    max_abs_error = float(abs_err.max())
    mean_abs_error = float(abs_err.mean())
    allclose = bool(torch.allclose(ref, out, rtol=2e-2, atol=2e-2))
    del ref, out, abs_err
    torch.cuda.empty_cache()

    prototype = benchmark_cuda(
        "flash_attention_v3_prototype",
        lambda: flash_attention_v3_prototype(q_packed, k_packed, v_packed, cu_seqlens, causal=causal),
        warmup=2,
        repeat=repeat,
    )
    v3 = benchmark_cuda(
        "flash_attention_v3",
        lambda: flash_attention_v3(q_packed, k_packed, v_packed, cu_seqlens, causal=causal),
        warmup=2,
        repeat=repeat,
    )

    return VarlenResult(
        lengths=lengths.detach().cpu().tolist(),
        total_tokens=q_packed.shape[1],
        prototype_ms=prototype.median_ms,
        v3_ms=v3.median_ms,
        v3_vs_prototype=prototype.median_ms / v3.median_ms,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        allclose=allclose,
    )


def write_markdown(
    path: Path,
    results: list[tuple[int, VarlenResult]],
    batch: int,
    heads: int,
    head_dim: int,
    causal: bool,
) -> None:
    lines = [
        "# FlashAttention V3：Varlen Triton Kernel 实验",
        "",
        "## 实验设置",
        "",
        f"- batch: `{batch}`",
        f"- heads: `{heads}`",
        f"- head_dim: `{head_dim}`",
        f"- causal: `{causal}`",
        "",
        "## 结果",
        "",
        "| max_seq_len | lengths | total_tokens | prototype ms | v3 ms | prototype / v3 | max abs err | mean abs err | allclose |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for max_seq_len, result in results:
        lines.append(
            f"| {max_seq_len} | `{result.lengths}` | {result.total_tokens} | "
            f"{result.prototype_ms:.4f} | {result.v3_ms:.4f} | {result.v3_vs_prototype:.2%} | "
            f"{result.max_abs_error:.6f} | {result.mean_abs_error:.6f} | {result.allclose} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/code/results/flash_attention_v3_varlen.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)

    print(
        f"batch={args.batch}, heads={args.heads}, head_dim={args.head_dim}, "
        f"causal={args.causal}, repeat={args.repeat}"
    )
    results: list[tuple[int, VarlenResult]] = []
    for max_seq_len in args.sizes:
        result = run_one(args.batch, args.heads, max_seq_len, args.head_dim, args.causal, args.repeat)
        results.append((max_seq_len, result))
        print(
            f"max_seq_len={max_seq_len} lengths={result.lengths} total_tokens={result.total_tokens} "
            f"prototype={result.prototype_ms:.4f}ms v3={result.v3_ms:.4f}ms "
            f"prototype/v3={result.v3_vs_prototype:.2%} err={result.max_abs_error:.6f} "
            f"allclose={result.allclose}"
        )

    write_markdown(args.output_md, results, args.batch, args.heads, args.head_dim, args.causal)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
