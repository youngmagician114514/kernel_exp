from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from kernel_exp_project.flash_attention import attention_reference, explicit_attention, flash_attention, flash_attention_v2
from kernel_exp_project.utils import benchmark_cuda, require_cuda


@dataclass
class FlashResult:
    seq_len: int
    repeat: int
    standard_ms: float
    sdpa_ms: float | None
    triton_v1_ms: float
    triton_v2_ms: float
    standard_tflops: float
    sdpa_tflops: float | None
    triton_v1_tflops: float
    triton_v2_tflops: float
    triton_v1_vs_standard: float
    triton_v2_vs_standard: float
    triton_v1_vs_sdpa: float | None
    triton_v2_vs_sdpa: float | None
    v1_max_abs_error: float
    v2_max_abs_error: float
    v1_mean_abs_error: float
    v2_mean_abs_error: float
    v1_allclose: bool
    v2_allclose: bool


def attention_flops(batch: int, heads: int, seq_len: int, head_dim: int, causal: bool) -> float:
    dense_flops = 4.0 * batch * heads * seq_len * seq_len * head_dim
    return dense_flops * 0.5 if causal else dense_flops


def tflops_from_ms(flops: float, ms: float) -> float:
    return flops / (ms * 1e9)


def auto_repeat(seq_len: int) -> int:
    if seq_len <= 256:
        return 30
    if seq_len <= 512:
        return 20
    if seq_len <= 1024:
        return 10
    return 5


def estimate_standard_score_gb(batch: int, heads: int, seq_len: int) -> float:
    # The explicit baseline materializes score/prob matrices. It is a lower-bound estimate.
    return 2 * batch * heads * seq_len * seq_len * 2 / 1024**3


def run_one(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    repeat: int,
    include_sdpa: bool,
) -> FlashResult:
    q = torch.randn((batch, heads, seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    torch.cuda.synchronize()

    ref = attention_reference(q, k, v, causal=causal)
    out_v1 = flash_attention(q, k, v, causal=causal)
    out_v2 = flash_attention_v2(q, k, v, causal=causal)
    torch.cuda.synchronize()
    abs_err_v1 = (ref - out_v1).abs()
    abs_err_v2 = (ref - out_v2).abs()
    v1_max_abs_error = float(abs_err_v1.max())
    v2_max_abs_error = float(abs_err_v2.max())
    v1_mean_abs_error = float(abs_err_v1.mean())
    v2_mean_abs_error = float(abs_err_v2.mean())
    v1_allclose = bool(torch.allclose(ref, out_v1, rtol=2e-2, atol=2e-2))
    v2_allclose = bool(torch.allclose(ref, out_v2, rtol=2e-2, atol=2e-2))
    del ref, out_v1, out_v2, abs_err_v1, abs_err_v2
    torch.cuda.empty_cache()

    standard = benchmark_cuda(
        "explicit_attention",
        lambda: explicit_attention(q, k, v, causal=causal),
        warmup=3,
        repeat=repeat,
    )
    triton_v1 = benchmark_cuda(
        "triton_flash_attention_v1",
        lambda: flash_attention(q, k, v, causal=causal),
        warmup=3,
        repeat=repeat,
    )
    triton_v2 = benchmark_cuda(
        "triton_flash_attention_v2",
        lambda: flash_attention_v2(q, k, v, causal=causal),
        warmup=3,
        repeat=repeat,
    )

    sdpa_ms: float | None = None
    if include_sdpa:
        sdpa = benchmark_cuda(
            "torch_sdpa",
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
            warmup=3,
            repeat=repeat,
        )
        sdpa_ms = sdpa.median_ms

    flops = attention_flops(batch, heads, seq_len, head_dim, causal)
    standard_tflops = tflops_from_ms(flops, standard.median_ms)
    triton_v1_tflops = tflops_from_ms(flops, triton_v1.median_ms)
    triton_v2_tflops = tflops_from_ms(flops, triton_v2.median_ms)
    sdpa_tflops = None if sdpa_ms is None else tflops_from_ms(flops, sdpa_ms)

    del q, k, v
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return FlashResult(
        seq_len=seq_len,
        repeat=repeat,
        standard_ms=standard.median_ms,
        sdpa_ms=sdpa_ms,
        triton_v1_ms=triton_v1.median_ms,
        triton_v2_ms=triton_v2.median_ms,
        standard_tflops=standard_tflops,
        sdpa_tflops=sdpa_tflops,
        triton_v1_tflops=triton_v1_tflops,
        triton_v2_tflops=triton_v2_tflops,
        triton_v1_vs_standard=triton_v1_tflops / standard_tflops,
        triton_v2_vs_standard=triton_v2_tflops / standard_tflops,
        triton_v1_vs_sdpa=None if sdpa_tflops is None else triton_v1_tflops / sdpa_tflops,
        triton_v2_vs_sdpa=None if sdpa_tflops is None else triton_v2_tflops / sdpa_tflops,
        v1_max_abs_error=v1_max_abs_error,
        v2_max_abs_error=v2_max_abs_error,
        v1_mean_abs_error=v1_mean_abs_error,
        v2_mean_abs_error=v2_mean_abs_error,
        v1_allclose=v1_allclose,
        v2_allclose=v2_allclose,
    )


def write_markdown(path: Path, results: list[FlashResult], batch: int, heads: int, head_dim: int, causal: bool) -> None:
    def fmt_optional_ms(value: float | None) -> str:
        return "NA" if value is None else f"{value:.4f}"

    def fmt_optional_tflops(value: float | None) -> str:
        return "NA" if value is None else f"{value:.2f}"

    def fmt_optional_ratio(value: float | None) -> str:
        return "NA" if value is None else f"{value:.2%}"

    lines = [
        "# FlashAttention Version 2：V1/V2 对比实验记录",
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
        "| 序列长度 | repeat | 显式 PyTorch ms | PyTorch SDPA ms | Triton V1 ms | Triton V2 ms | Triton V1 / 显式 | Triton V2 / 显式 | Triton V1 / SDPA | Triton V2 / SDPA | V1 max abs err | V2 max abs err | V1 allclose | V2 allclose |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.seq_len} | {result.repeat} | "
            f"{result.standard_ms:.4f} | {fmt_optional_ms(result.sdpa_ms)} | "
            f"{result.triton_v1_ms:.4f} | {result.triton_v2_ms:.4f} | "
            f"{result.triton_v1_vs_standard:.2%} | {result.triton_v2_vs_standard:.2%} | "
            f"{fmt_optional_ratio(result.triton_v1_vs_sdpa)} | "
            f"{fmt_optional_ratio(result.triton_v2_vs_sdpa)} | "
            f"{result.v1_max_abs_error:.6f} | {result.v2_max_abs_error:.6f} | "
            f"{result.v1_allclose} | {result.v2_allclose} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--include-sdpa", action="store_true")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/code/results/flash_attention_v1_raw.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)

    print(
        f"batch={args.batch}, heads={args.heads}, head_dim={args.head_dim}, "
        f"causal={args.causal}, include_sdpa={args.include_sdpa}"
    )
    results: list[FlashResult] = []
    for seq_len in args.sizes:
        repeat = args.repeat if args.repeat > 0 else auto_repeat(seq_len)
        print(
            f"\nseq_len={seq_len}, repeat={repeat}, "
            f"explicit_score_prob_lower_bound={estimate_standard_score_gb(args.batch, args.heads, seq_len):.3f}GB",
            flush=True,
        )
        result = run_one(args.batch, args.heads, seq_len, args.head_dim, args.causal, repeat, args.include_sdpa)
        results.append(result)
        print(
            f"explicit={result.standard_ms:.4f}ms "
            f"v1={result.triton_v1_ms:.4f}ms v2={result.triton_v2_ms:.4f}ms "
            f"v1/explicit={result.triton_v1_vs_standard:.2%} "
            f"v2/explicit={result.triton_v2_vs_standard:.2%} "
            f"v1_err={result.v1_max_abs_error:.6f} v2_err={result.v2_max_abs_error:.6f}"
        )
        if result.sdpa_ms is not None and result.triton_v1_vs_sdpa is not None and result.triton_v2_vs_sdpa is not None:
            print(
                f"sdpa={result.sdpa_ms:.4f}ms "
                f"v1/sdpa={result.triton_v1_vs_sdpa:.2%} "
                f"v2/sdpa={result.triton_v2_vs_sdpa:.2%}"
            )

    write_markdown(args.output_md, results, args.batch, args.heads, args.head_dim, args.causal)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
