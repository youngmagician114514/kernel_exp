from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from kernel_exp_project.flash_attention import (
    attention_reference_v3,
    flash_attention_v3,
    pack_padded_qkv,
    _select_flash_attention_v3_config,
)
from kernel_exp_project.utils import benchmark_cuda, require_cuda


@dataclass
class CompareResult:
    max_seq_len: int
    lengths: list[int]
    total_tokens: int
    v3_config: tuple[int, int, int, int]
    explicit_ms: float | None
    sdpa_ms: float
    v3_ms: float
    v3_vs_explicit: float | None
    v3_vs_sdpa: float
    max_abs_error: float | None
    mean_abs_error: float | None
    allclose: bool | None


def build_lengths(batch: int, max_seq_len: int) -> torch.Tensor:
    base = [max_seq_len, max_seq_len - max_seq_len // 4, max_seq_len // 2, max_seq_len // 4]
    values = [max(1, base[i % len(base)]) for i in range(batch)]
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def make_sdpa_mask(lengths: torch.Tensor, max_seq_len: int, causal: bool) -> torch.Tensor:
    q_pos = torch.arange(max_seq_len, device=lengths.device)[None, :, None]
    k_pos = torch.arange(max_seq_len, device=lengths.device)[None, None, :]
    valid_q = q_pos < lengths[:, None, None]
    valid_k = k_pos < lengths[:, None, None]
    mask = valid_q & valid_k
    if causal:
        mask = mask & (k_pos <= q_pos)
    return mask[:, None, :, :]


def explicit_varlen_padded(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor, causal: bool) -> torch.Tensor:
    batch, heads, seq_len, head_dim = q.shape
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (1.0 / head_dim**0.5)
    valid_q = torch.arange(seq_len, device=q.device)[None, None, :, None] < lengths[:, None, None, None]
    valid_k = torch.arange(seq_len, device=q.device)[None, None, None, :] < lengths[:, None, None, None]
    scores = scores.masked_fill(~(valid_q & valid_k), -float("inf"))
    if causal:
        causal_mask = torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v.float()).to(q.dtype)
    out = out * valid_q.to(out.dtype)
    return out


def auto_repeat(max_seq_len: int) -> int:
    if max_seq_len <= 256:
        return 8
    if max_seq_len <= 512:
        return 6
    if max_seq_len <= 2048:
        return 4
    if max_seq_len <= 8192:
        return 2
    return 1
    


def run_one(
    batch: int,
    heads: int,
    max_seq_len: int,
    head_dim: int,
    causal: bool,
    include_explicit: bool,
    include_reference: bool,
) -> CompareResult:
    q = torch.randn((batch, heads, max_seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    lengths = build_lengths(batch, max_seq_len)
    q_packed, k_packed, v_packed, cu_seqlens = pack_padded_qkv(q, k, v, lengths)
    repeat = auto_repeat(max_seq_len)
    mask = make_sdpa_mask(lengths, max_seq_len, causal)
    v3_config = _select_flash_attention_v3_config(max_seq_len, head_dim, causal)

    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    allclose: bool | None = None
    if include_reference:
        ref = attention_reference_v3(q_packed, k_packed, v_packed, cu_seqlens, causal=causal)
        out = flash_attention_v3(q_packed, k_packed, v_packed, cu_seqlens, causal=causal)
        torch.cuda.synchronize()
        abs_err = (ref - out).abs()
        max_abs_error = float(abs_err.max())
        mean_abs_error = float(abs_err.mean())
        allclose = bool(torch.allclose(ref, out, rtol=2e-2, atol=2e-2))
        del ref, out, abs_err
        torch.cuda.empty_cache()

    explicit_ms: float | None = None
    v3_vs_explicit: float | None = None
    if include_explicit:
        explicit = benchmark_cuda(
            "explicit_varlen_padded",
            lambda: explicit_varlen_padded(q, k, v, lengths, causal),
            warmup=2,
            repeat=repeat,
        )
        explicit_ms = explicit.median_ms
    sdpa = benchmark_cuda(
        "masked_sdpa",
        lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False),
        warmup=2,
        repeat=repeat,
    )
    v3 = benchmark_cuda(
        "flash_attention_v3",
        lambda: flash_attention_v3(q_packed, k_packed, v_packed, cu_seqlens, causal=causal),
        warmup=2,
        repeat=repeat,
    )

    return CompareResult(
        max_seq_len=max_seq_len,
        lengths=lengths.detach().cpu().tolist(),
        total_tokens=q_packed.shape[1],
        v3_config=v3_config,
        explicit_ms=explicit_ms,
        sdpa_ms=sdpa.median_ms,
        v3_ms=v3.median_ms,
        v3_vs_explicit=None if explicit_ms is None else explicit_ms / v3.median_ms,
        v3_vs_sdpa=sdpa.median_ms / v3.median_ms,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
        allclose=allclose,
    )


def write_markdown(path: Path, results: list[CompareResult], batch: int, heads: int, head_dim: int, causal: bool) -> None:
    include_explicit = any(result.explicit_ms is not None for result in results)
    include_reference = any(result.max_abs_error is not None for result in results)
    lines = [
        "# FlashAttention V3：与 Padded Baseline / Masked SDPA 的公平实验",
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
    ]
    if include_explicit:
        if include_reference:
            lines.extend(
                [
                    "| max_seq_len | lengths | total_tokens | V3 config | explicit ms | masked SDPA ms | V3 ms | V3 / explicit | V3 / SDPA | max abs err | mean abs err | allclose |",
                    "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
                ]
            )
        else:
            lines.extend(
                [
                    "| max_seq_len | lengths | total_tokens | V3 config | explicit ms | masked SDPA ms | V3 ms | V3 / explicit | V3 / SDPA |",
                    "|---:|---|---:|---|---:|---:|---:|---:|---:|",
                ]
            )
    else:
        if include_reference:
            lines.extend(
                [
                    "| max_seq_len | lengths | total_tokens | V3 config | masked SDPA ms | V3 ms | V3 / SDPA | max abs err | mean abs err | allclose |",
                    "|---:|---|---:|---|---:|---:|---:|---:|---:|:---:|",
                ]
            )
        else:
            lines.extend(
                [
                    "| max_seq_len | lengths | total_tokens | V3 config | masked SDPA ms | V3 ms | V3 / SDPA |",
                    "|---:|---|---:|---|---:|---:|---:|",
                ]
            )
    for result in results:
        if include_explicit:
            if include_reference:
                lines.append(
                    f"| {result.max_seq_len} | `{result.lengths}` | {result.total_tokens} | "
                    f"`BM={result.v3_config[0]}, BN={result.v3_config[1]}, W={result.v3_config[2]}, S={result.v3_config[3]}` | "
                    f"{result.explicit_ms:.4f} | {result.sdpa_ms:.4f} | {result.v3_ms:.4f} | "
                    f"{result.v3_vs_explicit:.2%} | {result.v3_vs_sdpa:.2%} | "
                    f"{result.max_abs_error:.6f} | {result.mean_abs_error:.6f} | {result.allclose} |"
                )
            else:
                lines.append(
                    f"| {result.max_seq_len} | `{result.lengths}` | {result.total_tokens} | "
                    f"`BM={result.v3_config[0]}, BN={result.v3_config[1]}, W={result.v3_config[2]}, S={result.v3_config[3]}` | "
                    f"{result.explicit_ms:.4f} | {result.sdpa_ms:.4f} | {result.v3_ms:.4f} | "
                    f"{result.v3_vs_explicit:.2%} | {result.v3_vs_sdpa:.2%} |"
                )
        else:
            if include_reference:
                lines.append(
                    f"| {result.max_seq_len} | `{result.lengths}` | {result.total_tokens} | "
                    f"`BM={result.v3_config[0]}, BN={result.v3_config[1]}, W={result.v3_config[2]}, S={result.v3_config[3]}` | "
                    f"{result.sdpa_ms:.4f} | {result.v3_ms:.4f} | {result.v3_vs_sdpa:.2%} | "
                    f"{result.max_abs_error:.6f} | {result.mean_abs_error:.6f} | {result.allclose} |"
                )
            else:
                lines.append(
                    f"| {result.max_seq_len} | `{result.lengths}` | {result.total_tokens} | "
                    f"`BM={result.v3_config[0]}, BN={result.v3_config[1]}, W={result.v3_config[2]}, S={result.v3_config[3]}` | "
                    f"{result.sdpa_ms:.4f} | {result.v3_ms:.4f} | {result.v3_vs_sdpa:.2%} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--skip-explicit", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/doc/flash_attention_v3_fair_compare.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)

    results = [
        run_one(
            args.batch,
            args.heads,
            size,
            args.head_dim,
            args.causal,
            include_explicit=not args.skip_explicit,
            include_reference=not args.skip_reference,
        )
        for size in args.sizes
    ]
    for result in results:
        if result.explicit_ms is not None and result.v3_vs_explicit is not None:
            extra = (
                f" err={result.max_abs_error:.6f} allclose={result.allclose}"
                if result.max_abs_error is not None and result.allclose is not None
                else ""
            )
            print(
                f"max_seq_len={result.max_seq_len} total_tokens={result.total_tokens} "
                f"v3_config={result.v3_config} "
                f"explicit={result.explicit_ms:.4f}ms sdpa={result.sdpa_ms:.4f}ms v3={result.v3_ms:.4f}ms "
                f"v3/explicit={result.v3_vs_explicit:.2%} v3/sdpa={result.v3_vs_sdpa:.2%}"
                f"{extra}"
            )
        else:
            extra = (
                f" err={result.max_abs_error:.6f} allclose={result.allclose}"
                if result.max_abs_error is not None and result.allclose is not None
                else ""
            )
            print(
                f"max_seq_len={result.max_seq_len} total_tokens={result.total_tokens} "
                f"v3_config={result.v3_config} "
                f"sdpa={result.sdpa_ms:.4f}ms v3={result.v3_ms:.4f}ms "
                f"v3/sdpa={result.v3_vs_sdpa:.2%}"
                f"{extra}"
            )
    write_markdown(args.output_md, results, args.batch, args.heads, args.head_dim, args.causal)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
