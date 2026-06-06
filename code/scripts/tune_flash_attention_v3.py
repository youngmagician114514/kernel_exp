from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from kernel_exp_project.flash_attention import (
    attention_reference_v3,
    flash_attention_v3,
    pack_padded_qkv,
)
from kernel_exp_project.utils import benchmark_cuda, require_cuda


@dataclass(frozen=True)
class FlashV3Config:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


@dataclass
class FlashV3Result:
    max_seq_len: int
    repeat: int
    total_tokens: int
    lengths: list[int]
    config: FlashV3Config
    median_ms: float
    max_abs_error: float
    mean_abs_error: float
    allclose: bool


CONFIGS = [
    FlashV3Config(16, 64, 4, 4),
    FlashV3Config(16, 64, 8, 4),
    FlashV3Config(16, 64, 4, 3),
    FlashV3Config(16, 64, 8, 3),
    FlashV3Config(32, 64, 4, 4),
    FlashV3Config(32, 64, 4, 3),
    FlashV3Config(32, 64, 8, 4),
    FlashV3Config(32, 64, 8, 3),
    FlashV3Config(32, 128, 4, 3),
    FlashV3Config(32, 128, 8, 3),
]


def build_lengths(batch: int, max_seq_len: int) -> torch.Tensor:
    base = [max_seq_len, max_seq_len - max_seq_len // 4, max_seq_len // 2, max_seq_len // 4]
    values = [max(1, base[i % len(base)]) for i in range(batch)]
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def auto_repeat(max_seq_len: int) -> int:
    if max_seq_len <= 256:
        return 10
    if max_seq_len <= 512:
        return 8
    return 4


def run_one(
    batch: int,
    heads: int,
    max_seq_len: int,
    head_dim: int,
    causal: bool,
    repeat: int,
    config: FlashV3Config,
) -> FlashV3Result:
    q = torch.randn((batch, heads, max_seq_len, head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    lengths = build_lengths(batch, max_seq_len)
    q_packed, k_packed, v_packed, cu_seqlens = pack_padded_qkv(q, k, v, lengths)

    ref = attention_reference_v3(q_packed, k_packed, v_packed, cu_seqlens, causal=causal)
    out = flash_attention_v3(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens,
        causal=causal,
        block_m=config.block_m,
        block_n=config.block_n,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    torch.cuda.synchronize()
    abs_err = (ref - out).abs()

    bench = benchmark_cuda(
        "flash_attention_v3",
        lambda: flash_attention_v3(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens,
            causal=causal,
            block_m=config.block_m,
            block_n=config.block_n,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        ),
        warmup=2,
        repeat=repeat,
    )

    return FlashV3Result(
        max_seq_len=max_seq_len,
        repeat=repeat,
        total_tokens=q_packed.shape[1],
        lengths=lengths.detach().cpu().tolist(),
        config=config,
        median_ms=bench.median_ms,
        max_abs_error=float(abs_err.max()),
        mean_abs_error=float(abs_err.mean()),
        allclose=bool(torch.allclose(ref, out, rtol=2e-2, atol=2e-2)),
    )


def config_text(config: FlashV3Config) -> str:
    return f"BM={config.block_m}, BN={config.block_n}, W={config.num_warps}, S={config.num_stages}"


def write_markdown(path: Path, results_by_size: dict[int, list[FlashV3Result]]) -> None:
    lines = [
        "# FlashAttention V3：Varlen 路径参数搜索",
        "",
        "## 结果",
        "",
        "| max_seq_len | total_tokens | 最优配置 | median ms | max abs err | mean abs err | allclose |",
        "|---:|---:|---|---:|---:|---:|:---:|",
    ]
    for max_seq_len in sorted(results_by_size):
        best = min(results_by_size[max_seq_len], key=lambda item: item.median_ms)
        lines.append(
            f"| {max_seq_len} | {best.total_tokens} | `{config_text(best.config)}` | "
            f"{best.median_ms:.4f} | {best.max_abs_error:.6f} | {best.mean_abs_error:.6f} | {best.allclose} |"
        )

    lines.extend(["", "## 完整记录", ""])
    for max_seq_len in sorted(results_by_size):
        lines.extend(
            [
                f"### max_seq_len = {max_seq_len}",
                "",
                "| rank | 配置 | median ms | max abs err | mean abs err | allclose |",
                "|---:|---|---:|---:|---:|:---:|",
            ]
        )
        for rank, result in enumerate(sorted(results_by_size[max_seq_len], key=lambda item: item.median_ms), start=1):
            lines.append(
                f"| {rank} | `{config_text(result.config)}` | {result.median_ms:.4f} | "
                f"{result.max_abs_error:.6f} | {result.mean_abs_error:.6f} | {result.allclose} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--output-md", type=Path, default=Path("kernel_exp/doc/flash_attention_v3_tuning.md"))
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)

    results_by_size: dict[int, list[FlashV3Result]] = {}
    for max_seq_len in args.sizes:
        repeat = auto_repeat(max_seq_len)
        records: list[FlashV3Result] = []
        print(f"\nmax_seq_len={max_seq_len}, repeat={repeat}", flush=True)
        for config in CONFIGS:
            try:
                result = run_one(
                    args.batch,
                    args.heads,
                    max_seq_len,
                    args.head_dim,
                    args.causal,
                    repeat,
                    config,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"skip max_seq_len={max_seq_len} config={config_text(config)} {type(exc).__name__}: {exc}")
                continue
            records.append(result)
            print(
                f"{max_seq_len:4d} {config.block_m:3d} {config.block_n:4d} {config.num_warps:2d} "
                f"{config.num_stages:2d} {result.median_ms:8.4f} {result.max_abs_error:10.6f} "
                f"{result.mean_abs_error:10.6f} {result.allclose}",
                flush=True,
            )
        if not records:
            raise RuntimeError(f"no valid configs for max_seq_len={max_seq_len}")
        results_by_size[max_seq_len] = records

    write_markdown(args.output_md, results_by_size)
    print(f"\nwritten {args.output_md}")


if __name__ == "__main__":
    main()
