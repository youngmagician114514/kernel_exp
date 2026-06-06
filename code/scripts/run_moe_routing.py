from __future__ import annotations

import argparse

import torch

from kernel_exp_project.moe_routing import (
    build_random_experts,
    grouped_expert_ffn,
    identity_moe_reference,
    moe_forward_grouped,
    moe_forward_grouped_v2,
    moe_forward_naive,
    moe_forward_triton_grouped,
    moe_forward_triton_persistent,
    moe_reference_error,
    moe_triton_error,
    moe_triton_grouped_error,
    moe_triton_persistent_error,
    moe_forward_triton,
    permute_tokens,
    topk_gating,
)
from kernel_exp_project.utils import benchmark_cuda, print_result, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ffn-hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "silu"])
    parser.add_argument("--skip-triton", action="store_true")
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    hidden = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.float16)
    logits = torch.randn((args.tokens, args.experts), device="cuda", dtype=torch.float16)
    weights = build_random_experts(
        args.experts,
        args.hidden,
        args.ffn_hidden,
        device=hidden.device,
        dtype=hidden.dtype,
        activation=args.activation,
    )

    err = identity_moe_reference(hidden, logits, k=args.top_k)
    err_ffn = moe_reference_error(hidden, logits, weights, k=args.top_k)
    err_triton = moe_triton_error(hidden, logits, weights, k=args.top_k)
    err_triton_grouped = moe_triton_grouped_error(hidden, logits, weights, k=args.top_k)
    err_triton_persistent = moe_triton_persistent_error(hidden, logits, weights, k=args.top_k)

    def route_once():
        topk_indices, topk_weights = topk_gating(logits, k=args.top_k)
        return permute_tokens(hidden, topk_indices, topk_weights)

    result = benchmark_cuda("moe_route", route_once, repeat=args.repeat)
    naive_result = benchmark_cuda(
        "moe_forward_naive",
        lambda: moe_forward_naive(hidden, logits, weights, k=args.top_k),
        repeat=args.repeat,
    )
    grouped_result = benchmark_cuda(
        "moe_forward_grouped",
        lambda: moe_forward_grouped(hidden, logits, weights, k=args.top_k),
        repeat=args.repeat,
    )
    grouped_v2_result = benchmark_cuda(
        "moe_forward_grouped_v2",
        lambda: moe_forward_grouped_v2(hidden, logits, weights, k=args.top_k),
        repeat=args.repeat,
    )
    triton_result = None
    triton_grouped_result = None
    triton_persistent_result = None
    if not args.skip_triton:
        triton_result = benchmark_cuda(
            "moe_forward_triton",
            lambda: moe_forward_triton(hidden, logits, weights, k=args.top_k),
            repeat=args.repeat,
        )
        triton_grouped_result = benchmark_cuda(
            "moe_forward_triton_grouped",
            lambda: moe_forward_triton_grouped(hidden, logits, weights, k=args.top_k),
            repeat=args.repeat,
        )
        triton_persistent_result = benchmark_cuda(
            "moe_forward_triton_persistent",
            lambda: moe_forward_triton_persistent(hidden, logits, weights, k=args.top_k),
            repeat=args.repeat,
        )
    routed = route_once()
    counts = torch.bincount(routed.expert_ids, minlength=args.experts).detach().cpu().tolist()

    print(
        f"tokens={args.tokens}, hidden={args.hidden}, ffn_hidden={args.ffn_hidden}, "
        f"experts={args.experts}, top_k={args.top_k}, activation={args.activation}"
    )
    print_result(result)
    print_result(naive_result)
    print_result(grouped_result)
    print_result(grouped_v2_result)
    if triton_result is not None:
        print_result(triton_result)
    if triton_grouped_result is not None:
        print_result(triton_grouped_result)
    if triton_persistent_result is not None:
        print_result(triton_persistent_result)
    print("expert route counts", counts)
    print(f"identity moe max error {err:.6f}")
    print(f"moe forward grouped-vs-naive max error {err_ffn:.6f}")
    if triton_result is not None:
        print(f"moe forward triton-vs-grouped_v2 max error {err_triton:.6f}")
    if triton_grouped_result is not None:
        print(f"moe forward triton_grouped-vs-grouped_v2 max error {err_triton_grouped:.6f}")
    if triton_persistent_result is not None:
        print(f"moe forward triton_persistent-vs-grouped_v2 max error {err_triton_persistent:.6f}")


if __name__ == "__main__":
    main()
