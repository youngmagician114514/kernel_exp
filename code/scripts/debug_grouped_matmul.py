from __future__ import annotations

import torch

from kernel_exp_project.moe_routing import (
    build_grouped_batches,
    build_grouped_matmul_problems,
    build_random_experts,
    permute_tokens,
    topk_gating,
)
from kernel_exp_project.triton_gemm import batched_matmul, grouped_matmul


def main() -> None:
    torch.manual_seed(0)
    num_tokens, hidden, ffn_hidden, experts, top_k = 128, 64, 128, 8, 2
    x = torch.randn((num_tokens, hidden), device="cuda", dtype=torch.float16)
    logits = torch.randn((num_tokens, experts), device="cuda", dtype=torch.float16)
    weights = build_random_experts(experts, hidden, ffn_hidden, device=x.device, dtype=x.dtype)

    topk_indices, topk_weights = topk_gating(logits, k=top_k)
    routed = permute_tokens(x, topk_indices, topk_weights)
    batches = build_grouped_batches(routed, experts)
    problems = build_grouped_matmul_problems(routed, experts, block_m=32, block_n=64, output_dim=ffn_hidden)
    active = batches.active_expert_ids.to(torch.int64)
    w1 = weights.w1[active]

    ref_padded = batched_matmul(batches.padded_tokens, w1)
    out_grouped = grouped_matmul(
        routed.tokens,
        w1,
        problems.expert_ids,
        problems.token_starts,
        problems.token_counts,
        problems.tile_offsets,
        block_m=32,
        block_n=64,
    )

    ref = torch.empty_like(out_grouped)
    for batch_idx, expert_id in enumerate(active.tolist()):
        start = int(routed.expert_offsets[expert_id].item())
        end = int(routed.expert_offsets[expert_id + 1].item())
        ref[start:end] = ref_padded[batch_idx, : end - start]

    err = (ref - out_grouped).abs()
    max_err = float(err.max())
    mean_err = float(err.mean())
    max_idx = torch.nonzero(err == err.max(), as_tuple=False)[0]
    row = int(max_idx[0].item())
    col = int(max_idx[1].item())
    expert_id = int(routed.expert_ids[row].item())

    print("active_experts", active.tolist())
    print("expert_counts", (routed.expert_offsets[1:] - routed.expert_offsets[:-1]).tolist())
    print("tile_offsets", problems.tile_offsets.tolist())
    print("max_err", max_err)
    print("mean_err", mean_err)
    print("argmax", [row, col], "expert", expert_id)
    print("expert_range", int(routed.expert_offsets[expert_id].item()), int(routed.expert_offsets[expert_id + 1].item()))
    print("ref_value", float(ref[row, col]))
    print("grouped_value", float(out_grouped[row, col]))


if __name__ == "__main__":
    main()
