from __future__ import annotations

from dataclasses import dataclass

import torch

from kernel_exp_project.triton_gemm import batched_matmul, grouped_matmul, matmul, persistent_grouped_matmul


@dataclass
class RoutedTokens:
    tokens: torch.Tensor
    expert_ids: torch.Tensor
    source_token_ids: torch.Tensor
    route_weights: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    expert_offsets: torch.Tensor


@dataclass
class ExpertWeights:
    w1: torch.Tensor
    w2: torch.Tensor
    activation: str = "gelu"


@dataclass
class GroupedExpertTensors:
    routed: RoutedTokens
    intermediate: torch.Tensor
    output: torch.Tensor


@dataclass
class GroupedBatches:
    active_expert_ids: torch.Tensor
    expert_token_counts: torch.Tensor
    max_tokens_per_expert: int
    padded_tokens: torch.Tensor


@dataclass
class GroupedMatmulProblems:
    expert_ids: torch.Tensor
    weight_expert_ids: torch.Tensor
    token_starts: torch.Tensor
    token_counts: torch.Tensor
    tile_offsets: torch.Tensor


def topk_gating(logits: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("logits must be [num_tokens, num_experts]")
    weights = torch.softmax(logits.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(weights, k=k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_indices, topk_weights.to(logits.dtype)


def permute_tokens(hidden: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> RoutedTokens:
    num_tokens, hidden_dim = hidden.shape
    flat_experts = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    source_token_ids = torch.arange(num_tokens, device=hidden.device).repeat_interleave(topk_indices.shape[1])
    order = torch.argsort(flat_experts, stable=True)
    num_experts = int(topk_indices.max().item()) + 1 if topk_indices.numel() > 0 else 0
    expert_counts = torch.bincount(flat_experts, minlength=num_experts)
    expert_offsets = torch.empty((num_experts + 1,), device=hidden.device, dtype=torch.int32)
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0).to(torch.int32)

    return RoutedTokens(
        tokens=hidden[source_token_ids[order]],
        expert_ids=flat_experts[order],
        source_token_ids=source_token_ids[order],
        route_weights=flat_weights[order],
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        expert_offsets=expert_offsets,
    )


def unpermute_tokens(expert_outputs: torch.Tensor, routed: RoutedTokens, num_tokens: int) -> torch.Tensor:
    weighted = expert_outputs * routed.route_weights[:, None]
    out = torch.zeros((num_tokens, expert_outputs.shape[1]), device=expert_outputs.device, dtype=expert_outputs.dtype)
    out.index_add_(0, routed.source_token_ids, weighted)
    return out


def identity_moe_reference(hidden: torch.Tensor, logits: torch.Tensor, k: int = 2) -> float:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    out = unpermute_tokens(routed.tokens, routed, hidden.shape[0])
    return float((out - hidden).abs().max())


def build_random_experts(
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    activation: str = "gelu",
) -> ExpertWeights:
    if activation not in {"gelu", "silu"}:
        raise ValueError("activation must be 'gelu' or 'silu'")
    w1 = torch.randn((num_experts, hidden_dim, ffn_dim), device=device, dtype=dtype) / hidden_dim**0.5
    w2 = torch.randn((num_experts, ffn_dim, hidden_dim), device=device, dtype=dtype) / ffn_dim**0.5
    return ExpertWeights(w1=w1, w2=w2, activation=activation)


def apply_expert_ffn(tokens: torch.Tensor, weights: ExpertWeights, expert_id: int) -> torch.Tensor:
    x = tokens
    h = x @ weights.w1[expert_id]
    if weights.activation == "gelu":
        h = torch.nn.functional.gelu(h.float()).to(tokens.dtype)
    else:
        h = torch.nn.functional.silu(h.float()).to(tokens.dtype)
    return h @ weights.w2[expert_id]


def moe_forward_naive(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> torch.Tensor:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    outputs = torch.empty_like(routed.tokens)
    num_experts = weights.w1.shape[0]
    for expert_id in range(num_experts):
        start = int(routed.expert_offsets[expert_id].item())
        end = int(routed.expert_offsets[expert_id + 1].item())
        if start == end:
            continue
        outputs[start:end] = apply_expert_ffn(routed.tokens[start:end], weights, expert_id)
    return unpermute_tokens(outputs, routed, hidden.shape[0])


def moe_forward_grouped(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> torch.Tensor:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    grouped = grouped_expert_ffn(hidden, routed, weights, use_triton=False)
    return unpermute_tokens(grouped.output, routed, hidden.shape[0])


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "gelu":
        return torch.nn.functional.gelu(x.float()).to(x.dtype)
    if activation == "silu":
        return torch.nn.functional.silu(x.float()).to(x.dtype)
    raise ValueError("activation must be 'gelu' or 'silu'")


def build_grouped_batches(routed: RoutedTokens, num_experts: int) -> GroupedBatches:
    hidden_dim = routed.tokens.shape[1]
    expert_token_counts = routed.expert_offsets[1:] - routed.expert_offsets[:-1]
    active_expert_ids = torch.nonzero(expert_token_counts > 0, as_tuple=False).flatten().to(torch.int64)
    if active_expert_ids.numel() == 0:
        return GroupedBatches(
            active_expert_ids=active_expert_ids,
            expert_token_counts=expert_token_counts,
            max_tokens_per_expert=0,
            padded_tokens=torch.empty((0, 0, hidden_dim), device=routed.tokens.device, dtype=routed.tokens.dtype),
        )

    max_tokens_per_expert = int(expert_token_counts[active_expert_ids].max().item())
    padded_tokens = torch.zeros(
        (active_expert_ids.numel(), max_tokens_per_expert, hidden_dim),
        device=routed.tokens.device,
        dtype=routed.tokens.dtype,
    )
    for batch_idx, expert_id in enumerate(active_expert_ids.tolist()):
        start = int(routed.expert_offsets[expert_id].item())
        end = int(routed.expert_offsets[expert_id + 1].item())
        count = end - start
        padded_tokens[batch_idx, :count] = routed.tokens[start:end]
    return GroupedBatches(
        active_expert_ids=active_expert_ids,
        expert_token_counts=expert_token_counts,
        max_tokens_per_expert=max_tokens_per_expert,
        padded_tokens=padded_tokens,
    )


def build_grouped_matmul_problems(
    routed: RoutedTokens,
    num_experts: int,
    block_m: int,
    block_n: int,
    output_dim: int,
) -> GroupedMatmulProblems:
    expert_token_counts = routed.expert_offsets[1:] - routed.expert_offsets[:-1]
    active_expert_ids = torch.nonzero(expert_token_counts > 0, as_tuple=False).flatten().to(torch.int32)
    if active_expert_ids.numel() == 0:
        empty = torch.empty((0,), device=routed.tokens.device, dtype=torch.int32)
        return GroupedMatmulProblems(
            expert_ids=empty,
            weight_expert_ids=empty,
            token_starts=empty,
            token_counts=empty,
            tile_offsets=torch.zeros((1,), device=routed.tokens.device, dtype=torch.int32),
        )

    weight_expert_ids = active_expert_ids.clone()
    token_starts = routed.expert_offsets[weight_expert_ids.to(torch.int64)].to(torch.int32)
    token_counts = expert_token_counts[weight_expert_ids.to(torch.int64)].to(torch.int32)
    order = torch.argsort(token_counts, descending=True, stable=True)
    weight_expert_ids = weight_expert_ids[order]
    token_starts = token_starts[order]
    token_counts = token_counts[order]
    tiles_m = torch.div(token_counts + block_m - 1, block_m, rounding_mode="floor")
    tiles_n = (output_dim + block_n - 1) // block_n
    tiles_per_problem = tiles_m * tiles_n
    tile_offsets = torch.empty((active_expert_ids.numel() + 1,), device=routed.tokens.device, dtype=torch.int32)
    tile_offsets[0] = 0
    tile_offsets[1:] = torch.cumsum(tiles_per_problem, dim=0)
    local_expert_ids = torch.arange(weight_expert_ids.numel(), device=routed.tokens.device, dtype=torch.int32)
    return GroupedMatmulProblems(
        expert_ids=local_expert_ids,
        weight_expert_ids=weight_expert_ids,
        token_starts=token_starts,
        token_counts=token_counts,
        tile_offsets=tile_offsets,
    )


def _unpack_grouped_output(grouped_output: torch.Tensor, grouped_batches: GroupedBatches, routed: RoutedTokens) -> torch.Tensor:
    hidden_dim = grouped_output.shape[-1]
    output = torch.empty((routed.tokens.shape[0], hidden_dim), device=grouped_output.device, dtype=grouped_output.dtype)
    for batch_idx, expert_id in enumerate(grouped_batches.active_expert_ids.tolist()):
        start = int(routed.expert_offsets[expert_id].item())
        end = int(routed.expert_offsets[expert_id + 1].item())
        count = end - start
        output[start:end] = grouped_output[batch_idx, :count]
    return output


def grouped_expert_gemm_1(routed: RoutedTokens, weights: ExpertWeights, *, use_triton: bool = False) -> torch.Tensor:
    num_routed_tokens = routed.tokens.shape[0]
    ffn_hidden = weights.w1.shape[2]
    intermediate = torch.empty((num_routed_tokens, ffn_hidden), device=routed.tokens.device, dtype=routed.tokens.dtype)
    num_experts = weights.w1.shape[0]
    for expert_id in range(num_experts):
        start = int(routed.expert_offsets[expert_id].item())
        end = int(routed.expert_offsets[expert_id + 1].item())
        if start == end:
            continue
        tokens = routed.tokens[start:end]
        w1 = weights.w1[expert_id]
        if use_triton:
            intermediate[start:end] = matmul(tokens, w1)
        else:
            intermediate[start:end] = tokens @ w1
    return intermediate


def grouped_expert_gemm_2(intermediate: torch.Tensor, routed: RoutedTokens, weights: ExpertWeights, *, use_triton: bool = False) -> torch.Tensor:
    num_routed_tokens = routed.tokens.shape[0]
    hidden_dim = weights.w2.shape[2]
    output = torch.empty((num_routed_tokens, hidden_dim), device=intermediate.device, dtype=intermediate.dtype)
    num_experts = weights.w2.shape[0]
    for expert_id in range(num_experts):
        start = int(routed.expert_offsets[expert_id].item())
        end = int(routed.expert_offsets[expert_id + 1].item())
        if start == end:
            continue
        hidden = intermediate[start:end]
        w2 = weights.w2[expert_id]
        if use_triton:
            output[start:end] = matmul(hidden, w2)
        else:
            output[start:end] = hidden @ w2
    return output


def grouped_expert_ffn(
    hidden: torch.Tensor,
    routed: RoutedTokens,
    weights: ExpertWeights,
    *,
    use_triton: bool = False,
) -> GroupedExpertTensors:
    intermediate = grouped_expert_gemm_1(routed, weights, use_triton=use_triton)
    intermediate = _apply_activation(intermediate, weights.activation)
    output = grouped_expert_gemm_2(intermediate, routed, weights, use_triton=use_triton)
    return GroupedExpertTensors(routed=routed, intermediate=intermediate, output=output)


def grouped_expert_ffn_batched(routed: RoutedTokens, weights: ExpertWeights) -> GroupedExpertTensors:
    grouped_batches = build_grouped_batches(routed, weights.w1.shape[0])
    if grouped_batches.active_expert_ids.numel() == 0:
        empty = torch.empty((0, weights.w1.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        out = torch.empty((0, weights.w2.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        return GroupedExpertTensors(routed=routed, intermediate=empty, output=out)

    w1 = weights.w1[grouped_batches.active_expert_ids]
    w2 = weights.w2[grouped_batches.active_expert_ids]

    intermediate_padded = torch.bmm(grouped_batches.padded_tokens, w1)
    intermediate_padded = _apply_activation(intermediate_padded, weights.activation)
    output_padded = torch.bmm(intermediate_padded, w2)

    intermediate = _unpack_grouped_output(intermediate_padded, grouped_batches, routed)
    output = _unpack_grouped_output(output_padded, grouped_batches, routed)
    return GroupedExpertTensors(routed=routed, intermediate=intermediate, output=output)


def grouped_expert_ffn_triton_batched(routed: RoutedTokens, weights: ExpertWeights) -> GroupedExpertTensors:
    grouped_batches = build_grouped_batches(routed, weights.w1.shape[0])
    if grouped_batches.active_expert_ids.numel() == 0:
        empty = torch.empty((0, weights.w1.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        out = torch.empty((0, weights.w2.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        return GroupedExpertTensors(routed=routed, intermediate=empty, output=out)

    w1 = weights.w1[grouped_batches.active_expert_ids]
    w2 = weights.w2[grouped_batches.active_expert_ids]

    intermediate_padded = batched_matmul(grouped_batches.padded_tokens, w1)
    intermediate_padded = _apply_activation(intermediate_padded, weights.activation)
    output_padded = batched_matmul(intermediate_padded, w2)

    intermediate = _unpack_grouped_output(intermediate_padded, grouped_batches, routed)
    output = _unpack_grouped_output(output_padded, grouped_batches, routed)
    return GroupedExpertTensors(routed=routed, intermediate=intermediate, output=output)


def grouped_expert_ffn_triton_grouped(routed: RoutedTokens, weights: ExpertWeights) -> GroupedExpertTensors:
    return grouped_expert_ffn_triton_grouped_configurable(
        routed,
        weights,
        block_m=32,
        block_n=64,
        block_k=32,
        num_warps=4,
        num_stages=4,
    )


def grouped_expert_ffn_triton_grouped_configurable(
    routed: RoutedTokens,
    weights: ExpertWeights,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
) -> GroupedExpertTensors:
    problems = build_grouped_matmul_problems(
        routed,
        weights.w1.shape[0],
        block_m=block_m,
        block_n=block_n,
        output_dim=weights.w1.shape[2],
    )
    if problems.expert_ids.numel() == 0:
        empty = torch.empty((0, weights.w1.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        out = torch.empty((0, weights.w2.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        return GroupedExpertTensors(routed=routed, intermediate=empty, output=out)

    w1 = weights.w1[problems.weight_expert_ids.to(torch.int64)]
    w2 = weights.w2[problems.weight_expert_ids.to(torch.int64)]

    intermediate = grouped_matmul(
        routed.tokens,
        w1,
        problems.expert_ids,
        problems.token_starts,
        problems.token_counts,
        problems.tile_offsets,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    intermediate = _apply_activation(intermediate, weights.activation)
    problems_2 = build_grouped_matmul_problems(
        routed,
        weights.w2.shape[0],
        block_m=block_m,
        block_n=block_n,
        output_dim=weights.w2.shape[2],
    )
    output = grouped_matmul(
        intermediate,
        w2,
        problems_2.expert_ids,
        problems_2.token_starts,
        problems_2.token_counts,
        problems_2.tile_offsets,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return GroupedExpertTensors(routed=routed, intermediate=intermediate, output=output)


def grouped_expert_ffn_triton_persistent(
    routed: RoutedTokens,
    weights: ExpertWeights,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    num_programs: int,
) -> GroupedExpertTensors:
    problems = build_grouped_matmul_problems(
        routed,
        weights.w1.shape[0],
        block_m=block_m,
        block_n=block_n,
        output_dim=weights.w1.shape[2],
    )
    if problems.expert_ids.numel() == 0:
        empty = torch.empty((0, weights.w1.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        out = torch.empty((0, weights.w2.shape[2]), device=routed.tokens.device, dtype=routed.tokens.dtype)
        return GroupedExpertTensors(routed=routed, intermediate=empty, output=out)

    w1 = weights.w1[problems.weight_expert_ids.to(torch.int64)]
    w2 = weights.w2[problems.weight_expert_ids.to(torch.int64)]

    intermediate = persistent_grouped_matmul(
        routed.tokens,
        w1,
        problems.expert_ids,
        problems.token_starts,
        problems.token_counts,
        problems.tile_offsets,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        num_programs=num_programs,
    )
    intermediate = _apply_activation(intermediate, weights.activation)
    problems_2 = build_grouped_matmul_problems(
        routed,
        weights.w2.shape[0],
        block_m=block_m,
        block_n=block_n,
        output_dim=weights.w2.shape[2],
    )
    output = persistent_grouped_matmul(
        intermediate,
        w2,
        problems_2.expert_ids,
        problems_2.token_starts,
        problems_2.token_counts,
        problems_2.tile_offsets,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        num_programs=num_programs,
    )
    return GroupedExpertTensors(routed=routed, intermediate=intermediate, output=output)


def moe_forward_grouped_v2(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> torch.Tensor:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    grouped = grouped_expert_ffn_batched(routed, weights)
    if grouped.output.numel() == 0:
        return torch.zeros_like(hidden)
    return unpermute_tokens(grouped.output, routed, hidden.shape[0])


def moe_forward_triton(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> torch.Tensor:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    grouped = grouped_expert_ffn(hidden, routed, weights, use_triton=True)
    return unpermute_tokens(grouped.output, routed, hidden.shape[0])


def moe_forward_triton_grouped(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    weights: ExpertWeights,
    *,
    k: int = 2,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    grouped = grouped_expert_ffn_triton_grouped_configurable(
        routed,
        weights,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if grouped.output.numel() == 0:
        return torch.zeros_like(hidden)
    return unpermute_tokens(grouped.output, routed, hidden.shape[0])


def moe_forward_triton_persistent(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    weights: ExpertWeights,
    *,
    k: int = 2,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 4,
    num_programs: int = 0,
) -> torch.Tensor:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    grouped = grouped_expert_ffn_triton_persistent(
        routed,
        weights,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        num_programs=num_programs,
    )
    if grouped.output.numel() == 0:
        return torch.zeros_like(hidden)
    return unpermute_tokens(grouped.output, routed, hidden.shape[0])


def moe_forward_grouped(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> torch.Tensor:
    topk_indices, topk_weights = topk_gating(logits, k=k)
    routed = permute_tokens(hidden, topk_indices, topk_weights)
    grouped = grouped_expert_ffn(hidden, routed, weights, use_triton=False)
    if grouped.output.numel() == 0:
        return torch.zeros_like(hidden)
    return unpermute_tokens(grouped.output, routed, hidden.shape[0])


def moe_reference_error(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> float:
    naive = moe_forward_naive(hidden, logits, weights, k=k)
    grouped = moe_forward_grouped(hidden, logits, weights, k=k)
    return float((naive - grouped).abs().max())


def moe_triton_error(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> float:
    grouped_v2 = moe_forward_grouped_v2(hidden, logits, weights, k=k)
    triton_out = moe_forward_triton(hidden, logits, weights, k=k)
    return float((grouped_v2 - triton_out).abs().max())


def moe_triton_grouped_error(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> float:
    grouped_v2 = moe_forward_grouped_v2(hidden, logits, weights, k=k)
    triton_out = moe_forward_triton_grouped(hidden, logits, weights, k=k)
    return float((grouped_v2 - triton_out).abs().max())


def moe_triton_persistent_error(hidden: torch.Tensor, logits: torch.Tensor, weights: ExpertWeights, *, k: int = 2) -> float:
    grouped_v2 = moe_forward_grouped_v2(hidden, logits, weights, k=k)
    triton_out = moe_forward_triton_persistent(hidden, logits, weights, k=k)
    return float((grouped_v2 - triton_out).abs().max())
