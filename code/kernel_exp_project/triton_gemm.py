from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((block_m, block_n), tl.float32)

    for k0 in range(0, k, block_k):
        a_tile = tl.load(a, mask=(offs_m[:, None] < m) & (k0 + offs_k[None, :] < k), other=0.0)
        b_tile = tl.load(b, mask=(k0 + offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
        acc += tl.dot(a_tile, b_tile)
        a += block_k * stride_ak
        b += block_k * stride_bk

    c = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c, acc.to(tl.float16), mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def _batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    batch: tl.constexpr,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    if pid_batch >= batch:
        return

    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a_batch = a_ptr + pid_batch * stride_ab
    b_batch = b_ptr + pid_batch * stride_bb
    c_batch = c_ptr + pid_batch * stride_cb

    a = a_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b = b_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((block_m, block_n), tl.float32)

    for k0 in range(0, k, block_k):
        a_tile = tl.load(a, mask=(offs_m[:, None] < m) & (k0 + offs_k[None, :] < k), other=0.0)
        b_tile = tl.load(b, mask=(k0 + offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
        acc += tl.dot(a_tile, b_tile)
        a += block_k * stride_ak
        b += block_k * stride_bk

    c = c_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c, acc.to(tl.float16), mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def _grouped_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    problem_expert_ids_ptr,
    problem_token_starts_ptr,
    problem_token_counts_ptr,
    problem_tile_offsets_ptr,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_problems: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    tiles_n = tl.cdiv(n, block_n)

    problem_idx = 0
    for i in range(1, num_problems):
        tile_start = tl.load(problem_tile_offsets_ptr + i)
        problem_idx += pid >= tile_start

    problem_start = tl.load(problem_token_starts_ptr + problem_idx)
    problem_tokens = tl.load(problem_token_counts_ptr + problem_idx)
    problem_expert = tl.load(problem_expert_ids_ptr + problem_idx)
    tile_start = tl.load(problem_tile_offsets_ptr + problem_idx)
    local_tile = pid - tile_start

    pid_m = local_tile // tiles_n
    pid_n = local_tile % tiles_n

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)

    a = a_ptr + (problem_start + offs_m)[:, None] * stride_am + offs_k[None, :] * stride_ak
    b = b_ptr + problem_expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((block_m, block_n), tl.float32)

    for k0 in range(0, k, block_k):
        a_tile = tl.load(a, mask=(offs_m[:, None] < problem_tokens) & (k0 + offs_k[None, :] < k), other=0.0)
        b_tile = tl.load(b, mask=(k0 + offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
        acc += tl.dot(a_tile, b_tile)
        a += block_k * stride_ak
        b += block_k * stride_bk

    c = c_ptr + (problem_start + offs_m)[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c, acc.to(tl.float16), mask=(offs_m[:, None] < problem_tokens) & (offs_n[None, :] < n))


@triton.jit
def _persistent_grouped_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    problem_expert_ids_ptr,
    problem_token_starts_ptr,
    problem_token_counts_ptr,
    problem_tile_offsets_ptr,
    work_counter_ptr,
    total_tiles,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_problems: tl.constexpr,
    max_iters: tl.constexpr,
):
    tiles_n = tl.cdiv(n, block_n)

    for _ in range(max_iters):
        tile_idx = tl.atomic_add(work_counter_ptr, 1)
        active = tile_idx < total_tiles
        problem_idx = 0
        for i in range(1, num_problems):
            tile_start = tl.load(problem_tile_offsets_ptr + i)
            problem_idx += active & (tile_idx >= tile_start)

        problem_start = tl.load(problem_token_starts_ptr + problem_idx)
        problem_tokens = tl.load(problem_token_counts_ptr + problem_idx)
        problem_expert = tl.load(problem_expert_ids_ptr + problem_idx)
        tile_start = tl.load(problem_tile_offsets_ptr + problem_idx)
        local_tile = tile_idx - tile_start

        pid_m = local_tile // tiles_n
        pid_n = local_tile % tiles_n

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        a = a_ptr + (problem_start + offs_m)[:, None] * stride_am + offs_k[None, :] * stride_ak
        b = b_ptr + problem_expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((block_m, block_n), tl.float32)

        for k0 in range(0, k, block_k):
            a_tile = tl.load(
                a,
                mask=active & (offs_m[:, None] < problem_tokens) & (k0 + offs_k[None, :] < k),
                other=0.0,
            )
            b_tile = tl.load(
                b,
                mask=active & (k0 + offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )
            acc += tl.dot(a_tile, b_tile)
            a += block_k * stride_ak
            b += block_k * stride_bk

        c = c_ptr + (problem_start + offs_m)[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(
            c,
            acc.to(tl.float16),
            mask=active & (offs_m[:, None] < problem_tokens) & (offs_n[None, :] < n),
        )


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    group_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("this teaching kernel expects float16 inputs")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("inputs must be CUDA tensors")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_m,
        block_n,
        block_k,
        group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


def batched_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    group_m: int = 8,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("batched_matmul expects rank-3 tensors")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"batch mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.shape[2] != b.shape[1]:
        raise ValueError(f"incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("this teaching kernel expects float16 inputs")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("inputs must be CUDA tensors")

    batch, m, k = a.shape
    _, _, n = b.shape
    c = torch.empty((batch, m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), batch)
    _batched_matmul_kernel[grid](
        a,
        b,
        c,
        batch,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        block_m,
        block_n,
        block_k,
        group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


def grouped_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    problem_expert_ids: torch.Tensor,
    problem_token_starts: torch.Tensor,
    problem_token_counts: torch.Tensor,
    problem_tile_offsets: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("grouped_matmul expects a:[total_tokens,k] and b:[num_experts,k,n]")
    if problem_expert_ids.ndim != 1 or problem_token_starts.ndim != 1 or problem_token_counts.ndim != 1:
        raise ValueError("problem descriptors must be rank-1 tensors")
    if problem_tile_offsets.ndim != 1:
        raise ValueError("problem_tile_offsets must be rank-1 tensor")
    if not (a.is_cuda and b.is_cuda and problem_expert_ids.is_cuda and problem_token_starts.is_cuda and problem_token_counts.is_cuda and problem_tile_offsets.is_cuda):
        raise ValueError("all inputs must be CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("this teaching kernel expects float16 inputs")
    if problem_expert_ids.dtype != torch.int32 or problem_token_starts.dtype != torch.int32 or problem_token_counts.dtype != torch.int32 or problem_tile_offsets.dtype != torch.int32:
        raise ValueError("problem descriptors must be int32 tensors")
    if problem_expert_ids.numel() != problem_token_starts.numel() or problem_expert_ids.numel() != problem_token_counts.numel():
        raise ValueError("descriptor tensors must have the same length")
    if problem_tile_offsets.numel() != problem_expert_ids.numel() + 1:
        raise ValueError("problem_tile_offsets must have length num_problems + 1")
    if problem_expert_ids.numel() == 0:
        return torch.empty((0, b.shape[2]), device=a.device, dtype=torch.float16)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")

    total_tokens, k = a.shape
    _, _, n = b.shape
    c = torch.empty((total_tokens, n), device=a.device, dtype=torch.float16)
    total_tiles = int(problem_tile_offsets[-1].item())
    _grouped_matmul_kernel[(total_tiles,)](
        a,
        b,
        c,
        problem_expert_ids,
        problem_token_starts,
        problem_token_counts,
        problem_tile_offsets,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        block_m,
        block_n,
        block_k,
        num_problems=problem_expert_ids.numel(),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


def persistent_grouped_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    problem_expert_ids: torch.Tensor,
    problem_token_starts: torch.Tensor,
    problem_token_counts: torch.Tensor,
    problem_tile_offsets: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 4,
    num_programs: int = 0,
) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("persistent_grouped_matmul expects a:[total_tokens,k] and b:[num_experts,k,n]")
    if problem_expert_ids.ndim != 1 or problem_token_starts.ndim != 1 or problem_token_counts.ndim != 1:
        raise ValueError("problem descriptors must be rank-1 tensors")
    if problem_tile_offsets.ndim != 1:
        raise ValueError("problem_tile_offsets must be rank-1 tensor")
    if not (a.is_cuda and b.is_cuda and problem_expert_ids.is_cuda and problem_token_starts.is_cuda and problem_token_counts.is_cuda and problem_tile_offsets.is_cuda):
        raise ValueError("all inputs must be CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("this teaching kernel expects float16 inputs")
    if problem_expert_ids.dtype != torch.int32 or problem_token_starts.dtype != torch.int32 or problem_token_counts.dtype != torch.int32 or problem_tile_offsets.dtype != torch.int32:
        raise ValueError("problem descriptors must be int32 tensors")
    if problem_expert_ids.numel() != problem_token_starts.numel() or problem_expert_ids.numel() != problem_token_counts.numel():
        raise ValueError("descriptor tensors must have the same length")
    if problem_tile_offsets.numel() != problem_expert_ids.numel() + 1:
        raise ValueError("problem_tile_offsets must have length num_problems + 1")
    if problem_expert_ids.numel() == 0:
        return torch.empty((0, b.shape[2]), device=a.device, dtype=torch.float16)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")

    total_tokens, k = a.shape
    _, _, n = b.shape
    c = torch.empty((total_tokens, n), device=a.device, dtype=torch.float16)
    total_tiles = int(problem_tile_offsets[-1].item())
    if num_programs <= 0:
        sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
        num_programs = min(total_tiles, max(sm_count * 4, 1))
    max_iters = (total_tiles + num_programs - 1) // num_programs
    work_counter = torch.zeros((1,), device=a.device, dtype=torch.int32)
    _persistent_grouped_matmul_kernel[(num_programs,)](
        a,
        b,
        c,
        problem_expert_ids,
        problem_token_starts,
        problem_token_counts,
        problem_tile_offsets,
        work_counter,
        total_tiles,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        block_m,
        block_n,
        block_k,
        num_problems=problem_expert_ids.numel(),
        max_iters=max_iters,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
