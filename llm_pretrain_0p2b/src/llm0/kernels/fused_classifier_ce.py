from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_ce_stats_kernel(
    x_ptr,
    w_ptr,
    target_ptr,
    block_max_ptr,
    block_sum_ptr,
    block_target_ptr,
    n_rows: tl.constexpr,
    n_vocab: tl.constexpr,
    hidden_dim: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_wv: tl.constexpr,
    stride_wd: tl.constexpr,
    n_vocab_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_v: tl.constexpr,
    block_d: tl.constexpr,
):
    # Adapted from code/kernel_exp_project/triton_gemm.py: tiled tl.dot,
    # fused with blockwise CE softmax statistics instead of storing logits.
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_v = pid_v * block_v + tl.arange(0, block_v)
    offs_d = tl.arange(0, block_d)

    acc = tl.zeros((block_m, block_v), tl.float32)
    for d0 in range(0, hidden_dim, block_d):
        d = d0 + offs_d
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + d[None, :] * stride_xd,
            mask=(offs_m[:, None] < n_rows) & (d[None, :] < hidden_dim),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptr + offs_v[None, :] * stride_wv + d[:, None] * stride_wd,
            mask=(offs_v[None, :] < n_vocab) & (d[:, None] < hidden_dim),
            other=0.0,
        )
        acc += tl.dot(x_tile, w_tile)

    logits = tl.where(offs_v[None, :] < n_vocab, acc, -float("inf"))
    row_max = tl.max(logits, axis=1)
    row_sum = tl.sum(tl.exp(logits - row_max[:, None]), axis=1)

    target = tl.load(target_ptr + offs_m, mask=offs_m < n_rows, other=0)
    target_mask = (target[:, None] == offs_v[None, :]) & (offs_m[:, None] < n_rows)
    target_logits = tl.max(tl.where(target_mask, logits, -float("inf")), axis=1)

    stat_offsets = (pid_m * n_vocab_blocks + pid_v) * block_m + tl.arange(0, block_m)
    row_mask = offs_m < n_rows
    tl.store(block_max_ptr + stat_offsets, row_max, mask=row_mask)
    tl.store(block_sum_ptr + stat_offsets, row_sum, mask=row_mask)
    tl.store(block_target_ptr + stat_offsets, target_logits, mask=row_mask)


@triton.jit
def _linear_ce_reduce_kernel(
    block_max_ptr,
    block_sum_ptr,
    block_target_ptr,
    losses_ptr,
    lse_ptr,
    n_rows: tl.constexpr,
    n_vocab_blocks: tl.constexpr,
    block_m: tl.constexpr,
    block_b: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_b = tl.arange(0, block_b)
    stat_offsets = (pid_m * n_vocab_blocks + offs_b[None, :]) * block_m + tl.arange(0, block_m)[:, None]
    mask = (offs_m[:, None] < n_rows) & (offs_b[None, :] < n_vocab_blocks)

    block_max = tl.load(block_max_ptr + stat_offsets, mask=mask, other=-float("inf"))
    block_sum = tl.load(block_sum_ptr + stat_offsets, mask=mask, other=0.0)
    block_target = tl.load(block_target_ptr + stat_offsets, mask=mask, other=-float("inf"))

    row_max = tl.max(block_max, axis=1)
    denom = tl.sum(block_sum * tl.exp(block_max - row_max[:, None]), axis=1)
    row_lse = row_max + tl.log(denom)
    target_logit = tl.max(block_target, axis=1)
    loss = row_lse - target_logit

    row_mask = offs_m < n_rows
    tl.store(losses_ptr + offs_m, loss, mask=row_mask)
    tl.store(lse_ptr + offs_m, row_lse, mask=row_mask)


@triton.jit
def _linear_ce_backward_kernel(
    x_ptr,
    w_ptr,
    target_ptr,
    lse_ptr,
    grad_scale_ptr,
    grad_x_ptr,
    grad_w_ptr,
    n_rows: tl.constexpr,
    n_vocab: tl.constexpr,
    hidden_dim: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_wv: tl.constexpr,
    stride_wd: tl.constexpr,
    stride_gxm: tl.constexpr,
    stride_gxd: tl.constexpr,
    stride_gwv: tl.constexpr,
    stride_gwd: tl.constexpr,
    block_m: tl.constexpr,
    block_v: tl.constexpr,
    block_k: tl.constexpr,
    block_d: tl.constexpr,
):
    # Backward fused classifier/CE. One program computes a token-row tile and
    # one vocab tile, then accumulates grad_hidden and grad_weight with atomics.
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_v = pid_v * block_v + tl.arange(0, block_v)
    offs_k = tl.arange(0, block_k)

    logits = tl.zeros((block_m, block_v), tl.float32)
    for k0 in range(0, hidden_dim, block_k):
        k = k0 + offs_k
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xd,
            mask=(offs_m[:, None] < n_rows) & (k[None, :] < hidden_dim),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptr + offs_v[None, :] * stride_wv + k[:, None] * stride_wd,
            mask=(offs_v[None, :] < n_vocab) & (k[:, None] < hidden_dim),
            other=0.0,
        )
        logits += tl.dot(x_tile, w_tile)

    row_mask = offs_m < n_rows
    vocab_mask = offs_v < n_vocab
    lse = tl.load(lse_ptr + offs_m, mask=row_mask, other=0.0)
    target = tl.load(target_ptr + offs_m, mask=row_mask, other=0)
    scale = tl.load(grad_scale_ptr)

    grad_logits = tl.exp(logits - lse[:, None])
    target_mask = target[:, None] == offs_v[None, :]
    grad_logits -= tl.where(target_mask, 1.0, 0.0)
    grad_logits = tl.where(row_mask[:, None] & vocab_mask[None, :], grad_logits * scale, 0.0)

    offs_d = tl.arange(0, block_d)
    for d0 in range(0, hidden_dim, block_d):
        d = d0 + offs_d
        w_d = tl.load(
            w_ptr + offs_v[:, None] * stride_wv + d[None, :] * stride_wd,
            mask=(offs_v[:, None] < n_vocab) & (d[None, :] < hidden_dim),
            other=0.0,
        )
        grad_x = tl.dot(grad_logits.to(tl.float32), w_d.to(tl.float32))
        tl.atomic_add(
            grad_x_ptr + offs_m[:, None] * stride_gxm + d[None, :] * stride_gxd,
            grad_x,
            sem="relaxed",
            mask=(offs_m[:, None] < n_rows) & (d[None, :] < hidden_dim),
        )

        x_d = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + d[None, :] * stride_xd,
            mask=(offs_m[:, None] < n_rows) & (d[None, :] < hidden_dim),
            other=0.0,
        )
        grad_w = tl.dot(tl.trans(grad_logits.to(tl.float32)), x_d.to(tl.float32))
        tl.atomic_add(
            grad_w_ptr + offs_v[:, None] * stride_gwv + d[None, :] * stride_gwd,
            grad_w,
            sem="relaxed",
            mask=(offs_v[:, None] < n_vocab) & (d[None, :] < hidden_dim),
        )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _fused_ce_forward(
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    targets_1d: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_2d.ndim != 2 or weight.ndim != 2 or targets_1d.ndim != 1:
        raise ValueError("expected hidden [N,D], weight [V,D], targets [N]")
    if hidden_2d.shape[1] != weight.shape[1]:
        raise ValueError(f"hidden dim mismatch: {tuple(hidden_2d.shape)} vs {tuple(weight.shape)}")
    if hidden_2d.shape[0] != targets_1d.numel():
        raise ValueError("targets must have one entry per hidden row")
    if not hidden_2d.is_cuda or not weight.is_cuda or not targets_1d.is_cuda:
        raise ValueError("fused classifier CE expects CUDA tensors")
    compute_dtype = hidden_2d.dtype if hidden_2d.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    x = hidden_2d.to(dtype=compute_dtype).contiguous()
    w = weight.to(dtype=compute_dtype).contiguous() if weight.dtype != compute_dtype or not weight.is_contiguous() else weight
    targets = targets_1d.contiguous()

    n_rows, hidden_dim = x.shape
    n_vocab = w.shape[0]
    block_m = 8
    block_v = min(max(128, int(chunk_size)), 256)
    block_d = 64
    n_row_blocks = triton.cdiv(n_rows, block_m)
    n_vocab_blocks = triton.cdiv(n_vocab, block_v)

    stat_shape = (n_row_blocks, n_vocab_blocks, block_m)
    block_max = torch.empty(stat_shape, device=x.device, dtype=torch.float32)
    block_sum = torch.empty(stat_shape, device=x.device, dtype=torch.float32)
    block_target = torch.empty(stat_shape, device=x.device, dtype=torch.float32)
    losses = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    lse = torch.empty((n_rows,), device=x.device, dtype=torch.float32)

    _linear_ce_stats_kernel[(n_row_blocks, n_vocab_blocks)](
        x,
        w,
        targets,
        block_max,
        block_sum,
        block_target,
        n_rows,
        n_vocab,
        hidden_dim,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        n_vocab_blocks,
        block_m,
        block_v,
        block_d,
        num_warps=8,
        num_stages=4,
    )

    block_b = _next_power_of_2(n_vocab_blocks)
    _linear_ce_reduce_kernel[(n_row_blocks,)](
        block_max,
        block_sum,
        block_target,
        losses,
        lse,
        n_rows,
        n_vocab_blocks,
        block_m,
        block_b,
        num_warps=4,
    )
    return losses.mean(), lse


class FusedLinearCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk_size: int) -> torch.Tensor:
        original_shape = hidden.shape
        hidden_2d = hidden.reshape(-1, original_shape[-1])
        targets_1d = targets.reshape(-1)
        loss, lse = _fused_ce_forward(hidden_2d, weight, targets_1d, int(chunk_size))
        ctx.save_for_backward(hidden_2d, weight, targets_1d, lse)
        ctx.original_shape = original_shape
        ctx.chunk_size = int(chunk_size)
        ctx.compute_dtype = hidden_2d.dtype if hidden_2d.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_2d, weight, targets_1d, lse = ctx.saved_tensors
        n_rows, hidden_dim = hidden_2d.shape
        n_vocab = weight.shape[0]
        compute_dtype = ctx.compute_dtype
        x_compute = hidden_2d.to(dtype=compute_dtype).contiguous()
        w_compute = weight.to(dtype=compute_dtype).contiguous()
        targets = targets_1d.contiguous()
        lse = lse.contiguous()
        grad_hidden = torch.zeros_like(hidden_2d, dtype=torch.float32)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32)
        grad_scale = (grad_output.float() / float(n_rows)).reshape(())

        block_m = 16
        block_v = 64
        block_k = 64
        block_d = 32
        grid = (triton.cdiv(n_rows, block_m), triton.cdiv(n_vocab, block_v))
        _linear_ce_backward_kernel[grid](
            x_compute,
            w_compute,
            targets,
            lse,
            grad_scale,
            grad_hidden,
            grad_weight,
            n_rows,
            n_vocab,
            hidden_dim,
            x_compute.stride(0),
            x_compute.stride(1),
            w_compute.stride(0),
            w_compute.stride(1),
            grad_hidden.stride(0),
            grad_hidden.stride(1),
            grad_weight.stride(0),
            grad_weight.stride(1),
            block_m,
            block_v,
            block_k,
            block_d,
            num_warps=4,
            num_stages=3,
        )
        return grad_hidden.to(dtype=hidden_2d.dtype).reshape(ctx.original_shape), grad_weight.to(dtype=weight.dtype), None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    return FusedLinearCrossEntropyFn.apply(hidden, weight, targets, int(chunk_size))
