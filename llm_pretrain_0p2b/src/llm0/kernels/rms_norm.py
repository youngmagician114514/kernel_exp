from __future__ import annotations

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton installed.
    triton = None
    tl = None


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x_float = x.float()
    rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_float * rstd * weight.float()).to(dtype=x.dtype)


if triton is not None:

    @triton.jit
    def _rms_norm_forward_kernel(x_ptr, weight_ptr, y_ptr, hidden: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < hidden
        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / hidden
        rstd = tl.rsqrt(mean_square + eps)
        y = x * rstd * weight
        tl.store(y_ptr + row * hidden + offs, y, mask=mask)

    @triton.jit
    def _rms_norm_backward_kernel(
        x_ptr,
        weight_ptr,
        grad_y_ptr,
        grad_x_ptr,
        grad_weight_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, block)
        mask = offs < hidden

        x = tl.load(x_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad_y = tl.load(grad_y_ptr + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)

        mean_square = tl.sum(x * x, axis=0) / hidden
        rstd = tl.rsqrt(mean_square + eps)
        grad_weighted = grad_y * weight
        inner = tl.sum(grad_weighted * x, axis=0) / hidden
        grad_x = rstd * grad_weighted - x * rstd * rstd * rstd * inner
        grad_weight = grad_y * x * rstd

        tl.store(grad_x_ptr + row * hidden + offs, grad_x, mask=mask)
        tl.atomic_add(grad_weight_ptr + offs, grad_weight, sem="relaxed", mask=mask)


def _num_warps(block: int) -> int:
    if block >= 4096:
        return 8
    if block >= 2048:
        return 4
    return 1


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if triton is None or not x.is_cuda:
        return rms_norm_reference(x, weight, eps)
    if x.shape[-1] != weight.numel():
        raise ValueError("weight must match the last dimension of x")

    hidden = x.shape[-1]
    block = triton.next_power_of_2(hidden)
    if block > 65536:
        raise ValueError(f"hidden={hidden} is too large for the simple RMSNorm kernel")

    x_2d = x.contiguous().view(-1, hidden)
    y_2d = torch.empty_like(x_2d)
    _rms_norm_forward_kernel[(x_2d.shape[0],)](
        x_2d,
        weight.contiguous(),
        y_2d,
        hidden,
        eps,
        block,
        num_warps=_num_warps(block),
    )
    return y_2d.view_as(x)


def rms_norm_backward_reference(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_float = x.float()
    grad_float = grad_output.float()
    weight_float = weight.float()
    rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)

    grad_weighted = grad_float * weight_float
    inner = (grad_weighted * x_float).mean(dim=-1, keepdim=True)
    grad_x = rstd * grad_weighted - x_float * rstd.pow(3) * inner

    reduce_dims = tuple(range(grad_output.ndim - 1))
    grad_weight = (grad_float * x_float * rstd).sum(dim=reduce_dims)
    return grad_x.to(dtype=x.dtype), grad_weight.to(dtype=weight.dtype)


def rms_norm_backward_triton(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or not x.is_cuda:
        return rms_norm_backward_reference(grad_output, x, weight, eps)
    if x.shape != grad_output.shape:
        raise ValueError("grad_output must have the same shape as x")
    if x.shape[-1] != weight.numel():
        raise ValueError("weight must match the last dimension of x")

    hidden = x.shape[-1]
    block = triton.next_power_of_2(hidden)
    if block > 65536:
        raise ValueError(f"hidden={hidden} is too large for the simple RMSNorm kernel")

    x_2d = x.contiguous().view(-1, hidden)
    grad_2d = grad_output.contiguous().view(-1, hidden)
    grad_x_2d = torch.empty_like(x_2d)
    grad_weight = torch.zeros((hidden,), device=x.device, dtype=torch.float32)
    _rms_norm_backward_kernel[(x_2d.shape[0],)](
        x_2d,
        weight.contiguous(),
        grad_2d,
        grad_x_2d,
        grad_weight,
        hidden,
        eps,
        block,
        num_warps=_num_warps(block),
    )
    return grad_x_2d.view_as(x), grad_weight.to(dtype=weight.dtype)


class TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        y = rms_norm_triton(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        grad_x, grad_weight = rms_norm_backward_triton(grad_output, x, weight, eps)
        return grad_x, grad_weight, None


class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return TritonRMSNormFn.apply(x, self.weight, self.eps)
