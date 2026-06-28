from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton installed.
    triton = None
    tl = None


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


if triton is not None:

    @triton.jit
    def _swiglu_forward_kernel(gate_ptr, up_ptr, out_ptr, n_elements: tl.constexpr, block: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        mask = offs < n_elements
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        out = gate * sigmoid * up
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        gate_ptr,
        up_ptr,
        grad_out_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        n_elements: tl.constexpr,
        block: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * block + tl.arange(0, block)
        mask = offs < n_elements
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        silu = gate * sigmoid
        dsilu = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_gate = grad_out * up * dsilu
        grad_up = grad_out * silu
        tl.store(grad_gate_ptr + offs, grad_gate, mask=mask)
        tl.store(grad_up_ptr + offs, grad_up, mask=mask)


def _block_size(n_elements: int) -> int:
    if n_elements < 1024:
        return triton.next_power_of_2(n_elements)  # type: ignore[union-attr]
    return 1024


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if triton is None or not gate.is_cuda:
        return swiglu_reference(gate, up)
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")

    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    out = torch.empty_like(gate_flat)
    n_elements = gate_flat.numel()
    block = _block_size(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    _swiglu_forward_kernel[grid](gate_flat, up_flat, out, n_elements, block, num_warps=4)
    return out.view_as(gate)


def swiglu_backward_triton(
    grad_output: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or not gate.is_cuda:
        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        swiglu_reference(gate_ref, up_ref).backward(grad_output)
        if gate_ref.grad is None or up_ref.grad is None:
            raise RuntimeError("unexpected missing gradients in swiglu fallback")
        return gate_ref.grad.to(dtype=gate.dtype), up_ref.grad.to(dtype=up.dtype)
    if grad_output.shape != gate.shape or gate.shape != up.shape:
        raise ValueError("grad_output, gate, and up must have the same shape")

    gate_flat = gate.contiguous().view(-1)
    up_flat = up.contiguous().view(-1)
    grad_flat = grad_output.contiguous().view(-1)
    grad_gate = torch.empty_like(gate_flat)
    grad_up = torch.empty_like(up_flat)
    n_elements = gate_flat.numel()
    block = _block_size(n_elements)
    grid = (triton.cdiv(n_elements, block),)
    _swiglu_backward_kernel[grid](gate_flat, up_flat, grad_flat, grad_gate, grad_up, n_elements, block, num_warps=4)
    return grad_gate.view_as(gate), grad_up.view_as(up)


class TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        return swiglu_triton(gate, up)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        return swiglu_backward_triton(grad_output, gate, up)
