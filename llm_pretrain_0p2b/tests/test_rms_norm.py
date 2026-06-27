from __future__ import annotations

import torch

from llm0.kernels.rms_norm import TritonRMSNormFn, rms_norm_reference, rms_norm_triton


def compare_forward(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    ref = rms_norm_reference(x, weight)
    tri = rms_norm_triton(x, weight)
    torch.cuda.synchronize()
    max_err = float((ref - tri).abs().max())
    mean_err = float((ref - tri).abs().mean())
    print(f"forward dtype={dtype} shape={shape} max_err={max_err:.6e} mean_err={mean_err:.6e}")
    tolerance = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    if max_err > tolerance:
        raise AssertionError(f"forward max_err {max_err} > {tolerance}")


def compare_backward(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    x_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w_ref = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    y_ref = rms_norm_reference(x_ref, w_ref)
    y_tri = TritonRMSNormFn.apply(x_tri, w_tri, 1e-5)
    y_ref.backward(grad)
    y_tri.backward(grad)
    torch.cuda.synchronize()

    x_err = float((x_ref.grad - x_tri.grad).abs().max())
    w_err = float((w_ref.grad - w_tri.grad).abs().max())
    print(f"backward dtype={dtype} shape={shape} x_err={x_err:.6e} w_err={w_err:.6e}")
    tolerance = 7e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    if x_err > tolerance:
        raise AssertionError(f"backward x_err {x_err} > {tolerance}")
    if w_err > tolerance * max(1, shape[0] if len(shape) == 2 else shape[0] * shape[1]):
        raise AssertionError(f"backward w_err {w_err} is too large")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Triton RMSNorm tests")

    cases = [
        (torch.float32, (32, 128)),
        (torch.float16, (32, 128)),
        (torch.float16, (4, 64, 512)),
        (torch.bfloat16, (16, 1536)),
    ]
    for dtype, shape in cases:
        compare_forward(dtype, shape)
        compare_backward(dtype, shape)
    print("rms_norm_ok")


if __name__ == "__main__":
    main()
