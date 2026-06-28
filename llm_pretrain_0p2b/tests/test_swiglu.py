from __future__ import annotations

import torch

from llm0.kernels.swiglu import TritonSwiGLUFn, swiglu_reference, swiglu_triton


def compare_forward(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    gate = torch.randn(shape, device="cuda", dtype=dtype)
    up = torch.randn(shape, device="cuda", dtype=dtype)
    ref = swiglu_reference(gate, up)
    tri = swiglu_triton(gate, up)
    torch.cuda.synchronize()
    max_err = float((ref - tri).abs().max())
    mean_err = float((ref - tri).abs().mean())
    print(f"forward dtype={dtype} shape={shape} max_err={max_err:.6e} mean_err={mean_err:.6e}")
    tolerance = 8e-2 if dtype is torch.bfloat16 else (8e-3 if dtype is torch.float16 else 1e-5)
    if max_err > tolerance:
        raise AssertionError(f"forward max_err {max_err} > {tolerance}")


def compare_backward(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    gate_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    gate_tri = gate_ref.detach().clone().requires_grad_(True)
    up_tri = up_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    y_ref = swiglu_reference(gate_ref, up_ref)
    y_tri = TritonSwiGLUFn.apply(gate_tri, up_tri)
    y_ref.backward(grad)
    y_tri.backward(grad)
    torch.cuda.synchronize()

    gate_err = float((gate_ref.grad - gate_tri.grad).abs().max())
    up_err = float((up_ref.grad - up_tri.grad).abs().max())
    gate_mean_err = float((gate_ref.grad - gate_tri.grad).abs().mean())
    up_mean_err = float((up_ref.grad - up_tri.grad).abs().mean())
    print(
        f"backward dtype={dtype} shape={shape} gate_err={gate_err:.6e} up_err={up_err:.6e} "
        f"gate_mean_err={gate_mean_err:.6e} up_mean_err={up_mean_err:.6e}"
    )
    tolerance = 1.5e-1 if dtype is torch.bfloat16 else (8e-3 if dtype is torch.float16 else 1e-5)
    mean_tolerance = 2e-3 if dtype is torch.bfloat16 else tolerance
    if gate_err > tolerance:
        raise AssertionError(f"backward gate_err {gate_err} > {tolerance}")
    if up_err > tolerance:
        raise AssertionError(f"backward up_err {up_err} > {tolerance}")
    if gate_mean_err > mean_tolerance or up_mean_err > mean_tolerance:
        raise AssertionError(f"backward mean_err gate={gate_mean_err}, up={up_mean_err} > {mean_tolerance}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Triton SwiGLU tests")

    torch.manual_seed(0)
    cases = [
        (torch.float32, (32, 128)),
        (torch.float16, (32, 128)),
        (torch.float16, (4, 64, 512)),
        (torch.bfloat16, (8, 1024, 2560)),
    ]
    for dtype, shape in cases:
        compare_forward(dtype, shape)
        compare_backward(dtype, shape)
    print("swiglu_ok")


if __name__ == "__main__":
    main()
