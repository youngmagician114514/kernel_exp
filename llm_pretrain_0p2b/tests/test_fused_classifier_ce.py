from __future__ import annotations

import torch
import torch.nn.functional as F

from llm0.kernels.fused_classifier_ce import fused_linear_cross_entropy


def compare(dtype: torch.dtype, rows: int, hidden: int, vocab: int, chunk_size: int) -> None:
    torch.manual_seed(123)
    x_ref = torch.randn((rows, hidden), device="cuda", dtype=dtype, requires_grad=True)
    w_ref = torch.randn((vocab, hidden), device="cuda", dtype=torch.float32, requires_grad=True) * 0.02
    w_ref = w_ref.detach().requires_grad_(True)
    targets = torch.randint(0, vocab, (rows,), device="cuda", dtype=torch.long)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = F.linear(x_ref, w_ref)
        ref = F.cross_entropy(logits.float(), targets)
        tri = fused_linear_cross_entropy(x_tri, w_tri, targets, chunk_size=chunk_size)
    ref.backward()
    tri.backward()
    torch.cuda.synchronize()

    loss_err = abs(float(ref.detach() - tri.detach()))
    x_err = float((x_ref.grad - x_tri.grad).abs().max())
    w_err = float((w_ref.grad - w_tri.grad).abs().max())
    x_mean = float((x_ref.grad - x_tri.grad).abs().mean())
    w_mean = float((w_ref.grad - w_tri.grad).abs().mean())
    print(
        f"dtype={dtype} rows={rows} hidden={hidden} vocab={vocab} "
        f"loss_err={loss_err:.6e} x_err={x_err:.6e} w_err={w_err:.6e} "
        f"x_mean={x_mean:.6e} w_mean={w_mean:.6e}"
    )
    if dtype is torch.bfloat16:
        loss_tol, max_tol, mean_tol = 5e-2, 8e-2, 2e-3
    else:
        loss_tol, max_tol, mean_tol = 2e-2, 5e-2, 1e-3
    if loss_err > loss_tol:
        raise AssertionError(f"loss_err {loss_err} > {loss_tol}")
    if x_err > max_tol or w_err > max_tol:
        raise AssertionError(f"max grad err x={x_err}, w={w_err} > {max_tol}")
    if x_mean > mean_tol or w_mean > mean_tol:
        raise AssertionError(f"mean grad err x={x_mean}, w={w_mean} > {mean_tol}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for fused classifier CE tests")
    compare(torch.float16, rows=64, hidden=128, vocab=1024, chunk_size=256)
    compare(torch.bfloat16, rows=64, hidden=128, vocab=1024, chunk_size=256)
    compare(torch.bfloat16, rows=128, hidden=256, vocab=4096, chunk_size=1024)
    print("fused_classifier_ce_ok")


if __name__ == "__main__":
    main()
