from __future__ import annotations

import torch

from kernel_exp_project.flash_attention import max_abs_error as flash_attention_max_abs_error
from kernel_exp_project.flash_attention import max_abs_error_v2 as flash_attention_v2_max_abs_error
from kernel_exp_project.flash_attention import max_abs_error_v3 as flash_attention_v3_max_abs_error
from kernel_exp_project.flash_attention import max_abs_error_v3_prototype as flash_attention_v3_prototype_max_abs_error
from kernel_exp_project.flash_attention import pack_padded_qkv
from kernel_exp_project.moe_routing import (
    build_random_experts,
    identity_moe_reference,
    moe_reference_error,
    moe_triton_error,
    moe_triton_grouped_error,
    moe_triton_persistent_error,
)
from kernel_exp_project.online_softmax import max_abs_error
from kernel_exp_project.triton_gemm import matmul
from kernel_exp_project.utils import require_cuda


def main() -> None:
    require_cuda()
    torch.manual_seed(0)

    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    err_gemm = float(((a @ b) - matmul(a, b)).abs().max())

    x = torch.randn((16, 257), device="cuda", dtype=torch.float16)
    err_softmax = max_abs_error(x, block_size=64)

    q = torch.randn((1, 2, 64, 32), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    err_flash = flash_attention_max_abs_error(q, k, v, causal=True)
    err_flash_v2 = flash_attention_v2_max_abs_error(q, k, v, causal=True)
    q_var = torch.randn((2, 2, 32, 32), device="cuda", dtype=torch.float16)
    k_var = torch.randn_like(q_var)
    v_var = torch.randn_like(q_var)
    lengths = torch.tensor([19, 27], device="cuda", dtype=torch.int32)
    q_packed, k_packed, v_packed, cu_seqlens = pack_padded_qkv(q_var, k_var, v_var, lengths)
    err_flash_v3_prototype = flash_attention_v3_prototype_max_abs_error(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens,
        causal=True,
    )
    err_flash_v3 = flash_attention_v3_max_abs_error(q_packed, k_packed, v_packed, cu_seqlens, causal=True)

    hidden = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    logits = torch.randn((128, 8), device="cuda", dtype=torch.float16)
    err_moe = identity_moe_reference(hidden, logits, k=2)
    moe_weights = build_random_experts(8, 64, 128, device=hidden.device, dtype=hidden.dtype, activation="gelu")
    err_moe_ffn = moe_reference_error(hidden, logits, moe_weights, k=2)
    err_moe_triton = moe_triton_error(hidden, logits, moe_weights, k=2)
    err_moe_triton_grouped = moe_triton_grouped_error(hidden, logits, moe_weights, k=2)
    err_moe_triton_persistent = moe_triton_persistent_error(hidden, logits, moe_weights, k=2)

    print("smoke_gemm_max_error", err_gemm)
    print("smoke_online_softmax_max_error", err_softmax)
    print("smoke_flash_attention_max_error", err_flash)
    print("smoke_flash_attention_v2_max_error", err_flash_v2)
    print("smoke_flash_attention_v3_prototype_max_error", err_flash_v3_prototype)
    print("smoke_flash_attention_v3_max_error", err_flash_v3)
    print("smoke_identity_moe_max_error", err_moe)
    print("smoke_moe_forward_grouped_max_error", err_moe_ffn)
    print("smoke_moe_forward_triton_max_error", err_moe_triton)
    print("smoke_moe_forward_triton_grouped_max_error", err_moe_triton_grouped)
    print("smoke_moe_forward_triton_persistent_max_error", err_moe_triton_persistent)
    print("smoke_ok")


if __name__ == "__main__":
    main()
