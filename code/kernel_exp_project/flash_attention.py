from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    scale,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    causal: tl.constexpr,
    causal_block_skip: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    batch_id = pid_bh // num_heads
    head_id = pid_bh - batch_id * num_heads
    q_block_start = pid_m * block_m

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)

    q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh
    k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh
    v_base = v_ptr + batch_id * stride_vb + head_id * stride_vh
    o_base = o_ptr + batch_id * stride_ob + head_id * stride_oh

    q = tl.load(
        q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    acc = tl.zeros((block_m, block_d), dtype=tl.float32)
    m_i = tl.where(offs_m < seq_len, -float("inf"), 0.0)
    l_i = tl.where(offs_m < seq_len, 0.0, 1.0)

    kv_loop_end = seq_len
    if causal and causal_block_skip:
        kv_loop_end = tl.minimum(seq_len, (pid_m + 1) * block_m)
        # Blocks strictly below the diagonal are fully valid and avoid per-element causal masking.
        full_blocks_end = tl.maximum(q_block_start - block_n + 1, 0)
        masked_blocks_start = tl.cdiv(full_blocks_end, block_n) * block_n

        for start_n in range(0, full_blocks_end, block_n):
            cols = start_n + offs_n
            k = tl.load(
                k_base + cols[None, :] * stride_ks + offs_d[:, None] * stride_kd,
                mask=(cols[None, :] < seq_len) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, k) * scale
            scores = tl.where(cols[None, :] < seq_len, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_base + cols[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=(cols[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new
            l_i = l_new

        for start_n in range(masked_blocks_start, kv_loop_end, block_n):
            cols = start_n + offs_n
            k = tl.load(
                k_base + cols[None, :] * stride_ks + offs_d[:, None] * stride_kd,
                mask=(cols[None, :] < seq_len) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, k) * scale
            scores = tl.where(cols[None, :] < seq_len, scores, -float("inf"))
            scores = tl.where(cols[None, :] <= offs_m[:, None], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_base + cols[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=(cols[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new
            l_i = l_new
    else:
        for start_n in range(0, kv_loop_end, block_n):
            cols = start_n + offs_n
            k = tl.load(
                k_base + cols[None, :] * stride_ks + offs_d[:, None] * stride_kd,
                mask=(cols[None, :] < seq_len) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, k) * scale
            scores = tl.where(cols[None, :] < seq_len, scores, -float("inf"))

            if causal:
                scores = tl.where(cols[None, :] <= offs_m[:, None], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_base + cols[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=(cols[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new
            l_i = l_new

    out = acc / l_i[:, None]
    tl.store(
        o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        out.to(tl.float16),
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )


@triton.jit
def _flash_attention_varlen_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    cu_seqlens_ptr,
    block_batch_ids_ptr,
    block_q_offsets_ptr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    scale,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    causal: tl.constexpr,
    causal_block_skip: tl.constexpr,
):
    pid_block = tl.program_id(axis=0)
    head_id = tl.program_id(axis=1)

    batch_id = tl.load(block_batch_ids_ptr + pid_block)
    q_block_start = tl.load(block_q_offsets_ptr + pid_block)

    seq_start = tl.load(cu_seqlens_ptr + batch_id)
    seq_end = tl.load(cu_seqlens_ptr + batch_id + 1)
    seq_len = seq_end - seq_start

    offs_m_local = q_block_start + tl.arange(0, block_m)
    offs_m_global = seq_start + offs_m_local
    offs_n_local = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)

    q = tl.load(
        q_ptr + offs_m_global[:, None] * stride_qt + head_id * stride_qh + offs_d[None, :] * stride_qd,
        mask=(offs_m_local[:, None] < seq_len) & (offs_d[None, :] < head_dim),
        other=0.0,
    )

    acc = tl.zeros((block_m, block_d), dtype=tl.float32)
    m_i = tl.where(offs_m_local < seq_len, -float("inf"), 0.0)
    l_i = tl.where(offs_m_local < seq_len, 0.0, 1.0)

    kv_loop_end = seq_len
    if causal and causal_block_skip:
        kv_loop_end = tl.minimum(seq_len, q_block_start + block_m)
        full_blocks_end = tl.maximum(q_block_start - block_n + 1, 0)
        masked_blocks_start = tl.cdiv(full_blocks_end, block_n) * block_n

        for start_n in range(0, full_blocks_end, block_n):
            cols_local = start_n + offs_n_local
            cols_global = seq_start + cols_local
            k = tl.load(
                k_ptr + cols_global[None, :] * stride_kt + head_id * stride_kh + offs_d[:, None] * stride_kd,
                mask=(cols_local[None, :] < seq_len) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, k) * scale
            scores = tl.where(cols_local[None, :] < seq_len, scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_ptr + cols_global[:, None] * stride_vt + head_id * stride_vh + offs_d[None, :] * stride_vd,
                mask=(cols_local[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new
            l_i = l_new

        for start_n in range(masked_blocks_start, kv_loop_end, block_n):
            cols_local = start_n + offs_n_local
            cols_global = seq_start + cols_local
            k = tl.load(
                k_ptr + cols_global[None, :] * stride_kt + head_id * stride_kh + offs_d[:, None] * stride_kd,
                mask=(cols_local[None, :] < seq_len) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, k) * scale
            scores = tl.where(cols_local[None, :] < seq_len, scores, -float("inf"))
            scores = tl.where(cols_local[None, :] <= offs_m_local[:, None], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_ptr + cols_global[:, None] * stride_vt + head_id * stride_vh + offs_d[None, :] * stride_vd,
                mask=(cols_local[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new
            l_i = l_new
    else:
        for start_n in range(0, kv_loop_end, block_n):
            cols_local = start_n + offs_n_local
            cols_global = seq_start + cols_local
            k = tl.load(
                k_ptr + cols_global[None, :] * stride_kt + head_id * stride_kh + offs_d[:, None] * stride_kd,
                mask=(cols_local[None, :] < seq_len) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, k) * scale
            scores = tl.where(cols_local[None, :] < seq_len, scores, -float("inf"))

            if causal:
                scores = tl.where(cols_local[None, :] <= offs_m_local[:, None], scores, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                v_ptr + cols_global[:, None] * stride_vt + head_id * stride_vh + offs_d[None, :] * stride_vd,
                mask=(cols_local[:, None] < seq_len) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            m_i = m_new
            l_i = l_new

    out = acc / l_i[:, None]
    tl.store(
        o_ptr + offs_m_global[:, None] * stride_ot + head_id * stride_oh + offs_d[None, :] * stride_od,
        out.to(tl.float16),
        mask=(offs_m_local[:, None] < seq_len) & (offs_d[None, :] < head_dim),
    )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _select_flash_attention_v2_config(
    seq_len: int,
    head_dim: int,
    causal: bool,
) -> tuple[int, int, int, int]:
    if causal and head_dim == 64:
        if seq_len <= 768:
            return 16, 64, 8, 4
        if seq_len <= 1536:
            return 32, 128, 4, 3
        return 32, 64, 4, 3
    return 16, 64, 4, 4


def _select_flash_attention_v3_config(
    max_seq_len: int,
    head_dim: int,
    causal: bool,
) -> tuple[int, int, int, int]:
    if causal and head_dim == 64:
        if max_seq_len <= 192:
            return 16, 64, 8, 4
        if max_seq_len <= 384:
            return 32, 64, 8, 4
        return 16, 64, 4, 3
    return 16, 64, 4, 4


def _build_varlen_block_map(cu_seqlens: torch.Tensor, block_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    cu_host = cu_seqlens.detach().cpu().tolist()
    block_batch_ids: list[int] = []
    block_q_offsets: list[int] = []
    for batch_id in range(len(cu_host) - 1):
        seq_len = cu_host[batch_id + 1] - cu_host[batch_id]
        for block_idx in range(triton.cdiv(seq_len, block_m)):
            block_batch_ids.append(batch_id)
            block_q_offsets.append(block_idx * block_m)
    return (
        torch.tensor(block_batch_ids, device=cu_seqlens.device, dtype=torch.int32),
        torch.tensor(block_q_offsets, device=cu_seqlens.device, dtype=torch.int32),
    )


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [batch, heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have the same shape, got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("this teaching kernel expects float16 q, k, v")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, v must be CUDA tensors")


def _check_packed_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed q, k, v must have shape [heads, total_tokens, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"packed q, k, v must have the same shape, got {q.shape}, {k.shape}, {v.shape}")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise ValueError("this teaching kernel expects float16 packed q, k, v")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("packed q, k, v must be CUDA tensors")
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must have shape [batch + 1]")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32")
    if not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be a CUDA tensor")
    cu_host = cu_seqlens.detach().cpu()
    if int(cu_host[0].item()) != 0:
        raise ValueError("cu_seqlens must start with 0")
    if int(cu_host[-1].item()) != q.shape[1]:
        raise ValueError("cu_seqlens[-1] must equal total_tokens")
    if bool((cu_host[1:] < cu_host[:-1]).any()):
        raise ValueError("cu_seqlens must be non-decreasing")


def pack_padded_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_qkv(q, k, v)
    if lengths.ndim != 1 or lengths.dtype != torch.int32:
        raise ValueError("lengths must have shape [batch] and dtype int32")
    if not lengths.is_cuda:
        raise ValueError("lengths must be a CUDA tensor")

    batch, heads, seq_len, _ = q.shape
    if lengths.numel() != batch:
        raise ValueError("lengths.shape[0] must equal batch size")
    if bool((lengths < 0).any()) or bool((lengths > seq_len).any()):
        raise ValueError("lengths must be within [0, seq_len]")

    lengths_host = lengths.detach().cpu().tolist()
    total_tokens = int(sum(lengths_host))
    q_packed = torch.empty((heads, total_tokens, q.shape[-1]), device=q.device, dtype=q.dtype)
    k_packed = torch.empty_like(q_packed)
    v_packed = torch.empty_like(q_packed)
    cu_seqlens = torch.empty((batch + 1,), device=q.device, dtype=torch.int32)
    cu_seqlens[0] = 0

    offset = 0
    for batch_id, valid_len in enumerate(lengths_host):
        if valid_len > 0:
            q_packed[:, offset : offset + valid_len, :] = q[batch_id, :, :valid_len, :]
            k_packed[:, offset : offset + valid_len, :] = k[batch_id, :, :valid_len, :]
            v_packed[:, offset : offset + valid_len, :] = v[batch_id, :, :valid_len, :]
        offset += valid_len
        cu_seqlens[batch_id + 1] = offset
    return q_packed, k_packed, v_packed, cu_seqlens


def unpack_padded_output(out_packed: torch.Tensor, cu_seqlens: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    if out_packed.ndim != 3:
        raise ValueError("out_packed must have shape [heads, total_tokens, head_dim]")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
        raise ValueError("cu_seqlens must have shape [batch + 1] and dtype int32")
    batch = cu_seqlens.numel() - 1
    heads, total_tokens, head_dim = out_packed.shape
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError("cu_seqlens[-1] must equal total_tokens")

    out = torch.zeros((batch, heads, max_seq_len, head_dim), device=out_packed.device, dtype=out_packed.dtype)
    for batch_id in range(batch):
        start = int(cu_seqlens[batch_id].item())
        end = int(cu_seqlens[batch_id + 1].item())
        seq_len = end - start
        if seq_len > 0:
            out[batch_id, :, :seq_len, :] = out_packed[:, start:end, :]
    return out


def _flash_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 4,
    causal_block_skip: bool = False,
) -> torch.Tensor:
    _check_qkv(q, k, v)
    batch, heads, seq_len, head_dim = q.shape
    if head_dim > 128:
        raise ValueError("this teaching kernel supports head_dim <= 128")
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")

    block_d = _next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    out = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, block_m), batch * heads)

    _flash_attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        seq_len,
        head_dim,
        heads,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_m,
        block_n,
        block_d,
        causal,
        causal_block_skip,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    block_m: int = 16,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    return _flash_attention_impl(
        q,
        k,
        v,
        causal=causal,
        scale=scale,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
        causal_block_skip=False,
    )


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    _, _, seq_len, head_dim = q.shape
    default_block_m, default_block_n, default_num_warps, default_num_stages = _select_flash_attention_v2_config(
        seq_len,
        head_dim,
        causal,
    )
    return _flash_attention_impl(
        q,
        k,
        v,
        causal=causal,
        scale=scale,
        block_m=default_block_m if block_m is None else block_m,
        block_n=default_block_n if block_n is None else block_n,
        num_warps=default_num_warps if num_warps is None else num_warps,
        num_stages=default_num_stages if num_stages is None else num_stages,
        causal_block_skip=True,
    )


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    _check_qkv(q, k, v)
    _, _, seq_len, head_dim = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    if causal:
        mask = torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -float("inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(probs, v)


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    _check_qkv(q, k, v)
    _, _, seq_len, head_dim = q.shape
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    if causal:
        mask = torch.ones((seq_len, seq_len), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def flash_attention_v3_prototype(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    _check_packed_qkv(q, k, v, cu_seqlens)
    heads, _, head_dim = q.shape
    batch = cu_seqlens.numel() - 1
    if head_dim > 128:
        raise ValueError("this teaching kernel supports head_dim <= 128")

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    out = torch.empty_like(q)
    for batch_id, seq_len in enumerate(lengths):
        start = int(cu_seqlens[batch_id].item())
        end = int(cu_seqlens[batch_id + 1].item())
        if seq_len == 0:
            continue
        seq_q = q[:, start:end, :].unsqueeze(0)
        seq_k = k[:, start:end, :].unsqueeze(0)
        seq_v = v[:, start:end, :].unsqueeze(0)
        seq_out = flash_attention_v2(
            seq_q,
            seq_k,
            seq_v,
            causal=causal,
            scale=scale,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        out[:, start:end, :] = seq_out.squeeze(0)
    return out


def flash_attention_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    _check_packed_qkv(q, k, v, cu_seqlens)
    heads, total_tokens, head_dim = q.shape
    batch = cu_seqlens.numel() - 1
    if head_dim > 128:
        raise ValueError("this teaching kernel supports head_dim <= 128")

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    max_seq_len = max(lengths) if lengths else 0
    default_block_m, default_block_n, default_num_warps, default_num_stages = _select_flash_attention_v3_config(
        max_seq_len,
        head_dim,
        causal,
    )
    block_m = default_block_m if block_m is None else block_m
    block_n = default_block_n if block_n is None else block_n
    num_warps = default_num_warps if num_warps is None else num_warps
    num_stages = default_num_stages if num_stages is None else num_stages
    block_d = _next_power_of_2(head_dim)
    scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    block_batch_ids, block_q_offsets = _build_varlen_block_map(cu_seqlens, block_m)

    out = torch.empty((heads, total_tokens, head_dim), device=q.device, dtype=q.dtype)
    grid = (block_batch_ids.numel(), heads)
    _flash_attention_varlen_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        cu_seqlens,
        block_batch_ids,
        block_q_offsets,
        head_dim,
        heads,
        scale,
        q.stride(1),
        q.stride(0),
        q.stride(2),
        k.stride(1),
        k.stride(0),
        k.stride(2),
        v.stride(1),
        v.stride(0),
        v.stride(2),
        out.stride(1),
        out.stride(0),
        out.stride(2),
        block_m,
        block_n,
        block_d,
        causal,
        True,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def attention_reference_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    _check_packed_qkv(q, k, v, cu_seqlens)
    out = torch.empty_like(q)
    batch = cu_seqlens.numel() - 1
    for batch_id in range(batch):
        start = int(cu_seqlens[batch_id].item())
        end = int(cu_seqlens[batch_id + 1].item())
        seq_q = q[:, start:end, :].unsqueeze(0)
        seq_k = k[:, start:end, :].unsqueeze(0)
        seq_v = v[:, start:end, :].unsqueeze(0)
        seq_out = attention_reference(seq_q, seq_k, seq_v, causal=causal)
        out[:, start:end, :] = seq_out.squeeze(0)
    return out


def max_abs_error(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> float:
    expected = attention_reference(q, k, v, causal=causal)
    actual = flash_attention(q, k, v, causal=causal)
    torch.cuda.synchronize()
    return float((expected - actual).abs().max())


def max_abs_error_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> float:
    expected = attention_reference(q, k, v, causal=causal)
    actual = flash_attention_v2(q, k, v, causal=causal)
    torch.cuda.synchronize()
    return float((expected - actual).abs().max())


def max_abs_error_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    causal: bool = False,
) -> float:
    expected = attention_reference_v3(q, k, v, cu_seqlens, causal=causal)
    actual = flash_attention_v3(q, k, v, cu_seqlens, causal=causal)
    torch.cuda.synchronize()
    return float((expected - actual).abs().max())


def max_abs_error_v3_prototype(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    causal: bool = False,
) -> float:
    expected = attention_reference_v3(q, k, v, cu_seqlens, causal=causal)
    actual = flash_attention_v3_prototype(q, k, v, cu_seqlens, causal=causal)
    torch.cuda.synchronize()
    return float((expected - actual).abs().max())
