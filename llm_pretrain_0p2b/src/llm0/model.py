from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from llm0.config import LLMConfig
from llm0.kernels.fused_classifier_ce import fused_linear_cross_entropy
from llm0.kernels.rms_norm import TritonRMSNorm
from llm0.kernels.swiglu import TritonSwiGLUFn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rstd = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rstd * self.weight.float()).to(dtype=x.dtype)


def make_norm(config: LLMConfig) -> nn.Module:
    if config.use_triton_rmsnorm:
        return TritonRMSNorm(config.d_model, eps=config.norm_eps)
    return RMSNorm(config.d_model, eps=config.norm_eps)


def maybe_autocast_activation(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return x
    try:
        enabled = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype()
    if enabled and dtype in (torch.float16, torch.bfloat16) and x.dtype != dtype:
        return x.to(dtype=dtype)
    return x


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos().to(dtype=dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype=dtype)[None, :, None, :]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.d_model, config.kv_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.d_model, config.kv_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.attn_out_bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim)

        q = apply_rope(q, cos[:, :seq_len], sin[:, :seq_len])
        k = apply_rope(k, cos[:, :seq_len], sin[:, :seq_len])

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            enable_gqa=self.n_kv_head != self.n_head,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.mlp_bias)
        self.use_triton_swiglu = config.use_triton_swiglu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = TritonSwiGLUFn.apply(gate, up) if self.use_triton_swiglu else F.silu(gate) * up
        return self.down_proj(hidden)


class Block(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.norm_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = make_norm(config)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), cos, sin)
        x = x + self.mlp(self.norm_2(x))
        return x


class LLMForCausalLM(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.use_fused_classifier_ce = config.use_fused_classifier_ce
        self.fused_ce_chunk_size = config.fused_ce_chunk_size
        self.drop = nn.Dropout(config.dropout)
        rope_cos, rope_sin = build_rope_cache(
            seq_len=config.seq_len,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = make_norm(config)
        self.lm_head = None if config.tie_embeddings else nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        seq_len = input_ids.shape[1]
        if seq_len > self.config.seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds configured seq_len={self.config.seq_len}")

        h = self.tok_embeddings(input_ids)
        h = maybe_autocast_activation(h)
        h = self.drop(h)
        cos = self.rope_cos[:, :seq_len].to(device=h.device, dtype=h.dtype)
        sin = self.rope_sin[:, :seq_len].to(device=h.device, dtype=h.dtype)
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                h = checkpoint(block, h, cos, sin, use_reentrant=False)
            else:
                h = block(h, cos, sin)
        h = self.norm(h)

        if targets is not None and self.use_fused_classifier_ce:
            if self.lm_head is not None:
                raise ValueError("fused classifier CE currently expects tied embeddings")
            loss = fused_linear_cross_entropy(
                h, self.tok_embeddings.weight, targets, chunk_size=self.fused_ce_chunk_size
            )
            return h.new_empty((0,)), loss

        if self.lm_head is None:
            logits = F.linear(h, self.tok_embeddings.weight)
        else:
            logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -self.config.seq_len :]
            logits, _ = self(input_cond)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return input_ids

    def estimate_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def init_summary(self) -> str:
        params = self.estimate_parameters()
        return f"params={params / 1e6:.2f}M, expected={self.config.estimate_parameters() / 1e6:.2f}M"
