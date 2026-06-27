from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from llm0.config import LLMConfig
from llm0.kernels.rms_norm import TritonRMSNorm


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

        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.mlp_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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
        self.drop = nn.Dropout(config.dropout)
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
        h = self.drop(h)
        cos, sin = build_rope_cache(
            seq_len=seq_len,
            head_dim=self.config.head_dim,
            theta=self.config.rope_theta,
            device=h.device,
            dtype=h.dtype,
        )
        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                h = checkpoint(block, h, cos, sin, use_reentrant=False)
            else:
                h = block(h, cos, sin)
        h = self.norm(h)

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
