from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LLMConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layer: int
    n_head: int
    n_kv_head: int
    d_ff: int
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    use_triton_rmsnorm: bool = False
    use_triton_swiglu: bool = False
    use_fused_classifier_ce: bool = False
    fused_ce_chunk_size: int = 8192
    qkv_bias: bool = False
    attn_out_bias: bool = False
    mlp_bias: bool = False
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim

    @classmethod
    def from_file(cls, path: str | Path) -> "LLMConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def estimate_parameters(self) -> int:
        embed = self.vocab_size * self.d_model
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model

        q_proj = self.d_model * self.d_model
        k_proj = self.d_model * self.kv_dim
        v_proj = self.d_model * self.kv_dim
        o_proj = self.d_model * self.d_model
        attn_bias = (self.d_model + 2 * self.kv_dim) if self.qkv_bias else 0
        attn_out_bias = self.d_model if self.attn_out_bias else 0
        attn = q_proj + k_proj + v_proj + o_proj + attn_bias + attn_out_bias

        mlp = 3 * self.d_model * self.d_ff
        if self.mlp_bias:
            mlp += 2 * self.d_ff + self.d_model
        norms = 2 * self.d_model
        per_layer = attn + mlp + norms

        final_norm = self.d_model
        return embed + lm_head + self.n_layer * per_layer + final_norm
