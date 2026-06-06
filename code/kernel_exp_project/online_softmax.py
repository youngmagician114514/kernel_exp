from __future__ import annotations

import torch


def online_softmax(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Blockwise online softmax over the last dimension."""
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")

    original_shape = x.shape
    rows = x.reshape(-1, original_shape[-1])
    running_max = torch.full((rows.shape[0], 1), -float("inf"), device=x.device, dtype=torch.float32)
    running_sum = torch.zeros((rows.shape[0], 1), device=x.device, dtype=torch.float32)
    x32 = rows.float()

    for start in range(0, rows.shape[1], block_size):
        block = x32[:, start : start + block_size]
        block_max = block.max(dim=1, keepdim=True).values
        new_max = torch.maximum(running_max, block_max)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(block - new_max).sum(
            dim=1, keepdim=True
        )
        running_max = new_max

    out = torch.exp(x32 - running_max) / running_sum
    return out.reshape(original_shape).to(x.dtype)


def max_abs_error(x: torch.Tensor, block_size: int = 64) -> float:
    expected = torch.softmax(x.float(), dim=-1).to(x.dtype)
    actual = online_softmax(x, block_size=block_size)
    return float((expected - actual).abs().max())

