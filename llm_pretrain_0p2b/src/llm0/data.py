from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


IGNORE_INDEX = -100
TOKEN_DTYPES: dict[str, Any] = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int32": np.int32,
    "int64": np.int64,
}


@dataclass
class Batch:
    input_ids: torch.Tensor
    targets: torch.Tensor


def _read_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _dtype_from_meta(meta: dict[str, Any], key: str, default: str) -> Any:
    dtype_name = str(meta.get(key, default))
    if dtype_name not in TOKEN_DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name!r}; choose from {sorted(TOKEN_DTYPES)}")
    return TOKEN_DTYPES[dtype_name]


class PackedMemmapDataset:
    """Autoregressive dataset backed by train.bin / val.bin token memmaps."""

    def __init__(self, data_dir: str | Path, seq_len: int) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.meta = _read_meta(self.data_dir)
        self.token_dtype = _dtype_from_meta(self.meta, "dtype", "uint16")
        self.train = self._open_split("train")
        self.val = self._open_split("val")
        self._check_split("train", self.train)
        self._check_split("val", self.val)

    def _open_split(self, split: str) -> np.memmap:
        path = self.data_dir / f"{split}.bin"
        if not path.exists():
            raise FileNotFoundError(f"missing split file: {path}")
        return np.memmap(path, dtype=self.token_dtype, mode="r")

    def _check_split(self, split: str, data: np.memmap) -> None:
        if len(data) <= self.seq_len + 1:
            raise ValueError(f"{split} split is too small for the configured sequence length")

    def split_size(self, split: str) -> int:
        data = self.train if split == "train" else self.val
        return int(len(data))

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        max_start = len(data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,)).tolist()
        x = np.stack([np.asarray(data[i : i + self.seq_len], dtype=np.int64) for i in starts])
        y = np.stack([np.asarray(data[i + 1 : i + 1 + self.seq_len], dtype=np.int64) for i in starts])
        return (
            torch.from_numpy(x).to(device=device, dtype=torch.long, non_blocking=True),
            torch.from_numpy(y).to(device=device, dtype=torch.long, non_blocking=True),
        )


class SupervisedMemmapDataset:
    """SFT dataset with flattened input_ids and response-only labels."""

    def __init__(self, data_dir: str | Path, seq_len: int) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.meta = _read_meta(self.data_dir)
        self.token_dtype = _dtype_from_meta(self.meta, "dtype", "uint32")
        self.label_dtype = _dtype_from_meta(self.meta, "label_dtype", "int32")
        self.train = self._open_pair("train")
        self.val = self._open_pair("val")
        self._check_split("train", self.train[0], self.train[1])
        self._check_split("val", self.val[0], self.val[1])

    def _open_pair(self, split: str) -> tuple[np.memmap, np.memmap]:
        input_path = self.data_dir / f"{split}_input_ids.bin"
        label_path = self.data_dir / f"{split}_labels.bin"
        if not input_path.exists():
            raise FileNotFoundError(f"missing split file: {input_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"missing split file: {label_path}")
        input_ids = np.memmap(input_path, dtype=self.token_dtype, mode="r")
        labels = np.memmap(label_path, dtype=self.label_dtype, mode="r")
        return input_ids, labels

    def _check_split(self, split: str, input_ids: np.memmap, labels: np.memmap) -> None:
        if len(input_ids) != len(labels):
            raise ValueError(f"{split} input_ids and labels have different lengths")
        if len(input_ids) <= self.seq_len + 1:
            raise ValueError(f"{split} split is too small for the configured sequence length")

    def split_size(self, split: str) -> int:
        input_ids, _ = self.train if split == "train" else self.val
        return int(len(input_ids))

    def _sample_start_with_label(self, labels: np.memmap, max_start: int) -> int:
        for _ in range(100):
            start = int(torch.randint(0, max_start, ()).item())
            target_window = labels[start + 1 : start + 1 + self.seq_len]
            if np.any(target_window != IGNORE_INDEX):
                return start
        return int(torch.randint(0, max_start, ()).item())

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.train if split == "train" else self.val
        max_start = len(input_ids) - self.seq_len - 1
        starts = [self._sample_start_with_label(labels, max_start) for _ in range(batch_size)]
        x = np.stack([np.asarray(input_ids[i : i + self.seq_len], dtype=np.int64) for i in starts])
        y = np.stack([np.asarray(labels[i + 1 : i + 1 + self.seq_len], dtype=np.int64) for i in starts])
        return (
            torch.from_numpy(x).to(device=device, dtype=torch.long, non_blocking=True),
            torch.from_numpy(y).to(device=device, dtype=torch.long, non_blocking=True),
        )


class ByteMemmapDataset(PackedMemmapDataset):
    """Backward-compatible name for older byte-level experiments."""


def build_dataset(data_dir: str | Path, seq_len: int) -> PackedMemmapDataset | SupervisedMemmapDataset:
    data_path = Path(data_dir)
    if (data_path / "train_input_ids.bin").exists() or (data_path / "train_labels.bin").exists():
        return SupervisedMemmapDataset(data_path, seq_len)
    return PackedMemmapDataset(data_path, seq_len)


def random_token_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len + 1),
        device=device,
        dtype=torch.long,
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
