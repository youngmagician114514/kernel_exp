from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack plain text or JSONL text into train.bin / val.bin.")
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="text/jsonl files or directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer name/path; omit for UTF-8 byte tokenizer")
    parser.add_argument("--jsonl-field", type=str, default="text")
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--append-eos", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=0, help="optional cap for quick scaling experiments")
    return parser.parse_args(argv)


def iter_files(paths: list[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for suffix in ("*.txt", "*.md", "*.jsonl"):
                yield from sorted(path.rglob(suffix))
        else:
            yield path


def iter_texts(files: Iterable[Path], jsonl_field: str) -> Iterable[str]:
    for path in files:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text = row.get(jsonl_field)
                    if text is None:
                        continue
                    yield str(text)
        else:
            yield path.read_text(encoding="utf-8", errors="ignore")


def load_tokenizer(name: str | None):
    if name is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def encode_text(text: str, tokenizer) -> list[int]:
    if tokenizer is None:
        return list(text.encode("utf-8"))
    return list(tokenizer.encode(text, add_special_tokens=False))


def token_dtype(max_token_id: int) -> tuple[str, object]:
    if max_token_id < 2**16:
        return "uint16", np.uint16
    return "uint32", np.uint32


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio should be in (0, 0.5)")

    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = None if tokenizer is None else tokenizer.eos_token_id
    tokens: list[int] = []
    files = list(iter_files(args.input))
    if not files:
        raise FileNotFoundError("no input files found")

    for text in iter_texts(files, args.jsonl_field):
        ids = encode_text(text, tokenizer)
        if args.append_eos and eos_id is not None:
            ids.append(int(eos_id))
        tokens.extend(ids)
        if args.max_tokens > 0 and len(tokens) >= args.max_tokens:
            tokens = tokens[: args.max_tokens]
            break

    if len(tokens) < 1024:
        raise ValueError(f"too few tokens after encoding: {len(tokens)}")

    rng = random.Random(args.seed)
    split_idx = int(len(tokens) * (1.0 - args.val_ratio))
    if split_idx <= 0 or split_idx >= len(tokens):
        raise ValueError("invalid train/val split")
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    max_id = max(tokens)
    dtype_name, dtype = token_dtype(max_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.asarray(train_tokens, dtype=dtype).tofile(args.output_dir / "train.bin")
    np.asarray(val_tokens, dtype=dtype).tofile(args.output_dir / "val.bin")

    vocab_size = 256 if tokenizer is None else int(getattr(tokenizer, "vocab_size", max_id + 1))
    meta = {
        "kind": "packed_lm",
        "tokenizer": "utf8_byte_level" if tokenizer is None else args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "max_token_id": int(max_id),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "files": [str(path) for path in files],
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "append_eos": args.append_eos,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
