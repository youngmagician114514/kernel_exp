from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np


DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--out-dir", type=Path, default=Path("llm_pretrain_0p2b/data/tiny_shakespeare"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"downloading {args.url}")
    with urlopen(args.url, timeout=30) as response:
        text_bytes = response.read()

    raw_path = raw_dir / "input.txt"
    raw_path.write_bytes(text_bytes)

    tokens = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.uint16)
    split = int(len(tokens) * (1.0 - args.val_ratio))
    train = tokens[:split]
    val = tokens[split:]
    train.tofile(args.out_dir / "train.bin")
    val.tofile(args.out_dir / "val.bin")

    meta = {
        "source_url": args.url,
        "tokenizer": "utf8_byte_level",
        "vocab_size": 256,
        "dtype": "uint16",
        "raw_bytes": len(text_bytes),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "val_ratio": args.val_ratio,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
