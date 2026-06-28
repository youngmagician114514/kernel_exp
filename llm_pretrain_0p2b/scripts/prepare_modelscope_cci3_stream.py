from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from array import array
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer


TREE_URL = "https://modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo/tree"
FILE_URL = "https://modelscope.cn/api/v1/datasets/BAAI/CCI3-HQ/repo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream BAAI/CCI3-HQ from ModelScope into train.bin/val.bin.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", type=str, default="master")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--start-file-index", type=int, default=0)
    parser.add_argument("--flush-tokens", type=int, default=1_000_000)
    parser.add_argument("--log-every-docs", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=0)
    return parser.parse_args()


def fetch_json(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def iter_file_paths(revision: str, page_size: int) -> Iterable[tuple[str, int]]:
    page_number = 1
    while True:
        query = urllib.parse.urlencode(
            {
                "Revision": revision,
                "Root": "/",
                "Recursive": "True",
                "PageNumber": page_number,
                "PageSize": page_size,
            }
        )
        payload = fetch_json(f"{TREE_URL}?{query}")
        files = ((payload.get("Data") or {}).get("Files") or [])
        if not files:
            break
        for item in files:
            path = item.get("Path") or item.get("Name")
            size = int(item.get("Size") or 0)
            if item.get("Type") == "blob" and path and path.endswith(".jsonl"):
                yield path, size
        total = payload.get("TotalCount")
        if total is not None and page_number * page_size >= int(total):
            break
        page_number += 1


def file_url(path: str, revision: str) -> str:
    query = urllib.parse.urlencode(
        {
            "Source": "SDK",
            "Revision": revision,
            "FilePath": path,
            "View": "False",
        }
    )
    return f"{FILE_URL}?{query}"


def flush(path: Path, values: list[int], dtype_code: str) -> None:
    if not values:
        return
    with path.open("ab") as handle:
        array(dtype_code, values).tofile(handle)
    values.clear()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_ratio < 0.5:
        raise ValueError("--val-ratio must be in (0, 0.5)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.bin"
    val_path = args.output_dir / "val.bin"
    if train_path.exists() or val_path.exists():
        raise FileExistsError(f"{args.output_dir} already has train.bin/val.bin; choose a new output dir")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    if int(getattr(tokenizer, "vocab_size", 0)) >= 2**16:
        dtype_name = "uint32"
        dtype_code = "I"
    else:
        dtype_name = "uint16"
        dtype_code = "H"

    rng = random.Random(args.seed)
    files = list(iter_file_paths(args.revision, args.page_size))
    files = files[args.start_file_index :]
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError("no CCI3-HQ jsonl shards found")

    train_buffer: list[int] = []
    val_buffer: list[int] = []
    train_tokens = 0
    val_tokens = 0
    total_tokens = 0
    docs = 0
    started = time.time()
    used_files: list[str] = []

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for file_idx, (path, size) in enumerate(files, start=args.start_file_index):
            if total_tokens >= args.target_tokens:
                break
            used_files.append(path)
            shard_start = time.time()
            shard_docs = 0
            shard_tokens = 0
            print(f"[file] index={file_idx} path={path} size={size}", flush=True)
            with urllib.request.urlopen(file_url(path, args.revision), timeout=120) as response:
                for raw_line in response:
                    if total_tokens >= args.target_tokens:
                        break
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    text = str(row.get("text") or "")
                    if not text:
                        continue
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if not ids:
                        continue
                    ids.append(int(eos_id))
                    remaining = args.target_tokens - total_tokens
                    if len(ids) > remaining:
                        ids = ids[:remaining]
                    if rng.random() < args.val_ratio:
                        val_buffer.extend(ids)
                        val_tokens += len(ids)
                    else:
                        train_buffer.extend(ids)
                        train_tokens += len(ids)
                    total_tokens += len(ids)
                    shard_tokens += len(ids)
                    docs += 1
                    shard_docs += 1
                    if len(train_buffer) >= args.flush_tokens:
                        flush(train_path, train_buffer, dtype_code)
                    if len(val_buffer) >= max(1, args.flush_tokens // 10):
                        flush(val_path, val_buffer, dtype_code)
                    if args.log_every_docs > 0 and docs % args.log_every_docs == 0:
                        elapsed = max(time.time() - started, 1e-6)
                        print(
                            f"[progress] docs={docs} tokens={total_tokens} "
                            f"train={train_tokens} val={val_tokens} tok_per_sec={total_tokens / elapsed:.1f}",
                            flush=True,
                        )
            manifest.write(
                json.dumps(
                    {
                        "index": file_idx,
                        "path": path,
                        "size": size,
                        "docs": shard_docs,
                        "tokens": shard_tokens,
                        "elapsed_sec": time.time() - shard_start,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest.flush()

    flush(train_path, train_buffer, dtype_code)
    flush(val_path, val_buffer, dtype_code)

    meta = {
        "kind": "packed_lm",
        "source": "modelscope://BAAI/CCI3-HQ",
        "revision": args.revision,
        "tokenizer": args.tokenizer,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "dtype": dtype_name,
        "max_token_id": int(getattr(tokenizer, "vocab_size", 0)) - 1,
        "target_tokens": args.target_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": total_tokens,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "files": used_files,
        "elapsed_sec": time.time() - started,
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

