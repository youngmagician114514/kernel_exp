from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np


IGNORE_INDEX = -100


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack chat SFT JSONL into response-masked memmaps.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with messages/conversations or instruction/output rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer name/path; omit for UTF-8 byte tokenizer")
    parser.add_argument("--system", type=str, default="你是一个有帮助的中文助手。")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0)
    return parser.parse_args(argv)


def load_tokenizer(name: str | None):
    if name is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def encode(text: str, tokenizer) -> list[int]:
    if tokenizer is None:
        return list(text.encode("utf-8"))
    return list(tokenizer.encode(text, add_special_tokens=False))


def normalize_messages(row: dict[str, Any], default_system: str) -> list[dict[str, str]]:
    raw = row.get("messages") or row.get("conversations")
    if raw is not None:
        messages: list[dict[str, str]] = []
        for item in raw:
            role = item.get("role", item.get("from", "user"))
            content = item.get("content", item.get("value", ""))
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant", "bot"}:
                role = "assistant"
            elif role != "system":
                role = "user"
            messages.append({"role": role, "content": str(content)})
        return messages

    instruction = str(row.get("instruction", row.get("prompt", ""))).strip()
    extra_input = str(row.get("input", "")).strip()
    output = row.get("output", row.get("response", row.get("answer", "")))
    user_content = instruction if not extra_input else f"{instruction}\n{extra_input}"
    return [
        {"role": "system", "content": default_system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": str(output)},
    ]


def append_segment(input_ids: list[int], labels: list[int], ids: list[int], label_ids: list[int] | None) -> None:
    input_ids.extend(ids)
    if label_ids is None:
        labels.extend([IGNORE_INDEX] * len(ids))
    else:
        labels.extend(label_ids)


def encode_conversation(messages: list[dict[str, str]], tokenizer) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        header = f"<|im_start|>{role}\n"
        footer = "<|im_end|>\n"
        if role == "assistant":
            header_ids = encode(header, tokenizer)
            answer_ids = encode(content + footer, tokenizer)
            append_segment(input_ids, labels, header_ids, None)
            append_segment(input_ids, labels, answer_ids, answer_ids)
        else:
            segment_ids = encode(header + content + footer, tokenizer)
            append_segment(input_ids, labels, segment_ids, None)
    return input_ids, labels


def read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def token_dtype(max_token_id: int) -> tuple[str, object]:
    if max_token_id < 2**16:
        return "uint16", np.uint16
    return "uint32", np.uint32


def write_split(output_dir: Path, split: str, examples: list[tuple[list[int], list[int]]], dtype) -> tuple[int, int]:
    input_ids: list[int] = []
    labels: list[int] = []
    for ids, lbl in examples:
        input_ids.extend(ids)
        labels.extend(lbl)
    np.asarray(input_ids, dtype=dtype).tofile(output_dir / f"{split}_input_ids.bin")
    np.asarray(labels, dtype=np.int32).tofile(output_dir / f"{split}_labels.bin")
    return len(input_ids), sum(1 for item in labels if item != IGNORE_INDEX)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer)
    rows = read_rows(args.input, args.max_examples)
    if len(rows) < 2:
        raise ValueError("need at least two SFT examples to create train/val splits")

    examples = [encode_conversation(normalize_messages(row, args.system), tokenizer) for row in rows]
    examples = [(ids, labels) for ids, labels in examples if len(ids) > 0 and any(x != IGNORE_INDEX for x in labels)]
    if len(examples) < 2:
        raise ValueError("no usable assistant-labeled examples found")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    val_count = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:val_count]
    train_examples = examples[val_count:]
    max_id = max(max(ids) for ids, _ in examples)
    dtype_name, dtype = token_dtype(max_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_tokens, train_label_tokens = write_split(args.output_dir, "train", train_examples, dtype)
    val_tokens, val_label_tokens = write_split(args.output_dir, "val", val_examples, dtype)

    vocab_size = 256 if tokenizer is None else int(getattr(tokenizer, "vocab_size", max_id + 1))
    meta = {
        "kind": "sft_response_masked",
        "tokenizer": "utf8_byte_level" if tokenizer is None else args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "label_dtype": "int32",
        "ignore_index": IGNORE_INDEX,
        "train_tokens": train_tokens,
        "train_labeled_tokens": train_label_tokens,
        "val_tokens": val_tokens,
        "val_labeled_tokens": val_label_tokens,
        "examples": len(examples),
        "val_examples": len(val_examples),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
