from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from llm0.config import LLMConfig
from llm0.model import LLMForCausalLM


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="optional config override; checkpoint config is used by default")
    parser.add_argument("--tokenizer", type=str, default=None, help="HF tokenizer name/path; omit for UTF-8 byte tokenizer")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system", type=str, default="你是一个有帮助的中文助手。")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def load_tokenizer(name: str | None):
    if name is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def format_prompt(prompt: str, system: str, chat: bool) -> str:
    if not chat:
        return prompt
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def encode(text: str, tokenizer) -> list[int]:
    if tokenizer is None:
        return list(text.encode("utf-8"))
    return list(tokenizer.encode(text, add_special_tokens=False))


def decode(ids: list[int], tokenizer) -> str:
    if tokenizer is None:
        return bytes([x for x in ids if 0 <= x < 256]).decode("utf-8", errors="ignore")
    return tokenizer.decode(ids, skip_special_tokens=False)


def strip_compiled_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state):
        return state
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = LLMConfig.from_file(args.config) if args.config is not None else LLMConfig(**ckpt["config"])
    tokenizer = load_tokenizer(args.tokenizer)

    model = LLMForCausalLM(config).to(device)
    model.load_state_dict(strip_compiled_prefix(ckpt["model"]))
    model.eval()

    prompt_text = format_prompt(args.prompt, args.system, args.chat)
    input_ids = torch.tensor([encode(prompt_text, tokenizer)], dtype=torch.long, device=device)
    eos_token_id = None if tokenizer is None else tokenizer.eos_token_id
    output = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=eos_token_id,
    )
    print(decode(output[0].tolist(), tokenizer))


if __name__ == "__main__":
    main()
