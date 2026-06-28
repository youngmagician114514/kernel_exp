from __future__ import annotations

from importlib import metadata
from pathlib import Path

import torch

from llm0.config import LLMConfig


ROOT = Path(__file__).resolve().parents[1]


def optional_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def main() -> None:
    print("torch", torch.__version__)
    print("torch cuda", torch.version.cuda)
    print("cuda available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            print("device", idx, torch.cuda.get_device_name(idx), torch.cuda.get_device_capability(idx))
    print("triton", optional_version("triton"))
    print("transformers", optional_version("transformers"))
    print("datasets", optional_version("datasets"))
    print("trl", optional_version("trl"))
    print("peft", optional_version("peft"))

    for name in [
        "byte_debug",
        "debug",
        "medium_debug",
        "llm_0p2b",
        "llm_0p2b_1k_3090",
        "llm_0p2b_2k",
        "llm_0p2b_4k",
    ]:
        cfg = LLMConfig.from_file(ROOT / "configs" / f"{name}.json")
        params = cfg.estimate_parameters()
        print(
            f"config {name}: params={params / 1e6:.2f}M, "
            f"layers={cfg.n_layer}, hidden={cfg.d_model}, heads={cfg.n_head}/{cfg.n_kv_head}, "
            f"seq_len={cfg.seq_len}, vocab={cfg.vocab_size}"
        )


if __name__ == "__main__":
    main()
