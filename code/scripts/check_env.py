from __future__ import annotations

from importlib import metadata

import torch


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

    packages = {
        "triton": "triton",
        "cupy": "cupy-cuda12x",
        "flashinfer": "flashinfer-python",
        "vllm": "vllm",
    }
    for label, distribution in packages.items():
        print(label, optional_version(distribution))


if __name__ == "__main__":
    main()
