from __future__ import annotations

import argparse

import torch

from kernel_exp_project.online_softmax import max_abs_error, online_softmax
from kernel_exp_project.utils import benchmark_cuda, print_result, require_cuda


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    require_cuda()
    torch.manual_seed(0)
    x = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.float16)

    err = max_abs_error(x, block_size=args.block_size)
    torch_result = benchmark_cuda("torch.softmax", lambda: torch.softmax(x.float(), dim=-1), repeat=args.repeat)
    online_result = benchmark_cuda(
        "online_softmax",
        lambda: online_softmax(x, block_size=args.block_size),
        repeat=args.repeat,
    )

    print(f"shape={tuple(x.shape)}, block_size={args.block_size}")
    print_result(torch_result)
    print_result(online_result)
    print(f"max error {err:.6f}")


if __name__ == "__main__":
    main()

