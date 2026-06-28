from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run inside the cloned vllm_srv environment.")
    return torch.device("cuda")


def benchmark_cuda(
    name: str,
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    repeat: int = 50,
) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    return BenchmarkResult(
        name=name,
        mean_ms=statistics.mean(samples),
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / (ms * 1e9)


def print_result(result: BenchmarkResult) -> None:
    print(
        f"{result.name:18s} mean={result.mean_ms:8.4f} ms "
        f"median={result.median_ms:8.4f} ms min={result.min_ms:8.4f} ms max={result.max_ms:8.4f} ms"
    )

