from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import torch

from llm0.config import LLMConfig
from llm0.data import build_dataset, random_token_batch
from llm0.model import LLMForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "debug.json")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4, help="micro batch size per gradient accumulation step")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="skip grad clipping and grad norm logging when <= 0")
    parser.add_argument("--adamw-fused", action="store_true", help="use fused CUDA AdamW when available")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--use-triton-rmsnorm", action="store_true")
    parser.add_argument("--use-triton-swiglu", action="store_true")
    parser.add_argument("--use-fused-classifier-ce", action="store_true")
    parser.add_argument("--fused-ce-chunk-size", type=int, default=8192)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default=None, choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-model-only", action="store_true", help="load model weights from checkpoint but reinitialize optimizer")
    return parser.parse_args(argv)


def checkpoint_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "float32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch_dtype(dtype_name))


def save_checkpoint(
    path: Path,
    model: LLMForCausalLM,
    optimizer: torch.optim.Optimizer,
    config: LLMConfig,
    step: int,
    loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "loss": loss,
            "config": config.to_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "model": checkpoint_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def prune_old_checkpoints(save_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(save_dir.glob("step_*.pt"), key=lambda path: path.stat().st_mtime)
    for old_path in checkpoints[:-keep]:
        old_path.unlink(missing_ok=True)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "max_memory_allocated_mb": 0.0,
            "max_memory_reserved_mb": 0.0,
        }
    return {
        "memory_allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "memory_reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_memory_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def lr_for_step(step: int, args: argparse.Namespace) -> float:
    if args.warmup_steps > 0 and step <= args.warmup_steps:
        return args.lr * step / args.warmup_steps
    if args.min_lr <= 0.0 or args.steps <= args.warmup_steps:
        return args.lr
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return args.min_lr + coeff * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(
    model: LLMForCausalLM,
    dataset,
    batch_size: int,
    device: torch.device,
    eval_iters: int,
    dtype_name: str,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_iters):
        input_ids, targets = dataset.get_batch("val", batch_size, device)
        with autocast_context(device, dtype_name):
            _, loss = model(input_ids, targets)
        if loss is None:
            raise RuntimeError("loss is unexpectedly None")
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    config = LLMConfig.from_file(args.config)
    config.use_triton_rmsnorm = bool(args.use_triton_rmsnorm)
    config.use_triton_swiglu = bool(args.use_triton_swiglu)
    config.use_fused_classifier_ce = bool(args.use_fused_classifier_ce)
    config.fused_ce_chunk_size = int(args.fused_ce_chunk_size)

    device = torch.device(args.device)
    dataset = build_dataset(args.data_dir, config.seq_len) if args.data_dir is not None else None
    model = LLMForCausalLM(config).to(device)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    adamw_kwargs: dict[str, object] = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.adamw_fused and device.type == "cuda":
        adamw_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

    start_step = 0
    last_loss = float("nan")
    resume_tokens = 0
    if args.resume is not None:
        checkpoint_device = "cpu" if args.resume_model_only else device
        checkpoint = torch.load(args.resume, map_location=checkpoint_device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        if not args.resume_model_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        last_loss = float(checkpoint.get("loss", float("nan")))
        old_args = checkpoint.get("args", {})
        old_batch = int(old_args.get("batch_size", args.batch_size))
        old_grad_accum = int(old_args.get("grad_accum_steps", args.grad_accum_steps))
        resume_tokens = start_step * old_batch * old_grad_accum * config.seq_len
        opt_text = "model_only" if args.resume_model_only else "model_and_optimizer"
        print(f"resumed {args.resume} at step={start_step}, inferred_resume_tokens={resume_tokens}, optimizer_resume={opt_text}")
        del checkpoint

    if args.compile:
        compile_mode = None if args.compile_mode in (None, "default") else args.compile_mode
        model = torch.compile(model, mode=compile_mode)  # type: ignore[assignment]

    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.results_dir / run_name
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    global_batch_tokens = args.batch_size * args.grad_accum_steps * config.seq_len
    print(model.init_summary())
    print(
        f"device={device}, dtype={args.dtype}, seq_len={config.seq_len}, micro_batch={args.batch_size}, "
        f"grad_accum={args.grad_accum_steps}, global_tokens_per_step={global_batch_tokens}, "
        f"triton_rmsnorm={config.use_triton_rmsnorm}, triton_swiglu={config.use_triton_swiglu}, "
        f"fused_classifier_ce={config.use_fused_classifier_ce}, fused_ce_chunk={config.fused_ce_chunk_size}, "
        f"activation_checkpointing={args.gradient_checkpointing}"
    )
    if dataset is not None:
        print(
            f"dataset={args.data_dir}, train_tokens={dataset.split_size('train')}, "
            f"val_tokens={dataset.split_size('val')}"
        )
    print(f"metrics={metrics_path}")

    model.train()
    cuda_sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()
    last_val_loss: float | None = None
    total_tokens = resume_tokens

    final_step = start_step

    for step in range(start_step + 1, args.steps + 1):
        final_step = step
        current_lr = lr_for_step(step, args)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        cuda_sync(device)
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            if dataset is None:
                input_ids, targets = random_token_batch(config.vocab_size, args.batch_size, config.seq_len, device)
            else:
                input_ids, targets = dataset.get_batch("train", args.batch_size, device)

            if args.compile and args.compile_mode == "reduce-overhead" and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            with autocast_context(device, args.dtype):
                _, loss = model(input_ids, targets)
            if loss is None:
                raise RuntimeError("loss is unexpectedly None")
            accum_loss += float(loss.detach().cpu())
            (loss / args.grad_accum_steps).backward()

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.tensor(float("nan"), device=device)
        optimizer.step()
        cuda_sync(device)
        step_time_ms = (time.perf_counter() - step_start) * 1000.0

        total_tokens += global_batch_tokens
        elapsed = max(time.perf_counter() - train_start, 1e-6)
        last_loss = accum_loss / args.grad_accum_steps

        if dataset is not None and args.eval_every > 0 and step % args.eval_every == 0:
            last_val_loss = estimate_loss(model, dataset, args.batch_size, device, args.eval_iters, args.dtype)
            cuda_sync(device)

        if args.save_dir is not None and args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(args.save_dir / f"step_{step:06d}.pt", model, optimizer, config, step, last_loss, args)
            prune_old_checkpoints(args.save_dir, args.keep_last_checkpoints)

        if step % args.log_every == 0 or step == args.steps:
            step_tokens_per_sec = global_batch_tokens / max(step_time_ms / 1000.0, 1e-6)
            run_tokens_per_sec = (total_tokens - resume_tokens) / elapsed
            row: dict[str, object] = {
                "step": step,
                "train_loss": last_loss,
                "val_loss": last_val_loss,
                "grad_norm": float(grad_norm),
                "lr": current_lr,
                "step_time_ms": step_time_ms,
                "tokens": total_tokens,
                "run_tokens": total_tokens - resume_tokens,
                "tokens_per_sec": run_tokens_per_sec,
                "step_tokens_per_sec": step_tokens_per_sec,
                "elapsed_sec": elapsed,
                "micro_batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "global_batch_tokens": global_batch_tokens,
                "dtype": args.dtype,
                "gradient_checkpointing": args.gradient_checkpointing,
                "use_triton_rmsnorm": config.use_triton_rmsnorm,
                "use_triton_swiglu": config.use_triton_swiglu,
                "use_fused_classifier_ce": config.use_fused_classifier_ce,
                "fused_ce_chunk_size": config.fused_ce_chunk_size,
                "adamw_fused": bool(args.adamw_fused),
                **memory_stats(device),
            }
            append_jsonl(metrics_path, row)
            val_text = "" if last_val_loss is None else f" val_loss={last_val_loss:.4f}"
            print(
                f"step={step:04d} loss={last_loss:.4f}{val_text} lr={current_lr:.2e} "
                f"grad_norm={float(grad_norm):.4f} step_ms={step_time_ms:.2f} "
                f"tokens_per_sec={step_tokens_per_sec:.1f} "
                f"peak_mem={row['max_memory_allocated_mb']:.1f}MB"
            )


    total_time_sec = time.perf_counter() - train_start
    summary: dict[str, object] = {
        "run_name": run_name,
        "config": str(args.config),
        "data_dir": None if args.data_dir is None else str(args.data_dir),
        "steps": args.steps,
        "final_step": final_step,
        "start_step": start_step,
        "micro_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "global_batch_tokens": global_batch_tokens,
        "seq_len": config.seq_len,
        "resume_tokens": resume_tokens,
        "total_tokens": total_tokens,
        "total_time_sec": total_time_sec,
        "avg_tokens_per_sec": (total_tokens - resume_tokens) / max(total_time_sec, 1e-6),
        "last_train_loss": last_loss,
        "last_val_loss": last_val_loss,
        "dtype": args.dtype,
        "use_triton_rmsnorm": config.use_triton_rmsnorm,
        "use_triton_swiglu": config.use_triton_swiglu,
        "gradient_checkpointing": args.gradient_checkpointing,
        "adamw_fused": bool(args.adamw_fused),
        "grad_clip": args.grad_clip,
        "keep_last_checkpoints": args.keep_last_checkpoints,
        "resume_model_only": bool(args.resume_model_only),
        **memory_stats(device),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_path}")

    if args.save_dir is not None:
        ckpt_path = args.save_dir / "last.pt"
        save_checkpoint(ckpt_path, model, optimizer, config, final_step, last_loss, args)
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
