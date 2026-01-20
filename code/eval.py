from __future__ import annotations

"""
Evaluation script for ModularLlama.

Features:
- Loads a training checkpoint (or raw model weights) and evaluates on a token buffer.
- Computes mean cross-entropy loss and perplexity.
- Optional greedy generation sanity check using incremental `forward_step`.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from llama_macro import ModularLlama
from blocks.registry import available_blocks, get_block_factory


DeviceChoice = Literal["auto", "cpu", "mps", "cuda"]
PrecisionChoice = Literal["auto", "fp32", "bf16", "fp16"]


def _select_device(choice: DeviceChoice) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_precision(device: torch.device, precision: PrecisionChoice) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if device.type == "cuda":
        if precision == "auto":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
    return torch.float32


def _load_token_buffer_pt(path: str) -> torch.Tensor:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"token_buffer_path not found: {p}")
    flat = torch.load(p, map_location="cpu")
    if not isinstance(flat, torch.Tensor) or flat.dim() != 1:
        raise ValueError(f"Expected a 1D torch.Tensor at {p}, got {type(flat)} with shape {getattr(flat,'shape',None)}")
    if flat.dtype not in (torch.int32, torch.int64, torch.int16, torch.uint16, torch.uint8):
        raise ValueError(f"Expected integer dtype token buffer; got {flat.dtype}")
    return flat.to(dtype=torch.int32).contiguous()


def _iter_spans(flat: torch.Tensor, *, batch_size: int, seq_len: int, max_batches: int) -> torch.Tensor:
    """
    Deterministic slicing of spans from the beginning of the buffer.
    Produces (B,T+1) chunks until max_batches is hit.
    """
    span = seq_len + 1
    total = flat.numel()
    step = batch_size * span
    n_batches = min(max_batches, max(0, (total // step)))
    for i in range(n_batches):
        start = i * step
        chunk = flat[start : start + step].view(batch_size, span).to(dtype=torch.long)
        yield chunk


def _load_checkpoint(path: str) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)}")
    return ckpt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ModularLlama on a token buffer.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a train.py checkpoint .pt")
    p.add_argument("--token-buffer-path", type=str, required=True, help="Path to a 1D token buffer tensor (.pt)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--no-grad-checkpoint", action="store_true", help="Disable grad checkpointing (irrelevant in eval).")
    p.add_argument(
        "--block",
        type=str,
        default=None,
        choices=[b.name for b in available_blocks()],
        help="Override block choice instead of using what was saved in the checkpoint config.",
    )
    p.add_argument(
        "--block-kwargs",
        type=str,
        default=None,
        help="Optional JSON object passed to the block factory builder (only if --block is set).",
    )
    p.add_argument("--generate", action="store_true", help="Run a small greedy generation sanity check.")
    p.add_argument("--gen-len", type=int, default=64)
    p.add_argument("--gen-start-token", type=int, default=1, help="Start token id for generation (placeholder).")
    return p.parse_args()


@dataclass
class EvalResult:
    mean_loss: float
    perplexity: float
    tokens: int


@torch.no_grad()
def evaluate(
    *,
    model: ModularLlama,
    flat: torch.Tensor,
    batch_size: int,
    seq_len: int,
    max_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for chunk in _iter_spans(flat, batch_size=batch_size, seq_len=seq_len, max_batches=max_batches):
        x = chunk[:, :-1].to(device, non_blocking=True)
        y = chunk[:, 1:].to(device, non_blocking=True)

        if amp_dtype == torch.float32:
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        else:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        b, t = x.shape
        total_loss += float(loss.item()) * (b * t)
        total_tokens += int(b * t)

    mean_loss = total_loss / max(1, total_tokens)
    ppl = float(math.exp(mean_loss)) if mean_loss < 50 else float("inf")
    return EvalResult(mean_loss=mean_loss, perplexity=ppl, tokens=total_tokens)


@torch.no_grad()
def greedy_generate(
    *,
    model: ModularLlama,
    start_token: int,
    gen_len: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    token_t = torch.tensor([int(start_token)], device=device, dtype=torch.long)
    state = model.init_state(batch_size=1, device=device, dtype=torch.float32, pos=0)
    out: list[int] = [int(start_token)]
    for _ in range(int(gen_len)):
        logits, state = model.forward_step(token_t, state=state)
        next_id = int(torch.argmax(logits[0], dim=-1).item())
        out.append(next_id)
        token_t = torch.tensor([next_id], device=device, dtype=torch.long)
    return out


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)
    amp_dtype = _resolve_precision(device, args.precision)

    ckpt = _load_checkpoint(args.checkpoint)
    cfg = ckpt.get("config", {})

    # Derive model hparams from training config.
    dim = int(cfg.get("dim", 768))
    n_layers = int(cfg.get("n_layers", 12))
    max_seq_len = int(cfg.get("max_seq_len", max(args.seq_len, 2048)))
    vocab_size = int(cfg.get("dry_run_vocab_size", 50257))
    # If trainer inferred vocab_size, prefer that.
    if "vocab_size" in cfg:
        try:
            vocab_size = int(cfg["vocab_size"])
        except Exception:
            pass

    # Block selection: use checkpoint config unless user overrides.
    block_name = str(args.block) if args.block is not None else str(cfg.get("block", "identity"))
    if args.block is not None and args.block_kwargs:
        block_kwargs = json.loads(args.block_kwargs)
    else:
        block_kwargs = cfg.get("block_kwargs") or {}

    model = ModularLlama(
        vocab_size=vocab_size,
        n_layers=n_layers,
        dim=dim,
        max_seq_len=max_seq_len,
        block_factory=get_block_factory(block_name, kwargs=block_kwargs),
        grad_checkpoint=not bool(args.no_grad_checkpoint),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    flat = _load_token_buffer_pt(args.token_buffer_path)
    res = evaluate(
        model=model,
        flat=flat,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        max_batches=int(args.max_batches),
        device=device,
        amp_dtype=amp_dtype,
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "token_buffer_path": str(args.token_buffer_path),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "max_batches": int(args.max_batches),
        "tokens": res.tokens,
        "mean_loss": res.mean_loss,
        "perplexity": res.perplexity,
    }
    print(json.dumps(summary, indent=2))

    if args.generate:
        ids = greedy_generate(model=model, start_token=int(args.gen_start_token), gen_len=int(args.gen_len), device=device)
        print(f"greedy_ids: {ids[:16]} ... (len={len(ids)})")


if __name__ == "__main__":
    main()

