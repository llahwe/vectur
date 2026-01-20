from __future__ import annotations

"""
Efficient, parameterized PyTorch training script for ModularLlama.

Key features:
- Token-buffer training for speed (sample contiguous spans from a flat token tensor).
- Optional HuggingFace streaming buffer builder (requires `datasets` + `transformers` and network).
- Grad accumulation, mixed precision (CUDA bf16/fp16), gradient clipping.
- Activation checkpointing per layer (via ModularLlama.grad_checkpoint).
- Checkpointing/resume with RNG state.

This script can train `ModularLlama` with any registered sequence block via `--block`.
"""

import argparse
import dataclasses
import datetime as dt
import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from llama_macro import ModularLlama
from blocks.registry import available_blocks, get_block_factory


ModelSize = Literal["s", "m", "l"]
DeviceChoice = Literal["auto", "cpu", "mps", "cuda"]
PrecisionChoice = Literal["auto", "fp32", "bf16", "fp16"]
DataSource = Literal["random", "hf_buffer", "pt_buffer"]

_SIZE_TARGET_PARAMS: dict[ModelSize, int] = {
    # Target TOTAL parameter counts (roughly, depends on vocab size).
    # These are intended to be "Llama-like" scale buckets for fair comparisons across sequence blocks.
    "s": 100_000_000,
    "m": 500_000_000,
    "l": 1_000_000_000,
}


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(choice: DeviceChoice) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    # Prefer MPS on Mac, else CUDA, else CPU.
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _count_params(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
    return sum(int(p.numel()) for p in module.parameters())


def _param_breakdown(model: ModularLlama) -> dict[str, Any]:
    """
    Produce a parameter breakdown for logging.

    This assumes `ModularLlama.layers[i].sequence_block` exists (true for current macro).
    """
    total = _count_params(model, trainable_only=False)
    trainable = _count_params(model, trainable_only=True)

    # Per-block params (each layer has its own instance).
    block_per_layer = [_count_params(layer.sequence_block, trainable_only=False) for layer in model.layers]
    block_total = int(sum(block_per_layer))

    # Other macro components.
    mlp_total = int(sum(_count_params(layer.mlp, trainable_only=False) for layer in model.layers))
    norms_total = int(
        sum(
            _count_params(layer.attention_norm, trainable_only=False) + _count_params(layer.ffn_norm, trainable_only=False)
            for layer in model.layers
        )
        + _count_params(model.norm, trainable_only=False)
    )
    emb_total = _count_params(model.tok_embeddings, trainable_only=False)
    out_total = _count_params(model.output, trainable_only=False)

    remainder = int(total - (block_total + mlp_total + norms_total + emb_total + out_total))

    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "embeddings_params": int(emb_total),
        "output_head_params": int(out_total),
        "layers": {
            "n_layers": int(len(model.layers)),
            "sequence_block_total_params": int(block_total),
            "sequence_block_params_per_layer": block_per_layer,
            "mlp_total_params": int(mlp_total),
            "norms_total_params": int(norms_total),
        },
        "remainder_params": int(remainder),
    }


def _preset_dim_from_size(model_size: ModelSize) -> int:
    """
    Width presets chosen to be Tensor Core friendly (multiples of 128) and to land near:
    - s: ~100M params
    - m: ~500M params
    - l: ~1B params

    Note: exact totals depend on vocab size and block choice; we solve n_layers later.
    """
    if model_size == "s":
        return 512
    if model_size == "m":
        return 1024
    if model_size == "l":
        return 1536
    raise ValueError(f"Unknown model_size: {model_size}")


def _default_block_kwargs_for_size(*, block: str, size: ModelSize, dim: int) -> dict[str, Any]:
    """
    Reasonable GPU/VRAM-friendly defaults per block type when user doesn't specify --block-kwargs.
    These are *not* meant to be exhaustive; users can always override via JSON.
    """
    _ = size
    _ = dim
    if block == "attention":
        # Prefer standard head_dim=64 (good Flash/SDPA behavior) and let the block auto-pick attn_dim
        # to match VecTur's per-layer parameter count at this dim.
        return {"head_dim": 64, "bias": False, "attn_dropout": 0.0, "proj_dropout": 0.0}
    if block == "lstm":
        # Keep the block-internal stack small; the macro already provides depth.
        return {"num_layers": 1, "dropout": 0.0, "bias": True, "proj_to_dim": True}
    if block == "moneta":
        # Chunk inner loop to keep activation memory bounded for long sequences.
        # Defaults already do this, but we make it explicit here.
        return {"chunk_size": 256, "tbptt_horizon_chunks": 4, "grad_checkpoint_inner": True}
    if block == "ntm":
        # NTM memory width/slots are the main VRAM/compute levers; keep them modest and enable
        # chunk+checkpoint for long sequences.
        # (mem_width defaults to min(dim, 128) in the block if omitted.)
        return {
            "mem_slots": 128,
            "n_read_heads": 1,
            "n_write_heads": 1,
            "shift_range": 1,
            "learned_init_memory": True,
            "chunk_size": 128,
            "grad_checkpoint_inner": True,
            "tbptt_horizon_chunks": 4,
        }
    if block == "vectur" or block == "vecstur":
        # Keep paper-default block hyperparams. (t_max affects compute, not params.)
        return {"k": 8, "t_max": 4, "expansion": 4}
    if block.startswith("identity"):
        return {}
    # Fallback for unknown/experimental blocks.
    return {}


def _estimate_total_params(
    *,
    vocab_size: int,
    dim: int,
    n_layers: int,
    hidden_dim: int,
    block_params_per_layer: int,
) -> int:
    """
    Estimate total parameters without instantiating the full model.

    We do instantiate ONE sequence block to get exact block param count at this dim (block-dependent),
    but avoid building a huge full model just to size it.
    """
    d = int(dim)
    v = int(vocab_size)
    L = int(n_layers)
    hd = int(hidden_dim)

    # Embeddings + output head (both are weight-only matrices in llama_macro.py).
    emb_out = 2 * v * d
    # Final norm weight.
    final_norm = d
    # Per-layer norms: RMSNorm weights (attention_norm + ffn_norm).
    norms_per_layer = 2 * d
    # Per-layer SwiGLU MLP params: w13 (D -> 2H) + w2 (H -> D) == 3*D*H
    mlp_per_layer = 3 * d * hd

    return int(emb_out + final_norm + L * (norms_per_layer + mlp_per_layer + int(block_params_per_layer)))


def _solve_n_layers_for_target(
    *,
    target_params: int,
    vocab_size: int,
    dim: int,
    hidden_dim: int,
    block: str,
    block_kwargs: dict[str, Any],
    min_layers: int = 1,
    max_layers: int = 256,
) -> int:
    """
    Pick n_layers so total params are close to the target for this block.
    """
    d = int(dim)
    v = int(vocab_size)
    hd = int(hidden_dim)
    target = int(target_params)

    # Base (non-layer) params:
    base = int(2 * v * d + d)

    blk = get_block_factory(block, kwargs=block_kwargs)(d)
    block_per_layer = int(_count_params(blk, trainable_only=False))
    per_layer = int(2 * d + 3 * d * hd + block_per_layer)

    if per_layer <= 0:
        return int(min_layers)

    # Initial guess, then clamp.
    L0 = int(round((target - base) / float(per_layer)))
    L0 = int(max(min_layers, min(max_layers, L0)))

    # Check a small neighborhood to compensate for rounding.
    best_L = L0
    best_err = float("inf")
    for L in range(max(min_layers, L0 - 4), min(max_layers, L0 + 4) + 1):
        total = _estimate_total_params(vocab_size=vocab_size, dim=d, n_layers=L, hidden_dim=hd, block_params_per_layer=block_per_layer)
        err = abs(total - target)
        if err < best_err:
            best_err = err
            best_L = L
    return int(best_L)


def _resolve_precision(device: torch.device, precision: PrecisionChoice) -> tuple[torch.dtype, bool]:
    """
    Returns (amp_dtype, use_grad_scaler).
    - For bf16: no grad scaler.
    - For fp16: use grad scaler on CUDA.
    """
    if precision == "fp32":
        return torch.float32, False

    if device.type == "cuda":
        if precision == "auto":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, False
            return torch.float16, True
        if precision == "bf16":
            return torch.bfloat16, False
        if precision == "fp16":
            return torch.float16, True

    # MPS/CPU: keep it stable.
    if precision in ("auto", "bf16", "fp16"):
        return torch.float32, False

    raise ValueError(f"Unknown precision: {precision}")


@dataclass(frozen=True)
class TrainConfig:
    # Run
    run_name: str | None = None
    runs_dir: str = "runs"
    seed: int = 1337

    # Model
    model_size: ModelSize = "s"
    dim: int = 768
    n_layers: int = 12
    hidden_dim: int | None = None  # defaults to 4*dim
    block: str = "identity"
    block_kwargs: dict[str, Any] | None = None

    # Sequence
    seq_len: int = 1024
    max_seq_len: int = 4096

    # Data
    data_source: DataSource = "pt_buffer"
    # pt_buffer: path to a saved int32/long 1D tensor of token ids
    token_buffer_path: str = "token_buffer.pt"
    # hf_buffer: HuggingFace dataset tokenizer settings
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    dataset_split: str = "train"
    tokenizer_name: str = "gpt2"
    buffer_tokens: int = 2_000_000

    # Training
    microbatch_size: int = 1
    grad_accum_steps: int = 16
    global_batch_size: int | None = None
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    # Runtime
    device: DeviceChoice = "auto"
    precision: PrecisionChoice = "auto"
    compile: bool = False
    grad_checkpoint: bool = True
    tf32: bool = True
    max_steps: int = 0  # 0 => no limit
    max_time_seconds: int = 3600
    log_every_steps: int = 20

    # Checkpointing
    ckpt_every_seconds: int = 300
    ckpt_keep_last: int = 8
    ckpt_prune: bool = True
    resume: str | None = None

    # Dry run
    dry_run: bool = False
    dry_run_steps: int = 2
    dry_run_seq_len: int = 128
    dry_run_microbatch_size: int = 2
    dry_run_vocab_size: int = 50257


def _build_or_load_token_buffer_hf(*, cfg: TrainConfig, run_dir: Path) -> tuple[torch.Tensor, Any]:
    """
    Builds a flat token buffer ONCE from a streaming HF dataset, saves it, and returns it.
    Requires: `datasets`, `transformers`, and network access.
    """
    from datasets import load_dataset  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    buffer_path = run_dir / "token_buffer.pt"
    meta_path = run_dir / "token_buffer_meta.json"

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    tokenizer.model_max_length = 10**9
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if buffer_path.exists() and meta_path.exists():
        flat = torch.load(buffer_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("buffer_tokens", -1)) != int(flat.numel()):
            raise RuntimeError(
                f"Token buffer mismatch: meta says {meta.get('buffer_tokens')} but tensor has {flat.numel()} tokens"
            )
        return flat, tokenizer

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split, streaming=True)
    needed = int(cfg.buffer_tokens)
    pieces: list[torch.Tensor] = []
    total = 0
    started = time.time()

    for ex in ds:
        text = ex.get("text")
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if ids.numel() == 0:
            continue
        pieces.append(ids)
        total += int(ids.numel())
        if total >= needed:
            break

    if total < needed:
        raise RuntimeError(f"Could only collect {total} tokens, need {needed}.")

    flat = torch.cat(pieces, dim=0)[:needed].to(dtype=torch.int32).contiguous()
    torch.save(flat, buffer_path)
    _write_json(
        meta_path,
        {
            "dataset_name": cfg.dataset_name,
            "dataset_config": cfg.dataset_config,
            "dataset_split": cfg.dataset_split,
            "tokenizer_name": cfg.tokenizer_name,
            "buffer_tokens": int(flat.numel()),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "build_seconds": time.time() - started,
        },
    )
    return flat, tokenizer


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


def _sample_batch_from_flat(
    flat: torch.Tensor, *, microbatch_size: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized sampling of contiguous spans from a flat token buffer.

    Returns:
      x: (B, T) input ids
      y: (B, T) next-token targets
    """
    span = seq_len + 1
    max_start = flat.numel() - span - 1
    if max_start <= 0:
        raise ValueError("Token buffer is too small for requested seq_len.")

    starts = torch.randint(0, max_start, (microbatch_size,), device="cpu")
    offsets = torch.arange(span, device="cpu").view(1, span)
    idx = starts.view(microbatch_size, 1) + offsets  # (B, span)
    chunk = flat[idx].to(dtype=torch.long)  # (B, span)
    x = chunk[:, :-1].to(device, non_blocking=True)
    y = chunk[:, 1:].to(device, non_blocking=True)
    return x, y


def _make_random_batch(
    *, microbatch_size: int, seq_len: int, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (microbatch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(0, vocab_size, (microbatch_size, seq_len), device=device, dtype=torch.long)
    return x, y


def _checkpoint(
    *,
    ckpt_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    cfg: TrainConfig,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"

    payload: dict[str, Any] = {
        "step": step,
        "config": dataclasses.asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "saved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    torch.save(payload, path)
    torch.save(payload, ckpt_dir / "latest.pt")
    return path


def _prune_checkpoints(*, ckpt_dir: Path, keep_last: int) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_re = re.compile(r"^step_(\d{8})\.pt$")
    steps: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("step_*.pt"):
        m = step_re.match(p.name)
        if m:
            steps.append((int(m.group(1)), p))
    steps.sort(key=lambda t: t[0])

    preserve = set(s for s, _ in steps[-keep_last:])
    for s, p in steps:
        if s in preserve:
            continue
        if p.name == "latest.pt":
            continue
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def _try_get_git_metadata() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parent
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root).decode()
        dirty = bool(status.strip())
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root).decode().strip()
        return {"git_commit": commit, "git_dirty": dirty, "git_branch": branch}
    except Exception:
        return {}


def _try_resume(
    resume_path: Path,
    *,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> int:
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    if "rng" in ckpt:
        rng = ckpt["rng"]
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    model.to(device)
    return int(ckpt.get("step", 0))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ModularLlama (macro architecture) with swappable blocks.")

    # Run
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=1337)

    # Model
    p.add_argument("--model-size", type=str, default="s", choices=["s", "m", "l"])
    p.add_argument("--dim", type=int, default=None, help="Override dim for custom model.")
    p.add_argument("--n-layers", type=int, default=None, help="Override n_layers for custom model.")
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument(
        "--block",
        type=str,
        default="identity",
        choices=[b.name for b in available_blocks()],
        help="Which sequence block to use inside each layer (stub registry).",
    )
    p.add_argument(
        "--block-kwargs",
        type=str,
        default=None,
        help="Optional JSON object passed to the block factory builder, e.g. '{\"use_linear\": true}'.",
    )

    # Sequence
    p.add_argument("--seq-len", type=int, default=1024)

    # Data
    p.add_argument("--data-source", type=str, default="pt_buffer", choices=["random", "hf_buffer", "pt_buffer"])
    p.add_argument("--token-buffer-path", type=str, default="token_buffer.pt")
    p.add_argument("--dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", type=str, default="sample-10BT")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--tokenizer-name", type=str, default="gpt2")
    p.add_argument("--buffer-tokens", type=int, default=2_000_000)

    # Training
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)

    # Runtime
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-time-seconds", type=int, default=3600)
    p.add_argument("--log-every-steps", type=int, default=20)

    # Checkpointing/resume
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--ckpt-every-seconds", type=int, default=300)
    p.add_argument("--ckpt-keep-last", type=int, default=8)
    p.add_argument("--no-ckpt-prune", action="store_true")

    # Dry-run
    p.add_argument("--dry-run", action="store_true", help="Short offline sanity run and exit.")
    p.add_argument("--dry-run-steps", type=int, default=2)
    p.add_argument("--dry-run-seq-len", type=int, default=128)
    p.add_argument("--dry-run-microbatch-size", type=int, default=2)
    p.add_argument("--dry-run-vocab-size", type=int, default=50257)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    base_dim = _preset_dim_from_size(args.model_size)
    dim = int(args.dim) if args.dim is not None else int(base_dim)
    # n_layers is finalized later (after we know vocab_size and have finalized block_kwargs).
    n_layers = int(args.n_layers) if args.n_layers is not None else 1

    cfg = TrainConfig(
        run_name=args.run_name,
        runs_dir=args.runs_dir,
        seed=int(args.seed),
        model_size=args.model_size,
        dim=dim,
        n_layers=n_layers,
        hidden_dim=(int(args.hidden_dim) if args.hidden_dim is not None else None),
        block=str(args.block),
        block_kwargs=(json.loads(args.block_kwargs) if args.block_kwargs else None),
        seq_len=int(args.seq_len),
        max_seq_len=int(args.max_seq_len),
        data_source=args.data_source,
        token_buffer_path=args.token_buffer_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        tokenizer_name=args.tokenizer_name,
        buffer_tokens=int(args.buffer_tokens),
        microbatch_size=int(args.microbatch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        global_batch_size=(int(args.global_batch_size) if args.global_batch_size is not None else None),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip_norm=float(args.grad_clip_norm),
        device=args.device,
        precision=args.precision,
        compile=bool(args.compile),
        grad_checkpoint=not bool(args.no_grad_checkpoint),
        tf32=not bool(args.no_tf32),
        max_steps=int(args.max_steps),
        max_time_seconds=int(args.max_time_seconds),
        log_every_steps=int(args.log_every_steps),
        resume=args.resume,
        ckpt_every_seconds=int(args.ckpt_every_seconds),
        ckpt_keep_last=int(args.ckpt_keep_last),
        ckpt_prune=not bool(args.no_ckpt_prune),
        dry_run=bool(args.dry_run),
        dry_run_steps=int(args.dry_run_steps),
        dry_run_seq_len=int(args.dry_run_seq_len),
        dry_run_microbatch_size=int(args.dry_run_microbatch_size),
        dry_run_vocab_size=int(args.dry_run_vocab_size),
    )

    if cfg.max_seq_len < cfg.seq_len and not cfg.dry_run:
        raise ValueError(f"max_seq_len must be >= seq_len, got {cfg.max_seq_len} < {cfg.seq_len}")

    device = _select_device(cfg.device)
    _seed_all(cfg.seed)

    # If user specifies global_batch_size, compute grad_accum_steps from it.
    if cfg.global_batch_size is not None:
        if cfg.global_batch_size <= 0:
            raise ValueError(f"global_batch_size must be > 0, got {cfg.global_batch_size}")
        mb = int(cfg.dry_run_microbatch_size if cfg.dry_run else cfg.microbatch_size)
        if cfg.global_batch_size % mb != 0:
            raise ValueError(
                f"global_batch_size must be divisible by microbatch_size "
                f"(got global_batch_size={cfg.global_batch_size}, microbatch_size={mb})"
            )
        object.__setattr__(cfg, "grad_accum_steps", cfg.global_batch_size // mb)  # type: ignore[misc]

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    amp_dtype, use_scaler = _resolve_precision(device, cfg.precision)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=bool(use_scaler and device.type == "cuda"))
        if hasattr(torch, "amp")
        else torch.cuda.amp.GradScaler(enabled=bool(use_scaler and device.type == "cuda"))
    )

    run_name = cfg.run_name or _now_tag()
    run_dir = Path(cfg.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"

    # Data
    tokenizer = None
    flat = None
    if cfg.dry_run or cfg.data_source == "random":
        vocab_size = int(cfg.dry_run_vocab_size)
    elif cfg.data_source == "hf_buffer":
        flat, tokenizer = _build_or_load_token_buffer_hf(cfg=cfg, run_dir=run_dir)
        # tokenizer may report a larger embedding size than actual ids used; still a good default.
        vocab_size = int(getattr(tokenizer, "vocab_size", cfg.dry_run_vocab_size))
    elif cfg.data_source == "pt_buffer":
        flat = _load_token_buffer_pt(cfg.token_buffer_path)
        vocab_size = int(flat.max().item()) + 1
    else:
        raise ValueError(f"Unknown data_source: {cfg.data_source}")

    eff_mb = int(cfg.dry_run_microbatch_size if cfg.dry_run else cfg.microbatch_size)
    eff_global_bs = int(eff_mb * cfg.grad_accum_steps)

    # Finalize model dims/depth now that vocab_size is known.
    merged_block_kwargs = _default_block_kwargs_for_size(block=cfg.block, size=cfg.model_size, dim=cfg.dim)
    if cfg.block_kwargs:
        merged_block_kwargs.update(cfg.block_kwargs)
    object.__setattr__(cfg, "block_kwargs", merged_block_kwargs)  # type: ignore[misc]

    if args.hidden_dim is None:
        object.__setattr__(cfg, "hidden_dim", int(4 * cfg.dim))  # type: ignore[misc]

    if args.n_layers is None:
        target = int(_SIZE_TARGET_PARAMS[cfg.model_size])
        solved_layers = _solve_n_layers_for_target(
            target_params=target,
            vocab_size=vocab_size,
            dim=cfg.dim,
            hidden_dim=int(cfg.hidden_dim or 4 * cfg.dim),
            block=cfg.block,
            block_kwargs=cfg.block_kwargs or {},
            min_layers=1,
            max_layers=256,
        )
        object.__setattr__(cfg, "n_layers", int(solved_layers))  # type: ignore[misc]

    model = ModularLlama(
        vocab_size=vocab_size,
        n_layers=cfg.n_layers,
        dim=cfg.dim,
        max_seq_len=max(cfg.max_seq_len, cfg.seq_len),
        hidden_dim=cfg.hidden_dim,
        block_factory=get_block_factory(cfg.block, kwargs=cfg.block_kwargs or {}),
        grad_checkpoint=bool(cfg.grad_checkpoint),
    ).to(device)

    if cfg.compile:
        # Compilation cost amortizes over longer runs.
        model = torch.compile(model)  # type: ignore[assignment]

    _write_json(
        run_dir / "config.json",
        dataclasses.asdict(cfg)
        | {
            "selected_device": str(device),
            "vocab_size": vocab_size,
            "effective_microbatch_size": eff_mb,
            "effective_global_batch_size": eff_global_bs,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(torch.backends.mps.is_available() and torch.backends.mps.is_built()),
            "compiled": bool(cfg.compile),
            "param_counts": _param_breakdown(model),
            **_try_get_git_metadata(),
        },
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_step = 0
    if cfg.resume:
        start_step = _try_resume(Path(cfg.resume), device=device, model=model, optimizer=optimizer, scaler=scaler)

    model.train()
    metrics_path = run_dir / "metrics.jsonl"
    t_start = time.time()
    t_last_ckpt = time.time()
    t_last_log = time.time()
    tokens_since_log = 0

    step = int(start_step)
    target_steps = int(cfg.dry_run_steps if cfg.dry_run else cfg.max_steps)

    while True:
        elapsed = time.time() - t_start
        if cfg.dry_run and step >= target_steps:
            break
        if not cfg.dry_run:
            if cfg.max_time_seconds > 0 and elapsed >= cfg.max_time_seconds:
                break
            if cfg.max_steps > 0 and step >= cfg.max_steps:
                break

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(int(cfg.grad_accum_steps)):
            if cfg.dry_run or cfg.data_source == "random":
                x, y = _make_random_batch(
                    microbatch_size=int(cfg.dry_run_microbatch_size if cfg.dry_run else cfg.microbatch_size),
                    seq_len=int(cfg.dry_run_seq_len if cfg.dry_run else cfg.seq_len),
                    vocab_size=int(vocab_size),
                    device=device,
                )
            else:
                assert flat is not None
                x, y = _sample_batch_from_flat(
                    flat,
                    microbatch_size=int(cfg.microbatch_size),
                    seq_len=int(cfg.seq_len),
                    device=device,
                )

            if amp_dtype == torch.float32:
                logits = model(x)  # (B,T,V)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            else:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss_to_backprop = loss / int(cfg.grad_accum_steps)
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            total_loss += float(loss.item())
            tokens_since_log += int(x.shape[0] * x.shape[1])

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1

        if step % int(cfg.log_every_steps) == 0:
            dt_log = max(1e-9, time.time() - t_last_log)
            toks_per_s = tokens_since_log / dt_log
            record = {
                "step": step,
                "elapsed_s": time.time() - t_start,
                "loss": total_loss / int(cfg.grad_accum_steps),
                "tokens_per_s": toks_per_s,
                "device": str(device),
                "precision": str(amp_dtype),
                "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            _append_jsonl(metrics_path, record)
            print(
                f"[step {step}] loss={record['loss']:.4f} tok/s={record['tokens_per_s']:.0f} grad_norm={record['grad_norm']:.3f}"
            )
            t_last_log = time.time()
            tokens_since_log = 0

        if not cfg.dry_run and (time.time() - t_last_ckpt) >= float(cfg.ckpt_every_seconds):
            path = _checkpoint(
                ckpt_dir=ckpt_dir,
                model=model,
                optimizer=optimizer,
                scaler=(scaler if scaler.is_enabled() else None),
                step=step,
                cfg=cfg,
            )
            if cfg.ckpt_prune:
                _prune_checkpoints(ckpt_dir=ckpt_dir, keep_last=int(cfg.ckpt_keep_last))
            print(f"Saved checkpoint: {path}")
            t_last_ckpt = time.time()

    # Final ckpt for non-dry runs.
    if not cfg.dry_run:
        path = _checkpoint(
            ckpt_dir=ckpt_dir,
            model=model,
            optimizer=optimizer,
            scaler=(scaler if scaler.is_enabled() else None),
            step=step,
            cfg=cfg,
        )
        if cfg.ckpt_prune:
            _prune_checkpoints(ckpt_dir=ckpt_dir, keep_last=int(cfg.ckpt_keep_last))
        print(f"Finished. Final checkpoint: {path}")
    else:
        print("Dry run finished successfully.")


if __name__ == "__main__":
    main()

