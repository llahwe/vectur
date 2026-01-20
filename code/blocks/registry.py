from __future__ import annotations

"""
Registry for swappable "sequence blocks" used inside `ModularLlama`.

Goal: make block selection a runtime config choice (CLI / config file),
without changing model wiring. Real blocks will be implemented later.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import torch.nn as nn

from llama_macro import IdentitySequenceBlock
from blocks.attention_block import make_attention_block
from blocks.lstm_block import make_lstm_block
from blocks.moneta_block import make_moneta_block
from blocks.ntm_block import make_ntm_block
from blocks.vectur_block import make_vecstur_block, make_vectur_block


BlockFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class BlockSpec:
    name: str
    description: str


def _not_implemented(name: str) -> BlockFactory:
    def factory(dim: int) -> nn.Module:
        raise NotImplementedError(
            f"Block '{name}' is registered but not implemented yet. "
            f"Implement it under `code/blocks/` and update the registry mapping."
        )

    return factory


def _identity_factory(*, use_linear: bool = False) -> BlockFactory:
    return lambda dim: IdentitySequenceBlock(dim, use_linear=use_linear)


_REGISTRY: Dict[str, tuple[BlockSpec, Callable[[Mapping[str, Any]], BlockFactory]]] = {
    # Working placeholders
    "identity": (
        BlockSpec(name="identity", description="No-op block (passes activations through)"),
        lambda kwargs: _identity_factory(use_linear=bool(kwargs.get("use_linear", False))),
    ),
    "identity_linear": (
        BlockSpec(name="identity_linear", description="No-op block with a learned linear projection"),
        lambda kwargs: _identity_factory(use_linear=True),
    ),
    # Future blocks (stubs)
    "attention": (
        BlockSpec(name="attention", description="Causal self-attention block (SDPA + RoPE; configurable size)"),
        lambda kwargs: (
            lambda dim: make_attention_block(
                dim=dim,
                attn_dim=(int(kwargs["attn_dim"]) if "attn_dim" in kwargs else None),
                head_dim=int(kwargs.get("head_dim", 64)),
                n_heads=(int(kwargs["n_heads"]) if "n_heads" in kwargs else None),
                bias=bool(kwargs.get("bias", False)),
                attn_dropout=float(kwargs.get("attn_dropout", 0.0)),
                proj_dropout=float(kwargs.get("proj_dropout", 0.0)),
            )
        ),
    ),
    "lstm": (
        BlockSpec(name="lstm", description="LSTM sequence block (fused nn.LSTM; configurable size)"),
        lambda kwargs: (
            lambda dim: make_lstm_block(
                dim=dim,
                hidden_size=(int(kwargs["hidden_size"]) if "hidden_size" in kwargs else None),
                num_layers=int(kwargs.get("num_layers", 1)),
                dropout=float(kwargs.get("dropout", 0.0)),
                bias=bool(kwargs.get("bias", True)),
                proj_to_dim=bool(kwargs.get("proj_to_dim", True)),
            )
        ),
    ),
    "moneta": (
        BlockSpec(name="moneta", description="MONETA-style recurrent memory block (ported from miras)"),
        lambda kwargs: (
            lambda dim: make_moneta_block(
                dim=dim,
                p=int(kwargs.get("p", 3)),
                q=int(kwargs.get("q", 4)),
                eps=float(kwargs.get("eps", 1e-6)),
                detach_state_every=int(kwargs.get("detach_state_every", 256)),
                chunk_size=(int(kwargs["chunk_size"]) if "chunk_size" in kwargs else None),
                tbptt_horizon_chunks=int(kwargs.get("tbptt_horizon_chunks", 4)),
                grad_checkpoint_inner=bool(kwargs.get("grad_checkpoint_inner", True)),
            )
        ),
    ),
    "ntm": (
        BlockSpec(name="ntm", description="Neural Turing Machine style controller+memory block (best-effort adapter)"),
        lambda kwargs: (
            lambda dim: make_ntm_block(
                dim=dim,
                controller_hidden=(int(kwargs["controller_hidden"]) if "controller_hidden" in kwargs else None),
                mem_slots=int(kwargs.get("mem_slots", 128)),
                mem_width=(int(kwargs["mem_width"]) if "mem_width" in kwargs else None),
                n_read_heads=int(kwargs.get("n_read_heads", 1)),
                n_write_heads=int(kwargs.get("n_write_heads", 1)),
                shift_range=int(kwargs.get("shift_range", 1)),
                bias=bool(kwargs.get("bias", True)),
                learned_init_memory=bool(kwargs.get("learned_init_memory", True)),
                chunk_size=(int(kwargs["chunk_size"]) if "chunk_size" in kwargs else None),
                grad_checkpoint_inner=bool(kwargs.get("grad_checkpoint_inner", False)),
                tbptt_horizon_chunks=int(kwargs.get("tbptt_horizon_chunks", 0)),
            )
        ),
    ),
    "vectur": (
        BlockSpec(name="vectur", description="VecTur block (stub)"),
        lambda kwargs: (
            lambda dim: make_vectur_block(
                dim=dim,
                k=int(kwargs.get("k", 8)),
                t_max=int(kwargs.get("t_max", 4)),
                expansion=int(kwargs.get("expansion", 4)),
            )
        ),
    ),
    "vecstur": (
        BlockSpec(name="vecstur", description="VecSTur block (stub)"),
        lambda kwargs: (
            lambda dim: make_vecstur_block(
                dim=dim,
                k=int(kwargs.get("k", 8)),
                t_max=int(kwargs.get("t_max", 4)),
                expansion=int(kwargs.get("expansion", 4)),
                z_ratio=float(kwargs.get("z_ratio", 1.0)),
            )
        ),
    ),
}


def available_blocks() -> list[BlockSpec]:
    return [spec for spec, _builder in _REGISTRY.values()]


def get_block_factory(name: str, *, kwargs: Mapping[str, Any] | None = None) -> BlockFactory:
    key = str(name).strip()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown block '{name}'. Known blocks: {known}")
    spec, builder = _REGISTRY[key]
    _ = spec  # kept for debugging/introspection
    return builder(kwargs or {})

