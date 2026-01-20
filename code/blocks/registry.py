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
        BlockSpec(name="attention", description="Transformer attention block (stub)"),
        lambda _kwargs: _not_implemented("attention"),
    ),
    "lstm": (
        BlockSpec(name="lstm", description="LSTM-style recurrent block (stub)"),
        lambda _kwargs: _not_implemented("lstm"),
    ),
    "moneta": (
        BlockSpec(name="moneta", description="MONETA / MIRAS-style block (stub)"),
        lambda _kwargs: _not_implemented("moneta"),
    ),
    "ntm": (
        BlockSpec(name="ntm", description="Neural Turing Machine / external memory block (stub)"),
        lambda _kwargs: _not_implemented("ntm"),
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

