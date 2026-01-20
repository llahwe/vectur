"""
Block registry for ModularLlama experiments.

This package is intentionally lightweight:
- `registry.py` provides a name -> factory mapping
- Placeholder stubs exist for future blocks (attention, lstm, moneta, ntm, vectur, vecstur, ...)
"""

from .registry import BlockSpec, available_blocks, get_block_factory

__all__ = ["BlockSpec", "available_blocks", "get_block_factory"]

