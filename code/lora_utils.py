from __future__ import annotations

"""
LoRA utilities for applying parameter-efficient fine-tuning to ModularLlama.

This module provides functions to apply LoRA adapters to both the macro architecture
(MLP, output head) and sequence blocks (attention, LSTM, etc.).
"""

from typing import Any

import torch.nn as nn


def _apply_lora_to_linear(
    module: nn.Module,
    *,
    target_modules: list[str],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
) -> None:
    """
    Apply LoRA to Linear layers in a module using peft.
    
    This is a helper that wraps peft's LoraConfig and get_peft_model.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    except ImportError as e:
        raise ImportError(
            "LoRA requires the 'peft' library. Install it with:\n"
            "  python3 -m pip install peft\n"
        ) from e
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
    )
    
    # Apply LoRA to the module
    peft_model = get_peft_model(module, lora_config)
    return peft_model


def apply_lora_to_model(
    model: nn.Module,
    *,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules_macro: list[str] | None = None,
    target_modules_blocks: list[str] | None = None,
    apply_to_blocks: bool = True,
) -> nn.Module:
    """
    Apply LoRA adapters to a ModularLlama model.
    
    Args:
        model: ModularLlama model instance
        r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling factor (default: 32, typically 2*r)
        lora_dropout: LoRA dropout rate (default: 0.05)
        target_modules_macro: List of module names to target in macro architecture.
                             If None, defaults to ["w13", "w2", "output"] (MLP + output head)
        target_modules_blocks: List of module names to target in sequence blocks.
                               If None, defaults to common patterns like ["wqkv", "wo", "proj"]
        apply_to_blocks: Whether to apply LoRA to sequence blocks (default: True)
    
    Returns:
        Model with LoRA adapters applied (wrapped in PEFT model)
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    except ImportError as e:
        raise ImportError(
            "LoRA requires the 'peft' library. Install it with:\n"
            "  python3 -m pip install peft\n"
        ) from e
    
    # Default target modules for macro architecture (MLP layers and output head)
    if target_modules_macro is None:
        target_modules_macro = ["w13", "w2", "output"]
    
    # Default target modules for blocks (common patterns across different block types)
    if target_modules_blocks is None:
        target_modules_blocks = ["wqkv", "wo", "proj"]
    
    # Collect all unique module names that match our targets
    # PEFT uses regex patterns, so we can use module names directly
    all_target_modules: list[str] = []
    seen: set[str] = set()
    
    # Find all Linear layers and check if they match our target patterns
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract the last component of the name (the actual layer name)
            parts = name.split(".")
            if parts:
                module_name = parts[-1]
                
                # Check if it's a macro architecture target
                if module_name in target_modules_macro:
                    if module_name not in seen:
                        seen.add(module_name)
                        all_target_modules.append(module_name)
                
                # Check if it's a block target (and we're applying to blocks)
                elif apply_to_blocks and module_name in target_modules_blocks:
                    if module_name not in seen:
                        seen.add(module_name)
                        all_target_modules.append(module_name)
    
    if not all_target_modules:
        # Fallback: use a more general approach - target all Linear layers by name pattern
        # This will match any Linear layer
        all_target_modules = ["Linear"]
        print("Warning: No specific target modules found, applying LoRA to all Linear layers")
    else:
        print(f"LoRA target modules: {all_target_modules}")
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=all_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    return peft_model


def _find_linear_modules_in_model(model: nn.Module) -> list[str]:
    """
    Find all Linear layer names in the model for LoRA targeting.
    Returns a list of unique module names that are Linear layers.
    """
    linear_names: list[str] = []
    seen: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get the last component of the name (the actual layer name)
            parts = name.split(".")
            if parts:
                layer_name = parts[-1]
                if layer_name not in seen:
                    seen.add(layer_name)
                    linear_names.append(layer_name)
    return linear_names
