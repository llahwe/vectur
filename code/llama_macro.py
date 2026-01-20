"""
Templatized Llama-style *macro* architecture with a swappable per-layer "sequence block".

Design goals:
- Keep the macro (embeddings, RoPE, residual layout, RMSNorm, SwiGLU MLP) fixed.
- Allow block swaps via a `block_factory(dim) -> nn.Module` hook.
- Support both full-sequence training (`forward(tokens)`) and incremental decoding (`forward_step`),
  where blocks can optionally implement `init_state` / `forward_step`.

This file intentionally does NOT implement any specific alternative blocks yet.
We provide a simple default block (an identity "no-op" sequence block) so the model runs end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class SupportsStep(Protocol):
    """
    Optional protocol for blocks that support fast incremental decoding.

    A block may expose:
    - init_state(batch_size, device, dtype) -> Any
    - forward_step(x_t, freqs_cis_t, state) -> (y_t, new_state)
    """

    def init_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Any: ...

    def forward_step(
        self, x_t: torch.Tensor, *, freqs_cis_t: torch.Tensor, state: Any
    ) -> Tuple[torch.Tensor, Any]: ...


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLUMLP(nn.Module):
    """
    Llama-style SwiGLU MLP: (SiLU(W1 x) ⊙ (W3 x)) W2.
    Implemented with a fused projection for efficiency.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x3 = self.w13(x).chunk(2, dim=-1)
        return self.w2(nn.functional.silu(x1) * x3)


class IdentitySequenceBlock(nn.Module):
    """
    Default placeholder block: does nothing besides a linear projection (optional).
    This is a stand-in until you implement attention / LSTM / MONETA / NTM / VecTur blocks.
    """

    def __init__(self, dim: int, *, use_linear: bool = False):
        super().__init__()
        self.use_linear = bool(use_linear)
        self.proj = nn.Linear(dim, dim, bias=False) if self.use_linear else None

    def forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        return self.proj(x) if self.proj is not None else x

    def init_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:  # noqa: ARG002
        return None

    def forward_step(
        self, x_t: torch.Tensor, *, freqs_cis_t: torch.Tensor, state: None  # noqa: ARG002
    ) -> Tuple[torch.Tensor, None]:
        y = self.proj(x_t) if self.proj is not None else x_t
        return y, None


class LlamaLayer(nn.Module):
    """
    One residual layer:
      h = x + SequenceBlock(RMSNorm(x))
      y = h + MLP(RMSNorm(h))
    """

    def __init__(self, dim: int, *, sequence_block: nn.Module, hidden_dim: int):
        super().__init__()
        self.dim = int(dim)
        self.sequence_block = sequence_block
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor) -> torch.Tensor:
        seq_in = self.attention_norm(x)
        y = self.sequence_block(seq_in, freqs_cis=freqs_cis)
        # Some blocks (e.g. VecTur) may return a full tape (B,N_T,D). The macro-architecture
        # is responsible for reshaping/slicing back to the input length.
        if y.dim() == 3 and y.shape[1] != x.shape[1]:
            y = y[:, : x.shape[1], :]
        h = x + y
        return h + self.mlp(self.ffn_norm(h))

    def forward_step(
        self,
        x_t: torch.Tensor,  # (b, d)
        *,
        freqs_cis_t: torch.Tensor,  # (1, d/2) complex
        state: Any,
    ) -> Tuple[torch.Tensor, Any]:
        seq_in = self.attention_norm(x_t)
        # If the block supports forward_step, use it. Otherwise fall back to full forward on a length-1 "sequence".
        if hasattr(self.sequence_block, "forward_step"):
            y_t, new_state = self.sequence_block.forward_step(seq_in, freqs_cis_t=freqs_cis_t, state=state)
        else:
            # fallback path: treat (b,d) as (b,1,d) then squeeze
            y_seq = self.sequence_block(seq_in.unsqueeze(1), freqs_cis=freqs_cis_t)
            if y_seq.dim() == 3 and y_seq.shape[1] != 1:
                y_seq = y_seq[:, :1, :]
            y_t = y_seq.squeeze(1)
            new_state = state
        h = x_t + y_t
        out = h + self.mlp(self.ffn_norm(h))
        return out, new_state


@dataclass
class ModularLlamaState:
    pos: int
    layer_states: list[Any]


class ModularLlama(nn.Module):
    """
    Minimal Llama-like decoder-only model with block swaps.

    Notes:
    - We precompute RoPE frequencies `freqs_cis` as complex numbers (common implementation).
    - Blocks receive `freqs_cis` (full seq) or `freqs_cis_t` (one position) and may ignore it.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        n_layers: int,
        dim: int,
        block_factory: Optional[Callable[[int], nn.Module]] = None,
        max_seq_len: int = 4096,
        hidden_dim: Optional[int] = None,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")

        self.grad_checkpoint = bool(grad_checkpoint)
        self.dim = int(dim)
        self.vocab_size = int(vocab_size)
        self.n_layers = int(n_layers)
        self.max_seq_len = int(max_seq_len)

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        freqs_cis = self._precompute_freqs_cis(dim, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        if block_factory is None:
            block_factory = lambda d: IdentitySequenceBlock(d)

        self.layers = nn.ModuleList(
            [
                LlamaLayer(dim, sequence_block=block_factory(dim), hidden_dim=hidden_dim or 4 * dim)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def _precompute_freqs_cis(self, dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T) -> logits: (B, T, V)
        """
        if tokens.dim() != 2:
            raise ValueError(f"tokens must have shape (B,T), got {tuple(tokens.shape)}")
        b, t = tokens.shape
        if t > self.max_seq_len:
            raise ValueError(f"seq_len {t} > max_seq_len {self.max_seq_len}")

        h = self.tok_embeddings(tokens)  # (B,T,D)
        freqs_cis = self.freqs_cis[:t]  # (T, D/2) complex

        for layer in self.layers:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                h = checkpoint(lambda _h: layer(_h, freqs_cis=freqs_cis), h, use_reentrant=False)
            else:
                h = layer(h, freqs_cis=freqs_cis)

        return self.output(self.norm(h))

    def init_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        pos: int = 0,
    ) -> ModularLlamaState:
        layer_states: list[Any] = []
        for layer in self.layers:
            blk = layer.sequence_block
            if hasattr(blk, "init_state"):
                layer_states.append(blk.init_state(batch_size=batch_size, device=device, dtype=dtype))
            else:
                layer_states.append(None)
        return ModularLlamaState(pos=int(pos), layer_states=layer_states)

    @torch.no_grad()
    def forward_step(
        self,
        token_t: torch.Tensor,  # (B,)
        *,
        state: ModularLlamaState,
    ) -> Tuple[torch.Tensor, ModularLlamaState]:
        """
        Incremental (1-token) forward: returns logits for the current position.
        """
        if token_t.dim() != 1:
            raise ValueError(f"token_t must have shape (B,), got {tuple(token_t.shape)}")

        h = self.tok_embeddings(token_t)  # (B,D)
        pos = int(state.pos)
        if pos >= self.max_seq_len:
            raise ValueError(f"pos {pos} >= max_seq_len {self.max_seq_len}")
        freqs_cis_t = self.freqs_cis[pos : pos + 1]  # (1, D/2) complex

        new_layer_states: list[Any] = []
        for layer, layer_state in zip(self.layers, state.layer_states):
            h, new_s = layer.forward_step(h, freqs_cis_t=freqs_cis_t, state=layer_state)
            new_layer_states.append(new_s)

        logits = self.output(self.norm(h))  # (B,V)
        return logits, ModularLlamaState(pos=pos + 1, layer_states=new_layer_states)


def default_block_factory(dim: int) -> nn.Module:
    """
    Default factory used by scripts when no explicit block is requested.
    """
    return IdentitySequenceBlock(dim)

