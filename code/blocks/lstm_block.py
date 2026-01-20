from __future__ import annotations

"""
LSTM sequence block (PyTorch).

Design goals:
- API-compatible with `code/llama_macro.py` sequence blocks:
    forward(x: (B,T,D), freqs_cis=...) -> (B,T,D)
- Use a fused `nn.LSTM` for efficient full-sequence training (cuDNN fast path on CUDA).
- Make size configurable (hidden size, number of layers, dropout).
- Default to an "apples-to-apples" parameter count vs the default `VecTurBlock`.

Notes:
- RoPE is ignored here; it is handled at the macro-architecture level.
- This is a standard (Schmidhuber/Hochreiter) LSTM; no peepholes, no bidirectionality (causal).
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LSTMBlockConfig:
    dim: int
    hidden_size: int | None = None
    num_layers: int = 1
    dropout: float = 0.0
    bias: bool = True
    proj_to_dim: bool = True


def _count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def _lstm_param_count(*, dim: int, hidden_size: int, num_layers: int, bias: bool, proj_to_dim: bool) -> int:
    """
    Parameter count of a stacked unidirectional LSTM + optional output projection H->D.

    Matches `nn.LSTM` parameterization:
      - weight_ih_l[k]: (4H, input_size)
      - weight_hh_l[k]: (4H, H)
      - bias_*: 2*(4H) if bias=True
    """
    d = int(dim)
    h = int(hidden_size)
    L = int(num_layers)
    if L <= 0:
        raise ValueError(f"num_layers must be > 0, got {L}")
    if h <= 0:
        raise ValueError(f"hidden_size must be > 0, got {h}")

    total = 0
    for layer in range(L):
        inp = d if layer == 0 else h
        total += 4 * h * inp  # weight_ih
        total += 4 * h * h  # weight_hh
        if bias:
            total += 8 * h  # bias_ih + bias_hh

    if proj_to_dim and h != d:
        # Keep bias=False to stay closer to the rest of the codebase (many linears use bias=False).
        total += d * h
    return int(total)


def _target_params_like_vectur(*, dim: int, k: int = 8, t_max: int = 4, expansion: int = 4) -> int:
    """
    Compute the parameter count of the default VecTur block for this `dim`.

    We do this dynamically so LSTM defaults stay "apples-to-apples" even if the VecTur
    implementation changes.
    """
    # Local import to avoid circularity at module import time.
    from blocks.vectur_block import make_vectur_block

    ref = make_vectur_block(dim=int(dim), k=int(k), t_max=int(t_max), expansion=int(expansion))
    return _count_params(ref)


def _pick_hidden_size_for_target(
    *,
    dim: int,
    target_params: int,
    num_layers: int,
    bias: bool,
    proj_to_dim: bool,
    max_hidden: int,
) -> int:
    """
    Pick hidden_size so LSTM param count is close to `target_params`.
    Uses a monotone binary search (param count grows with hidden_size).
    """
    d = int(dim)
    target = int(target_params)
    L = int(num_layers)
    if target <= 0:
        return max(1, d)

    lo = 1
    hi = max(2, int(max_hidden))

    # Ensure hi is large enough.
    while _lstm_param_count(dim=d, hidden_size=hi, num_layers=L, bias=bias, proj_to_dim=proj_to_dim) < target:
        lo = hi
        hi *= 2
        if hi > int(max_hidden):
            hi = int(max_hidden)
            break

    # If even max_hidden is smaller than target, just return max_hidden.
    if _lstm_param_count(dim=d, hidden_size=hi, num_layers=L, bias=bias, proj_to_dim=proj_to_dim) < target:
        return int(hi)

    # Binary search.
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _lstm_param_count(dim=d, hidden_size=mid, num_layers=L, bias=bias, proj_to_dim=proj_to_dim) >= target:
            hi = mid
        else:
            lo = mid

    # Pick closer of lo/hi.
    p_lo = _lstm_param_count(dim=d, hidden_size=lo, num_layers=L, bias=bias, proj_to_dim=proj_to_dim)
    p_hi = _lstm_param_count(dim=d, hidden_size=hi, num_layers=L, bias=bias, proj_to_dim=proj_to_dim)
    return int(hi if abs(p_hi - target) <= abs(p_lo - target) else lo)


class LSTMBlock(nn.Module):
    """
    LSTM sequence block: (B,T,D) -> (B,T,D).
    """

    def __init__(self, cfg: LSTMBlockConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.dim)
        L = int(cfg.num_layers)
        if L <= 0:
            raise ValueError(f"num_layers must be > 0, got {L}")

        # If hidden_size is unset, choose a size that roughly matches VecTur's parameter count.
        if cfg.hidden_size is None:
            target = _target_params_like_vectur(dim=d)
            h = _pick_hidden_size_for_target(
                dim=d,
                target_params=target,
                num_layers=L,
                bias=bool(cfg.bias),
                proj_to_dim=bool(cfg.proj_to_dim),
                # Keep a sane cap so we don't accidentally explode memory for very large dims,
                # but allow enough headroom to match larger blocks (e.g. VecTur's default).
                max_hidden=max(8 * d, 256),
            )
        else:
            h = int(cfg.hidden_size)

        if h <= 0:
            raise ValueError(f"hidden_size must be > 0, got {h}")

        dropout = float(cfg.dropout)
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got {dropout}")

        # NOTE: `dropout` in nn.LSTM is only applied between layers (so only active if num_layers > 1).
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=h,
            num_layers=L,
            bias=bool(cfg.bias),
            batch_first=True,
            dropout=(dropout if L > 1 else 0.0),
            bidirectional=False,
        )

        self.proj: nn.Module
        if bool(cfg.proj_to_dim) and h != d:
            self.proj = nn.Linear(h, d, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        if x.dim() != 3:
            raise ValueError(f"LSTMBlock expects (B,T,D), got {tuple(x.shape)}")
        # Ensures cuDNN uses a contiguous weight buffer on CUDA for best performance.
        # (No-op on CPU; safe to call repeatedly.)
        if x.is_cuda:
            self.lstm.flatten_parameters()
        y, _state = self.lstm(x)
        return self.proj(y)

    def init_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Any:  # noqa: ARG002
        # (num_layers, batch, hidden)
        L = int(self.lstm.num_layers)
        h = int(self.lstm.hidden_size)
        h0 = torch.zeros((L, int(batch_size), h), device=device, dtype=dtype)
        c0 = torch.zeros((L, int(batch_size), h), device=device, dtype=dtype)
        return (h0, c0)

    def _step_one_layer(
        self,
        x_t: torch.Tensor,  # (B, input_size)
        h_t: torch.Tensor,  # (B, H)
        c_t: torch.Tensor,  # (B, H)
        *,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One LSTM cell step using the parameters inside `nn.LSTM`, without duplicating weights.
        """
        w_ih = getattr(self.lstm, f"weight_ih_l{layer}")
        w_hh = getattr(self.lstm, f"weight_hh_l{layer}")
        b_ih = getattr(self.lstm, f"bias_ih_l{layer}", None)
        b_hh = getattr(self.lstm, f"bias_hh_l{layer}", None)

        gates = torch.nn.functional.linear(x_t, w_ih, b_ih) + torch.nn.functional.linear(h_t, w_hh, b_hh)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def forward_step(
        self,
        x_t: torch.Tensor,  # (B,D)
        *,
        freqs_cis_t: torch.Tensor,  # noqa: ARG002
        state: Any,
    ) -> tuple[torch.Tensor, Any]:
        """
        Efficient 1-token forward using cached recurrent state.

        This avoids llama_macro's fallback path (which would run an entire length-1 LSTM forward each step),
        and is the key to making decoding compute/VRAM efficient.
        """
        if x_t.dim() != 2:
            raise ValueError(f"LSTMBlock.forward_step expects (B,D), got {tuple(x_t.shape)}")
        b, d = x_t.shape
        if d != int(self.lstm.input_size):
            raise ValueError(f"Input dim mismatch: got D={d}, expected {int(self.lstm.input_size)}")

        if state is None:
            # Best-effort default if caller didn't initialize state.
            state = self.init_state(batch_size=b, device=x_t.device, dtype=x_t.dtype)
        h_all, c_all = state
        if not (isinstance(h_all, torch.Tensor) and isinstance(c_all, torch.Tensor)):
            raise TypeError("LSTMBlock state must be (h, c) tensors")
        if h_all.dim() != 3 or c_all.dim() != 3:
            raise ValueError("LSTMBlock state tensors must have shape (num_layers, B, H)")

        L = int(self.lstm.num_layers)
        if int(h_all.shape[0]) != L or int(c_all.shape[0]) != L:
            raise ValueError(f"State num_layers mismatch: got {h_all.shape[0]}/{c_all.shape[0]}, expected {L}")

        # Run through the stacked layers.
        x_in = x_t
        h_outs: list[torch.Tensor] = []
        c_outs: list[torch.Tensor] = []
        p_drop = float(self.lstm.dropout)

        for layer in range(L):
            h_prev = h_all[layer]
            c_prev = c_all[layer]
            h_next, c_next = self._step_one_layer(x_in, h_prev, c_prev, layer=layer)
            h_outs.append(h_next)
            c_outs.append(c_next)

            x_in = h_next
            # Match nn.LSTM behavior: dropout between layers (not on last layer).
            if layer < L - 1 and p_drop > 0.0:
                x_in = torch.nn.functional.dropout(x_in, p=p_drop, training=self.training)

        h_new = torch.stack(h_outs, dim=0)
        c_new = torch.stack(c_outs, dim=0)

        y_t = self.proj(x_in)
        return y_t, (h_new, c_new)


def make_lstm_block(
    *,
    dim: int,
    hidden_size: int | None = None,
    num_layers: int = 1,
    dropout: float = 0.0,
    bias: bool = True,
    proj_to_dim: bool = True,
) -> LSTMBlock:
    return LSTMBlock(
        LSTMBlockConfig(
            dim=int(dim),
            hidden_size=(int(hidden_size) if hidden_size is not None else None),
            num_layers=int(num_layers),
            dropout=float(dropout),
            bias=bool(bias),
            proj_to_dim=bool(proj_to_dim),
        )
    )

