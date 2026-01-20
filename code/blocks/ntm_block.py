from __future__ import annotations

"""
Neural Turing Machine (Graves et al.) style sequence block (PyTorch).

This adapts an NTM to the `vectur/code/llama_macro.py` "sequence block" interface:
  forward(x: (B,T,D), freqs_cis=...) -> (B,T,D)

Caveat:
- Classic NTM is a controller+memory architecture, not originally designed as a drop-in
  transformer sub-layer. This implementation is a best-effort "NTM-flavored" block
  that is causal and produces per-token outputs.

Design goals:
- Stay close to Graves NTM addressing:
  content lookup -> interpolation -> circular shift -> sharpening
- Efficient enough for research: vectorized ops over batch/head/memory; Python loop over time.
- Configurable sizes (memory slots/width, heads, controller hidden).
- Default controller hidden size is auto-chosen to match the parameter count of the default
  `VecTurBlock` for the same `dim`, for apples-to-apples comparisons.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NTMConfig:
    dim: int
    controller_hidden: int | None = None
    mem_slots: int = 128
    mem_width: int | None = None  # if None, defaults to min(dim, 128)
    n_read_heads: int = 1
    n_write_heads: int = 1  # classic NTM uses 1 write head
    shift_range: int = 1  # supports shifts in [-R..R], default R=1 => 3-way shift
    bias: bool = True
    # If True, include a learnable initial memory (N,W). If False, start from zeros.
    learned_init_memory: bool = True
    # VRAM controls for training:
    # - chunk_size: if set, process time loop in chunks (enables checkpointing + TBPTT)
    # - grad_checkpoint_inner: checkpoint each chunk to reduce activation memory (recompute on backward)
    # - tbptt_horizon_chunks: detach the recurrent state every H chunks (truncated BPTT)
    chunk_size: int | None = None
    grad_checkpoint_inner: bool = False
    tbptt_horizon_chunks: int = 0


def _count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def _target_params_like_vectur(*, dim: int, k: int = 8, t_max: int = 4, expansion: int = 4) -> int:
    from blocks.vectur_block import make_vectur_block

    ref = make_vectur_block(dim=int(dim), k=int(k), t_max=int(t_max), expansion=int(expansion))
    return _count_params(ref)


def _lstm_cell_param_count(*, input_size: int, hidden_size: int, bias: bool) -> int:
    # LSTMCell has:
    #  weight_ih: (4H, I), weight_hh: (4H, H), bias_ih/bias_hh: (4H) each if bias
    i = int(input_size)
    h = int(hidden_size)
    total = 4 * h * i + 4 * h * h
    if bias:
        total += 8 * h
    return int(total)


def _ntm_param_count(
    *,
    dim: int,
    controller_hidden: int,
    mem_slots: int,
    mem_width: int,
    n_read_heads: int,
    n_write_heads: int,
    shift_range: int,
    bias: bool,
    learned_init_memory: bool,
) -> int:
    """
    Rough-but-accurate parameter count for this module's components.
    """
    d = int(dim)
    h = int(controller_hidden)
    n = int(mem_slots)
    w = int(mem_width)
    r = int(n_read_heads)
    wh = int(n_write_heads)
    s = int(2 * int(shift_range) + 1)

    # Controller
    total = _lstm_cell_param_count(input_size=d, hidden_size=h, bias=True)  # we always use bias=True in LSTMCell

    # Interface projection (controller -> params)
    # For each read head:
    #   key(w), beta(1), g(1), shift(s), gamma(1)
    # For each write head:
    #   key(w), beta(1), g(1), shift(s), gamma(1), erase(w), add(w)
    per_read = w + 1 + 1 + s + 1
    per_write = w + 1 + 1 + s + 1 + w + w
    iface = r * per_read + wh * per_write
    total += h * iface + (iface if bias else 0)

    # Read vector(s) + controller output -> dim
    total += (h + r * w) * d + (d if bias else 0)

    # Optional learnable initial memory
    if learned_init_memory:
        total += n * w
    return int(total)


def _pick_controller_hidden_for_target(
    *,
    dim: int,
    target_params: int,
    mem_slots: int,
    mem_width: int,
    n_read_heads: int,
    n_write_heads: int,
    shift_range: int,
    bias: bool,
    learned_init_memory: bool,
    max_hidden: int,
) -> int:
    """
    Monotone-ish search for controller hidden size that matches target params.
    """
    d = int(dim)
    target = int(target_params)
    if target <= 0:
        return max(1, d)

    def p(h: int) -> int:
        return _ntm_param_count(
            dim=d,
            controller_hidden=int(h),
            mem_slots=int(mem_slots),
            mem_width=int(mem_width),
            n_read_heads=int(n_read_heads),
            n_write_heads=int(n_write_heads),
            shift_range=int(shift_range),
            bias=bool(bias),
            learned_init_memory=bool(learned_init_memory),
        )

    lo, hi = 8, max(16, int(max_hidden))
    # Ensure hi is large enough (or capped).
    while p(hi) < target and hi < int(max_hidden):
        lo = hi
        hi = min(int(max_hidden), hi * 2)
        if hi == lo:
            break

    if p(hi) < target:
        return int(hi)

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if p(mid) >= target:
            hi = mid
        else:
            lo = mid

    return int(hi if abs(p(hi) - target) <= abs(p(lo) - target) else lo)


def _circular_convolution(w: torch.Tensor, s: torch.Tensor, shift_range: int) -> torch.Tensor:
    """
    Circular shift of weights by a learned distribution.

    w: (B,H,N)
    s: (B,H,S) where S = 2R+1, sums to 1, corresponds to shifts [-R..R]
    """
    b, h, n = w.shape
    r = int(shift_range)
    S = 2 * r + 1
    if s.shape[-1] != S:
        raise ValueError(f"shift distribution size mismatch: expected {S}, got {s.shape[-1]}")

    out = w.new_zeros((b, h, n))
    # Small S (default 3) => loop is cheap; avoids FFT overhead.
    for i, shift in enumerate(range(-r, r + 1)):
        out.add_(s[:, :, i].unsqueeze(-1) * torch.roll(w, shifts=shift, dims=-1))
    return out


def _address(
    *,
    k: torch.Tensor,  # (B,H,W)
    beta: torch.Tensor,  # (B,H,1)
    g: torch.Tensor,  # (B,H,1)
    s: torch.Tensor,  # (B,H,S)
    gamma: torch.Tensor,  # (B,H,1)
    M: torch.Tensor,  # (B,N,W)
    w_prev: torch.Tensor,  # (B,H,N)
    shift_range: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    NTM addressing pipeline producing weights w_t: (B,H,N).
    """
    # Content addressing (cosine similarity), implemented without normalizing the full memory tensor.
    # This avoids allocating a (B,N,W) normalized copy of M every timestep (major VRAM/compute win).
    # sim = (k · M) / (||k|| * ||M||)
    dot = torch.einsum("bhw,bnw->bhn", k, M)  # (B,H,N)
    inv_k = torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + eps)  # (B,H,1)
    inv_M = torch.rsqrt((M * M).sum(dim=-1, keepdim=True) + eps).squeeze(-1).unsqueeze(1)  # (B,1,N)
    sim = dot * inv_k * inv_M
    wc = torch.softmax(beta * sim, dim=-1)

    # Interpolation
    w = g * wc + (1.0 - g) * w_prev

    # Circular shift
    w = _circular_convolution(w, s, shift_range)

    # Sharpen
    w = w.clamp(min=eps).pow(gamma.clamp(min=1.0))
    w = w / (w.sum(dim=-1, keepdim=True) + eps)
    return w


class NTMBlock(nn.Module):
    def __init__(self, cfg: NTMConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.dim)

        n = int(cfg.mem_slots)
        if n <= 0:
            raise ValueError(f"mem_slots must be > 0, got {n}")
        w = int(cfg.mem_width) if cfg.mem_width is not None else min(d, 128)
        if w <= 0:
            raise ValueError(f"mem_width must be > 0, got {w}")

        r = int(cfg.n_read_heads)
        wh = int(cfg.n_write_heads)
        if r <= 0 or wh <= 0:
            raise ValueError(f"n_read_heads and n_write_heads must be > 0, got {r}, {wh}")

        sr = int(cfg.shift_range)
        if sr < 0:
            raise ValueError(f"shift_range must be >= 0, got {sr}")
        S = 2 * sr + 1

        # Pick controller hidden size to match VecTur params (default) if unset.
        if cfg.controller_hidden is None:
            target = _target_params_like_vectur(dim=d)
            h = _pick_controller_hidden_for_target(
                dim=d,
                target_params=target,
                mem_slots=n,
                mem_width=w,
                n_read_heads=r,
                n_write_heads=wh,
                shift_range=sr,
                bias=bool(cfg.bias),
                learned_init_memory=bool(cfg.learned_init_memory),
                max_hidden=max(8 * d, 256),
            )
        else:
            h = int(cfg.controller_hidden)
        if h <= 0:
            raise ValueError(f"controller_hidden must be > 0, got {h}")

        self.dim = d
        self.mem_slots = n
        self.mem_width = w
        self.n_read_heads = r
        self.n_write_heads = wh
        self.shift_range = sr
        self.shift_size = S
        self.chunk_size = (int(cfg.chunk_size) if cfg.chunk_size is not None else None)
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 when provided, got {self.chunk_size}")
        self.grad_checkpoint_inner = bool(cfg.grad_checkpoint_inner)
        self.tbptt_horizon_chunks = int(cfg.tbptt_horizon_chunks)

        # Controller: LSTMCell (classic NTM uses an LSTM controller)
        self.controller = nn.LSTMCell(input_size=d, hidden_size=h, bias=True)

        # Interface projection: controller hidden -> all head parameters
        per_read = w + 1 + 1 + S + 1
        per_write = w + 1 + 1 + S + 1 + w + w
        iface = r * per_read + wh * per_write
        self.iface = nn.Linear(h, iface, bias=bool(cfg.bias))

        # Combine controller output + read vectors -> model dim
        self.out = nn.Linear(h + r * w, d, bias=bool(cfg.bias))

        if bool(cfg.learned_init_memory):
            self.M0 = nn.Parameter(torch.zeros(n, w))
            nn.init.normal_(self.M0, mean=0.0, std=0.02)
        else:
            self.register_parameter("M0", None)

    def init_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Any:
        b = int(batch_size)
        h = int(self.controller.hidden_size)
        # Controller state
        h_t = torch.zeros((b, h), device=device, dtype=dtype)
        c_t = torch.zeros((b, h), device=device, dtype=dtype)
        # Memory + weights
        if self.M0 is not None:
            M = self.M0.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1).contiguous()
        else:
            M = torch.zeros((b, self.mem_slots, self.mem_width), device=device, dtype=dtype)
        w_r = torch.full((b, self.n_read_heads, self.mem_slots), 1.0 / float(self.mem_slots), device=device, dtype=dtype)
        w_w = torch.full((b, self.n_write_heads, self.mem_slots), 1.0 / float(self.mem_slots), device=device, dtype=dtype)
        return (h_t, c_t, M, w_r, w_w)

    def _step(
        self,
        x_t: torch.Tensor,  # (B,D)
        state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        h_t, c_t, M, w_r_prev, w_w_prev = state

        h_t, c_t = self.controller(x_t, (h_t, c_t))

        # Decode interface parameters
        p = self.iface(h_t)  # (B, iface)
        b = int(p.shape[0])
        n = int(self.mem_slots)
        w = int(self.mem_width)
        S = int(self.shift_size)

        offs = 0

        def take(shape: tuple[int, ...]) -> torch.Tensor:
            nonlocal offs
            size = 1
            for s_ in shape:
                size *= int(s_)
            out = p[:, offs : offs + size].view(b, *shape)
            offs += size
            return out

        # Read head params
        k_r = take((self.n_read_heads, w))
        beta_r = F.softplus(take((self.n_read_heads, 1)))
        g_r = torch.sigmoid(take((self.n_read_heads, 1)))
        s_r = torch.softmax(take((self.n_read_heads, S)), dim=-1)
        gamma_r = 1.0 + F.softplus(take((self.n_read_heads, 1)))

        # Write head params
        k_w = take((self.n_write_heads, w))
        beta_w = F.softplus(take((self.n_write_heads, 1)))
        g_w = torch.sigmoid(take((self.n_write_heads, 1)))
        s_w = torch.softmax(take((self.n_write_heads, S)), dim=-1)
        gamma_w = 1.0 + F.softplus(take((self.n_write_heads, 1)))
        erase = torch.sigmoid(take((self.n_write_heads, w)))
        add = torch.tanh(take((self.n_write_heads, w)))

        # Addressing to get weights
        w_r = _address(
            k=k_r,
            beta=beta_r,
            g=g_r,
            s=s_r,
            gamma=gamma_r,
            M=M,
            w_prev=w_r_prev,
            shift_range=self.shift_range,
        )
        w_w = _address(
            k=k_w,
            beta=beta_w,
            g=g_w,
            s=s_w,
            gamma=gamma_w,
            M=M,
            w_prev=w_w_prev,
            shift_range=self.shift_range,
        )

        # Read: r_t = w_r M  -> (B,R,W)
        r_t = torch.einsum("brn,bnw->brw", w_r, M)

        # Write: erase + add (broadcast over memory slots)
        # M = M * Π_h (1 - w_w[h] ⊗ erase[h]) + Σ_h w_w[h] ⊗ add[h]
        # We'll apply heads sequentially (wh is small; default 1).
        for i in range(self.n_write_heads):
            w_i = w_w[:, i].unsqueeze(-1)  # (B,N,1)
            e_i = erase[:, i].unsqueeze(1)  # (B,1,W)
            a_i = add[:, i].unsqueeze(1)  # (B,1,W)
            M = M * (1.0 - w_i * e_i) + w_i * a_i

        # Output: combine controller hidden + reads
        y_t = self.out(torch.cat([h_t, r_t.reshape(b, -1)], dim=-1))
        return y_t, (h_t, c_t, M, w_r, w_w)

    def _run_chunk(
        self,
        x: torch.Tensor,  # (B,C,D)
        h_t: torch.Tensor,
        c_t: torch.Tensor,
        M: torch.Tensor,
        w_r: torch.Tensor,
        w_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run a time chunk and return (y, new_state...).
        Designed to be checkpointed for VRAM savings.
        """
        b, c, d = x.shape
        y = x.new_empty((b, c, d))
        state = (h_t, c_t, M, w_r, w_w)
        for t in range(c):
            y_t, state = self._step(x[:, t], state)
            y[:, t] = y_t
        h_t, c_t, M, w_r, w_w = state
        return y, h_t, c_t, M, w_r, w_w

    def forward_step(
        self,
        x_t: torch.Tensor,  # (B,D)
        *,
        freqs_cis_t: torch.Tensor,  # noqa: ARG002
        state: Any,
    ) -> Tuple[torch.Tensor, Any]:
        y_t, new_state = self._step(x_t, state)
        return y_t, new_state

    def forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        if x.dim() != 3:
            raise ValueError(f"NTMBlock expects (B,T,D), got {tuple(x.shape)}")
        b, t, d = x.shape
        if int(d) != int(self.dim):
            raise ValueError(f"dim mismatch: got {d}, expected {self.dim}")

        state = self.init_state(batch_size=b, device=x.device, dtype=x.dtype)
        y = x.new_empty((b, t, d))

        # Optional VRAM-saving path: chunk + (optional) checkpoint + TBPTT.
        if self.chunk_size is not None and t > 0:
            from torch.utils.checkpoint import checkpoint

            do_ckpt = self.grad_checkpoint_inner and self.training and torch.is_grad_enabled()
            chunk_idx = 0
            for start in range(0, t, self.chunk_size):
                end = min(t, start + self.chunk_size)
                x_c = x[:, start:end]

                h_t, c_t, M, w_r, w_w = state
                if do_ckpt:
                    y_c, h_t, c_t, M, w_r, w_w = checkpoint(
                        lambda _x, _h, _c, _M, _wr, _ww: self._run_chunk(_x, _h, _c, _M, _wr, _ww),
                        x_c,
                        h_t,
                        c_t,
                        M,
                        w_r,
                        w_w,
                        use_reentrant=False,
                    )
                else:
                    y_c, h_t, c_t, M, w_r, w_w = self._run_chunk(x_c, h_t, c_t, M, w_r, w_w)

                y[:, start:end] = y_c
                state = (h_t, c_t, M, w_r, w_w)

                chunk_idx += 1
                if self.tbptt_horizon_chunks > 0 and (chunk_idx % self.tbptt_horizon_chunks == 0) and end < t:
                    h_t, c_t, M, w_r, w_w = state
                    state = (h_t.detach(), c_t.detach(), M.detach(), w_r.detach(), w_w.detach())
            return y

        # Default path: simple time loop.
        for i in range(t):
            y_i, state = self._step(x[:, i], state)
            y[:, i] = y_i
        return y


def make_ntm_block(
    *,
    dim: int,
    controller_hidden: int | None = None,
    mem_slots: int = 128,
    mem_width: int | None = None,
    n_read_heads: int = 1,
    n_write_heads: int = 1,
    shift_range: int = 1,
    bias: bool = True,
    learned_init_memory: bool = True,
    chunk_size: int | None = None,
    grad_checkpoint_inner: bool = False,
    tbptt_horizon_chunks: int = 0,
) -> NTMBlock:
    return NTMBlock(
        NTMConfig(
            dim=int(dim),
            controller_hidden=(int(controller_hidden) if controller_hidden is not None else None),
            mem_slots=int(mem_slots),
            mem_width=(int(mem_width) if mem_width is not None else None),
            n_read_heads=int(n_read_heads),
            n_write_heads=int(n_write_heads),
            shift_range=int(shift_range),
            bias=bool(bias),
            learned_init_memory=bool(learned_init_memory),
            chunk_size=(int(chunk_size) if chunk_size is not None else None),
            grad_checkpoint_inner=bool(grad_checkpoint_inner),
            tbptt_horizon_chunks=int(tbptt_horizon_chunks),
        )
    )

