from __future__ import annotations

"""
Transformer attention sequence block (PyTorch).

Design goals:
- API-compatible with `code/llama_macro.py` sequence blocks:
    forward(x: (B,T,D), freqs_cis=...) -> (B,T,D)
- Use PyTorch SDPA (`scaled_dot_product_attention`) for efficiency.
- Apply RoPE using the provided `freqs_cis` (complex-valued precompute from `llama_macro.py`).
- Make size configurable, and default to an "apples-to-apples" parameter count vs the default VecTur block.

Notes:
- This is *only* the attention sub-layer. The macro architecture already provides
  pre-norm (RMSNorm) + residual wiring + MLP.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AttentionBlockConfig:
    dim: int
    # Total attention inner dimension = n_heads * head_dim.
    # If omitted, we choose it to roughly match the param count of VecTur's default block.
    attn_dim: int | None = None
    head_dim: int = 64
    n_heads: int | None = None
    bias: bool = False
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0


def _count_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def _target_params_like_vectur(*, dim: int, k: int = 8, t_max: int = 4, expansion: int = 4) -> int:
    """
    Compute the parameter count of the default VecTur block for this `dim`.
    """
    from blocks.vectur_block import make_vectur_block

    ref = make_vectur_block(dim=int(dim), k=int(k), t_max=int(t_max), expansion=int(expansion))
    return _count_params(ref)


def _round_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return int(x)
    return int(max(m, (int(x + m // 2) // m) * m))


def _pick_attn_dim_for_target(*, dim: int, target_params: int, head_dim: int, max_attn_dim: int) -> int:
    """
    Attention block params (no bias) ~= 4 * D * A for q,k,v,o projections where:
      Wq: D->A, Wk: D->A, Wv: D->A, Wo: A->D

    Choose A (multiple of head_dim) so param count is close to target.
    """
    d = int(dim)
    target = int(target_params)
    hd = int(head_dim)
    if hd <= 0:
        raise ValueError(f"head_dim must be > 0, got {hd}")
    if target <= 0:
        return _round_to_multiple(d, hd)

    raw = max(hd, int(round(target / max(1, (4 * d)))))
    a = _round_to_multiple(raw, hd)
    return int(min(max(a, hd), int(max_attn_dim)))


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to the last dimension of x using complex multiplication.

    x: (B,T,H,HD) with HD even
    freqs_cis: (T, HD/2) complex or (1, HD/2) complex
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x to have shape (B,T,H,HD), got {tuple(x.shape)}")
    b, t, h, hd = x.shape
    if hd % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got head_dim={hd}")

    # freqs_cis may be passed as (1, D/2) for forward_step fallback; broadcast over T.
    if freqs_cis.dim() != 2:
        raise ValueError(f"Expected freqs_cis to have shape (T,HD/2) complex, got {tuple(freqs_cis.shape)}")
    if freqs_cis.shape[0] not in (1, t):
        raise ValueError(f"freqs_cis length mismatch: got {freqs_cis.shape[0]} vs T={t}")

    # Convert last dim to complex pairs.
    x_float = x.float()
    x2 = x_float.reshape(b, t, h, hd // 2, 2)
    x_complex = torch.view_as_complex(x2)  # (B,T,H,HD/2)

    # Use only the rope dims we need (head_dim/2).
    freqs = freqs_cis[:, : hd // 2]
    x_out = x_complex * freqs.view(freqs.shape[0], 1, 1, freqs.shape[1])
    out2 = torch.view_as_real(x_out).reshape(b, t, h, hd)
    return out2.type_as(x)


class AttentionBlock(nn.Module):
    """
    Causal self-attention block: (B,T,D) -> (B,T,D).
    """

    def __init__(self, cfg: AttentionBlockConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.dim)

        # Resolve attn_dim / heads.
        if cfg.attn_dim is None:
            target = _target_params_like_vectur(dim=d)
            a = _pick_attn_dim_for_target(dim=d, target_params=target, head_dim=int(cfg.head_dim), max_attn_dim=8 * d)
        else:
            a = int(cfg.attn_dim)
        if a <= 0:
            raise ValueError(f"attn_dim must be > 0, got {a}")

        hd = int(cfg.head_dim)
        if hd <= 0:
            raise ValueError(f"head_dim must be > 0, got {hd}")
        if a % hd != 0:
            raise ValueError(f"attn_dim must be divisible by head_dim (got attn_dim={a}, head_dim={hd})")

        n_heads = int(cfg.n_heads) if cfg.n_heads is not None else (a // hd)
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}")
        if n_heads * hd != a:
            raise ValueError(f"n_heads * head_dim must equal attn_dim (got {n_heads}*{hd} != {a})")

        self.dim = d
        self.attn_dim = a
        self.n_heads = n_heads
        self.head_dim = hd

        bias = bool(cfg.bias)
        self.wq = nn.Linear(d, a, bias=bias)
        self.wk = nn.Linear(d, a, bias=bias)
        self.wv = nn.Linear(d, a, bias=bias)
        self.wo = nn.Linear(a, d, bias=bias)

        self.attn_dropout = float(cfg.attn_dropout)
        self.proj_dropout = float(cfg.proj_dropout)
        self.drop = nn.Dropout(self.proj_dropout) if self.proj_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"AttentionBlock expects (B,T,D), got {tuple(x.shape)}")
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(f"Input dim mismatch: got D={d}, expected {self.dim}")

        # Project to Q,K,V and reshape to (B,T,H,HD).
        q = self.wq(x).view(b, t, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, t, self.n_heads, self.head_dim)
        v = self.wv(x).view(b, t, self.n_heads, self.head_dim)

        # Apply RoPE to Q/K using the provided complex freqs (slice to head_dim).
        q = _apply_rope(q, freqs_cis=freqs_cis)
        k = _apply_rope(k, freqs_cis=freqs_cis)

        # SDPA expects (B,H,T,HD)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=(self.attn_dropout if self.training else 0.0),
            is_causal=True,
        )  # (B,H,T,HD)
        y = y.transpose(1, 2).contiguous().view(b, t, self.attn_dim)
        y = self.wo(y)
        return self.drop(y)


def make_attention_block(
    *,
    dim: int,
    attn_dim: int | None = None,
    head_dim: int = 64,
    n_heads: int | None = None,
    bias: bool = False,
    attn_dropout: float = 0.0,
    proj_dropout: float = 0.0,
) -> AttentionBlock:
    return AttentionBlock(
        AttentionBlockConfig(
            dim=int(dim),
            attn_dim=(int(attn_dim) if attn_dim is not None else None),
            head_dim=int(head_dim),
            n_heads=(int(n_heads) if n_heads is not None else None),
            bias=bool(bias),
            attn_dropout=float(attn_dropout),
            proj_dropout=float(proj_dropout),
        )
    )

