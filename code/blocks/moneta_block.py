from __future__ import annotations

"""
MONETA-like recurrent sequence block (PyTorch).

This is ported from `miras/miras/moneta.py` into the VecTur codebase so it can be
selected via the block registry.

API compatibility with `code/llama_macro.py`:
- forward(x: (B,T,D), freqs_cis=...) -> (B,T,D)
- optional init_state(...) and forward_step(...) for incremental decoding

Notes:
- RoPE is applied inside the block (to q/k), using the complex-valued `freqs_cis`
  precomputed by `ModularLlama`.
- The recurrence state is (D,D) per batch element, following the research sketch.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class MonetaState:
    """
    Cached state for fast incremental inference.

    - A/W: recurrence state (b, d, d)
    - *_hist: last 3 pre-conv inputs for depthwise conv (b, 3, d)
    """

    A: torch.Tensor
    W: torch.Tensor
    q_hist: torch.Tensor
    k_hist: torch.Tensor
    v_hist: torch.Tensor


class MonetaBlock(nn.Module):
    """
    Minimal / research-sketch MONETA-like block.

    Key invariant for this project: the recurrence state lives in (dim, dim)
    (per batch element), so the block input/output stays in feature space `dim`.
    """

    def __init__(
        self,
        dim: int,
        p: int = 3,
        q: int = 4,
        expansion_factor: int = 4,  # kept for future work; not used in recurrence
        eps: float = 1e-6,
        # Backwards-compat: interpret as a default chunk size when `chunk_size` is not provided.
        detach_state_every: int = 256,
        *,
        chunk_size: Optional[int] = None,
        tbptt_horizon_chunks: int = 4,
        grad_checkpoint_inner: bool = True,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires even dim, got dim={dim}")

        self.dim = int(dim)
        self.p = int(p)
        self.q = int(q)
        self.eps = float(eps)
        self.expansion_factor = int(expansion_factor)

        if chunk_size is None:
            chunk_size = int(detach_state_every)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got chunk_size={self.chunk_size}")

        self.tbptt_horizon_chunks = int(tbptt_horizon_chunks)
        self.grad_checkpoint_inner = bool(grad_checkpoint_inner)

        # 1) Linear projections (Llama-style bias=False)
        # Fuse Q/K/V projection into one GEMM (same math, fewer kernel launches).
        self.qkv_proj = nn.Linear(self.dim, 3 * self.dim, bias=False)

        # 2) Depthwise convs (kernel=4) over sequence length
        self.q_conv = nn.Conv1d(self.dim, self.dim, kernel_size=4, padding=3, groups=self.dim)
        self.k_conv = nn.Conv1d(self.dim, self.dim, kernel_size=4, padding=3, groups=self.dim)
        self.v_conv = nn.Conv1d(self.dim, self.dim, kernel_size=4, padding=3, groups=self.dim)

        # 3) Data-dependent recurrence parameters (scalar per token)
        self.param_gen = nn.Linear(self.dim, 2, bias=True)  # -> (eta, alpha) logits

        # Initial recurrence matrix (shared across batch), used at t=0.
        self.W0 = nn.Parameter(torch.empty(self.dim, self.dim))
        nn.init.eye_(self.W0)

        # 4) Output gating
        self.gate_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    def init_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MonetaState:
        b = int(batch_size)
        d = int(self.dim)
        A = torch.zeros((b, d, d), device=device, dtype=dtype)
        W = self.W0.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1).contiguous()
        zeros_hist = torch.zeros((b, 3, d), device=device, dtype=dtype)
        return MonetaState(A=A, W=W, q_hist=zeros_hist.clone(), k_hist=zeros_hist.clone(), v_hist=zeros_hist.clone())

    def apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        x: (b, n, dim) real
        freqs_cis: (n, dim/2) complex (as produced by `ModularLlama`)
        """
        x_ri = x.float().reshape(*x.shape[:-1], -1, 2).contiguous()
        x_complex = torch.view_as_complex(x_ri)
        freqs_cis = freqs_cis.view(1, x.shape[1], -1)
        x_rotated = x_complex * freqs_cis
        return torch.view_as_real(x_rotated).flatten(2).type_as(x)

    def _depthwise_conv_step(
        self,
        conv: nn.Conv1d,
        *,
        x_hist: torch.Tensor,  # (b, 3, d)
        x_t: torch.Tensor,  # (b, d)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the conv output at the current timestep, matching:
          conv(x_seq.transpose(1,2))[:, :, :n].transpose(1,2)
        for the last position.
        """
        stacked = torch.cat([x_hist, x_t.unsqueeze(1)], dim=1)  # (b,4,d)
        w = conv.weight.squeeze(1)  # (d,4)
        y_t = (stacked.permute(0, 2, 1) * w.unsqueeze(0)).sum(dim=-1)
        if conv.bias is not None:
            y_t = y_t + conv.bias.unsqueeze(0)
        new_hist = stacked[:, 1:, :].contiguous()
        return y_t, new_hist

    def forward_step(
        self,
        x_t: torch.Tensor,  # (b, d)
        *,
        freqs_cis_t: torch.Tensor,  # (1, d/2) complex
        state: MonetaState,
    ) -> Tuple[torch.Tensor, MonetaState]:
        b, d = x_t.shape
        if int(d) != int(self.dim):
            raise ValueError(f"dim mismatch: got {d}, expected {self.dim}")

        qkv_t = self.qkv_proj(x_t)  # (b, 3d)
        q_t, k_t, v_t = qkv_t.chunk(3, dim=-1)  # (b, d)

        q_t, q_hist = self._depthwise_conv_step(self.q_conv, x_hist=state.q_hist, x_t=q_t)
        k_t, k_hist = self._depthwise_conv_step(self.k_conv, x_hist=state.k_hist, x_t=k_t)
        v_t, v_hist = self._depthwise_conv_step(self.v_conv, x_hist=state.v_hist, x_t=v_t)

        # RoPE + L2 norm
        q1 = F.normalize(self.apply_rope(q_t.unsqueeze(1), freqs_cis_t), p=2, dim=-1).squeeze(1)
        k1 = F.normalize(self.apply_rope(k_t.unsqueeze(1), freqs_cis_t), p=2, dim=-1).squeeze(1)

        # eta/alpha for this token
        params_t = self.param_gen(x_t)  # (b, 2)
        eta_t, alpha_t = torch.chunk(params_t, 2, dim=-1)
        eta_t = torch.sigmoid(eta_t).view(b, 1, 1)
        alpha_t = torch.sigmoid(alpha_t).view(b, 1, 1)

        A = state.A
        W = state.W

        pred = torch.bmm(k1.unsqueeze(1), W).squeeze(1)  # (b, d)
        diff = pred - v_t

        if self.p == 3:
            grad_pred = 3 * (torch.tanh(10 * diff) * (diff * diff))
        else:
            grad_pred = self.p * (torch.tanh(10 * diff) * torch.abs(diff).pow(self.p - 1))

        grad_W = k1.unsqueeze(2) * grad_pred.unsqueeze(1)  # (b, d, d)
        A = alpha_t * A - eta_t * grad_W

        if self.q == 4:
            a4 = A.square().square()
            denom = torch.sqrt(a4.sum(dim=(-2, -1), keepdim=True)) + self.eps
            W = A / denom
        else:
            norm_q = torch.linalg.vector_norm(A, ord=self.q, dim=(-2, -1), keepdim=True)
            W = A / (norm_q.pow(self.q - 2) + self.eps)

        y_t = torch.bmm(q1.unsqueeze(1), W).squeeze(1)  # (b, d)
        out = self.out_proj(y_t) * torch.sigmoid(self.gate_proj(x_t))

        new_state = MonetaState(A=A, W=W, q_hist=q_hist, k_hist=k_hist, v_hist=v_hist)
        return out, new_state

    def _run_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        eta: torch.Tensor,
        alpha: torch.Tensor,
        A: torch.Tensor,
        W: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, d = q.shape
        y = q.new_empty((b, c, d))

        for t in range(c):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            etat, alphat = eta[:, t], alpha[:, t]

            pred = torch.bmm(kt.unsqueeze(1), W).squeeze(1)
            diff = pred - vt

            if self.p == 3:
                grad_pred = 3 * (torch.tanh(10 * diff) * (diff * diff))
            else:
                grad_pred = self.p * (torch.tanh(10 * diff) * torch.abs(diff).pow(self.p - 1))

            grad_W = kt.unsqueeze(2) * grad_pred.unsqueeze(1)
            A = alphat * A - etat * grad_W

            if self.q == 4:
                a4 = A.square().square()
                denom = torch.sqrt(a4.sum(dim=(-2, -1), keepdim=True)) + self.eps
                W = A / denom
            else:
                norm_q = torch.linalg.vector_norm(A, ord=self.q, dim=(-2, -1), keepdim=True)
                W = A / (norm_q.pow(self.q - 2) + self.eps)

            y[:, t] = torch.bmm(qt.unsqueeze(1), W).squeeze(1)

        return y, A, W

    def forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"MonetaBlock expects (B,T,D), got {tuple(x.shape)}")
        b, n, d = x.shape
        if int(d) != int(self.dim):
            raise ValueError(f"dim mismatch: got {d}, expected {self.dim}")

        # Projections -> depthwise convs (slice back to length n)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_conv(q.transpose(1, 2))[:, :, :n].transpose(1, 2)
        k = self.k_conv(k.transpose(1, 2))[:, :, :n].transpose(1, 2)
        v = self.v_conv(v.transpose(1, 2))[:, :, :n].transpose(1, 2)

        # RoPE + L2 norm
        q = F.normalize(self.apply_rope(q, freqs_cis), p=2, dim=-1)
        k = F.normalize(self.apply_rope(k, freqs_cis), p=2, dim=-1)

        # Data-dependent eta/alpha (scalar per token)
        params = self.param_gen(x)  # (b, n, 2)
        eta, alpha = torch.chunk(params, 2, dim=-1)
        eta = torch.sigmoid(eta).view(b, n, 1, 1)
        alpha = torch.sigmoid(alpha).view(b, n, 1, 1)

        # Recurrence state
        A = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        W = self.W0.to(dtype=x.dtype, device=x.device).unsqueeze(0).expand(b, -1, -1)

        y = x.new_empty((b, n, d))

        chunk_size = min(self.chunk_size, n) if n > 0 else self.chunk_size
        do_ckpt = self.grad_checkpoint_inner and self.training and torch.is_grad_enabled()

        chunk_idx = 0
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)

            q_c = q[:, start:end]
            k_c = k[:, start:end]
            v_c = v[:, start:end]
            eta_c = eta[:, start:end]
            alpha_c = alpha[:, start:end]

            if do_ckpt:
                y_c, A, W = checkpoint(
                    lambda _q, _k, _v, _eta, _alpha, _A, _W: self._run_chunk(_q, _k, _v, _eta, _alpha, _A, _W),
                    q_c,
                    k_c,
                    v_c,
                    eta_c,
                    alpha_c,
                    A,
                    W,
                    use_reentrant=False,
                )
            else:
                y_c, A, W = self._run_chunk(q_c, k_c, v_c, eta_c, alpha_c, A, W)

            y[:, start:end] = y_c

            chunk_idx += 1
            if self.tbptt_horizon_chunks > 0 and (chunk_idx % self.tbptt_horizon_chunks == 0) and end < n:
                A = A.detach()
                W = W.detach()

        return self.out_proj(y) * torch.sigmoid(self.gate_proj(x))


def make_moneta_block(
    *,
    dim: int,
    p: int = 3,
    q: int = 4,
    eps: float = 1e-6,
    detach_state_every: int = 256,
    chunk_size: int | None = None,
    tbptt_horizon_chunks: int = 4,
    grad_checkpoint_inner: bool = True,
) -> MonetaBlock:
    return MonetaBlock(
        dim=int(dim),
        p=int(p),
        q=int(q),
        eps=float(eps),
        detach_state_every=int(detach_state_every),
        chunk_size=(int(chunk_size) if chunk_size is not None else None),
        tbptt_horizon_chunks=int(tbptt_horizon_chunks),
        grad_checkpoint_inner=bool(grad_checkpoint_inner),
    )

