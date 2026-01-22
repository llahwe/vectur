from __future__ import annotations

"""
VecTur / VecSTur block (PyTorch).

This implements the *block* described in `paper/paper.tex` with concrete choices:
- M_T: per-token linear projection into tape symbol space
- M_Q: linear map into R^{d_Q} (implemented as a sum over per-token projections)
- M_I: linear map into (S^1)^k × R^k (implemented as a sum over per-token projections)
- Δ_T, Δ_Q, Δ_θ, Δ_w: 2-layer MLPs with expansion factor 4
- κ(x): positive scalar per example (softplus)
- I_t = (θ_t, w_t): head angles and weights; dense J(I_t) is never materialized

Efficiency notes:
- The paper's addressing map E(θ) is 2-sparse (linear interpolation between adjacent tape cells),
  so J(I_t) has at most 2k nonzeros.
- Reads/writes are implemented via gather + `index_add` on only those 2k locations (no dense Δ tensor).
- Optional per-step activation checkpointing can reduce MLP activation memory without truncating BPTT.

Stability notes:
- Gradient explosion can occur due to iterative unrolling and tape accumulation.
- Configurable clipping options are provided to stabilize training (enabled by default).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _mlp(in_dim: int, out_dim: int, *, expansion: int = 4) -> nn.Module:
    hid = int(expansion * in_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hid, bias=False),
        nn.SiLU(),
        nn.Linear(hid, out_dim, bias=False),
    )


@dataclass(frozen=True)
class VecTurConfig:
    dim: int
    k: int = 8
    t_max: int = 4
    expansion: int = 4
    stochastic: bool = False
    z_ratio: float = 1.0  # len(z) = round(z_ratio * len(x))
    eps: float = 1e-6
    halt_eps: float = 1e-3
    early_stop: bool = True
    checkpoint_steps: bool = True  # activation checkpointing for Δ networks
    # Gradient stability options (None = disabled, float = clip value)
    # Defaults are enabled to prevent gradient explosion
    clip_delta_q: float | None = 5.0  # Clip dq updates
    clip_delta_theta: float | None = 1.0  # Clip dtheta updates
    clip_delta_w: float | None = 2.0  # Clip dw updates
    clip_delta_t: float | None = 5.0  # Clip u_t (tape write) updates
    clip_tape: float | None = 10.0  # Clip tape values after writes
    clip_w: float | None = 10.0  # Clip w values
    normalize_q: bool = False  # Normalize Q state (L2) to prevent unbounded growth


class VecTurBlock(nn.Module):
    """
    Sequence block API compatible with `code/llama_macro.py`:
      forward(x: (B,T,D), freqs_cis=...) -> (B,N_T,D)   (full tape after halting)

    We ignore RoPE for now; the block is meant to be a drop-in alternative.
    """

    def __init__(self, cfg: VecTurConfig):
        super().__init__()
        self.cfg = cfg
        d = int(cfg.dim)

        # "Tape symbol" dimension and "control state" dimension.
        # We keep them equal to the model dimension for simplicity.
        self.d_t = d
        self.d_q = d

        # M_T: per-token projection into tape symbol space.
        self.m_t = nn.Linear(d, self.d_t, bias=False)
        # M_Q: linear map into control state (implemented as sum over token projections).
        self.m_q_tok = nn.Linear(d, self.d_q, bias=False)
        # M_I: linear maps into θ0 and w0 (implemented as sum over token projections).
        self.m_theta_tok = nn.Linear(d, int(cfg.k), bias=False)
        self.m_w_tok = nn.Linear(d, int(cfg.k), bias=False)

        # κ(x): positive scalar per example.
        self.kappa_net = _mlp(d, 1, expansion=cfg.expansion)

        # Δ functions
        self.delta_q = _mlp(self.d_t + self.d_q, self.d_q, expansion=cfg.expansion)
        self.delta_t = _mlp(self.d_t + self.d_q, self.d_t, expansion=cfg.expansion)  # U_t = Δ_T(S,Q)

        # Head updates: Δ_θ and Δ_w operate on per-head features.
        # We use the paper-friendly periodic features for θ: sin/cos.
        per_head_in = self.d_t + self.d_q + 3  # [S_t,Q_t,sinθ_i,cosθ_i,w_i]
        self.delta_theta = _mlp(per_head_in, 1, expansion=cfg.expansion)  # applied per head -> (B,k)
        self.delta_w = _mlp(per_head_in, 1, expansion=cfg.expansion)  # applied per head -> (B,k)

        # Output projection back to model dimension (kept as identity if d_t == d).
        self.out = nn.Linear(self.d_t, d, bias=False) if self.d_t != d else nn.Identity()

        # Halting target state q0 (learned).
        # Initialize with small values to prevent early instability
        self.q0 = nn.Parameter(torch.zeros(self.d_q))
        nn.init.normal_(self.q0, std=0.01)

    def _make_z(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.cfg.stochastic:
            return None
        b, t, d = x.shape
        z_len = max(1, int(round(self.cfg.z_ratio * t))) if t > 0 else 0
        if z_len <= 0:
            return None
        # Gaussian noise symbols (can be swapped for Rademacher).
        return torch.randn((b, z_len, d), device=x.device, dtype=x.dtype)

    def _gate(self, *, kappa: torch.Tensor, t: int, q: torch.Tensor) -> torch.Tensor:
        """
        Paper gate:
          g_t = sigmoid((-kappa * t) / max(eps, ||Q_t - q0||^2))

        kappa: (B,1), q: (B,d_q)
        """
        # ||Q_t - q0||^2 with eps for stability (use cfg.eps instead of 1.0 for better numerical stability).
        diff = q - self.q0.view(1, -1)
        dq2 = diff.pow(2).sum(dim=-1, keepdim=True)
        denom = torch.clamp(dq2, min=self.cfg.eps)
        g = torch.sigmoid((-kappa * float(t)) / denom)
        return g

    def forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        """
        x: (B,T,D) -> tape: (B,N_T,D)
        """
        if x.dim() != 3:
            raise ValueError(f"VecTurBlock expects (B,T,D), got {tuple(x.shape)}")

        b, t, d = x.shape
        if t == 0:
            return x

        # Build tape from x (+ optional stochastic symbols z).
        tape_x = self.m_t(x)  # (B,T,d_t)
        z = self._make_z(x)
        if z is not None:
            tape_z = self.m_t(z)  # (B,Tz,d_t)
            tape = torch.cat([tape_x, tape_z], dim=1)
        else:
            tape = tape_x
        n_tape = int(tape.shape[1])

        # Initialize Q, θ, w as linear functions of the sequence (sum over tokens).
        # This is a learnable linear map from vec(x) into the target spaces, without imposing constraints on w.
        q = self.m_q_tok(x).sum(dim=1)  # (B,d_q)
        theta = self.m_theta_tok(x).sum(dim=1)  # (B,k)
        w = self.m_w_tok(x).sum(dim=1)  # (B,k) unconstrained

        two_pi = float(2.0 * torch.pi)
        theta = torch.remainder(theta, two_pi)

        # κ(x) positive. We feed a cheap summary (sum) so κ participates in gradients.
        kappa_in = x.sum(dim=1)  # (B,D)
        kappa = torch.nn.functional.softplus(self.kappa_net(kappa_in)) + self.cfg.eps  # (B,1)

        k = min(int(self.cfg.k), int(n_tape))
        if k != int(self.cfg.k):
            theta = theta[:, :k]
            w = w[:, :k]

        # Flatten tape once; use index_add for sparse writes.
        tape_flat = tape.reshape(b * n_tape, self.d_t)
        base = (torch.arange(b, device=x.device, dtype=torch.long) * n_tape).view(b, 1)  # (B,1)

        def step_fn(tape_flat_: torch.Tensor, q_: torch.Tensor, theta_: torch.Tensor, w_: torch.Tensor, step: int):
            # Ensure θ in [0,2π).
            theta_mod = torch.remainder(theta_, two_pi)  # (B,k)
            u = (float(n_tape) * theta_mod) / two_pi  # (B,k)
            n = torch.floor(u).to(torch.long)  # (B,k)
            s = (u - torch.floor(u)).to(tape_flat_.dtype)  # (B,k) in [0,1)
            n_plus = (n + 1) % n_tape  # (B,k)

            # Read S_t using 2k gather + weighted sum.
            flat_n = (base + n).reshape(-1)  # (B*k,)
            flat_np = (base + n_plus).reshape(-1)  # (B*k,)
            t_n = tape_flat_.index_select(0, flat_n).reshape(b, k, self.d_t)
            t_np = tape_flat_.index_select(0, flat_np).reshape(b, k, self.d_t)
            weights_n = (w_ * (1.0 - s)).unsqueeze(-1)  # (B,k,1)
            weights_np = (w_ * s).unsqueeze(-1)  # (B,k,1)
            s_t = (t_n * weights_n + t_np * weights_np).sum(dim=1)  # (B,d_t)

            # Gate g_t.
            g = self._gate(kappa=kappa, t=step, q=q_)  # (B,1)

            # Early stop condition (control-flow). We stop outside step_fn.
            # Control update.
            dq = self.delta_q(torch.cat([s_t, q_], dim=-1))
            
            # Clip delta updates to prevent large steps that cause gradient explosion
            if self.cfg.clip_delta_q is not None:
                dq = torch.clamp(dq, min=-self.cfg.clip_delta_q, max=self.cfg.clip_delta_q)
            
            q_new = q_ + g * dq
            
            # Normalize Q state to prevent unbounded growth (optional)
            if self.cfg.normalize_q:
                q_norm = torch.linalg.vector_norm(q_new, dim=-1, keepdim=True)
                q_new = q_new / (q_norm + self.cfg.eps)

            # Head updates (per-head MLPs on [S,Q,sinθ,cosθ,w]).
            sin_th = torch.sin(theta_mod)
            cos_th = torch.cos(theta_mod)
            per_head = torch.cat(
                [
                    s_t.unsqueeze(1).expand(b, k, self.d_t),
                    q_new.unsqueeze(1).expand(b, k, self.d_q),
                    sin_th.unsqueeze(-1),
                    cos_th.unsqueeze(-1),
                    w_.unsqueeze(-1),
                ],
                dim=-1,
            )  # (B,k, d_t+d_q+3)

            dtheta = self.delta_theta(per_head).squeeze(-1)  # (B,k)
            dw = self.delta_w(per_head).squeeze(-1)  # (B,k)
            
            # Clip delta updates to prevent large steps
            if self.cfg.clip_delta_theta is not None:
                dtheta = torch.clamp(dtheta, min=-self.cfg.clip_delta_theta, max=self.cfg.clip_delta_theta)
            if self.cfg.clip_delta_w is not None:
                dw = torch.clamp(dw, min=-self.cfg.clip_delta_w, max=self.cfg.clip_delta_w)
            
            theta_new = torch.remainder(theta_ + g * dtheta, two_pi)
            w_new = w_ + g * dw
            
            # Clip w values to prevent extreme weights
            if self.cfg.clip_w is not None:
                w_new = torch.clamp(w_new, min=-self.cfg.clip_w, max=self.cfg.clip_w)

            # Tape write: U_t = Δ_T(S_t,Q_t) and scatter-add only to {n_i, n_i+1}.
            u_t = self.delta_t(torch.cat([s_t, q_new], dim=-1))  # (B,d_t)
            
            # Clip tape write updates to prevent large writes
            if self.cfg.clip_delta_t is not None:
                u_t = torch.clamp(u_t, min=-self.cfg.clip_delta_t, max=self.cfg.clip_delta_t)
            
            write_n = (g * (w_ * (1.0 - s))).unsqueeze(-1) * u_t.unsqueeze(1)  # (B,k,d_t)
            write_np = (g * (w_ * s)).unsqueeze(-1) * u_t.unsqueeze(1)  # (B,k,d_t)

            # Debug / shape safety
            vn_source = write_n.reshape(-1, self.d_t)
            if vn_source.shape[0] != flat_n.shape[0]:
                print(f"DEBUG: Shape mismatch in step_fn!")
                print(f"  flat_n: {flat_n.shape}")
                print(f"  write_n reshaped: {vn_source.shape}")
                print(f"  b={b}, k={k}, n_tape={n_tape}, d_t={self.d_t}")
            
            tape_flat_new = tape_flat_.index_add(0, flat_n, vn_source.to(tape_flat_.dtype))
            vn_source_p = write_np.reshape(-1, self.d_t)
            tape_flat_new = tape_flat_new.index_add(0, flat_np, vn_source_p.to(tape_flat_.dtype))
            
            # Clip tape values to prevent accumulation explosion
            if self.cfg.clip_tape is not None:
                tape_flat_new = torch.clamp(tape_flat_new, min=-self.cfg.clip_tape, max=self.cfg.clip_tape)
            
            return tape_flat_new, q_new, theta_new, w_new, g

        # Iterative transition
        for step in range(int(self.cfg.t_max)):
            if self.cfg.checkpoint_steps and self.training and torch.is_grad_enabled():
                tape_flat, q, theta, w, g = checkpoint(
                    lambda tf, qq, th, ww: step_fn(tf, qq, th, ww, step)[:5],  # type: ignore[misc]
                    tape_flat,
                    q,
                    theta,
                    w,
                    use_reentrant=False,
                )
            else:
                tape_flat, q, theta, w, g = step_fn(tape_flat, q, theta, w, step)

            if self.cfg.early_stop:
                # If all examples are below threshold, stop the whole batch early.
                if bool((g < float(self.cfg.halt_eps)).all().item()):
                    break

        tape_out = tape_flat.reshape(b, n_tape, self.d_t)
        return self.out(tape_out)


def make_vectur_block(
    *,
    dim: int,
    k: int = 8,
    t_max: int = 4,
    expansion: int = 4,
    clip_delta_q: float | None = 5.0,
    clip_delta_theta: float | None = 1.0,
    clip_delta_w: float | None = 2.0,
    clip_delta_t: float | None = 5.0,
    clip_tape: float | None = 10.0,
    clip_w: float | None = 10.0,
    normalize_q: bool = False,
) -> VecTurBlock:
    return VecTurBlock(
        VecTurConfig(
            dim=dim,
            k=k,
            t_max=t_max,
            expansion=expansion,
            stochastic=False,
            clip_delta_q=clip_delta_q,
            clip_delta_theta=clip_delta_theta,
            clip_delta_w=clip_delta_w,
            clip_delta_t=clip_delta_t,
            clip_tape=clip_tape,
            clip_w=clip_w,
            normalize_q=normalize_q,
        )
    )


def make_vecstur_block(
    *,
    dim: int,
    k: int = 8,
    t_max: int = 4,
    expansion: int = 4,
    z_ratio: float = 1.0,
    clip_delta_q: float | None = 5.0,
    clip_delta_theta: float | None = 1.0,
    clip_delta_w: float | None = 2.0,
    clip_delta_t: float | None = 5.0,
    clip_tape: float | None = 10.0,
    clip_w: float | None = 10.0,
    normalize_q: bool = False,
) -> VecTurBlock:
    return VecTurBlock(
        VecTurConfig(
            dim=dim,
            k=k,
            t_max=t_max,
            expansion=expansion,
            stochastic=True,
            z_ratio=float(z_ratio),
            clip_delta_q=clip_delta_q,
            clip_delta_theta=clip_delta_theta,
            clip_delta_w=clip_delta_w,
            clip_delta_t=clip_delta_t,
            clip_tape=clip_tape,
            clip_w=clip_w,
            normalize_q=normalize_q,
        )
    )
