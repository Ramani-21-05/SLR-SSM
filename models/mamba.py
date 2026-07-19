"""
A minimal, pure-PyTorch selective state-space block (Mamba-style).

This is a from-scratch, readable implementation of the core selective-scan
idea, NOT the official mamba-ssm package. It runs anywhere (CPU or GPU, no
custom CUDA kernel required), which matters for prototyping and for edge
deployment where compiling mamba-ssm's fused kernels may not be practical.

For production training speed, this class can be swapped for the official
`mamba_ssm.Mamba` block (pip install mamba-ssm, requires a CUDA build) --
the input/output shapes are compatible, so nothing else in the model needs
to change if you make that swap later.

Reference behaviour: selective SSM with input-dependent (delta, B, C),
causal depthwise conv for local context, and a gated output -- the same
ingredients as Mamba, implemented as a straightforward sequential scan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )

        # Input-dependent SSM parameters: delta, B, C are predicted per timestep.
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # A is a learned, negative, per-channel decay (log-parameterised for stability).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model), residual connection included."""
        residual = x
        B, T, _ = x.shape

        xz = self.in_proj(x)                      # (B, T, 2*d_inner)
        x_in, gate = xz.chunk(2, dim=-1)           # each (B, T, d_inner)

        x_in = x_in.transpose(1, 2)                # (B, d_inner, T)
        x_in = self.conv1d(x_in)[..., :T]          # causal conv, trim padding
        x_in = F.silu(x_in).transpose(1, 2)        # (B, T, d_inner)

        params = self.x_proj(x_in)                 # (B, T, 2*d_state + d_inner)
        delta_raw, B_ssm, C_ssm = torch.split(
            params, [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner)

        A = -torch.exp(self.A_log)                 # (d_inner, d_state), stays negative

        y = self._selective_scan(x_in, delta, A, B_ssm, C_ssm)
        y = y + x_in * self.D

        y = y * F.silu(gate)
        out = self.out_proj(y)

        return self.norm(out + residual)

    @staticmethod
    def _selective_scan(x, delta, A, B_ssm, C_ssm):
        """
        Sequential recurrence over time. Simpler than a fused parallel scan,
        but correct and easy to verify -- fine for research-scale sequences
        (STNet caps T at 200 frames).
        """
        Bsz, T, d_inner = x.shape
        d_state = A.shape[1]

        h = torch.zeros(Bsz, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        # Discretise A per-timestep via delta (zero-order hold approximation).
        dA = torch.exp(delta.unsqueeze(-1) * A)                 # (B, T, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B_ssm.unsqueeze(2)            # (B, T, d_inner, d_state)

        for t in range(T):
            h = h * dA[:, t] + x[:, t].unsqueeze(-1) * dB[:, t]
            y_t = torch.einsum("bds,bs->bd", h, C_ssm[:, t])
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, T, d_inner)
