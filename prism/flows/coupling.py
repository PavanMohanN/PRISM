"""Conditional affine-coupling flow (RealNVP-style).

A discrete normalizing flow that is exactly invertible with an analytic
log-determinant (no ODE solve). It conditions each coupling layer on an
observation embedding, giving a conditional density p(x | y). This is the
backbone of the cINN baseline and the discrete counterpart to PRISM's
continuous liquid-ODE flow.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .liquid_ode import _mlp


class _Coupling(nn.Module):
    """Affine coupling: transforms one half of x conditioned on the other half
    and the context. Exactly invertible; log-det is the sum of log-scales."""

    def __init__(self, dim, context_dim, hidden, depth, mask):
        super().__init__()
        self.register_buffer("mask", mask)
        in_dim = dim + context_dim
        self.net = _mlp([in_dim] + [hidden] * depth + [2 * dim])
        # zero-init last layer => starts as identity (stable training)
        last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def _st(self, x, context):
        xm = x * self.mask
        h = xm if context is None else torch.cat([xm, context], dim=-1)
        s, t = self.net(h).chunk(2, dim=-1)
        s = torch.tanh(s) * (1 - self.mask)        # bound + only on free dims
        t = t * (1 - self.mask)
        return s, t

    def forward(self, x, context):
        s, t = self._st(x, context)
        z = x * torch.exp(s) + t
        return z, s.sum(dim=-1)

    def inverse(self, z, context):
        s, t = self._st(z, context)               # mask half is identical
        x = (z - t) * torch.exp(-s)
        return x


class ConditionalCouplingFlow(nn.Module):
    def __init__(self, dim, context_dim=0, hidden=64, depth=2, n_layers=6):
        super().__init__()
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[::2] = 1.0 if i % 2 == 0 else 0.0
            mask[1::2] = 0.0 if i % 2 == 0 else 1.0
            if dim == 1:                           # degenerate: alternate scalar
                mask = torch.tensor([float(i % 2)])
            layers.append(_Coupling(dim, context_dim, hidden, depth, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, context=None):
        ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, dld = layer(x, context)
            ld = ld + dld
        return x, ld

    def inverse(self, z, context=None):
        for layer in reversed(self.layers):
            z = layer.inverse(z, context)
        return z
