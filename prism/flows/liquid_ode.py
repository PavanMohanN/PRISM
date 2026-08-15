"""Liquid (time-constant) ODE velocity field.

This is the right-hand side g(t, v) = dv/dt of PRISM's continuous-time flow.
It uses an LTC-style state-dependent ("liquid") effective time constant so the
decay rate of each coordinate depends on the current state, which is what
distinguishes PRISM's dynamics from a static neural-ODE field.

Setting ``liquid=False`` removes the state-dependent gate (static neural-ODE),
which is exactly the "liquid vs static" ablation reported in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class LiquidVelocity(nn.Module):
    """dv/dt for a continuous normalizing flow.

    liquid:   dv/dt = -(1/tau + s(v,t)) * v + s(v,t) * A
    static:   dv/dt = -(1/tau) * v        + h(v,t)

    where s in (0,1) is a learned state-dependent gate (the "liquid" term),
    A is a learned attractor, tau>0 a per-dim time constant, and h a plain
    MLP field used in the static ablation.
    """

    def __init__(self, dim: int, hidden: int = 64, depth: int = 2,
                 liquid: bool = True, context_dim: int = 0):
        super().__init__()
        self.dim = dim
        self.liquid = liquid
        self.context_dim = context_dim
        sizes = [dim + 1 + context_dim] + [hidden] * depth + [dim]
        self.net = _mlp(sizes)
        self.tau_raw = nn.Parameter(torch.zeros(dim))      # softplus -> tau>0
        self.A = nn.Parameter(torch.zeros(dim))            # attractor
        self._ctx = None
        # small init for stable ODE dynamics
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def set_context(self, c):
        """Fix the conditioning vector c (B, context_dim) for an integration."""
        self._ctx = c

    def forward(self, t, v):
        # t: scalar tensor; v: (B, dim)
        if t.dim() == 0:
            t = t.expand(v.shape[0], 1)
        else:
            t = t.reshape(-1, 1).expand(v.shape[0], 1)
        parts = [v, t]
        if self.context_dim > 0 and self._ctx is not None:
            parts.append(self._ctx)
        raw = self.net(torch.cat(parts, dim=-1))           # (B, dim)
        tau = F.softplus(self.tau_raw) + 1e-3
        if self.liquid:
            s = torch.sigmoid(raw)                         # gate in (0,1)
            return -(1.0 / tau + s) * v + s * self.A
        else:
            return -(1.0 / tau) * v + torch.tanh(raw)
