"""Standard-normal latent base used by PRISM's posterior block.

The flow maps parameters x to an output split [y_hat, z]; the latent block z is
encouraged toward N(0, I). To draw posterior samples for a given observation y,
we sample z ~ N(0, I), pair it with y, and invert the flow. This is the
invertible-network route to Bayesian inverse problems.
"""
from __future__ import annotations

import math
import torch


class GaussianBase:
    def __init__(self, dim: int):
        self.dim = dim

    def log_prob(self, z):
        return -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * math.log(2 * math.pi)

    def sample(self, n, device="cpu", dtype=torch.float32):
        return torch.randn(n, self.dim, device=device, dtype=dtype)


def rbf_mmd(x, y, sigmas=(0.5, 1.0, 2.0, 4.0)):
    """Unbiased-ish RBF MMD^2 between two sample sets (used as a light
    distribution-matching term for the reverse/posterior pass)."""
    def _k(a, b):
        aa = (a ** 2).sum(1, keepdim=True)
        bb = (b ** 2).sum(1, keepdim=True)
        d2 = (aa - 2 * a @ b.t() + bb.t()).clamp_min(0.0)
        out = 0.0
        for s in sigmas:
            out = out + torch.exp(-d2 / (2 * s * s))
        return out / len(sigmas)

    return _k(x, x).mean() + _k(y, y).mean() - 2 * _k(x, y).mean()
