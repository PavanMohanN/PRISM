"""PRISM Phase 1 -- cycle metrics (guide item 1).

We must stop conflating two different quantities:

  * LATENT reversibility (guaranteed by construction, Proposition 1):
        || F^{-1}(F(x_tilde)) - x_tilde ||
    -- a property of the ODE transport alone.

  * PHYSICAL cycle consistency (EMPIRICAL, not guaranteed):
        || h(g_psi(x)) - x ||
    where g_psi is the auxiliary forward surrogate and h is the amortized
    inverse path y -> mu_phi(y) -> F^{-1} -> c. There is no theorem forcing
    this to be small; it must be measured and reported as empirical.

Both helpers take plain callables so they are robust to the exact estimator API.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def _norm(a, b):
    a = torch.as_tensor(a, dtype=torch.get_default_dtype())
    b = torch.as_tensor(b, dtype=torch.get_default_dtype())
    return torch.linalg.vector_norm(a - b, dim=-1)


def latent_cycle_error(flow, X_tilde) -> dict:
    """|| F^{-1}(F(x_tilde)) - x_tilde ||  in the unconstrained coordinate."""
    X_tilde = torch.as_tensor(X_tilde, dtype=torch.get_default_dtype())
    z = flow.forward(X_tilde)
    x_rec = flow.inverse(z)
    err = _norm(x_rec, X_tilde)
    return {"latent_cycle_mean": err.mean().item(),
            "latent_cycle_max": err.max().item()}


def physical_cycle_error(forward_fn: Callable, inverse_fn: Callable, X) -> dict:
    """|| h(g_psi(x)) - x ||  in physical coordinates (EMPIRICAL).

    forward_fn:  x -> y_hat   (the auxiliary surrogate g_psi)
    inverse_fn:  y -> x_hat   (the amortized inverse path h, incl. projection)
    """
    X = torch.as_tensor(X, dtype=torch.get_default_dtype())
    y_hat = forward_fn(X)
    x_rec = torch.as_tensor(inverse_fn(y_hat), dtype=torch.get_default_dtype())
    err = _norm(x_rec, X)
    denom = torch.linalg.vector_norm(X, dim=-1).clamp_min(1e-12)
    rel = (err / denom)
    return {"physical_cycle_mean": err.mean().item(),
            "physical_cycle_max": err.max().item(),
            "physical_cycle_rel_mean": rel.mean().item()}


def cycle_report(flow, forward_fn, inverse_fn, X_tilde, X) -> dict:
    """Both metrics side by side, for the reversibility table/figure."""
    out = {}
    out.update(latent_cycle_error(flow, X_tilde))
    out.update(physical_cycle_error(forward_fn, inverse_fn, X))
    return out
