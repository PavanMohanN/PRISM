"""PRISM Phase 1 -- support transforms with correct change-of-variables.

Each transform c maps an UNCONSTRAINED coordinate u to the feasible set C, is a
smooth bijection onto the open feasible set, and exposes:

    forward(u)                      ->  x = c(u)           (physical, feasible)
    inverse(x)                      ->  u = c^{-1}(x)
    log_abs_det_forward(u)          ->  log|det dc/du|
    log_abs_det_inverse(x)          ->  log|det d c^{-1}/dx|   ( = -forward at u )

The inverse log-det is the term that MUST be added to the physical-space NLL
(guide item 8):

    log p(x|y) = log p(u|y) + log_abs_det_inverse(x),   u = c^{-1}(x).

All maps are validated against numerical normalization checks in
tests/test_transforms.py (pushforward density integrates to 1.0). The softmax is
deliberately NOT provided: it is non-injective and dimension-mismatched. Use
StickBreaking for simplex constraints (guide item 7).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SupportTransform:
    """Base class. constrained_dim / unconstrained_dim differ only for simplex."""

    #: physical (constrained) dimension
    constrained_dim: int
    #: latent (unconstrained) dimension the flow operates in
    unconstrained_dim: int

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # u -> x
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:  # x -> u
        raise NotImplementedError

    def log_abs_det_forward(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_abs_det_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_abs_det_forward(self.inverse(x))

    # convenience: is a physical point feasible? (should always be True for
    # anything produced by forward())
    def is_feasible(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Identity(SupportTransform):
    def __init__(self, dim: int):
        self.constrained_dim = self.unconstrained_dim = dim

    def forward(self, u):
        return u

    def inverse(self, x):
        return x

    def log_abs_det_forward(self, u):
        return torch.zeros(u.shape[:-1], device=u.device, dtype=u.dtype)

    def log_abs_det_inverse(self, x):
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)

    def is_feasible(self, x):
        return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)


class Positive(SupportTransform):
    """c(u) = softplus(u) -> (0, inf).  Stable inverse and log-det."""

    def __init__(self, dim: int):
        self.constrained_dim = self.unconstrained_dim = dim

    def forward(self, u):
        return F.softplus(u)

    def inverse(self, x):
        # u = log(e^x - 1) = x + log(1 - e^{-x})   (stable for x > 0)
        return x + torch.log(-torch.expm1(-x))

    def log_abs_det_forward(self, u):
        # log|c'(u)| = log sigmoid(u)   (summed over event dim)
        return F.logsigmoid(u).sum(dim=-1)

    def log_abs_det_inverse(self, x):
        # = -log(1 - e^{-x})  summed
        return (-torch.log(-torch.expm1(-x))).sum(dim=-1)

    def is_feasible(self, x):
        return (x > 0).all(dim=-1)


class Box(SupportTransform):
    """c(u) = a + (b-a) sigmoid(u) -> (a, b)."""

    def __init__(self, lower, upper, dim: int | None = None):
        self.lower = torch.as_tensor(lower, dtype=torch.get_default_dtype())
        self.upper = torch.as_tensor(upper, dtype=torch.get_default_dtype())
        d = dim if dim is not None else int(self.lower.numel())
        self.constrained_dim = self.unconstrained_dim = d

    def _ab(self, ref):
        a = self.lower.to(ref); b = self.upper.to(ref)
        return a, b

    def forward(self, u):
        a, b = self._ab(u)
        return a + (b - a) * torch.sigmoid(u)

    def inverse(self, x):
        a, b = self._ab(x)
        z = (x - a) / (b - a)
        return torch.log(z) - torch.log1p(-z)  # logit(z)

    def log_abs_det_forward(self, u):
        a, b = self._ab(u)
        width = torch.log(b - a)
        # log|c'(u)| = log(b-a) + log sigma(u) + log(1 - sigma(u))
        return (width + F.logsigmoid(u) + F.logsigmoid(-u)).sum(dim=-1)

    def log_abs_det_inverse(self, x):
        a, b = self._ab(x)
        z = (x - a) / (b - a)
        return (-(torch.log(b - a) + torch.log(z) + torch.log1p(-z))).sum(dim=-1)

    def is_feasible(self, x):
        a, b = self._ab(x)
        return ((x > a) & (x < b)).all(dim=-1)


class StickBreaking(SupportTransform):
    """Dimension-matched simplex map R^{d-1} -> int(Delta^{d-1}).

    z_k = sigmoid(u_k);  x_1 = z_1;  x_k = z_k * prod_{j<k}(1 - z_j);
    x_d = prod_{j<d}(1 - z_j).  Bijection onto the open simplex; softmax-free.
    (Guide item 7.)
    """

    def __init__(self, dim: int):
        assert dim >= 2
        self.constrained_dim = dim
        self.unconstrained_dim = dim - 1

    def forward(self, u):
        z = torch.sigmoid(u)                     # (..., d-1)
        rem = torch.cumprod(1 - z, dim=-1)       # prod_{j<=k}(1-z_j)
        rem_prev = torch.cat(
            [torch.ones_like(rem[..., :1]), rem[..., :-1]], dim=-1
        )                                        # prod_{j<k}(1-z_j)
        x_head = z * rem_prev                    # x_1..x_{d-1}
        x_last = rem[..., -1:]                   # x_d
        return torch.cat([x_head, x_last], dim=-1)

    def inverse(self, x):
        x_head = x[..., :-1]                      # x_1..x_{d-1}
        csum = torch.cumsum(x_head, dim=-1)
        rem_prev = 1 - torch.cat(
            [torch.zeros_like(csum[..., :1]), csum[..., :-1]], dim=-1
        )                                        # 1 - sum_{j<k} x_j
        z = x_head / rem_prev
        return torch.log(z) - torch.log1p(-z)

    def log_abs_det_forward(self, u):
        z = torch.sigmoid(u)
        d_minus_1 = u.shape[-1]
        # dz/du term
        term = (F.logsigmoid(u) + F.logsigmoid(-u)).sum(dim=-1)
        # triangular z->x term: sum_k (d-2-k) log(1 - z_k), k = 0..d-2
        k = torch.arange(d_minus_1, device=u.device, dtype=u.dtype)
        exponent = (d_minus_1 - 1) - k           # = d-2-k in 0-indexed
        term = term + (exponent * torch.log1p(-z)).sum(dim=-1)
        return term

    def is_feasible(self, x):
        return (x > 0).all(dim=-1) & torch.isclose(
            x.sum(dim=-1), torch.ones_like(x.sum(dim=-1)), atol=1e-5
        )


def make_transform(kind: str, dim: int, **kw) -> SupportTransform:
    kind = kind.lower()
    if kind in ("identity", "none"):
        return Identity(dim)
    if kind in ("positive", "pos", "softplus"):
        return Positive(dim)
    if kind in ("box", "interval"):
        return Box(kw["lower"], kw["upper"], dim)
    if kind in ("simplex", "stick_breaking", "stickbreaking"):
        return StickBreaking(dim)
    raise ValueError(f"unknown transform: {kind}")
