"""Hard physical-constraint layers.

Each constraint is a fixed, differentiable bijection between an unconstrained
coordinate space and the feasible set. PRISM models parameters in the
unconstrained space; the constraint's ``forward`` maps every prediction into
the feasible set, so the constraint-violation rate is exactly 0 by
construction (not via a penalty). ``inverse`` encodes feasible data back to the
unconstrained space for training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _softplus_inv(y, beta=1.0):
    # numerically-stable inverse of softplus
    return y + torch.log((-torch.expm1(-beta * y)).clamp_min(1e-12)) / beta


class Constraint:
    name = "identity"

    def forward(self, u):  # unconstrained -> feasible
        return u

    def inverse(self, x):  # feasible -> unconstrained
        return x

    def violation(self, x):  # per-sample max violation magnitude (>=0)
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)


class Identity(Constraint):
    name = "identity"


class Positive(Constraint):
    """x > 0 via softplus. Guarantees strict positivity."""
    name = "positive"

    def forward(self, u):
        return F.softplus(u) + 1e-8

    def inverse(self, x):
        return _softplus_inv((x - 1e-8).clamp_min(1e-8))

    def violation(self, x):
        return (-(x)).clamp_min(0.0).reshape(x.shape[0], -1).max(dim=1).values


class Box(Constraint):
    """lo <= x <= hi via scaled sigmoid."""
    name = "box"

    def __init__(self, lo=0.0, hi=1.0):
        self.lo, self.hi = float(lo), float(hi)

    def forward(self, u):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(u)

    def inverse(self, x):
        p = ((x - self.lo) / (self.hi - self.lo)).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def violation(self, x):
        below = (self.lo - x).clamp_min(0.0)
        above = (x - self.hi).clamp_min(0.0)
        v = torch.maximum(below, above).reshape(x.shape[0], -1)
        return v.max(dim=1).values


class Simplex(Constraint):
    """x_i >= 0 and sum_i x_i = 1 via softmax (conservation constraint)."""
    name = "simplex"

    def forward(self, u):
        return torch.softmax(u, dim=-1)

    def inverse(self, x):
        return torch.log(x.clamp_min(1e-8))

    def violation(self, x):
        neg = (-(x)).clamp_min(0.0).reshape(x.shape[0], -1).max(dim=1).values
        consv = (x.reshape(x.shape[0], -1).sum(dim=1) - 1.0).abs()
        return torch.maximum(neg, consv)


class Monotone(Constraint):
    """Non-decreasing vector via cumulative softplus (output guarantee)."""
    name = "monotone"

    def forward(self, u):
        first = u[:, :1]
        steps = F.softplus(u[:, 1:])
        return torch.cat([first, first + torch.cumsum(steps, dim=1)], dim=1)

    def inverse(self, x):
        first = x[:, :1]
        diffs = (x[:, 1:] - x[:, :-1]).clamp_min(1e-6)
        return torch.cat([first, _softplus_inv(diffs)], dim=1)

    def violation(self, x):
        d = x[:, 1:] - x[:, :-1]
        return (-d).clamp_min(0.0).reshape(x.shape[0], -1).max(dim=1).values


_REGISTRY = {
    "none": Identity, "identity": Identity, "positive": Positive,
    "box": Box, "simplex": Simplex, "monotone": Monotone,
}


def get_constraint(spec):
    """spec: str name, a Constraint instance, or ('box', {'lo':..,'hi':..})."""
    if isinstance(spec, Constraint):
        return spec
    if isinstance(spec, (tuple, list)):
        name, kw = spec
        return _REGISTRY[name](**kw)
    if spec is None:
        return Identity()
    return _REGISTRY[spec]()
