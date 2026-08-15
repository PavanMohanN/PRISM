"""Exactly-invertible continuous normalizing flow (liquid-ODE).

The map T: v(0) -> v(1) is obtained by integrating dv/dt = g(t, v) from 0 to 1.
The inverse T^{-1} integrates the *same* field from 1 to 0, so invertibility is
architectural: the cycle error equals the ODE solver tolerance, not a trained
penalty. The change-of-variables log-det is accumulated as an augmented ODE
state (exact trace for small dims, Hutchinson estimator otherwise).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint


def _exact_trace(dv, v):
    """Exact tr(d g / d v). O(dim) autograd passes; use for small dims."""
    tr = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    for i in range(v.shape[1]):
        gi = torch.autograd.grad(dv[:, i].sum(), v, create_graph=True,
                                 retain_graph=True)[0][:, i]
        tr = tr + gi
    return tr


def _hutch_trace(dv, v, eps):
    """Hutchinson estimator eps^T (dg/dv) eps; O(1) autograd passes."""
    vjp = torch.autograd.grad((dv * eps).sum(), v, create_graph=True,
                              retain_graph=True)[0]
    return (vjp * eps).sum(dim=1)


class _Aug(nn.Module):
    """Augmented dynamics returning (dv/dt, d logdet/dt)."""

    def __init__(self, g, trace_mode="exact"):
        super().__init__()
        self.g = g
        self.trace_mode = trace_mode
        self._eps = None  # fixed Hutchinson noise per integration

    def forward(self, t, state):
        v, _ = state
        with torch.enable_grad():
            v = v.requires_grad_(True)
            dv = self.g(t, v)
            if self.trace_mode == "exact":
                tr = _exact_trace(dv, v)
            else:
                if self._eps is None:
                    self._eps = torch.randint_like(v, low=0, high=2) * 2.0 - 1.0
                tr = _hutch_trace(dv, v, self._eps)
        return dv, -tr


class _Plain(nn.Module):
    """Dynamics returning only dv/dt (no log-det), for fast sampling/inversion."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, t, v):
        return self.g(t, v)


class LiquidODEFlow(nn.Module):
    """Continuous normalizing flow with exact reverse-time inversion.

    Parameters
    ----------
    velocity : nn.Module     the field g(t, v)
    solver   : str           torchdiffeq method ('dopri5', 'rk4', ...)
    rtol, atol : float        adaptive-solver tolerances
    n_steps  : int|None       if set with a fixed solver, uses step_size=1/n_steps
    trace_mode : 'exact'|'hutch'
    use_adjoint : bool        memory-efficient adjoint backprop
    """

    def __init__(self, velocity, solver="dopri5", rtol=1e-5, atol=1e-6,
                 n_steps=None, trace_mode="exact", use_adjoint=False):
        super().__init__()
        self.g = velocity
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.n_steps = n_steps
        self.trace_mode = trace_mode
        self.use_adjoint = use_adjoint

    def _opts(self):
        opts = {}
        if self.n_steps is not None and self.solver in ("rk4", "euler", "midpoint"):
            opts["options"] = {"step_size": 1.0 / self.n_steps}
        else:
            opts["rtol"] = self.rtol
            opts["atol"] = self.atol
        return opts

    def _integrate(self, func, y0, reverse=False):
        t = torch.tensor([0.0, 1.0], dtype=torch.float32,
                         device=self._device())
        if reverse:
            t = torch.flip(t, dims=[0])
        solve = odeint_adjoint if self.use_adjoint else odeint
        out = solve(func, y0, t, method=self.solver, **self._opts())
        return out

    def _device(self):
        return next(self.g.parameters()).device

    def forward(self, x, context=None, logdet=False):
        """T(x | context). If logdet, also return log|det dT/dx|."""
        if context is not None:
            self.g.set_context(context)
        if logdet:
            func = _Aug(self.g, self.trace_mode)
            func._eps = None
            ld0 = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            v, ld = self._integrate(func, (x, ld0), reverse=False)
            return v[-1], ld[-1]
        else:
            v = self._integrate(_Plain(self.g), x, reverse=False)
            return v[-1]

    def inverse(self, u, context=None, logdet=False):
        """T^{-1}(u | context). Integrates the same field backward in time."""
        if context is not None:
            self.g.set_context(context)
        if logdet:
            func = _Aug(self.g, self.trace_mode)
            func._eps = None
            ld0 = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
            v, ld = self._integrate(func, (u, ld0), reverse=True)
            return v[-1], ld[-1]
        else:
            v = self._integrate(_Plain(self.g), u, reverse=True)
            return v[-1]
