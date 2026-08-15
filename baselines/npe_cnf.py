"""NPE-CNF baseline — neural posterior estimation with a continuous flow.

A conditional continuous normalizing flow with a *static* (non-liquid) velocity
field. The observation conditions both the velocity (concatenated context) and
the base mean. Trained by exact maximum likelihood. This represents prior
continuous-flow posterior methods; the contrast with PRISM isolates the value
of the liquid dynamics, amortized base, and hard constraints.
"""
from __future__ import annotations

import numpy as np
import torch

from prism.models.base import BaseEstimator, _Scaler
from prism.flows.liquid_ode import LiquidVelocity
from prism.flows.invertible_flow import LiquidODEFlow
from ._common import mlp, resolve_device


class NPECNF(BaseEstimator):
    def __init__(self, c_dim=32, hidden=64, depth=2, solver="rk4", n_steps=5,
                 rtol=1e-5, atol=1e-6, lr=1e-3, epochs=200, batch_size=256,
                 w_fwd=1.0, device="auto", seed=0, verbose=False):
        self.c_dim = c_dim; self.hidden = hidden; self.depth = depth
        self.solver = solver; self.n_steps = n_steps; self.rtol = rtol; self.atol = atol
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.w_fwd = w_fwd; self.device = device; self.seed = seed; self.verbose = verbose

    def fit(self, X, Y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.device_ = resolve_device(self.device)
        X = self._as2d(X); Y = self._as2d(Y)
        self.sx_ = _Scaler().fit(X); self.sy_ = _Scaler().fit(Y)
        Xs = self._to_t(self.sx_.transform(X)); Ys = self._to_t(self.sy_.transform(Y))
        d, N = Xs.shape[1], Ys.shape[1]; self.d_ = d
        self.embed_ = mlp([N] + [self.hidden] * self.depth + [self.c_dim]).to(self.device_)
        self.mean_ = mlp([N] + [self.hidden] * self.depth + [d]).to(self.device_)
        self.fwd_head_ = mlp([d] + [self.hidden] * self.depth + [N]).to(self.device_)
        vel = LiquidVelocity(d, self.hidden, self.depth, liquid=False, context_dim=self.c_dim)
        self.flow_ = LiquidODEFlow(vel.to(self.device_), solver=self.solver,
                                   rtol=self.rtol, atol=self.atol, n_steps=self.n_steps,
                                   trace_mode="exact" if d <= 12 else "hutch").to(self.device_)
        params = (list(self.flow_.parameters()) + list(self.embed_.parameters()) +
                  list(self.mean_.parameters()) + list(self.fwd_head_.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr)
        n = Xs.shape[0]; bs = min(self.batch_size, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb, yb = Xs[idx], Ys[idx]
                opt.zero_grad()
                c = self.embed_(yb); mu0 = self.mean_(yb)
                z, ld = self.flow_.forward(xb, context=c, logdet=True)
                nll = (0.5 * ((z - mu0) ** 2).sum(1) + ld).mean()
                loss = nll + self.w_fwd * ((self.fwd_head_(xb) - yb) ** 2).mean()
                loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        xs = self._to_t(self.sx_.transform(self._as2d(X)))
        return self.sy_.inverse_transform(self.fwd_head_(xs).cpu().numpy())

    @torch.no_grad()
    def invert(self, Y):
        ys = self._to_t(self.sy_.transform(self._as2d(Y)))
        c = self.embed_(ys); mu0 = self.mean_(ys)
        xs = self.flow_.inverse(mu0, context=c)
        return self.sx_.inverse_transform(xs.cpu().numpy())

    @torch.no_grad()
    def predict_posterior(self, Y, n_samples=1000):
        ys = self._to_t(self.sy_.transform(self._as2d(Y)))
        c = self.embed_(ys); mu0 = self.mean_(ys); m = ys.shape[0]
        out = np.empty((m, n_samples, self.d_))
        for s in range(n_samples):
            z = mu0 + torch.randn_like(mu0)
            xs = self.flow_.inverse(z, context=c)
            out[:, s, :] = self.sx_.inverse_transform(xs.cpu().numpy())
        return out

    def sample(self, Y, n_samples=1000):
        return self.predict_posterior(Y, n_samples)
