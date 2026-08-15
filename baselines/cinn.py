"""cINN baseline — conditional invertible neural network (coupling flow).

A fully-implemented posterior estimator using a discrete affine-coupling flow
conditioned on the observation. Trained by exact maximum likelihood. Serves as
the primary "different flow family" comparison to PRISM's continuous liquid-ODE
flow. Unlike PRISM it has no hard-constraint layer, so its samples may violate
physical constraints (quantified in the constraint table).
"""
from __future__ import annotations

import numpy as np
import torch

from prism.models.base import BaseEstimator, _Scaler
from prism.flows.coupling import ConditionalCouplingFlow
from ._common import mlp, resolve_device


class CINN(BaseEstimator):
    def __init__(self, c_dim=32, hidden=64, depth=2, n_layers=6, lr=1e-3,
                 epochs=200, batch_size=256, w_fwd=1.0, device="auto",
                 seed=0, verbose=False):
        self.c_dim = c_dim; self.hidden = hidden; self.depth = depth
        self.n_layers = n_layers; self.lr = lr; self.epochs = epochs
        self.batch_size = batch_size; self.w_fwd = w_fwd
        self.device = device; self.seed = seed; self.verbose = verbose

    def fit(self, X, Y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.device_ = resolve_device(self.device)
        X = self._as2d(X); Y = self._as2d(Y)
        self.sx_ = _Scaler().fit(X); self.sy_ = _Scaler().fit(Y)
        Xs = self._to_t(self.sx_.transform(X)); Ys = self._to_t(self.sy_.transform(Y))
        d, N = Xs.shape[1], Ys.shape[1]
        self.d_ = d
        self.embed_ = mlp([N] + [self.hidden] * self.depth + [self.c_dim]).to(self.device_)
        self.fwd_head_ = mlp([d] + [self.hidden] * self.depth + [N]).to(self.device_)
        self.flow_ = ConditionalCouplingFlow(d, self.c_dim, self.hidden,
                                             self.depth, self.n_layers).to(self.device_)
        params = list(self.flow_.parameters()) + list(self.embed_.parameters()) + \
            list(self.fwd_head_.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        n = Xs.shape[0]; bs = min(self.batch_size, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = Xs[idx], Ys[idx]
                opt.zero_grad()
                c = self.embed_(yb)
                z, ld = self.flow_.forward(xb, c)
                nll = (0.5 * (z ** 2).sum(1) - ld).mean()
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
        c = self.embed_(ys)
        z0 = torch.zeros(ys.shape[0], self.d_, device=self.device_)
        xs = self.flow_.inverse(z0, c)
        return self.sx_.inverse_transform(xs.cpu().numpy())

    @torch.no_grad()
    def predict_posterior(self, Y, n_samples=1000):
        ys = self._to_t(self.sy_.transform(self._as2d(Y)))
        c = self.embed_(ys); m = ys.shape[0]
        out = np.empty((m, n_samples, self.d_))
        for s in range(n_samples):
            z = torch.randn(m, self.d_, device=self.device_)
            out[:, s, :] = self.sx_.inverse_transform(self.flow_.inverse(z, c).cpu().numpy())
        return out

    def sample(self, Y, n_samples=1000):
        return self.predict_posterior(Y, n_samples)

    @torch.no_grad()
    def cycle_consistency_error(self, X):
        """Relative reconstruction error of the coupling flow (analytic inverse)."""
        xs = self._to_t(self.sx_.transform(self._as2d(X)))
        c = self.embed_(self.fwd_head_(xs))
        z, _ = self.flow_.forward(xs, c)
        x_rec = self.flow_.inverse(z, c)
        num = (x_rec - xs).norm(dim=1)
        den = xs.norm(dim=1).clamp_min(1e-12)
        return float((num / den).mean())
