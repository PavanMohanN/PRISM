"""Amortized VI baseline — Gaussian posterior q(x|y) = N(mu(y), diag(sigma(y))).

A fast, unimodal amortized posterior trained by maximum likelihood on (x, y)
pairs. It cannot represent skewed/multimodal posteriors, isolating the value of
PRISM's flexible flow while sharing the amortized-inference setup.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from prism.models.base import BaseEstimator, _Scaler
from ._common import mlp, resolve_device


class AmortizedVI(BaseEstimator):
    def __init__(self, hidden=64, depth=2, lr=1e-3, epochs=200, batch_size=256,
                 device="auto", seed=0, verbose=False):
        self.hidden = hidden; self.depth = depth; self.lr = lr
        self.epochs = epochs; self.batch_size = batch_size
        self.device = device; self.seed = seed; self.verbose = verbose

    def fit(self, X, Y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.device_ = resolve_device(self.device)
        X = self._as2d(X); Y = self._as2d(Y)
        self.sx_ = _Scaler().fit(X); self.sy_ = _Scaler().fit(Y)
        Xs = self._to_t(self.sx_.transform(X)); Ys = self._to_t(self.sy_.transform(Y))
        d, N = Xs.shape[1], Ys.shape[1]; self.d_ = d
        self.head_ = mlp([N] + [self.hidden] * self.depth + [2 * d]).to(self.device_)
        opt = torch.optim.Adam(self.head_.parameters(), lr=self.lr)
        n = Xs.shape[0]; bs = min(self.batch_size, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; xb, yb = Xs[idx], Ys[idx]
                opt.zero_grad()
                mu, logs = self.head_(yb).chunk(2, dim=-1)
                sigma = F.softplus(logs) + 1e-3
                nll = (0.5 * ((xb - mu) / sigma) ** 2 + torch.log(sigma)).sum(1).mean()
                nll.backward(); opt.step()
        return self

    def _params(self, Y):
        ys = self._to_t(self.sy_.transform(self._as2d(Y)))
        mu, logs = self.head_(ys).chunk(2, dim=-1)
        return mu, F.softplus(logs) + 1e-3

    @torch.no_grad()
    def invert(self, Y):
        mu, _ = self._params(Y)
        return self.sx_.inverse_transform(mu.cpu().numpy())

    @torch.no_grad()
    def predict_posterior(self, Y, n_samples=1000):
        mu, sigma = self._params(Y); m = mu.shape[0]
        out = np.empty((m, n_samples, self.d_))
        for s in range(n_samples):
            z = mu + sigma * torch.randn_like(mu)
            out[:, s, :] = self.sx_.inverse_transform(z.cpu().numpy())
        return out

    def sample(self, Y, n_samples=1000):
        return self.predict_posterior(Y, n_samples)
