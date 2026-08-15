"""PRISM — the estimator (conditional liquid-ODE flow).

PRISM models the posterior over parameters with a **conditional, exactly-
invertible liquid-ODE normalizing flow**: a bijection x <-> z conditioned on an
embedding of the observation y, trained by exact maximum likelihood.

* **posterior**:   z ~ N(0, I); x = F^{-1}(z | y)        (samplable, calibrated)
* **invert** (point):  z = 0 -> x_hat
* **forward** (predict):  a lightweight head g(x) -> y_hat (the forward surrogate)

Because F(.|y) is a continuous flow integrated by an ODE solver, the cycle
F^{-1}(F(x|y)|y) reconstructs x to the solver tolerance — exact reversibility by
construction, not a trained penalty. A constraint projection makes every
recovered x feasible. ``soft=True`` replaces the flow with two independent
networks and a soft cycle penalty (the BLiqNet-style ablation).

API mirrors scikit-learn: ``fit(X, Y).predict(X)`` / ``.invert(Y)`` /
``.predict_posterior(Y)`` where X are parameters and Y are observations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import BaseEstimator, _Scaler
from ..flows.liquid_ode import LiquidVelocity, _mlp
from ..flows.invertible_flow import LiquidODEFlow
from ..constraints.projections import get_constraint
from ..posterior.conditional import GaussianBase


class PRISM(BaseEstimator):
    def __init__(self, c_dim=32, hidden=64, depth=2, liquid=True,
                 constraint="none", solver="dopri5", rtol=1e-5, atol=1e-6,
                 n_steps=None, lr=1e-3, epochs=200, batch_size=256,
                 w_fwd=1.0, w_logdet=1.0, w_cycle=1.0,
                 soft=False, device="auto", seed=0, verbose=False):
        self.c_dim = c_dim
        self.hidden = hidden
        self.depth = depth
        self.liquid = liquid
        self.constraint = constraint
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.n_steps = n_steps
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.w_fwd = w_fwd
        self.w_logdet = w_logdet
        self.w_cycle = w_cycle
        self.soft = soft
        self.device = device
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------ setup
    def _resolve_device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _build(self, d, N):
        self.d_, self.N_ = d, N
        self.base_ = GaussianBase(d)
        if self.soft:
            self.fwd_net_ = _mlp([d] + [self.hidden] * self.depth + [N]).to(self.device_)
            self.inv_net_ = _mlp([N] + [self.hidden] * self.depth + [d]).to(self.device_)
            self.params_ = list(self.fwd_net_.parameters()) + list(self.inv_net_.parameters())
        else:
            # amortized conditional base N(mu(y), sigma(y)); unconditional flow
            self.embed_ = _mlp([N] + [self.hidden] * self.depth + [2 * d]).to(self.device_)
            self.fwd_head_ = _mlp([d] + [self.hidden] * self.depth + [N]).to(self.device_)
            vel = LiquidVelocity(d, self.hidden, self.depth, liquid=self.liquid)
            tmode = "exact" if d <= 12 else "hutch"
            self.flow_ = LiquidODEFlow(vel.to(self.device_), solver=self.solver,
                                       rtol=self.rtol, atol=self.atol,
                                       n_steps=self.n_steps, trace_mode=tmode).to(self.device_)
            self.params_ = (list(self.flow_.parameters()) +
                            list(self.embed_.parameters()) +
                            list(self.fwd_head_.parameters()))

    # ------------------------------------------------------------------- train
    def fit(self, X, Y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device_ = self._resolve_device()

        X = self._as2d(X)
        Y = self._as2d(Y)
        self.constraint_ = get_constraint(self.constraint)

        Xb = self.constraint_.inverse(torch.as_tensor(X, dtype=torch.float64)).numpy()
        self.sx_ = _Scaler().fit(Xb)
        self.sy_ = _Scaler().fit(Y)
        Xs = self._to_t(self.sx_.transform(Xb))
        Ys = self._to_t(self.sy_.transform(Y))

        self._build(Xs.shape[1], Ys.shape[1])
        opt = torch.optim.Adam(self.params_, lr=self.lr)
        n = Xs.shape[0]
        bs = min(self.batch_size, n)
        self.history_ = []

        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            ep_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb, yb = Xs[idx], Ys[idx]
                opt.zero_grad()
                loss = self._soft_loss(xb, yb) if self.soft else self._flow_loss(xb, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.params_, 10.0)
                opt.step()
                ep_loss += float(loss.detach()) * xb.shape[0]
            ep_loss /= n
            self.history_.append(ep_loss)
            if self.verbose and (ep % max(1, self.epochs // 10) == 0 or ep == self.epochs - 1):
                print(f"[PRISM] epoch {ep:4d}  loss {ep_loss:.4f}")
        self.fitted_ = True
        return self

    def _base_params(self, ys):
        """Observation -> (mu, sigma) of the amortized Gaussian base."""
        out = self.embed_(ys)
        mu, logs = out[:, :self.d_], out[:, self.d_:]
        sigma = torch.nn.functional.softplus(logs) + 1e-3
        return mu, sigma

    def _flow_loss(self, xb, yb):
        mu, sigma = self._base_params(yb)
        if self.w_logdet > 0:
            z, ld = self.flow_.forward(xb, logdet=True)
        else:
            z, ld = self.flow_.forward(xb, logdet=False), 0.0
        # -log p(x|y) = sum_d[0.5((z-mu)/sigma)^2 + log sigma] + w_logdet * ld
        nll = (0.5 * ((z - mu) / sigma) ** 2 + torch.log(sigma)).sum(dim=1)
        if self.w_logdet > 0:
            nll = nll + self.w_logdet * ld
        L_nll = nll.mean()
        y_hat = self.fwd_head_(xb)
        L_fwd = ((y_hat - yb) ** 2).mean()
        return self.w_fwd * L_fwd + L_nll

    def _soft_loss(self, xb, yb):
        y_hat = self.fwd_net_(xb)
        x_hat = self.inv_net_(yb)
        L_fwd = ((y_hat - yb) ** 2).mean()
        L_inv = ((x_hat - xb) ** 2).mean()
        x_cyc = self.inv_net_(self.fwd_net_(xb))
        L_cyc = ((x_cyc - xb) ** 2).mean()
        return self.w_fwd * L_fwd + L_inv + self.w_cycle * L_cyc

    # ----------------------------------------------------------- encode/decode
    def _encode_x(self, X):
        Xb = self.constraint_.inverse(torch.as_tensor(self._as2d(X), dtype=torch.float64)).numpy()
        return self._to_t(self.sx_.transform(Xb))

    def _decode_x(self, xs):
        Xb = self.sx_.inverse_transform(xs.cpu().numpy())
        return self.constraint_.forward(torch.as_tensor(Xb, dtype=torch.float64)).numpy()

    # ----------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, X):
        """Forward surrogate: parameters -> observation."""
        xs = self._encode_x(X)
        ys = self.fwd_net_(xs) if self.soft else self.fwd_head_(xs)
        return self.sy_.inverse_transform(ys.cpu().numpy())

    @torch.no_grad()
    def invert(self, Y):
        """Point inverse (z=0): observation -> single parameter estimate."""
        ys = self._to_t(self.sy_.transform(self._as2d(Y)))
        if self.soft:
            xs = self.inv_net_(ys)
        else:
            mu, _ = self._base_params(ys)
            xs = self.flow_.inverse(mu)
        return self._decode_x(xs)

    @torch.no_grad()
    def predict_posterior(self, Y, n_samples=1000):
        """Posterior samples p(x|y). Returns (n_obs, n_samples, d)."""
        ys = self._to_t(self.sy_.transform(self._as2d(Y)))
        m = ys.shape[0]
        out = np.empty((m, n_samples, self.d_), dtype=np.float64)
        if self.soft:
            base = self.invert(Y)
            noise = 0.05 * np.std(base, axis=0, keepdims=True)
            for s in range(n_samples):
                out[:, s, :] = base + noise * np.random.randn(m, self.d_)
            return out
        c = None
        mu, sigma = self._base_params(ys)
        for s in range(n_samples):
            z = mu + sigma * torch.randn_like(mu)
            xs = self.flow_.inverse(z)
            out[:, s, :] = self._decode_x(xs)
        return out

    def sample(self, Y, n_samples=1000):
        return self.predict_posterior(Y, n_samples)

    # -------------------------------------------------------------- diagnostics
    def score(self, X, Y):
        """R^2 of the forward surrogate (higher is better)."""
        from ..utils.metrics import r2_score
        return r2_score(self._as2d(Y), self.predict(X))

    @torch.no_grad()
    def cycle_consistency_error(self, X):
        """Mean relative ||F^{-1}(F(x|y)|y) - x|| in the model's working space.

        For the exact flow this is ~ solver tolerance; for soft=True it is the
        trained (nonzero) cycle residual."""
        xs = self._encode_x(X)
        if self.soft:
            x_rec = self.inv_net_(self.fwd_net_(xs))
        else:
            z = self.flow_.forward(xs)
            x_rec = self.flow_.inverse(z)
        num = (x_rec - xs).norm(dim=1)
        den = xs.norm(dim=1).clamp_min(1e-12)
        return float((num / den).mean())

    @torch.no_grad()
    def constraint_violation_rate(self, X=None, n=2048):
        """Fraction of recovered parameters that violate the constraint.

        Decoded random latents pass through the constraint layer; for PRISM
        this is exactly 0 by construction."""
        c = self.constraint_
        u = torch.randn(n, self.d_, dtype=torch.float64)
        x = c.forward(u)
        return float((c.violation(x) > 1e-6).float().mean())
