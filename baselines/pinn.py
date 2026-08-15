"""PINN-lite baseline — physics-regularised MLP forward surrogate.

A compact MLP mapping the flattened parameter field to observations, trained
with a data loss plus a smoothness (Sobolev) penalty on the output as a light
physics-style regulariser. A full PDE-residual PINN is task-specific; this
serves as the lightweight stand-in for the forward-surrogate comparison.
"""
from __future__ import annotations

import numpy as np
import torch

from prism.models.base import _Scaler
from ._common import mlp, resolve_device


class PINN:
    def __init__(self, hidden=96, depth=3, w_smooth=1e-3, lr=1e-3, epochs=200,
                 batch_size=64, device="auto", seed=0, verbose=False):
        self.hidden = hidden; self.depth = depth; self.w_smooth = w_smooth
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.device = device; self.seed = seed; self.verbose = verbose

    def fit(self, fields, Y):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.device_ = resolve_device(self.device)
        fields = np.asarray(fields, float).reshape(len(fields), -1)
        Y = np.asarray(Y, float)
        self.fmu_, self.fsd_ = fields.mean(0), fields.std(0) + 1e-8
        self.sy_ = _Scaler().fit(Y)
        Fn = torch.as_tensor((fields - self.fmu_) / self.fsd_, dtype=torch.float32, device=self.device_)
        Yt = torch.as_tensor(self.sy_.transform(Y), dtype=torch.float32, device=self.device_)
        self.net_ = mlp([Fn.shape[1]] + [self.hidden] * self.depth + [Y.shape[1]]).to(self.device_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        n = Fn.shape[0]; bs = min(self.batch_size, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                pred = self.net_(Fn[idx])
                data = ((pred - Yt[idx]) ** 2).mean()
                smooth = ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()   # Sobolev penalty
                (data + self.w_smooth * smooth).backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, fields):
        fields = np.asarray(fields, float).reshape(len(fields), -1)
        Fn = torch.as_tensor((fields - self.fmu_) / self.fsd_, dtype=torch.float32, device=self.device_)
        return self.sy_.inverse_transform(self.net_(Fn).cpu().numpy())
