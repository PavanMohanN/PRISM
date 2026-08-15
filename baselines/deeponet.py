"""DeepONet baseline — branch/trunk operator forward surrogate.

A branch network encodes the (flattened) parameter field and a trunk network
encodes the sensor location; their inner product yields the observation at each
sensor. A lighter operator baseline alongside FNO.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from prism.models.base import _Scaler
from ._common import mlp, resolve_device


class DeepONet:
    def __init__(self, p=48, hidden=64, depth=2, lr=1e-3, epochs=200,
                 batch_size=64, device="auto", seed=0, verbose=False):
        self.p = p; self.hidden = hidden; self.depth = depth; self.lr = lr
        self.epochs = epochs; self.batch_size = batch_size
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
        N = Y.shape[1]
        coords = torch.linspace(0, 1, N, device=self.device_).unsqueeze(-1)
        self.coords_ = coords
        self.branch_ = mlp([Fn.shape[1]] + [self.hidden] * self.depth + [self.p]).to(self.device_)
        self.trunk_ = mlp([1] + [self.hidden] * self.depth + [self.p]).to(self.device_)
        self.bias_ = nn.Parameter(torch.zeros(N, device=self.device_))
        params = list(self.branch_.parameters()) + list(self.trunk_.parameters()) + [self.bias_]
        opt = torch.optim.Adam(params, lr=self.lr)
        n = Fn.shape[0]; bs = min(self.batch_size, n)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                b = self.branch_(Fn[idx])                  # (B,p)
                t = self.trunk_(self.coords_)              # (N,p)
                pred = b @ t.t() + self.bias_              # (B,N)
                loss = ((pred - Yt[idx]) ** 2).mean()
                loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, fields):
        fields = np.asarray(fields, float).reshape(len(fields), -1)
        Fn = torch.as_tensor((fields - self.fmu_) / self.fsd_, dtype=torch.float32, device=self.device_)
        pred = self.branch_(Fn) @ self.trunk_(self.coords_).t() + self.bias_
        return self.sy_.inverse_transform(pred.cpu().numpy())
