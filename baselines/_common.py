"""Shared utilities for baseline methods.

Baselines reuse PRISM's scaler and metrics so comparisons are apples-to-apples.
A baseline exposes a subset of the PRISM API:

* forward surrogates (FNO/DeepONet/PINN):   ``fit(X, Y)`` + ``predict(X)``
* posterior methods (cINN/NPE-CNF/VI/MCMC):  ``fit`` + ``invert`` + ``predict_posterior``
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from prism.models.base import _Scaler


def mlp(sizes, act=nn.Tanh, zero_last=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    net = nn.Sequential(*layers)
    if zero_last:
        last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    return net


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def as2d(A):
    A = np.asarray(A, dtype=np.float64)
    return A[:, None] if A.ndim == 1 else A


def fields_from_dataset(ds, X):
    """Decode a batch of parameter vectors into stacked field tensors.

    Uses the dataset's ``decode_field``; returns (b, *field_shape) float array.
    Only valid for PDE benchmarks that define a field decoder.
    """
    X = as2d(X)
    fields = [np.asarray(ds.decode_field(x), float) for x in X]
    return np.stack(fields, axis=0)


class TorchEstimator:
    """Minimal fit/predict training loop with standardisation (forward maps)."""

    def __init__(self, lr=1e-3, epochs=200, batch_size=256, device="auto",
                 seed=0, verbose=False):
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size
        self.device = device; self.seed = seed; self.verbose = verbose

    def _setup(self):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.device_ = resolve_device(self.device)

    def _train(self, inputs, targets, net, loss_fn=None):
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        n = inputs.shape[0]; bs = min(self.batch_size, n)
        loss_fn = loss_fn or (lambda p, t: ((p - t) ** 2).mean())
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device_)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = loss_fn(net(inputs[idx]), targets[idx])
                loss.backward(); opt.step()
        return net
