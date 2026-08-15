"""Minimal scikit-learn-style estimator base.

Gives PRISM a familiar API surface (``get_params``/``set_params``, ``fit``/
``predict``) and standard-scaler bookkeeping, without depending on sklearn's
estimator internals (we only use numpy + torch).
"""
from __future__ import annotations

import inspect
import numpy as np
import torch


class _Scaler:
    """Per-feature standardiser (like sklearn StandardScaler) in numpy."""

    def fit(self, A):
        A = np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            A = A[:, None]
        self.mean_ = A.mean(axis=0, keepdims=True)
        self.std_ = A.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, A):
        A = np.asarray(A, dtype=np.float64)
        sq = A.ndim == 1
        if sq:
            A = A[:, None]
        out = (A - self.mean_) / self.std_
        return out

    def inverse_transform(self, A):
        A = np.asarray(A, dtype=np.float64)
        return A * self.std_ + self.mean_


class BaseEstimator:
    """get_params/set_params via the __init__ signature (sklearn convention)."""

    @classmethod
    def _param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self._param_names()}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- helpers ----
    @staticmethod
    def _as2d(A):
        A = np.asarray(A, dtype=np.float64)
        return A[:, None] if A.ndim == 1 else A

    def _to_t(self, A):
        return torch.as_tensor(A, dtype=torch.float32, device=self.device_)

    def __repr__(self):
        kv = ", ".join(f"{k}={getattr(self, k)!r}" for k in self._param_names())
        return f"{type(self).__name__}({kv})"
