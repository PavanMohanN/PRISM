"""Accuracy / reconstruction metrics.

All functions accept numpy arrays or torch tensors and return python floats.
Kept dependency-light (numpy + scipy only) so they work everywhere.
"""
from __future__ import annotations

import numpy as np


def _np(x):
    """Coerce torch tensor or array-like to a detached float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def rmse(y_true, y_pred) -> float:
    a, b = _np(y_true), _np(y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true, y_pred) -> float:
    a, b = _np(y_true), _np(y_pred)
    return float(np.mean(np.abs(a - b)))


def relative_l2(y_true, y_pred, eps: float = 1e-12) -> float:
    """Mean over samples of ||y_pred - y_true||_2 / ||y_true||_2.

    Standard operator-learning error (FNO/PDEBench convention).
    Arrays are flattened per-sample over all but the first axis.
    """
    a, b = _np(y_true), _np(y_pred)
    a = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(1, -1)
    b = b.reshape(b.shape[0], -1) if b.ndim > 1 else b.reshape(1, -1)
    num = np.linalg.norm(b - a, axis=1)
    den = np.linalg.norm(a, axis=1) + eps
    return float(np.mean(num / den))


def r2_score(y_true, y_pred) -> float:
    a, b = _np(y_true).ravel(), _np(y_pred).ravel()
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def psnr(y_true, y_pred, data_range: float | None = None) -> float:
    a, b = _np(y_true), _np(y_pred)
    if data_range is None:
        data_range = float(a.max() - a.min()) or 1.0
    mse = np.mean((a - b) ** 2) + 1e-12
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim(y_true, y_pred, data_range: float | None = None) -> float:
    """Global (single-window) SSIM. Light, dependency-free approximation
    adequate for reporting field-recovery quality on benchmark grids."""
    a, b = _np(y_true).ravel(), _np(y_pred).ravel()
    if data_range is None:
        data_range = float(a.max() - a.min()) or 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / (den + 1e-12))


def aggregate(values) -> dict:
    """Return mean/std/n for a list of per-seed scalars (for ±std tables)."""
    v = _np(values).ravel()
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if v.size > 1 else 0.0),
            "n": int(v.size)}
