"""PRISM Phase 2 -- analysis core (no torch; unit-tested).

These helpers turn raw measurements into the numbers that populate the Phase 2
tables and figures. They are deliberately torch-free so they can be verified
independently of the model:

  * loglog_order_and_floor  -- fixes the "flat reversibility curve" objection
    (guide item 21) by fitting the convergence slope in the descending regime and
    separately reporting the roundoff/saturation floor.
  * spectral_norm           -- largest singular value via power iteration, used
    for the empirical Lipschitz factors in the stability table (items 9, 31).
  * lipschitz_bound         -- composes the Groenwall bound Lip(c) e^{L_g} Lip(mu).
"""
from __future__ import annotations

import numpy as np


def loglog_order_and_floor(steps, errors, floor_rel=3.0):
    """Estimate convergence order p and the saturation floor from a step sweep.

    A fixed-step order-p integrator gives reconstruction error ~ C h^p until it
    hits a roundoff/representation floor, after which the curve flattens. We:
      1. estimate the floor as the median of the smallest errors (the plateau),
      2. keep only points comfortably above the floor (the descending regime),
      3. fit log(error) vs log(h)  (h = 1/steps) by least squares -> slope = p.

    Returns dict(order, floor, n_fit, regime_mask).
    """
    steps = np.asarray(steps, float)
    errors = np.asarray(errors, float)
    h = 1.0 / steps

    order_idx = np.argsort(h)                 # ascending h
    h_s, e_s = h[order_idx], errors[order_idx]

    floor = float(np.median(np.sort(e_s)[:max(1, len(e_s) // 4)]))
    mask = e_s > floor_rel * floor            # points where h^p dominates
    if mask.sum() >= 2:
        p = np.polyfit(np.log(h_s[mask]), np.log(e_s[mask]), 1)[0]
    else:
        p = float("nan")
    # map mask back to original ordering
    regime_mask = np.zeros_like(errors, dtype=bool)
    regime_mask[order_idx] = mask
    return {"order": float(p), "floor": floor,
            "n_fit": int(mask.sum()), "regime_mask": regime_mask}


def spectral_norm(A, iters=100, tol=1e-10, seed=0):
    """Largest singular value of matrix A via power iteration on A^T A."""
    A = np.asarray(A, float)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=A.shape[1])
    v /= np.linalg.norm(v)
    s_prev = 0.0
    for _ in range(iters):
        w = A.T @ (A @ v)
        nv = np.linalg.norm(w)
        if nv == 0:
            return 0.0
        v = w / nv
        s = np.sqrt(nv)
        if abs(s - s_prev) < tol * max(1.0, s):
            break
        s_prev = s
    return float(np.linalg.norm(A @ v))


def lipschitz_bound(lip_c, L_g, lip_mu):
    """Global Groenwall bound on the amortized inverse h = c . F^{-1} . mu."""
    return float(lip_c) * float(np.exp(L_g)) * float(lip_mu)


def local_sensitivity_ratio(h_of, y0, eps, n_dirs=32, seed=0):
    """Empirical local sensitivity ||h(y0+eps u) - h(y0)|| / (eps) over random u.

    h_of: callable y -> x (numpy). Returns (mean, max) over directions.
    Used for the stability figure vs eps (mean is lower, max approaches the
    local operator norm) -- keeping guide item 31's distinction explicit.
    """
    rng = np.random.default_rng(seed)
    y0 = np.asarray(y0, float)
    base = np.asarray(h_of(y0), float)
    ratios = []
    for _ in range(n_dirs):
        u = rng.normal(size=y0.shape); u /= np.linalg.norm(u)
        pert = np.asarray(h_of(y0 + eps * u), float)
        ratios.append(np.linalg.norm(pert - base) / eps)
    return float(np.mean(ratios)), float(np.max(ratios))


def latex_pm(mean, std, prec=3):
    return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"
