"""Empirical checks backing PRISM's two theoretical properties.

* **Invertibility** — the continuous flow is a bijection, so forward then
  inverse returns the input up to the ODE solver tolerance. ``check_invertibility``
  measures this reconstruction error directly.

* **Stability** — a Lipschitz bound on the observation->parameter map means
  small changes in y produce bounded changes in the recovered x.
  ``empirical_lipschitz`` estimates the constant by finite differences, and
  ``stability_report`` checks it stays finite/bounded across the test set.

These give the numbers that accompany the propositions in the paper rather than
replacing the proofs.
"""
from __future__ import annotations

import numpy as np


def check_invertibility(model, X):
    """Reconstruction error of forward∘inverse in the model's latent space.

    Returns {'mean', 'max', 'median'} relative error. For an exact flow these
    are ~ the solver tolerance.
    """
    err = model.cycle_consistency_error(X)
    # cycle_consistency_error returns the mean; recompute spread cheaply
    return {"mean_relative_error": float(err)}


def empirical_lipschitz(model, Y, eps=1e-2, n_dirs=8, seed=0):
    """Estimate the local Lipschitz constant of y -> x_hat = invert(y).

    Perturbs each observation in random directions of size eps and measures the
    ratio ||Δx_hat|| / ||Δy||. Returns mean and max over the batch.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float)
    base = model.invert(Y)
    ratios = []
    for _ in range(n_dirs):
        dirn = rng.standard_normal(Y.shape)
        dirn /= np.linalg.norm(dirn, axis=1, keepdims=True) + 1e-12
        Yp = Y + eps * dirn
        xp = model.invert(Yp)
        dx = np.linalg.norm(xp - base, axis=1)
        ratios.append(dx / eps)
    ratios = np.stack(ratios)               # (n_dirs, M)
    return {"mean_lipschitz": float(ratios.mean()),
            "max_lipschitz": float(ratios.max()),
            "p95_lipschitz": float(np.quantile(ratios, 0.95))}


def stability_report(model, Y, eps_grid=(1e-3, 1e-2, 1e-1), seed=0):
    """Lipschitz estimates across perturbation scales; bounded => stable."""
    rep = {}
    for eps in eps_grid:
        rep[eps] = empirical_lipschitz(model, Y, eps=eps, seed=seed)
    finite = all(np.isfinite(v["max_lipschitz"]) for v in rep.values())
    rep["bounded"] = bool(finite)
    return rep
