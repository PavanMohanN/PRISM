"""PRISM Phase 1 -- fair constraint comparison (guide item 10).

Zero constraint violations is a property of the support transform, not of PRISM.
To make the comparison fair, we wrap ANY baseline (cINN, NPE-CNF, VI) so that it
is trained in the SAME unconstrained coordinate and decodes through the SAME
transform c. After wrapping, a correctly-implemented baseline should also achieve
~0 violations; the interesting comparison then moves to calibration, likelihood,
inverse error, and efficiency.

The wrapped estimator must expose the usual scikit-learn-style methods operating
in whatever coordinate it is given:

    base.fit(U, y)                  U = c^{-1}(X)   (unconstrained targets)
    base.predict(y)      -> U_hat
    base.predict_posterior(y, n)   -> U samples  (shape [N, n, dim] or [n, dim])

This wrapper handles the transform on both sides so every method is evaluated on
identical, feasible physical samples.
"""
from __future__ import annotations

import numpy as np
import torch

from prism.constraints.transforms import SupportTransform, Identity


def _to_t(a):
    if isinstance(a, torch.Tensor):
        return a
    return torch.as_tensor(np.asarray(a), dtype=torch.get_default_dtype())


class SupportConstrained:
    """Give a baseline feasibility-by-construction via an identical transform."""

    def __init__(self, base_estimator, transform: SupportTransform | None = None):
        self.base = base_estimator
        self.transform = transform or Identity(dim=1)

    # ---- fit in unconstrained coordinates ----
    def fit(self, X, y):
        Xt = _to_t(X)
        U = self.transform.inverse(Xt)
        self.base.fit(U.detach().cpu().numpy() if _wants_numpy(self.base) else U, y)
        return self

    def predict(self, y):
        U = _to_t(self.base.predict(y))
        return self.transform.forward(U)

    def predict_posterior(self, y, n_samples: int = 1000):
        U = _to_t(self.base.predict_posterior(y, n_samples))
        return self.transform.forward(U)

    # ---- now a property of the shared transform, should be ~0 for all methods ----
    def constraint_violation_rate(self, y, n_samples: int = 1000):
        X = self.predict_posterior(y, n_samples)
        feasible = self.transform.is_feasible(X)
        return 1.0 - feasible.float().mean().item()


def _wants_numpy(est) -> bool:
    # torch-native baselines accept tensors; sklearn-style want numpy.
    return getattr(est, "expects_numpy", False)


def wrap_all(baselines: dict, transform: SupportTransform) -> dict:
    """Wrap a dict {name: estimator} with the same transform for a fair table."""
    return {name: SupportConstrained(est, transform) for name, est in baselines.items()}
