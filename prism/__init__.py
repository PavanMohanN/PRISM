"""PRISM — Physically-constrained Reversible Inference via Stochastic
liquid-ODE Maps for ill-posed inverse problems.

Quickstart (scikit-learn style)::

    from prism import PRISM
    model = PRISM(constraint="positive").fit(X, Y)   # X=params, Y=observations
    Y_hat = model.predict(X)            # forward surrogate
    X_hat = model.invert(Y)            # point inverse (same weights)
    samples = model.predict_posterior(Y, n_samples=1000)   # p(X | Y)

The estimator lives in `prism.models.prism`. It is imported lazily so that
`import prism` stays cheap and does not require torch until a model is built.
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["PRISM", "__version__"]


def __getattr__(name):  # PEP 562 lazy import
    if name == "PRISM":
        from .models.prism import PRISM
        return PRISM
    raise AttributeError(f"module 'prism' has no attribute {name!r}")
