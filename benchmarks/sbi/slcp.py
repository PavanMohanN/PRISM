"""SLCP — Simple Likelihood, Complex Posterior.

theta ~ U(-3,3)^5. Mean m=(theta0,theta1); a 2x2 covariance is built from
theta2..theta4. The observation stacks 4 i.i.d. draws from N(m, cov) -> R^8.
The likelihood is tractable (so emcee gives a reference posterior), but the
posterior is complex and multimodal — a classic calibration benchmark.
"""
from __future__ import annotations

import numpy as np
from ..registry import Dataset, uniform_prior


def _cov(theta):
    s1 = theta[..., 2] ** 2
    s2 = theta[..., 3] ** 2
    rho = np.tanh(theta[..., 4])
    c = np.empty(theta.shape[:-1] + (2, 2))
    c[..., 0, 0] = s1 ** 2
    c[..., 1, 1] = s2 ** 2
    c[..., 0, 1] = c[..., 1, 0] = rho * s1 * s2
    return c


def _simulate(theta, rng):
    theta = np.atleast_2d(theta)
    b = theta.shape[0]
    m = theta[:, :2]
    cov = _cov(theta) + 1e-6 * np.eye(2)
    L = np.linalg.cholesky(cov)                       # (b,2,2)
    eps = rng.standard_normal((b, 4, 2))
    pts = m[:, None, :] + np.einsum("bij,bkj->bki", L, eps)   # (b,4,2)
    return pts.reshape(b, 8)


def _log_like(theta, x_obs):
    theta = np.asarray(theta, float)
    m = theta[:2]
    cov = _cov(theta) + 1e-6 * np.eye(2)
    try:
        inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
    except np.linalg.LinAlgError:
        return -np.inf
    if sign <= 0:
        return -np.inf
    pts = np.asarray(x_obs, float).reshape(4, 2)
    ll = 0.0
    for p in pts:
        r = p - m
        ll += -0.5 * r @ inv @ r - 0.5 * logdet - np.log(2 * np.pi)
    return float(ll)


def make(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    lo = -3 * np.ones(5); hi = 3 * np.ones(5)
    sample, logp, bounds = uniform_prior(lo, hi)
    theta = sample(n, rng)
    X = theta
    Y = _simulate(theta, rng)
    return Dataset(
        name="slcp", X=X, Y=Y, constraint="box",
        meta={"d": 5, "N": 8, "box": (lo.tolist(), hi.tolist()), "tractable": True,
              "multimodal": True},
        simulate=lambda th: _simulate(th, np.random.default_rng()),
        log_likelihood=_log_like, log_prior=logp,
        prior_sample=lambda m, r: sample(m, r), prior_bounds=bounds,
    )
