"""Gaussian-linear SBI benchmark.

Prior  theta ~ N(0, s0^2 I_d);  likelihood  x | theta ~ N(theta, sl^2 I_d).
The posterior is analytic Gaussian, giving an *exact* reference for C2ST and
coverage — the strict calibration anchor in the paper.
"""
from __future__ import annotations

import numpy as np
from ..registry import Dataset


def make(n=2000, seed=0, d=5, s0=1.0, sl=0.3):
    rng = np.random.default_rng(seed)
    theta = s0 * rng.standard_normal((n, d))
    X = theta
    Y = theta + sl * rng.standard_normal((n, d))

    def simulate(th):
        th = np.atleast_2d(th)
        return th + sl * np.random.default_rng().standard_normal(th.shape)

    def log_like(th, x_obs):
        th = np.asarray(th, float)
        r = x_obs - th
        return -0.5 * np.sum(r ** 2) / sl ** 2 - d * np.log(sl)

    def log_prior(th):
        th = np.asarray(th, float)
        return -0.5 * np.sum(th ** 2) / s0 ** 2

    def prior_sample(m, rng_):
        return s0 * rng_.standard_normal((m, d))

    def analytic_ref(x_obs, n=1000, seed=0):
        rng_ = np.random.default_rng(seed)
        prec = 1.0 / s0 ** 2 + 1.0 / sl ** 2
        var = 1.0 / prec
        mean = var * (x_obs / sl ** 2)
        return mean + np.sqrt(var) * rng_.standard_normal((n, d))

    lo = -4 * s0 * np.ones(d); hi = 4 * s0 * np.ones(d)
    return Dataset(
        name="gaussian_linear", X=X, Y=Y, constraint="none",
        meta={"d": d, "N": d, "tractable": True, "analytic_posterior": True},
        simulate=simulate, log_likelihood=log_like, log_prior=log_prior,
        prior_sample=prior_sample, prior_bounds=(lo, hi), _ref=analytic_ref,
    )
