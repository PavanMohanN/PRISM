"""Two Moons SBI benchmark.

A two-dimensional task whose posterior has two crescent-shaped modes — the
standard stress test for posterior *shape* recovery. Likelihood is intractable,
so calibration uses simulation-based calibration (SBC) and coverage, which need
only the simulator and the model's posterior.
"""
from __future__ import annotations

import numpy as np
from ..registry import Dataset, uniform_prior


def _simulate(theta, rng):
    theta = np.atleast_2d(theta)
    b = theta.shape[0]
    a = rng.uniform(-np.pi / 2, np.pi / 2, size=b)
    r = 0.1 + 0.01 * rng.standard_normal(b)
    p1 = r * np.cos(a) + 0.25
    p2 = r * np.sin(a)
    x1 = p1 - np.abs(theta[:, 0] + theta[:, 1]) / np.sqrt(2.0)
    x2 = p2 + (-theta[:, 0] + theta[:, 1]) / np.sqrt(2.0)
    return np.stack([x1, x2], axis=1)


def make(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([-1.0, -1.0]); hi = np.array([1.0, 1.0])
    sample, logp, bounds = uniform_prior(lo, hi)
    theta = sample(n, rng)
    X = theta
    Y = _simulate(theta, rng)

    def simulate(th):
        return _simulate(th, np.random.default_rng())

    return Dataset(
        name="two_moons", X=X, Y=Y, constraint="box",
        meta={"d": 2, "N": 2, "box": (lo.tolist(), hi.tolist()), "tractable": False,
              "multimodal": True},
        simulate=simulate, log_prior=logp,
        prior_sample=lambda m, r: sample(m, r), prior_bounds=bounds,
    )
