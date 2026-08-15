"""Damped harmonic oscillator inverse task.

Recover the two positive physical parameters (natural frequency omega, damping
ratio zeta) from a noisy displacement trajectory. A general dynamical / wave-
like inverse problem that exercises the *positivity* hard constraint. The
observation likelihood is Gaussian, so emcee provides a reference posterior.
"""
from __future__ import annotations

import numpy as np
from ..registry import Dataset

T_GRID = np.linspace(0.0, 6.0, 24)          # 24 observation times
NOISE = 0.03


def _traj(theta):
    """theta (b,2)=[omega,zeta] -> displacement at T_GRID, underdamped form."""
    theta = np.atleast_2d(theta)
    omega = theta[:, 0:1]
    zeta = np.clip(theta[:, 1:2], 1e-3, 0.999)
    wd = omega * np.sqrt(1.0 - zeta ** 2)
    t = T_GRID[None, :]
    return np.exp(-zeta * omega * t) * np.cos(wd * t)


def make(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    omega = rng.uniform(0.5, 3.0, size=(n, 1))
    zeta = rng.uniform(0.05, 0.5, size=(n, 1))
    theta = np.hstack([omega, zeta])
    X = theta
    Y = _traj(theta) + NOISE * rng.standard_normal((n, len(T_GRID)))

    lo = np.array([0.5, 0.05]); hi = np.array([3.0, 0.5])

    def simulate(th):
        th = np.atleast_2d(th)
        return _traj(th) + NOISE * np.random.default_rng().standard_normal((th.shape[0], len(T_GRID)))

    def log_like(th, x_obs):
        mu = _traj(np.atleast_2d(th))[0]
        r = np.asarray(x_obs, float) - mu
        return -0.5 * np.sum(r ** 2) / NOISE ** 2

    def log_prior(th):
        th = np.asarray(th, float)
        return 0.0 if np.all((th >= lo) & (th <= hi)) else -np.inf

    return Dataset(
        name="oscillator", X=X, Y=Y, constraint="positive",
        meta={"d": 2, "N": len(T_GRID), "tractable": True, "param_names": ["omega", "zeta"]},
        simulate=simulate, log_likelihood=log_like, log_prior=log_prior,
        prior_sample=lambda m, r: np.hstack([r.uniform(0.5, 3.0, (m, 1)),
                                             r.uniform(0.05, 0.5, (m, 1))]),
        prior_bounds=(lo, hi),
    )
