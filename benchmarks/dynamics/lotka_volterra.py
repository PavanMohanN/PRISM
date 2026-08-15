"""Lotka-Volterra inverse task.

Recover four positive ecological rate parameters (alpha, beta, gamma, delta)
from a noisy two-species population time series. Exercises the *positivity*
constraint on a nonlinear dynamical inverse problem; the Gaussian observation
likelihood admits an emcee reference posterior.
"""
from __future__ import annotations

import numpy as np
from ..registry import Dataset

T_MAX = 15.0
N_STEPS = 150                      # RK4 integration steps
N_OBS = 8                         # observation times
NOISE = 0.05                      # gaussian noise on log-populations
_OBS_IDX = np.linspace(0, N_STEPS, N_OBS, endpoint=False, dtype=int) + 1


def _rk4(theta, x0=(1.0, 0.5)):
    """Vectorised RK4 over a batch of parameter sets. Returns (b, N_OBS*2)."""
    theta = np.atleast_2d(theta)
    b = theta.shape[0]
    a, bb, g, d = (theta[:, i] for i in range(4))
    dt = T_MAX / N_STEPS
    s = np.empty((b, 2))
    s[:, 0] = x0[0]; s[:, 1] = x0[1]
    traj = np.empty((b, N_STEPS + 1, 2))
    traj[:, 0] = s

    def f(state):
        x, y = state[:, 0], state[:, 1]
        return np.stack([a * x - bb * x * y, -g * y + d * x * y], axis=1)

    for k in range(N_STEPS):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s = np.clip(s, 1e-4, 1e4)
        traj[:, k + 1] = s
    obs = traj[:, _OBS_IDX, :]                  # (b, N_OBS, 2)
    return np.log(obs).reshape(b, -1)           # log-populations


def make(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    # log-normal-ish positive priors centred near a stable cycle
    theta = np.exp(rng.normal(loc=[-0.1, -0.5, -0.1, -0.5], scale=0.4, size=(n, 4)))
    X = theta
    Y = _rk4(theta) + NOISE * rng.standard_normal((n, N_OBS * 2))

    lo = np.array([0.1, 0.05, 0.1, 0.05]); hi = np.array([3.0, 2.0, 3.0, 2.0])

    def simulate(th):
        th = np.atleast_2d(th)
        return _rk4(th) + NOISE * np.random.default_rng().standard_normal((th.shape[0], N_OBS * 2))

    def log_like(th, x_obs):
        mu = _rk4(np.atleast_2d(th))[0]
        r = np.asarray(x_obs, float) - mu
        return -0.5 * np.sum(r ** 2) / NOISE ** 2

    def log_prior(th):
        th = np.asarray(th, float)
        return 0.0 if np.all(th > 0) and np.all(th < 5) else -np.inf

    return Dataset(
        name="lotka_volterra", X=X, Y=Y, constraint="positive",
        meta={"d": 4, "N": N_OBS * 2, "tractable": True,
              "param_names": ["alpha", "beta", "gamma", "delta"]},
        simulate=simulate, log_likelihood=log_like, log_prior=log_prior,
        prior_sample=lambda m, r: np.exp(r.normal([-0.1, -0.5, -0.1, -0.5], 0.4, (m, 4))),
        prior_bounds=(lo, hi),
    )
