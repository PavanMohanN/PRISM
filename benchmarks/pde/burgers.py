"""Viscous Burgers inverse task.

Recover the initial-condition amplitudes (sine modes) from the solution
u(x, T) sampled at sensor locations. The forward map integrates
u_t = -u u_x + nu u_xx on a periodic domain with a batched Fourier-spectral
RK4 scheme. ``decode_field`` returns the recovered initial profile.
"""
from __future__ import annotations

import numpy as np
from ..registry import Dataset

NX = 64
L = 2 * np.pi
NU = 0.05
T = 0.5
NT = 200
K_MODES = 4
_x = np.linspace(0, L, NX, endpoint=False)
_k = 2 * np.pi * np.fft.fftfreq(NX, d=L / NX)            # wavenumbers
_sensors = np.linspace(0, NX, 16, endpoint=False, dtype=int)


def _u0(A):
    """amplitudes (b,K) -> initial field (b,NX)."""
    A = np.atleast_2d(A)
    modes = np.arange(1, K_MODES + 1)[None, :, None]
    return (A[:, :, None] * np.sin(modes * _x[None, None, :])).sum(axis=1)


def _rhs(u):
    uh = np.fft.fft(u, axis=1)
    ux = np.real(np.fft.ifft(1j * _k[None, :] * uh, axis=1))
    uxx = np.real(np.fft.ifft(-(_k[None, :] ** 2) * uh, axis=1))
    return -u * ux + NU * uxx


def _evolve(A):
    u = _u0(A)
    dt = T / NT
    for _ in range(NT):
        k1 = _rhs(u)
        k2 = _rhs(u + 0.5 * dt * k1)
        k3 = _rhs(u + 0.5 * dt * k2)
        k4 = _rhs(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def _forward(A):
    return _evolve(np.atleast_2d(A))[:, _sensors]


def make(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1.0, 1.0, size=(n, K_MODES))
    X = A
    Y = _forward(A)
    return Dataset(
        name="burgers", X=X, Y=Y, constraint="none",
        meta={"d": K_MODES, "N": len(_sensors), "nu": NU, "T": T, "tractable": False,
              "field_shape": [NX]},
        simulate=lambda th: _forward(th),
        prior_sample=lambda m, r: r.uniform(-1.0, 1.0, (m, K_MODES)),
        prior_bounds=(-1.0 * np.ones(K_MODES), 1.0 * np.ones(K_MODES)),
        decode_field=lambda a: _u0(np.asarray(a, float).ravel())[0],
    )
