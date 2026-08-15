"""1D Helmholtz (wave) inverse task.

Recover the smooth medium coefficients m(x) from the field u sampled at sensors,
where (d^2/dx^2 + k0^2 m(x)) u = f with a Gaussian source and homogeneous
Dirichlet boundaries. A linear wave-scattering inverse problem solved with a
tridiagonal system. ``decode_field`` returns the recovered medium profile.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from ..registry import Dataset

G = 64
K0 = 8.0
K_MODES = 5
_x = np.linspace(0, 1, G)
_h = _x[1] - _x[0]
_src = np.exp(-((_x - 0.5) ** 2) / (2 * 0.03 ** 2))      # gaussian source
_sensors = np.linspace(4, G - 5, 16, dtype=int)
_phi = np.stack([np.sin(np.pi * (p + 1) * _x) for p in range(K_MODES)])   # (K,G)


def _medium(c):
    return 1.0 + 0.3 * np.tensordot(np.atleast_2d(c), _phi, axes=(1, 0))   # (b,G)


def _solve_one(m):
    """Tridiagonal solve of u'' + K0^2 m u = f on interior nodes."""
    n = G - 2
    main = -2.0 / _h ** 2 + K0 ** 2 * m[1:-1]
    off = np.ones(n - 1) / _h ** 2
    ab = np.zeros((3, n))
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off
    u_int = sla.solve_banded((1, 1), ab, _src[1:-1])
    u = np.zeros(G); u[1:-1] = u_int
    return u


def _forward(C):
    M = _medium(C)
    Y = np.empty((M.shape[0], len(_sensors)))
    for i in range(M.shape[0]):
        Y[i] = _solve_one(M[i])[_sensors]
    return Y


def make(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.uniform(-1.0, 1.0, size=(n, K_MODES))
    X = C
    Y = _forward(C)
    return Dataset(
        name="helmholtz", X=X, Y=Y, constraint="none",
        meta={"d": K_MODES, "N": len(_sensors), "k0": K0, "tractable": False,
              "field_shape": [G]},
        simulate=lambda th: _forward(np.atleast_2d(th)),
        prior_sample=lambda m, r: r.uniform(-1.0, 1.0, (m, K_MODES)),
        prior_bounds=(-1.0 * np.ones(K_MODES), 1.0 * np.ones(K_MODES)),
        decode_field=lambda c: _medium(np.asarray(c, float).ravel())[0],
    )
