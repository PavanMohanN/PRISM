"""2D Darcy-flow inverse task.

Recover a (KL-parameterised, log-normal) permeability field from sparse pressure
measurements. The forward map solves -div(k grad u) = 1 with u=0 on the boundary
via a 5-point finite-difference scheme; ``decode_field`` rebuilds the full
permeability map from the recovered coefficients for the field-recovery figure.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..registry import Dataset

GRID = 16                                   # GxG grid
MODES = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]   # K=6 KL modes
_h = 1.0 / (GRID - 1)
_xy = np.linspace(0, 1, GRID)
_XX, _YY = np.meshgrid(_xy, _xy, indexing="ij")
# sensor nodes: interior 4x4 lattice
_si = np.linspace(2, GRID - 3, 4).astype(int)
_SENS = [(i, j) for i in _si for j in _si]


def _basis():
    B = np.stack([np.cos(np.pi * m * _XX) * np.cos(np.pi * n * _YY) for m, n in MODES])
    return B                                # (K, G, G)


_B = _basis()


def _field(c):
    """coeffs (K,) -> permeability k (G,G), strictly positive."""
    f = np.tensordot(c, _B, axes=(0, 0))
    return np.exp(0.5 * f)


def _solve(k):
    """5-point FD solve of -div(k grad u)=1, u=0 boundary. Returns u (G,G)."""
    G = GRID
    nint = (G - 2) ** 2
    idx = -np.ones((G, G), int)
    cnt = 0
    for i in range(1, G - 1):
        for j in range(1, G - 1):
            idx[i, j] = cnt; cnt += 1
    rows, cols, vals = [], [], []
    b = np.ones(nint)
    inv_h2 = 1.0 / _h ** 2
    for i in range(1, G - 1):
        for j in range(1, G - 1):
            p = idx[i, j]
            kE = 0.5 * (k[i, j] + k[i + 1, j])
            kW = 0.5 * (k[i, j] + k[i - 1, j])
            kN = 0.5 * (k[i, j] + k[i, j + 1])
            kS = 0.5 * (k[i, j] + k[i, j - 1])
            rows.append(p); cols.append(p); vals.append((kE + kW + kN + kS) * inv_h2)
            for (ii, jj, kk) in [(i + 1, j, kE), (i - 1, j, kW),
                                 (i, j + 1, kN), (i, j - 1, kS)]:
                if idx[ii, jj] >= 0:
                    rows.append(p); cols.append(idx[ii, jj]); vals.append(-kk * inv_h2)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(nint, nint))
    u_int = spla.spsolve(A, b)
    u = np.zeros((G, G)); 
    for i in range(1, G - 1):
        for j in range(1, G - 1):
            u[i, j] = u_int[idx[i, j]]
    return u


def _forward_batch(C):
    Y = np.empty((C.shape[0], len(_SENS)))
    for n_, c in enumerate(C):
        u = _solve(_field(c))
        Y[n_] = [u[i, j] for (i, j) in _SENS]
    return Y


def make(n=600, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((n, len(MODES)))
    X = C
    Y = _forward_batch(C)

    return Dataset(
        name="darcy", X=X, Y=Y, constraint="none",
        meta={"d": len(MODES), "N": len(_SENS), "grid": GRID, "tractable": False,
              "field_shape": [GRID, GRID]},
        simulate=lambda th: _forward_batch(np.atleast_2d(th)),
        prior_sample=lambda m, r: r.standard_normal((m, len(MODES))),
        prior_bounds=(-4 * np.ones(len(MODES)), 4 * np.ones(len(MODES))),
        decode_field=lambda c: _field(np.asarray(c, float).ravel()),
    )
