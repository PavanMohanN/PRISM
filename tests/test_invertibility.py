"""Headline claim #1: exact reversibility.

The bijective PRISM map must reconstruct its input through forward->inverse to
near the ODE solver tolerance, while the soft (BLiqNet-style) ablation cannot.
"""
import numpy as np
from prism import PRISM


def _toy(n=400, d=3, N=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.2, 2.0, size=(n, d))           # positive params
    W = rng.normal(size=(d, N))
    Y = np.tanh(X @ W) + 0.01 * rng.normal(size=(n, N))
    return X, Y


def test_cycle_consistency_exact_is_tiny():
    X, Y = _toy()
    m = PRISM(epochs=20, solver="rk4", n_steps=8, w_logdet=0.0,
              hidden=32, seed=0).fit(X, Y)
    err = m.cycle_consistency_error(X)
    assert err < 1e-3, f"exact-bijection cycle error too large: {err}"


def test_soft_cycle_is_larger_than_exact():
    X, Y = _toy()
    exact = PRISM(epochs=20, solver="rk4", n_steps=8, w_logdet=0.0,
                  hidden=32, seed=0).fit(X, Y)
    soft = PRISM(epochs=40, soft=True, hidden=32, seed=0).fit(X, Y)
    assert exact.cycle_consistency_error(X) < soft.cycle_consistency_error(X)
