"""Baselines satisfy their API contracts (posterior + operator surrogates)."""
import numpy as np
import pytest
from baselines import CINN, NPECNF, AmortizedVI, FNO, DeepONet, PINN


def _toy(n=300, d=2, N=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.3, 1.5, (n, d))
    W = rng.normal(size=(d, N))
    Y = np.tanh(X @ W) + 0.02 * rng.normal(size=(n, N))
    return X, Y


@pytest.mark.parametrize("ctor", [
    lambda: CINN(epochs=10, n_layers=4, seed=0),
    lambda: NPECNF(epochs=8, n_steps=4, seed=0),
    lambda: AmortizedVI(epochs=15, seed=0),
])
def test_posterior_baseline_api(ctor):
    X, Y = _toy()
    m = ctor().fit(X, Y)
    assert m.invert(Y).shape == X.shape
    post = m.predict_posterior(Y[:5], n_samples=20)
    assert post.shape == (5, 20, X.shape[1])


@pytest.mark.parametrize("ctor", [
    lambda: FNO(epochs=8, seed=0),
    lambda: DeepONet(epochs=10, seed=0),
    lambda: PINN(epochs=10, seed=0),
])
def test_operator_baseline_api(ctor):
    rng = np.random.default_rng(0)
    fields = rng.standard_normal((120, 16))         # 1D fields
    Y = fields[:, :4] * 2 + 0.1 * rng.standard_normal((120, 4))
    m = ctor().fit(fields, Y)
    assert m.predict(fields).shape == Y.shape
