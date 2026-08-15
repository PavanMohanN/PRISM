"""API contract: PRISM behaves like a scikit-learn estimator."""
import numpy as np
from prism import PRISM


def _toy(n=300, d=2, N=3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.3, 1.5, size=(n, d))
    W = rng.normal(size=(d, N))
    Y = np.sin(X @ W)
    return X, Y


def test_get_set_params():
    m = PRISM(hidden=16, epochs=3)
    assert m.get_params()["hidden"] == 16
    m.set_params(hidden=8)
    assert m.get_params()["hidden"] == 8


def test_fit_predict_shapes():
    X, Y = _toy()
    m = PRISM(epochs=5, solver="rk4", n_steps=4, hidden=16,
              w_logdet=0.0, constraint="positive").fit(X, Y)
    assert m.predict(X).shape == Y.shape
    assert m.invert(Y).shape == X.shape
    post = m.predict_posterior(Y[:10], n_samples=20)
    assert post.shape == (10, 20, X.shape[1])
    # constraint honoured on the point inverse
    assert (m.invert(Y) > 0).all()


def test_score_runs():
    X, Y = _toy()
    m = PRISM(epochs=5, solver="rk4", n_steps=4, hidden=16,
              w_logdet=0.0).fit(X, Y)
    s = m.score(X, Y)
    assert np.isfinite(s)
