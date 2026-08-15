"""Benchmark registry + end-to-end integration guard."""
import numpy as np
import pytest
from benchmarks.registry import get_dataset, list_datasets
from prism import PRISM
from prism.utils.metrics import r2_score


@pytest.mark.parametrize("name", list_datasets())
def test_dataset_loads(name):
    n = 200 if name == "darcy" else 400
    ds = get_dataset(name, n=n, seed=0)
    assert ds.X.shape[0] == ds.Y.shape[0] == n
    assert np.isfinite(ds.X).all() and np.isfinite(ds.Y).all()
    assert ds.Y.std() > 1e-6
    assert ds.X_train.shape[0] > 0 and ds.X_test.shape[0] > 0


def test_field_decoders():
    for name in ("darcy", "burgers", "helmholtz"):
        ds = get_dataset(name, n=50, seed=0)
        f = np.asarray(ds.decode_field(ds.X[0]))
        assert np.isfinite(f).all() and f.size > 1


def test_analytic_reference_posterior():
    ds = get_dataset("gaussian_linear", n=200, seed=0)
    ref = ds.reference_posterior(ds.Y_test[0], n=300)
    assert ref.shape == (300, ds.d)


def test_prism_recovers_oscillator():
    ds = get_dataset("oscillator", n=800, seed=0)
    m = PRISM(epochs=60, solver="rk4", n_steps=5, hidden=48,
              constraint="positive", seed=0).fit(ds.X_train, ds.Y_train)
    Xhat = m.invert(ds.Y_test)
    assert (Xhat > 0).all()                          # constraint honoured
    assert r2_score(ds.X_test[:, 0], Xhat[:, 0]) > 0.5   # real inversion
