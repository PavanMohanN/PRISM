"""Unified benchmark registry.

`get_dataset(name, n=...)` returns a :class:`Dataset` with parameters ``X``,
observations ``Y``, train/val/test splits, a constraint tag, and — where the
task supports it — a simulator, log-likelihood, prior, reference posterior, and
a field decoder for plotting. All generators are self-contained (no downloads).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class Dataset:
    name: str
    X: np.ndarray                       # (n, d) parameters (inverse target)
    Y: np.ndarray                       # (n, N) observations
    constraint: str = "none"
    meta: dict = field(default_factory=dict)
    simulate: Optional[Callable] = None        # theta (b,d) -> obs (b,N)
    log_likelihood: Optional[Callable] = None  # (theta(d,), x_obs(N,)) -> float
    log_prior: Optional[Callable] = None       # theta(d,) -> float
    prior_sample: Optional[Callable] = None    # (n, rng) -> (n, d)
    prior_bounds: Optional[tuple] = None       # (lo(d,), hi(d,))
    _ref: Optional[Callable] = None            # analytic reference, if any
    decode_field: Optional[Callable] = None    # theta(d,) -> field array

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float64)
        self.Y = np.asarray(self.Y, dtype=np.float64)
        self.d = self.X.shape[1]
        self.N = self.Y.shape[1]

    # -------- splitting --------
    def make_splits(self, test=0.1, val=0.1, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.X.shape[0])
        n = len(idx)
        n_te = int(round(test * n))
        n_va = int(round(val * n))
        te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
        self.X_train, self.Y_train = self.X[tr], self.Y[tr]
        self.X_val, self.Y_val = self.X[va], self.Y[va]
        self.X_test, self.Y_test = self.X[te], self.Y[te]
        return self

    # -------- reference posterior --------
    def reference_posterior(self, x_obs, n=1000, seed=0, **kw):
        if self._ref is not None:
            return self._ref(np.asarray(x_obs, float), n=n, seed=seed)
        if self.log_likelihood is not None and self.prior_bounds is not None:
            return emcee_reference(self.log_likelihood, self.log_prior,
                                   self.prior_bounds, x_obs, n=n, seed=seed, **kw)
        raise NotImplementedError(f"no reference posterior for '{self.name}'")


# ---------------------------------------------------------------- helpers
def uniform_prior(lo, hi):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    span = hi - lo

    def sample(n, rng):
        return lo + span * rng.random((n, len(lo)))

    def logp(theta):
        theta = np.asarray(theta, float)
        inside = np.all((theta >= lo) & (theta <= hi))
        return 0.0 if inside else -np.inf

    return sample, logp, (lo, hi)


def emcee_reference(log_like, log_prior, bounds, x_obs, n=1000, seed=0,
                    nsteps=1500, burn=500):
    """Reference posterior samples via emcee MCMC on a tractable likelihood."""
    import emcee
    lo, hi = bounds
    dim = len(lo)
    x_obs = np.asarray(x_obs, float)

    def logpost(theta):
        lp = log_prior(theta) if log_prior is not None else 0.0
        if not np.isfinite(lp):
            return -np.inf
        return lp + float(log_like(theta, x_obs))

    rng = np.random.default_rng(seed)
    nwalk = max(2 * dim + 2, 10)
    p0 = lo + (hi - lo) * rng.random((nwalk, dim))
    sampler = emcee.EnsembleSampler(nwalk, dim, logpost)
    sampler.run_mcmc(p0, nsteps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    sel = rng.choice(chain.shape[0], size=min(n, chain.shape[0]), replace=chain.shape[0] < n)
    return chain[sel]


# ---------------------------------------------------------------- registry
def _registry():
    from .sbi.gaussian_linear import make as gl
    from .sbi.two_moons import make as tm
    from .sbi.slcp import make as slcp
    from .dynamics.oscillator import make as osc
    from .dynamics.lotka_volterra import make as lv
    from .pde.darcy import make as darcy
    from .pde.burgers import make as burgers
    from .pde.helmholtz import make as helmholtz
    return {
        "gaussian_linear": gl, "two_moons": tm, "slcp": slcp,
        "oscillator": osc, "lotka_volterra": lv,
        "darcy": darcy, "burgers": burgers, "helmholtz": helmholtz,
    }


def list_datasets():
    return sorted(_registry().keys())


def get_dataset(name, n=2000, seed=0, test=0.1, val=0.1, **kw):
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown dataset '{name}'. available: {list_datasets()}")
    ds = reg[name](n=n, seed=seed, **kw)
    return ds.make_splits(test=test, val=val, seed=seed)
