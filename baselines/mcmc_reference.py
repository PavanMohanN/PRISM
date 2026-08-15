"""MCMC reference — gold-standard posterior via emcee on the true likelihood.

Not a learned method: it samples the exact posterior implied by a benchmark's
tractable likelihood and prior, providing the reference against which learned
posteriors (PRISM, cINN, NPE-CNF, VI) are scored. Slow, so it is applied to a
handful of test observations.
"""
from __future__ import annotations

import numpy as np

from benchmarks.registry import emcee_reference


class MCMCReference:
    def __init__(self, dataset, nsteps=1500, burn=500):
        if dataset.log_likelihood is None or dataset.prior_bounds is None:
            raise ValueError(f"dataset '{dataset.name}' has no tractable likelihood")
        self.ds = dataset; self.nsteps = nsteps; self.burn = burn

    def fit(self, *a, **k):
        return self                                   # nothing to train

    def predict_posterior(self, Y, n_samples=1000, seed=0):
        Y = np.atleast_2d(Y)
        out = np.empty((Y.shape[0], n_samples, self.ds.d))
        for i, y in enumerate(Y):
            out[i] = emcee_reference(self.ds.log_likelihood, self.ds.log_prior,
                                     self.ds.prior_bounds, y, n=n_samples,
                                     seed=seed + i, nsteps=self.nsteps, burn=self.burn)
        return out

    def invert(self, Y, n_samples=1000, seed=0):
        return self.predict_posterior(Y, n_samples=n_samples, seed=seed).mean(axis=1)

    def sample(self, Y, n_samples=1000, seed=0):
        return self.predict_posterior(Y, n_samples=n_samples, seed=seed)
