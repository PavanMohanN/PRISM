"""PRISM Phase 2 -- E3: stability quantification (guide items 9, 31).

Proposition 3 gives a GLOBAL bound Lip(h) <= Lip(c) e^{L_g} Lip(mu). Stated alone
it is uninformative. Here we:
  * compute each factor (Lip(c) analytic per transform; L_g and Lip(mu) via the
    spectral norm of network Jacobians over samples),
  * report the composed bound, and
  * measure the EMPIRICAL local sensitivity of h over held-out observations,
    reporting mean and max separately and sweeping the perturbation eps -- so we
    never equate a finite-sample max with the true global Lipschitz constant.

Run in D:\\ICLR:  py experiments\\phase2_stability.py --mode smoke
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from experiments.provenance import ResultManifest
from experiments._common import load_cfg
from experiments.api import make_method
from experiments.api import get_dataset
from experiments.phase2_analysis import spectral_norm, lipschitz_bound, local_sensitivity_ratio

TASKS = ["oscillator", "slcp"]
SEEDS = [0, 1, 2, 3, 4]
EPS = [1e-3, 1e-2, 1e-1]


def _transform_lipschitz(prism) -> float:
    """Analytic Lip(c): sup |c'(u)|.  softplus->1 ; box[a,b]->(b-a)/4 ; identity->1."""
    tf = getattr(prism, "transform_", None)
    name = type(tf).__name__ if tf is not None else "Identity"
    if name == "Box":
        width = float((tf.upper - tf.lower).max())
        return width / 4.0
    return 1.0   # softplus and identity are 1-Lipschitz


def _jacobian_spectral(fn, x0) -> float:
    """Spectral norm of the Jacobian of fn at x0 via torch autograd + power iter."""
    x0 = torch.as_tensor(x0, dtype=torch.get_default_dtype()).clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(lambda z: fn(z), x0)
    return spectral_norm(J.detach().cpu().numpy().reshape(J.shape[0], -1))


def run(task, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    prism = make_method("PRISM", task, cfg)
    prism.fit(ds.X_train, ds.y_train)

    lip_c = _transform_lipschitz(prism)

    # L_g: max spectral norm of d g_theta / d v over samples and a few times
    flow = getattr(prism, "flow_", getattr(prism, "flow", None))
    vel = lambda v: flow.velocity(v, t=0.5) if hasattr(flow, "velocity") else flow.g(v, 0.5)
    Xt = prism.transform_.inverse(torch.as_tensor(ds.X_test)) if hasattr(prism, "transform_") \
        else torch.as_tensor(ds.X_test)
    Lg = max(_jacobian_spectral(vel, Xt[i]) for i in range(min(16, len(Xt))))

    # Lip(mu): spectral norm of the amortization mean network over observations
    amort = getattr(prism, "amortize_", None)
    mu = (lambda y: amort(y)[0]) if amort is not None else (lambda y: prism.base_mean(y))
    Y = torch.as_tensor(ds.y_test)
    Lmu = max(_jacobian_spectral(mu, Y[i]) for i in range(min(16, len(Y))))

    bound = lipschitz_bound(lip_c, Lg, Lmu)

    # empirical local sensitivity of h: y -> x_hat
    h_of = lambda y: np.asarray(prism.invert(torch.as_tensor(y).unsqueeze(0))).ravel() \
        if hasattr(prism, "invert") else np.asarray(prism.predict(y[None])).ravel()
    sens = {}
    for e in EPS:
        means, maxes = zip(*[local_sensitivity_ratio(h_of, Y[i].numpy(), e, n_dirs=16)
                             for i in range(min(16, len(Y)))])
        sens[f"sens_mean_{e:g}"] = float(np.mean(means))
        sens[f"sens_max_{e:g}"] = float(np.max(maxes))

    out = {"lip_c": lip_c, "L_g": float(Lg), "lip_mu": float(Lmu),
           "bound": float(bound)}
    out.update(sens)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--manifest", default="results/manifest.json")
    args = ap.parse_args()
    cfg = load_cfg("phase2", args.mode)

    import os
    man = ResultManifest(args.manifest)
    if os.path.exists(args.manifest):
        man.load()
    for task in TASKS:
        man.run_seeds("phase2_stability", task, "PRISM",
                      fn=lambda s, t=task: run(t, s, cfg),
                      seeds=SEEDS, config={"mode": args.mode, "eps": EPS})
        b = man.agg("phase2_stability", task, "PRISM", "bound")[0]
        sm = man.agg("phase2_stability", task, "PRISM", f"sens_max_{EPS[-1]:g}")[0]
        print(f"  [{task}] global bound={b:.2e}  empirical max sensitivity={sm:.3f}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
