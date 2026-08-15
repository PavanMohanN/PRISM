# INTEGRATION — confirm these hooks before the full run

The revision experiment scripts depend on your model and dataset internals only
through **two files**: `experiments/api.py` (dataset + method + reference adapter)
and a small set of estimator method names. Confirm the items below once, then the
whole pipeline runs unchanged.

## 1. `experiments/api.py` — already bridged, verify the maps

`api.py` translates the scripts' calling convention to the real repo API. It is
written against the actual signatures in `experiments/_common.py` and
`benchmarks/registry.py`:

- `get_dataset(task, seed)` wraps `benchmarks.registry.get_dataset` and adds
  lowercase `.y_train/.y_val/.y_test` aliases over the repo's `.Y_*` fields.
- `make_method(name, task, cfg, **flags)` calls
  `experiments._common.make_method(name, cfg, constraint=..., **over)`; it derives
  the constraint from `CONSTRAINT_FOR` and maps `liquid=False` to the repo's
  `"PRISM-static"` variant.
- `reference_posterior` / `emcee_reference` call `Dataset.reference_posterior(y, n)`.
- `mcmc_reference` wraps `baselines.mcmc_reference.MCMCReference(ds).predict_posterior`.

**Confirm:** the `CONSTRAINT_FOR` map matches your tasks, and `"PRISM-static"` is
your static-flow variant name (it is, in the shipped `_common.make_method`).

## 2. Estimator methods used by the scripts

Your PRISM estimator (`prism/models/prism.py`) and baselines are expected to expose:

| Method / attribute | Used by | Status |
|---|---|---|
| `fit(X, Y)` | all | exists |
| `predict_posterior(y, n)` | all posterior metrics | exists |
| `invert(Y)` -> point estimate | reversibility, accuracy, generality | exists |
| `constraint_violation_rate(y, n)` | constraints, generality | exists |
| `predict_forward(X)` -> Y | reversibility (physical cycle), accuracy, super-res | **confirm/add** |
| `transform_` (support transform object) | reversibility, stability | **confirm/add** |
| `flow_` with settable step count + `forward/inverse` | reversibility sweep | **confirm** |
| `flow_.velocity(v, t)` (or `.g`) | stability (L_g) | **confirm** |
| `amortize_(y)` -> (mu, sigma) | stability (Lip(mu)) | **confirm** |

The scripts already guard several of these with `getattr` fallbacks; the ones
marked **confirm/add** are the small model-side work.

## 3. New PRISM constructor flags (only if you run those ablations)

Passed through `make_method` as `**over` to the PRISM constructor:

- `cond_base`, `cond_velocity` — conditioning ablation (Phase 3, E5) and hybrid
  velocity (E6). If you do not implement `cond_velocity`, skip
  `phase3_hybrid_velocity.py` and reduce `phase3_conditioning.py` to base-only vs a
  `cond_base=False` control.
- `use_projection` — projection ablation (Phase 3, E7).
- `base="mixture", n_mix=2` — optional mixture-base multimodal experiment
  (Phase 4, E10). Skip `phase4_mixture_base.py` if not implemented.

## 4. Dataset super-resolution hooks (Phase 5, E13 only)

`phase5_superres.py` queries the forward map at higher resolution via
`ds.upsample_params(X, factor)` and `ds.solve_high_res(X, factor)`. If your PDE
datasets expose resolution differently, adjust `_forward_at` in that file (or drop
E13 and cite the forward-operator study).

## 5. Config keys

The scripts call `load_cfg("phase2".."phase5", mode)`; `configs/phase2-5.yaml`
already exist with `smoke`/`full` sections. Ensure each provides `epochs`,
`hidden`, and `n_steps` (the keys `make_method` reads).

---

Once §1-§2 are confirmed, run `run_smoke.bat`. If a `getattr`/attribute error
appears, it names the exact hook to wire — fix it in `prism.py` or `api.py`, not in
the experiment scripts.
