# PRISM — Phase 2 Guide (core guaranteed-property experiments)

Phase 2 runs the three experiments that establish PRISM's *guaranteed* and
*measured* core properties, all on the Phase 1 provenance harness so every value
is multi-seed and traceable. It directly closes guide items **1, 4, 5, 9, 21, 31**.

## What's in this drop

| File | Path in repo | Produces | Guide items |
|---|---|---|---|
| `phase2_analysis.py` | `experiments/phase2_analysis.py` | order/floor fit, spectral norm, bound, sensitivity (torch-free) | 21, 9, 31 |
| `phase2_reversibility.py` | `experiments/phase2_reversibility.py` | E1: latent vs physical cycle sweep | 1, 4, 5, 21 |
| `phase2_constraints_fair.py` | `experiments/phase2_constraints_fair.py` | E2: fair constraint comparison | 10 |
| `phase2_stability.py` | `experiments/phase2_stability.py` | E3: Lipschitz factors + sensitivity | 9, 31 |
| `make_phase2_tables.py` | `tables/make_phase2_tables.py` | T-Cycle, T-Constraints-Fair, T-Stability | — |
| `make_phase2_figures.py` | `figures/make_phase2_figures.py` | F-Reversibility, F-Stability | 21, 31 |
| `test_phase2_analysis.py` | `tests/test_phase2_analysis.py` | analysis unit tests | — |

## Verified in the sandbox (evidence)

- **Analysis core** — RK4-like sweep recovers convergence order **4.000**, the
  floor is detected, `spectral_norm` matches `numpy.linalg.norm(A,2)`, the bound
  composes correctly, and local sensitivity is bounded by the operator norm. All
  unit tests pass.
- **Reporting pipeline end-to-end** — a synthetic 105-record manifest builds all
  three tables and both figures; cells render as e.g.
  `4.00 $\pm$ 0.00` and `9.9e-10 $\pm$ 3e-14`, with the latent/physical split and
  Title-Case task names correct.

The experiment scripts themselves call torch/your model and run in `D:\ICLR`
(no torch here); the correctness-critical analysis and reporting they depend on
is what was verified above.

## Repo hooks to confirm (attribute names)

The scripts reference a few model internals via `getattr` fallbacks; confirm or
adjust these once against your codebase (all flagged in-file):

- `phase2_reversibility.py`: `_set_solver_steps` (flow step-count attribute),
  `prism.predict_forward(x)` (surrogate `g_ψ`), `prism.invert(y)` (inverse `h`),
  `prism.flow_`, `prism.transform_`.
- `phase2_stability.py`: `flow.velocity(v,t)` (or `flow.g`), `prism.amortize_(y)`
  returning `(μ, σ)`.
- `phase2_constraints_fair.py`: `make_method("cINN"/"NPE-CNF"/"VI", ...)`,
  `emcee_reference`, and `prism.posterior.calibration.{c2st,coverage,gaussian_nll}`.

## Run order (in D:\ICLR)

1. `py -m pytest -q tests/test_phase2_analysis.py` (must pass).
2. `py experiments\phase2_reversibility.py --mode smoke`
3. `py experiments\phase2_constraints_fair.py --mode smoke`
4. `py experiments\phase2_stability.py --mode smoke`
5. `py tables\make_phase2_tables.py`
6. `py figures\make_phase2_figures.py`
7. Sanity-read: the reversibility figure should show a descending O(hᵖ) regime
   with slope ≈ integrator order, then a floor — not a flat line. If it is still
   flat, the step range is entirely below the floor: widen it or keep float64.
8. Re-run 2–6 with `--mode full` for the 5-seed camera-ready numbers.

## Phase 2 acceptance gate (before Phase 3)

- [ ] Reversibility figure shows the convergence regime + floor; fitted latent
      order is within ~0.5 of the integrator order (item 21 resolved).
- [ ] Latent and physical cycle are reported as **separate** quantities; no text
      claims physical exactness (item 1 resolved).
- [ ] After wrapping, cINN/NPE-CNF/VI all show ≈0 violations on the box/positive
      tasks — confirming feasibility is the transform's doing, and moving the
      story to calibration/NLL/latency (item 10 resolved).
- [ ] Stability table shows the loose global bound *and* the (much smaller)
      empirical mean/max sensitivity, with the distinction stated (items 9, 31).
- [ ] Every Phase 2 table/figure reads from `results/manifest.json`; no ad-hoc
      numbers.

Once these are green, Phase 3 runs the ablations that justify novelty
(liquid-vs-static across ≥5 tasks, conditioning ablation, hybrid velocity,
no-projection on a box task), all on the same harness.
