# PRISM — Phase 5 Guide (accuracy, efficiency, scaling, generality)

Phase 5 is the last experimental phase. It re-runs the headline accuracy numbers
with 5 seeds, turns the unsupported speedup claim into a demonstrated crossover,
fixes the super-resolution provenance bug, and replaces the arbitrary pass/fail
generality table with raw metrics. Closes guide items **16, 18, 20, 26, 28, 29**.

## What's in this drop

| File | Path in repo | Produces | Guide items |
|---|---|---|---|
| `phase5_analysis.py` | `experiments/phase5_analysis.py` | crossover, source audit, forward gap, primary metric (torch-free) | 20, 28, 29, 18 |
| `phase5_main_pde.py` | `experiments/phase5_main_pde.py` | E11: forward + inverse, 5 seeds | 16, 29 |
| `phase5_efficiency_scaling.py` | `experiments/phase5_efficiency_scaling.py` | E12: efficiency + crossover | 20, 26 |
| `phase5_superres.py` | `experiments/phase5_superres.py` | E13: super-res (audited PRISM source) | 28 |
| `phase5_generality.py` | `experiments/phase5_generality.py` | E14: raw-metric generality | 18 |
| `make_phase5_tables.py` | `tables/make_phase5_tables.py` | main-PDE, efficiency, generality tables | 26, 29 |
| `make_phase5_figures.py` | `figures/make_phase5_figures.py` | F-Fields, F-Pareto, F-SuperRes | 26, 28 |
| `test_phase5_analysis.py` | `tests/test_phase5_analysis.py` | analysis unit tests | — |

## Verified in the sandbox (evidence)

- **Analysis core** — crossover break-even computes to 66.7 observations and
  **shrinks with dimension (100 → 29 → 5)**; the source audit correctly identifies
  a set of numbers as FNO's when they are mislabeled PRISM; the forward gap and
  primary-metric selection work. All tests pass.
- **Reporting pipeline** — every Phase 5 table and figure builds from a synthetic
  manifest. The main-PDE caption states the forward gap as **"about 180% lower
  relative L2"**; PRISM is bolded where it wins the inverse but **NPE-CNF is bolded
  on Helmholtz** where it genuinely wins; the efficiency caption shows the crossover
  falling 100 → 29 → 5; the super-res figure is PRISM-labeled.

## Hooks to confirm

- `est.predict_forward(X)` — auxiliary forward surrogate (E11/E13/E14).
- `ds.upsample_params(X, factor)` and `ds.solve_high_res(X, factor)` — the
  super-resolution query at higher mesh resolution (E13). If your dataset exposes
  these differently, adjust `_forward_at`.
- `get_dataset("gaussian_linear", seed=s, dim=d)` — a `dim` argument for the
  scaling study (E12); if unsupported, wrap a dimension-parameterized generator.
- `mcmc_reference(task, y, n_samples)` — timed per-observation reference (E12).

## Run order (in D:\ICLR)

1. `py -m pytest -q tests/test_phase5_analysis.py`
2. `py experiments\phase5_main_pde.py --mode smoke`
3. `py experiments\phase5_efficiency_scaling.py --mode smoke`
4. `py experiments\phase5_superres.py --mode smoke`   (aborts if the audit fails)
5. `py experiments\phase5_generality.py --mode smoke`
6. `py tables\make_phase5_tables.py`
7. `py figures\make_phase5_figures.py`
8. Re-run 2–5 with `--mode full` for the 5-seed numbers.

## Phase 5 acceptance gate (before Phase 6)

- [ ] Main PDE table has 5-seed mean±std; the forward wording states the numeric
      gap and PRISM is not called "best" on forward (items 16, 29).
- [ ] The accuracy–latency figure is generated from the SAME records as the
      efficiency table; caption and data agree (item 26).
- [ ] The efficiency/scaling result shows the amortization crossover and its
      decrease with dimension — the speedup claim is demonstrated, not asserted,
      or removed (item 20).
- [ ] The super-resolution numbers pass the provenance audit (are PRISM's, not
      FNO's) and the figure is labeled PRISM (item 28).
- [ ] The generality table reports raw metrics with a posterior-appropriate
      primary metric; no binary pass/fail (item 18).
- [ ] Every table/figure reads from `results/manifest.json`.

Once green, **Phase 6** is the single consolidated full-scale run: execute all
phases end-to-end at `--mode full` with 5 seeds into one frozen manifest,
regenerate every table and figure from it, run the preflight audit (stale-string
grep, claim-vs-table check, SLCP-modality consistency), fill the reproducibility
appendix, and assemble the final ICLR 2027 manuscript.

## Scope note

As in every phase, the experiment scripts call torch/your model and run in
`D:\ICLR`; the analysis and the full table/figure pipeline are verified here.
