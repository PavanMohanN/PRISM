# PRISM — Phase 1 Integration Guide (code corrections + measurement harness)

Phase 1 makes the implementation match the corrected Phase 0 theory and builds
the infrastructure that guarantees the final results are fair, reproducible, and
internally consistent. Proceeding on the recommended gate defaults
(D2 = SiLU, D3 = fixed-step RK4 + step-size language, D4 = drop simplex from
claims but ship a correct stick-breaking utility, D5 = physical-space NLL).

## What's in this drop

New modules (copy into the matching paths under `D:\ICLR`):

| File | Path in repo | Purpose | Guide items |
|---|---|---|---|
| `transforms.py` | `prism/constraints/transforms.py` | correct support transforms + log-det (softmax removed) | 7, 8 |
| `test_transforms.py` | `tests/test_transforms.py` | normalization + round-trip + feasibility gate | 7, 8 |
| `metrics_cycle.py` | `prism/utils/metrics_cycle.py` | latent vs **physical** cycle metrics | 1 |
| `support_wrapper.py` | `baselines/support_wrapper.py` | identical transform for every baseline (fair comparison) | 10 |
| `npe_cnf_validated.py` | `baselines/npe_cnf_validated.py` | stronger NPE-CNF config + Gaussian-Linear sanity gate | 12 |
| `provenance.py` | `experiments/provenance.py` | one canonical result record + 5-seed harness | 16, 22, 26, 27, 28 |

In-place edits (small, surgical): see `phase1_patches.md` — SiLU activation,
physical-space NLL Jacobian, manifest routing, solver terminology.

## Verified in the sandbox (evidence for the gate)

These ran here on numpy/scipy/sklearn (torch parts run in your env):

- **Change-of-variables normalization** — pushforward densities integrate to
  exactly `1.00000000` for **positive**, **box**, and **simplex (d=3)**; log-det
  identities hold; round-trips at `~1e-15`. → constraint math is correct.
- **NPE-CNF sanity gate** — the analytic linear-Gaussian posterior passes its own
  check at **C2ST 0.504** with coverage `{0.5→0.50, 0.8→0.80, 0.9→0.91}`,
  confirming the gate correctly certifies a good estimator (and will flag a weak
  one).
- **Provenance manifest** — 5-seed aggregation renders `0.555 $\pm$ 0.003`, and
  save/load round-trips. → tables/figures can read one canonical value.

## Integration order

1. Drop in `transforms.py`; run `python tests/test_transforms.py` (and
   `pytest -q tests/test_transforms.py` once torch is present). All must pass.
2. Apply Patch 1 (SiLU) and Patch 2 (NLL Jacobian) from `phase1_patches.md`.
   Re-run the transform tests; confirm the torch round-trip/logdet tests pass.
3. Wrap baselines with `support_wrapper.wrap_all(baselines, transform)` in the
   constraint experiment. Expect all transformed baselines → ~0 violations.
4. Replace the NPE-CNF config with `RECOMMENDED_NPE_CNF`; run
   `sanity_check(...)` on Gaussian-Linear and **do not proceed** until it passes.
5. Convert `phaseN_*.py` to `ResultManifest.run_seeds(...)` over seeds
   `[0,1,2,3,4]`; point `make_tables.py` / `make_figures.py` at
   `manifest.agg/.fmt`.
6. Add `physical_cycle_error(...)` alongside the existing latent cycle metric in
   the reversibility experiment.

## Phase 1 acceptance gate (all must hold before Phase 2)

- [ ] `tests/test_transforms.py` passes fully **with torch** (normalization,
      round-trip, feasibility, logdet-match).
- [ ] `sanity_check` on Gaussian-Linear passes for the re-tuned NPE-CNF
      (C2ST ≤ 0.65, coverage within 0.10 of nominal).
- [ ] Every baseline, wrapped with the shared transform, reports ~0 violations on
      the box/positive tasks (confirming zero-violation is the transform's doing).
- [ ] Physical-space NLL includes the Jacobian term (spot-check one task).
- [ ] All experiments write to the manifest; a table cell and its figure read the
      **same** record (kills the Table 8/Fig 8, Fig 10/Table 9, Table 5/Table 7
      mismatches).
- [ ] Velocity field, amortization net, and forward head use SiLU (grep for
      `ReLU` returns only intended uses, if any).

Once these are green, Phase 2 runs the core "guaranteed-property" experiments
(reversibility sweep with the physical-cycle metric, fair constraint comparison,
stability quantification) on the new harness.

## Note on scope

I did not reproduce your existing source files verbatim (they live on your
machine, not in this session), so Patches 1–4 are written as precise find/replace
against the known structure rather than as full-file rewrites. If you paste the
current `liquid_ode.py` and `prism.py`, I'll return exact full-file versions with
the edits already applied.
