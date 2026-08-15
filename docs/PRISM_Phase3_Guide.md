# PRISM — Phase 3 Guide (ablations that justify the novelty)

Phase 3 converts three soft claims into controlled experiments on the Phase 1
harness, closing guide items **11, 13, 14, 15** and supplying the demonstration
that backs the "liquid" novelty (**30**).

## What's in this drop

| File | Path in repo | Produces | Guide items |
|---|---|---|---|
| `phase3_analysis.py` | `experiments/phase3_analysis.py` | paired verdicts + collapse diagnostics (torch-free) | 13, 15, 16 |
| `phase3_variants.py` | `experiments/phase3_variants.py` | variant construction hooks + param counting | 15 |
| `phase3_liquid_vs_static.py` | `experiments/phase3_liquid_vs_static.py` | E4: liquid vs static, 5 tasks | 15, 30 |
| `phase3_conditioning.py` | `experiments/phase3_conditioning.py` | E5: base / dynamics / both | 13 |
| `phase3_hybrid_velocity.py` | `experiments/phase3_hybrid_velocity.py` | E6: shared vs hybrid velocity | 14 |
| `phase3_projection.py` | `experiments/phase3_projection.py` | E7: projection on box tasks | 11 |
| `make_phase3_tables.py` | `tables/make_phase3_tables.py` | the four ablation tables | — |
| `test_phase3_analysis.py` | `tests/test_phase3_analysis.py` | analysis unit tests | — |

## Verified in the sandbox (evidence)

- **Analysis core** — paired-delta + `classify_regime` correctly return `helps`
  for a consistent improvement, `neutral` for mixed-sign noise, `hurts` for a
  consistent regression; the collapse diagnostics (`spread_ratio`,
  `y_dependence`) separate an informative posterior from a collapsed one; parity
  check works. All tests pass.
- **Reporting pipeline end-to-end** — a synthetic manifest builds all four tables;
  the liquid-vs-static verdict reads **helps** on SLCP/Two Moons and **neutral**
  on Darcy/Oscillator/Lotka-Volterra (the honest, task-dependent story), and the
  conditioning table shows the dynamics-only **collapse** (y-dependence ≈ 0.02,
  spread ratio ≈ 0.98) beside the informative base (0.50, 0.40).

## Variant hooks to confirm (one place)

All four experiments toggle PRISM through `experiments/phase3_variants.py`. Set
`FLAG_ALIASES` once to your estimator's kwarg names:

- `liquid` — liquid field vs static neural ODE (you already expose this).
- `cond_base` — observation-conditioned base (default True).
- `cond_velocity` — observation-conditioned velocity g(v,t,y). **This is the one
  new mode** to add if it doesn't exist; E6 needs it, and E5 uses it for the
  dynamics-only variant. If you don't want to implement it, drop E6 and reduce
  E5 to base-only vs a `cond_base=False` control.
- `use_projection` — apply the support transform (default True).

Also confirm `count_params` sees your modules (it scans estimator attributes for
`torch.nn.Module`s) and `emcee_reference(task, y)` exists for SLCP/Gaussian-Linear.

## Run order (in D:\ICLR)

1. `py -m pytest -q tests/test_phase3_analysis.py`
2. `py experiments\phase3_liquid_vs_static.py --mode smoke`
3. `py experiments\phase3_conditioning.py --mode smoke`
4. `py experiments\phase3_hybrid_velocity.py --mode smoke`
5. `py experiments\phase3_projection.py --mode smoke`
6. `py tables\make_phase3_tables.py`
7. Re-run 2–5 with `--mode full` for the 5-seed camera-ready numbers.

## Phase 3 acceptance gate (before Phase 4)

- [ ] Liquid vs static run on ≥5 tasks with **matched parameter counts**
      (parity within ~10%; the table prints both counts) and 5 seeds; the paper
      states the regime where liquid helps rather than claiming it helps
      universally (item 15).
- [ ] Conditioning ablation shows the dynamics-only variant collapsing
      (spread ratio ≈ 1, y-dependence ≈ 0) and the base variant informative;
      the text is reworded from a generic claim to this measured result (item 13).
- [ ] Hybrid-velocity result reported honestly, including a negative result if
      the hybrid does not beat the shared transport (item 14).
- [ ] No-projection ablation is on a **box** task and shows the violation rate
      jumping without projection while posterior quality is ~unchanged (item 11).
- [ ] Every table reads from `results/manifest.json`; verdicts use the paired
      seed test, not single runs.

Once green, Phase 4 covers posterior quality and calibration breadth
(Oscillator/Lotka-Volterra SBC, the Two Moons joint-geometry honesty, and the
optional mixture-base multimodal experiment if you opted it in at the D6 gate).

## Scope note

As in Phases 1–2, the experiment scripts call torch/your model and run in
`D:\ICLR`; the correctness-critical analysis and the full table pipeline are
what was verified here. If you paste your `prism.py`, I'll pin the variant flags
exactly and hand back the four scripts with no hooks left to confirm.
