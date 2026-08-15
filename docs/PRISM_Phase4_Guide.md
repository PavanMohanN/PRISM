# PRISM — Phase 4 Guide (posterior quality, calibration breadth, multimodal honesty)

Phase 4 broadens calibration to every domain, replaces the overclaimed
"best-calibrated" wording with honest per-metric bolding, and makes the Two Moons
story about *joint* geometry rather than marginal coverage. Closes guide items
**2 (empirical), 17, 19, 24, 25**.

## What's in this drop

| File | Path in repo | Produces | Guide items |
|---|---|---|---|
| `phase4_analysis.py` | `experiments/phase4_analysis.py` | SBC, coverage/ECE, MMD, mode coverage, marginal-vs-joint (torch-free) | 17, 19 |
| `phase4_calibration.py` | `experiments/phase4_calibration.py` | E8: calibration on 5 domains | 17, 2 |
| `phase4_joint_geometry.py` | `experiments/phase4_joint_geometry.py` | E9: Two Moons joint geometry | 19 |
| `phase4_mixture_base.py` | `experiments/phase4_mixture_base.py` | E10: mixture base (optional, D6) | 19 |
| `make_phase4_tables.py` | `tables/make_phase4_tables.py` | extended calibration + joint-geometry tables | 24 |
| `make_phase4_figures.py` | `figures/make_phase4_figures.py` | F-Calibration, F-Posteriors (Two Moons) | 17, 19 |
| `test_phase4_analysis.py` | `tests/test_phase4_analysis.py` | analysis unit tests | — |

## Verified in the sandbox (evidence)

- **Analysis core** — SBC uniformity gives high p for a calibrated posterior and
  low p for an over-confident one; coverage/ECE recovers ~nominal; MMD is ~0 for
  equal distributions and large for shifted ones; mode coverage separates a
  balanced bimodal set (balance 0.7+) from a collapsed one (< 0.05); and the
  marginal-vs-joint gap shows **marginal MMD 0.0009 vs joint MMD 0.0235**. All
  tests pass.
- **Reporting pipeline** — a synthetic manifest builds both tables and both
  figures. Honest bolding works: on Gaussian-Linear **VI is bolded for Cov@90 and
  ECE** while PRISM is bolded for C2ST/NLL/SBC-p. The joint-geometry table shows
  marginal ECE 0.063 (fine) with mode balance 0.18 (blurred).

## Hooks to confirm

- `reference_posterior(task, y0, n)` in `benchmarks/registry.py` — reference
  samples for Two Moons (and any task assessed by mode coverage). Add if missing;
  E9/E10 need it. `emcee_reference` already covers SLCP/Gaussian-Linear.
- Mixture base flag (E10 only): `make_method("PRISM", task, cfg, base="mixture",
  n_mix=2)`. Skip `phase4_mixture_base.py` entirely if you did not opt in at D6.
- `sklearn.cluster.KMeans` is used to locate the two reference modes.

## Wording decisions to apply (in the .tex, not code)

- **Item 24:** the calibration paragraph must read "competitive with the
  strongest baselines; strong coverage and C2ST on several tasks," and note VI's
  lower ECE on the Gaussian tasks. The table already bolds the true winner.
- **Item 25 (SLCP modality):** fix the contradiction — treat **SLCP as a single
  dominant mode** (complex but connected) and **Two Moons as the disconnected
  bimodal** case. Make Section 4.6, the figure captions, and the limitations use
  this convention consistently.

## Run order (in D:\ICLR)

1. `py -m pytest -q tests/test_phase4_analysis.py`
2. `py experiments\phase4_calibration.py --mode smoke`
3. `py experiments\phase4_joint_geometry.py --mode smoke`
4. `py experiments\phase4_mixture_base.py --mode smoke`   (optional)
5. `py tables\make_phase4_tables.py`
6. `py figures\make_phase4_figures.py`
7. Re-run 2–4 with `--mode full` for the 5-seed numbers.

## Phase 4 acceptance gate (before Phase 5)

- [ ] Calibration table includes Oscillator and Lotka-Volterra (SBC p + coverage),
      not just the three SBI tasks (item 17).
- [ ] Every "best" in the calibration table is the true per-metric winner; the
      text no longer claims PRISM is best-calibrated overall (item 24).
- [ ] Two Moons is reported with the marginal-vs-joint distinction; the text says
      plainly that a single connected base blurs the modes (item 19).
- [ ] SLCP is described consistently as single-dominant-mode everywhere (item 25).
- [ ] If the mixture base was run and recovers both modes, the abstract regains a
      multimodal clause; otherwise the honest limitation stands.
- [ ] All tables/figures read from `results/manifest.json`.

Once green, Phase 5 covers accuracy/efficiency/scaling/generality: main PDE with
5 seeds, the efficiency + amortization-scaling study (fixing the speedup claim),
the super-resolution source audit, and the raw-metric generality table.

## Scope note

As before, the experiment scripts call torch/your model and run in `D:\ICLR`;
the analysis and the full table/figure pipeline are verified here. Paste
`prism.py` / the registry and I'll pin the mixture-base flag and
`reference_posterior` exactly.
