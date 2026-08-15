# RUN_ORDER — PRISM ICLR 2027 pipeline

The exact order to run everything, and which phase each file belongs to. Run all
commands from `D:\ICLR`. `<mode>` is `smoke` (fast, CPU) or `full` (5 seeds,
camera-ready). The simplest path is the four master batch files; the per-file order
below is what they encode, for granular control or debugging.

## Fast path (recommended)

```
setup.bat
run_tests.bat
run_full.bat        (== experiments\run_all_phases.py --mode full --fresh)
build_paper.bat
```

`run_all_phases.py` executes every experiment below in order into one manifest,
regenerates all tables/figures, and runs the preflight audit.

---

## Phase 0 — Reframing & theory (NO code runs)

Paper-only. These are merged into the manuscript; nothing executes.

| File | Role |
|---|---|
| `phase0_theory.tex` | Assumption 1 + Propositions 1-3 (latent reversibility, numerical, stability) |
| `phase0_constraints.tex` | support transforms + physical-space NLL |
| `docs/paper_fragments/phase0_capability_matrix.tex` | objective capability matrix |
| `docs/paper_fragments/phase0_abstract_contributions.tex` | reframed abstract + contributions |

Already inlined/inputted by `prism_iclr2027.tex`.

## Phase 1 — Setup & measurement infrastructure

| Order | Command | Phase file |
|---|---|---|
| 1 | `setup.bat` | `requirements.txt`, `pyproject.toml` |
| 2 | `run_tests.bat` (`py -m pytest -q tests`) | `tests/test_transforms.py` (+ analysis tests) |
| 3 | `py baselines\npe_cnf_validated.py` | NPE-CNF sanity self-check |

Infrastructure used by every later phase: `experiments/api.py`,
`experiments/provenance.py`, `prism/constraints/transforms.py`,
`prism/utils/metrics_cycle.py`, `baselines/support_wrapper.py`.

## Phase 2 — Core guaranteed properties  (`batch\phase2.bat <mode>`)

| Order | Command |
|---|---|
| 4 | `py experiments\phase2_reversibility.py --mode <mode>` |
| 5 | `py experiments\phase2_constraints_fair.py --mode <mode>` |
| 6 | `py experiments\phase2_stability.py --mode <mode>` |
| 7 | `py tables\make_phase2_tables.py` |
| 8 | `py figures\make_phase2_figures.py` |

Analysis core: `experiments/phase2_analysis.py`.

## Phase 3 — Ablations  (`batch\phase3.bat <mode>`)

| Order | Command |
|---|---|
| 9 | `py experiments\phase3_liquid_vs_static.py --mode <mode>` |
| 10 | `py experiments\phase3_conditioning.py --mode <mode>` |
| 11 | `py experiments\phase3_hybrid_velocity.py --mode <mode>` |
| 12 | `py experiments\phase3_projection.py --mode <mode>` |
| 13 | `py tables\make_phase3_tables.py` |

Cores: `experiments/phase3_analysis.py`, `experiments/phase3_variants.py`.

## Phase 4 — Posterior quality  (`batch\phase4.bat <mode>`)

| Order | Command |
|---|---|
| 14 | `py experiments\phase4_calibration.py --mode <mode>` |
| 15 | `py experiments\phase4_joint_geometry.py --mode <mode>` |
| 16 | `py experiments\phase4_mixture_base.py --mode <mode>` (optional; skip if no mixture base) |
| 17 | `py tables\make_phase4_tables.py` |
| 18 | `py figures\make_phase4_figures.py` |

Analysis core: `experiments/phase4_analysis.py`.

## Phase 5 — Accuracy / efficiency / scaling / generality  (`batch\phase5.bat <mode>`)

| Order | Command |
|---|---|
| 19 | `py experiments\phase5_main_pde.py --mode <mode>` |
| 20 | `py experiments\phase5_efficiency_scaling.py --mode <mode>` |
| 21 | `py experiments\phase5_superres.py --mode <mode>` |
| 22 | `py experiments\phase5_generality.py --mode <mode>` |
| 23 | `py tables\make_phase5_tables.py` |
| 24 | `py figures\make_phase5_figures.py` |

Analysis core: `experiments/phase5_analysis.py`.

## Phase 6 — Consolidation, audit, manuscript

| Order | Command |
|---|---|
| 25 | `run_full.bat` (re-runs 4-24 into one frozen manifest, then all tables/figures) |
| 26 | `build_paper.bat` (pdflatex + bibtex + preflight audit) |
| 27 | `audit.bat` (re-run the audit any time) |

Tooling: `experiments/run_all_phases.py`, `tools/preflight_audit.py`,
`PRISM_reproducibility.tex`, `prism_iclr2027.tex`.

---

## Notes

- Run `run_smoke.bat` first to confirm the whole pipeline executes on your machine
  before spending the full 5-seed run.
- Tables and figures always read from `results/manifest.json`; never edit generated
  files in `paper_assets/` by hand.
- Confirm the model hooks in `INTEGRATION.md` before the full run — that is the only
  place code depends on your `prism.py` internals.
