# PRISM — Structurally Reversible Latent Transport for Ill-Posed Inverse Problems

Reference implementation, benchmark suite, and reproducible experiment pipeline for
the ICLR 2027 submission. PRISM couples a liquid time-constant neural-ODE flow
(reversible latent transport under stated regularity), explicit support transforms
(feasibility by construction), and an observation-conditioned base distribution
(amortized, calibrated posterior inference).

This repository is the **revised** codebase: it implements the full phased revision
that separates properties *guaranteed by construction* from those *demonstrated
empirically*, adds fair (identically-constrained) baselines, multi-seed reporting,
and a single-source result manifest so no number can disagree across the paper.

## Repository layout

```
D:\ICLR\
├── prism/                     installable package (model, flows, constraints, posterior, theory)
│   ├── constraints/transforms.py     support transforms + correct log-det  [revision]
│   └── utils/metrics_cycle.py        latent vs physical cycle metrics       [revision]
├── benchmarks/                8 procedural tasks (PDE / SBI / dynamics) + registry
├── baselines/                 cINN, NPE-CNF, VI, FNO, DeepONet, PINN, MCMC
│   ├── support_wrapper.py            identical-transform wrapper (fair comparison) [revision]
│   └── npe_cnf_validated.py          stronger NPE-CNF + Gaussian-Linear sanity gate [revision]
├── experiments/               experiment pipeline
│   ├── api.py                        integration adapter — CONFIRM the two maps here
│   ├── provenance.py                 result manifest + 5-seed harness
│   ├── phaseN_analysis.py            torch-free analysis cores (unit-tested)
│   ├── phaseN_*.py                   the revision experiments (Phases 2-5)
│   ├── run_all_phases.py             consolidated run + regenerate + audit
│   └── legacy/                       pre-revision scripts (reference only)
├── tables/  figures/          manifest -> LaTeX tables / PDF figures
├── tools/preflight_audit.py   stale-claim / ref / asset / contradiction scan
├── tests/                     unit tests (transforms + analysis cores)
├── configs/                   phase2-6.yaml (smoke / full sections)
├── results/                   manifest.json lands here (git-ignored)
├── paper_assets/              generated tables & figures land here (git-ignored)
├── docs/                      per-phase guides + the revision design + patches
├── prism_iclr2027.tex         assembled manuscript (inputs all phase tables)
├── phase0_theory.tex, phase0_constraints.tex, PRISM_reproducibility.tex, references.bib
├── *.bat, batch/              Windows run scripts (see RUN_ORDER.md)
└── requirements.txt, pyproject.toml, Makefile, LICENSE
```

## Quick start (from `D:\ICLR`)

```bat
setup.bat          :: install dependencies (once)
run_tests.bat      :: Phase 1 gate - unit tests must pass
run_smoke.bat      :: fast end-to-end proof of the pipeline (minutes, CPU)
run_full.bat       :: THE final results trial - all phases, 5 seeds, one manifest, audit
build_paper.bat    :: compile prism_iclr2027.tex + run the preflight audit
```

`RUN_ORDER.md` gives the exact file-by-file order and which phase each belongs to.
`INTEGRATION.md` lists the handful of model hooks to confirm against `prism.py`
before the full run.

## What was verified where

The correctness-critical analysis logic (change-of-variables normalization, SBC
uniformity, MMD, crossover, order/floor fitting, paired verdicts) is covered by
`tests/` and runs on numpy/scipy without a GPU. The experiment scripts call the
model and run on your machine. See `docs/PRISM_Phase*_Guide.md` for the per-phase
acceptance gates and the verified evidence.

## Dependencies

Python 3.10+ (tested on 3.12); `numpy, scipy, scikit-learn, matplotlib, pyyaml,
torch>=2.0, torchdiffeq, emcee`, plus `pytest` for tests. LaTeX (pdflatex + bibtex)
and the ICLR style files (`iclr2026_conference.sty/.bst`, `math_commands.tex`) are
required only to build the manuscript.

## License

MIT - see `LICENSE`.
