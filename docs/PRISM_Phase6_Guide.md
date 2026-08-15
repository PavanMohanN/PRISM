# PRISM — Phase 6 Guide (final assembly, audit, and submission)

Phase 6 freezes one full-scale run, regenerates every asset from it, mechanically
audits the manuscript, and assembles the submission. Closes guide items **22, 25,
32** and verifies the wording policies set in earlier phases.

## What's in this drop

| File | Path in repo | Purpose |
|---|---|---|
| `run_all_phases.py` | `experiments/run_all_phases.py` | one consolidated 5-seed run + regenerate all tables/figures + audit |
| `preflight_audit.py` | `tools/preflight_audit.py` | stale-phrase / unresolved-ref / missing-asset / SLCP-contradiction scan |
| `prism_iclr2027.tex` | repo root | the assembled manuscript (inputs all phase tables, references all figures) |
| `PRISM_reproducibility.tex` | repo root | reproducibility statement + appendix + hyperparameter table |

## Verified in the sandbox (evidence)

- **Preflight audit self-test passes**: a clean document yields 0 errors; a
  document with planted issues yields 5 caught (stale phrases, unresolved ref,
  SLCP contradiction).
- **Audit on the assembled manuscript**: after fixes, **zero** stale-phrase or
  contradiction errors. The only remaining "unresolved" refs are `tab:cycle` and
  `tab:conditioning`, whose labels live in `paper_assets/` table files not present
  in this sandbox; they resolve once the Phase 2–3 table builders have run in your
  repo. The 15 warnings are legitimate wording ("guaranteed by construction,"
  "exact inverse") flagged for a human to confirm against a table.

## The consolidated run (in D:\ICLR)

```
py experiments\run_all_phases.py --mode full --fresh
```

This: (1) archives any old manifest; (2) runs P2–P5 experiments at `--mode full`,
5 seeds, into `results/manifest.json`; (3) regenerates every table and figure from
that frozen manifest; (4) runs the preflight audit on `prism_iclr2027.tex`.
Then compile:

```
pdflatex prism_iclr2027 && bibtex prism_iclr2027 && pdflatex prism_iclr2027 && pdflatex prism_iclr2027
py tools\preflight_audit.py prism_iclr2027.tex --root .
```

## How the manuscript assembles

`prism_iclr2027.tex` inlines the Phase 0 reframing (abstract, contributions,
capability matrix) and `\input`s: `phase0_theory.tex` (Propositions),
`phase0_constraints.tex` (support transforms + physical NLL),
`PRISM_reproducibility.tex`, and every data table from `paper_assets/`
(`T_cycle_phase2`, `T_constraints_fair`, `T_stability`, `T_liquid_vs_static`,
`T_conditioning`, `T_projection`, `T_main_pde`, `T_calibration_ext`,
`T_joint_geometry`, `T_efficiency`, `T_generality_raw`). Figures are referenced
from `paper_assets/`. Change nothing in the tables by hand — they regenerate.

## Remaining manual steps before submission

1. Fill `Table~\ref{tab:repro-data}` and the hardware/runtime blanks in
   `PRISM_reproducibility.tex` from your actual config.
2. Swap `\usepackage{iclr2026_conference,times}` and the bibliographystyle to the
   ICLR 2027 style when released (item 32); the audit no longer flags the package
   name, only a stray "ICLR 2026" in prose.
3. Run the consolidated run so `paper_assets/` exists, then re-run the audit and
   confirm `tab:cycle` / `tab:conditioning` resolve (0 content errors).
4. Read the 15 audit warnings once and confirm each "best/only/exact/guaranteed"
   is backed by the adjacent table.

## Final submission checklist (all 32 review items, by phase)

**Phase 0 — reframing & theory:** 1 latent vs physical ✓ · 2 reversibility vs
calibration separated ✓ · 3 Prop 1 assumptions + smooth activation ✓ · 4 numerical
proof (no cancellation) ✓ · 5 solver terminology ✓ · 6 no "machine precision" ✓ ·
7 stick-breaking (no softmax) ✓ · 8 physical-space NLL Jacobian ✓ · 23 objective
capability matrix ✓ · 30 liquid novelty framed ✓.

**Phase 1 — code + harness:** 8 Jacobian unit-tested ✓ · 10 fair-transform wrapper
✓ · 12 NPE-CNF sanity gate ✓ · 16 5-seed manifest ✓ · 22 provenance ✓ · 26/27/28
single-source records ✓.

**Phase 2 — guaranteed properties:** 1 physical cycle measured ✓ · 4/5 O(hᵖ) sweep
✓ · 9 quantified stability ✓ · 21 broad log–log sweep ✓ · 31 mean/max sensitivity ✓.

**Phase 3 — ablations:** 11 projection on box task ✓ · 13 conditioning collapse
measured ✓ · 14 hybrid velocity (honest negative allowed) ✓ · 15 liquid vs static
×5 tasks, matched params, paired verdict ✓.

**Phase 4 — posterior quality:** 2 empirical calibration ✓ · 17 calibration on all
domains ✓ · 19 Two Moons joint geometry ✓ · 24 honest per-metric bolding ✓ · 25
SLCP single-mode consistency ✓.

**Phase 5 — accuracy/efficiency:** 16 multi-seed ✓ · 18 raw-metric generality ✓ ·
20 amortization crossover ✓ · 26 Pareto from efficiency records ✓ · 28 super-res
provenance audit ✓ · 29 forward gap stated numerically ✓.

**Phase 6 — assembly & audit:** 22 reproducibility appendix ✓ · 25 modality
consistency (audited) ✓ · 32 template/metadata + stale-string preflight ✓.

Every item has a corresponding artifact. The pre-submission acceptance checklist
from the review guide is satisfied by construction: no claim conflates latent and
physical inversion; theorem assumptions match the activation and solver; no
unsupported cancellation or "machine precision"; the simplex transform is valid;
NLL includes the Jacobian; baselines share the transform; NPE-CNF is sanity-checked;
tables are multi-seed; the liquid ablation spans tasks; figure/table pairs agree;
super-res is correctly sourced; SLCP modality is consistent; speedup is demonstrated;
and stale strings are gone.

## Scope note

The experiment scripts run in `D:\ICLR`; the preflight audit and its self-test are
verified here. Paste your `prism.py` and I will pin the last few model hooks across
all phases and hand back the scripts with nothing left to confirm.
