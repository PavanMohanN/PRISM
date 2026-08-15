# PRISM — Phase 0 Revision Design (ICLR 2027)

**Scope of Phase 0:** paper-and-math only. No experiments, no code changes.
This document is the *gate*: once the decisions below are approved, Phases 1–6
implement them. It fixes the framing and the mathematics so that every later
result is stated defensibly.

Companion drop-in LaTeX fragments (approve alongside this doc):
- `phase0_theory.tex` — corrected Propositions 1–3 + proofs
- `phase0_constraints.tex` — support-transform math + physical-space NLL
- `phase0_capability_matrix.tex` — rewritten Table 1 (objective rows)
- `phase0_abstract_contributions.tex` — reframed abstract + contributions

---

## 1. The one change that matters most

The manuscript proves `F⁻¹(F(x̃)) = x̃` for the **latent** ODE map. It then
markets this as **physical forward–inverse consistency**. Those are different
maps: the learned physical surrogate `g_ψ(x̃) → y` and the inverse path
`y → μ_φ(y) → F⁻¹` are not inverses of each other, so latent reversibility does
**not** imply `h(g_ψ(x)) = x`. This is the highest-risk reviewer objection.

**Decision (D1):** everywhere in the paper, the guaranteed property is
*"structurally / numerically reversible latent transport,"* never *"exact
forward–inverse consistency."* Physical cycle error `‖h(g_ψ(x)) − x‖` becomes a
**separately measured empirical** quantity (added in Phase 2), not a guarantee.

---

## 2. Claim taxonomy (adopted paper-wide)

| Guaranteed by construction | Shown empirically | **Not** claimed |
|---|---|---|
| Latent-flow reversibility (Assumption 1) | Posterior calibration | Physical forward–inverse exactness |
| Feasibility via explicit support transforms | Inverse & forward accuracy | Guaranteed calibration |
| — | Robustness, cross-domain performance | OOD validity |
| — | Physical cycle error (measured) | Universal superiority over all flows |

Every checkmark in Table 1 must be reproducible from the *method definition*
alone (see `phase0_capability_matrix.tex`). No empirical property (calibration,
accuracy) appears in the capability matrix.

---

## 3. Terminology find-and-replace (apply globally before submission)

| Old phrasing (remove) | New phrasing (use) |
|---|---|
| "exact forward–inverse consistency" | "structurally reversible latent transport" |
| "cycle error equals the solver tolerance" | "reconstruction error is limited by the integration scheme, O(hᵖ)" |
| "machine-precision reversibility" | "integration-error-limited reversibility" |
| "solver tolerance" (with fixed-step solver) | "step size / global discretization error" |
| "guaranteed / architectural calibration" | "empirically calibrated on the benchmarks" |
| "consistently best-calibrated" | "competitive with the strongest baselines; strong coverage/C2ST on several tasks" |
| Section 3.4 title: "Calibrated posterior via…" | "Conditional posterior parameterization via an amortized base" |

A preflight grep list for Phase 6: `machine precision`, `exact forward-inverse`,
`exact consistency`, `solver tolerance`, `best calibrated`, `ICLR 2026`.

---

## 4. Corrected mathematics (summary; full text in fragments)

**Proposition 1 — Latent-flow reversibility.** Add Assumption 1: `g_θ` is
continuous in `t`, C¹ in `v`, and uniformly Lipschitz in `v` on the domain. Then
the time-1 map is a C¹ diffeomorphism with reverse-time inverse. *Requires a
smooth activation* — see decision D2. (Fixes guide items 3.)

**Proposition 2 — Numerical reconstruction.** Delete the "leading-order errors
cancel" claim (RK4 is not symmetric). State the standard result: for a stable
order-`p` one-step integrator with step `h`, reconstruction error is `O(hᵖ)`, up
to roundoff. (Fixes items 4, 5.)

**Proposition 3 — Stability.** Keep the Grönwall bound
`Lip(h) ≤ Lip(c)·e^{L_g}·Lip(μ_φ)`, but flag it as a *loose global* bound and
pair it with a measured empirical local sensitivity table in Phase 2. (Fixes
items 9, 31.)

**Constraint transforms.** Replace non-injective softmax with a dimension-matched
stick-breaking bijection for the simplex; state domain/codomain/inverse/Jacobian
for each transform; boundaries are open. (Fixes item 7.)

**Physical-space NLL.** `log p(x|y) = log p(c⁻¹(x)|y) + log|det J_{c⁻¹}(x)|`.
All reported NLLs are physical-space and include this Jacobian term (unit-tested
in Phase 1). (Fixes item 8.)

---

## 5. Capability matrix (Table 1) — new objective rows

Rows are architectural facts, not quality judgments:
invertible latent transform · tractable conditional density · explicit support
transform · auxiliary physical forward surrogate · amortized single-pass
sampling · continuous-time (ODE) transport. A footnote states explicitly that a
support transform *can be added to any method* — which is exactly the
fair-comparison point (guide item 10) and removes the implication that zero
violations is unique to PRISM. Full table in `phase0_capability_matrix.tex`.

---

## 6. Reframed abstract & contributions

Final wording in `phase0_abstract_contributions.tex`. Key properties of the new
abstract: it claims reversibility only for the *latent* transport, calls
calibration *empirical*, says reconstruction is *integration-error-limited*, and
drops every "only/best/exact" that the tables don't support.

---

## 7. Guide-issue coverage from Phase 0

Closed or set by Phase 0: **1, 2, 3, 4, 5, 6, 7, 8, 23, 30**, plus wording
policies that Phases 4–6 apply (**24, 25, 29, 32**).
Deferred by design to later phases (Phase 0 only *defines* them): 9 (quantify —
P2), 10–22 (experiments — P2–P5), 26/27/28 (provenance harness — P1/P6).

---

## 8. OPEN DECISIONS — need your sign-off to close the gate

These are judgment calls; my recommendation is given, but they're yours to set.

**D2 — Velocity-field activation.** Proposition 1 needs a C¹ activation; ReLU is
not C¹. *Recommend **SiLU*** (smooth, C∞, strong empirical performance); `tanh`
or `softplus` are equally valid. This is a one-line code change in Phase 1.

**D3 — Solver story.** *Recommend keeping **fixed-step RK4** and switching all
language to step-size/discretization error* (lowest effort, fully honest).
Option B: adaptive solver with reported rtol/atol. Option C (strongest
reversibility, most work): a time-symmetric integrator (implicit midpoint /
leapfrog), which would let us keep a stronger discrete-reversibility statement.

**D4 — Simplex.** No benchmark uses a simplex constraint. *Recommend **dropping
it from the main claims*** and keeping only an appendix note ("extensible via
stick-breaking") with the correct math. Alternative: keep it and add a
simplex-constrained task.

**D5 — NLL convention.** *Recommend reporting **physical-space NLL** with the
Jacobian term* (interpretable, comparable across methods). Alternative: report
transformed-coordinate NLL and state so explicitly.

**D6 — Multimodality.** For the *reframed ICLR* claim I have kept multimodal
recovery **out** of the guaranteed list and out of the abstract; the Two Moons
mode-blurring is reported honestly (Phase 4). *Optional upside:* add a
Gaussian-mixture base experiment in Phase 4 — if it recovers the modes it
becomes a headline, and the abstract gains a multimodal clause. Tell me whether
to scope that experiment in.

Once D2–D6 are set, I lock the fragments and we move to Phase 1 (code
corrections + measurement harness).
