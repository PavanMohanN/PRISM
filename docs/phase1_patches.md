# PRISM Phase 1 — in-place patches to existing files

Two changes must be made inside files that already exist in `D:\ICLR`. They are
small and surgical. Everything else in Phase 1 is a new drop-in module.

---

## Patch 1 — Smooth activation in the liquid velocity field (D2, guide item 3)

Proposition 1 (Phase 0) requires a C¹ velocity field, so the activation in the
liquid MLP `s(v,t) = σ(MLP([v,t]))` must be smooth. ReLU is not C¹.

**File:** `prism/flows/liquid_ode.py` (class `LiquidVelocity`).

Find the MLP construction — it will look something like:

```python
# BEFORE
layers += [nn.Linear(h_prev, h), nn.ReLU()]
# ...or a functional call...
s = F.relu(self.fc(torch.cat([v, t], dim=-1)))
```

Replace every hidden-layer `ReLU` with `SiLU`:

```python
# AFTER
layers += [nn.Linear(h_prev, h), nn.SiLU()]
# ...or functional...
s = F.silu(self.fc(torch.cat([v, t], dim=-1)))
```

Notes
- Do **not** change the final `sigmoid` that produces the gate `s ∈ (0,1)`; that
  is the LTC gate, not a hidden activation.
- Apply the same ReLU→SiLU swap in the amortization network
  (`μ_φ, σ_φ`) and the forward head for consistency; it is not required by the
  theorem but keeps the model uniformly smooth.
- `tanh` or `softplus` are equally valid substitutes if you prefer (D2).

---

## Patch 2 — Physical-space NLL includes the transform Jacobian (D5, guide item 8)

Equation (cond) defines a density on the **unconstrained** coordinate `x̃`. Any
NLL reported for the **physical** parameter `x = c(x̃)` must add
`log|det J_{c⁻¹}(x)|`.

**File:** `prism/models/prism.py` (wherever the conditional log-density / NLL is
computed for reporting or scoring — e.g. `score`, `predict_posterior` logprob,
or a `log_prob` helper).

Wire in the new transform object (from `prism/constraints/transforms.py`). During
`fit`, construct `self.transform_ = make_transform(kind, dim, **kw)` from the
task's constraint spec (replacing the old `projections` object where it was used
for density).

Then, where physical-space NLL is returned:

```python
# BEFORE  (transformed-coordinate NLL only)
log_p = base.log_prob(z, mu_y, sigma_y) + flow_logdet          # = log p(x̃ | y)
nll = -log_p

# AFTER   (physical-space NLL, guide item 8)
log_p_tilde = base.log_prob(z, mu_y, sigma_y) + flow_logdet     # log p(x̃ | y)
log_p_phys  = log_p_tilde + self.transform_.log_abs_det_inverse(x)  # + log|det J_{c^-1}(x)|
nll = -log_p_phys
```

- `x` here is the **physical** parameter; `x̃ = self.transform_.inverse(x)`.
- If you deliberately keep transformed-coordinate NLL for some diagnostic, label
  it as such in the table and do **not** compare it against physical-space NLLs
  from other methods.
- The correctness of `log_abs_det_inverse` is proven by the normalization test
  (`tests/test_transforms.py`) — pushforward densities integrate to 1.0.

---

## Patch 3 — Route experiments through the provenance manifest (items 16/26/27/28)

**File:** `experiments/_common.py` and each `phaseN_*.py`.

Replace ad-hoc `save_results(...)` dictionaries with `ResultManifest.run_seeds`
(from `experiments/provenance.py`), looping seeds `[0,1,2,3,4]`. Tables and
figures must then read values via `manifest.agg(...)` / `manifest.fmt(...)`
instead of re-deriving them, so no value can appear twice with two numbers.

---

## Patch 4 — Solver-story terminology (D3, guide items 5, 6)

**File:** `experiments/_common.py` / config + any place that says "tolerance".

Keep the fixed-step RK4 integrator but rename the swept quantity to
**solver steps / step size** everywhere (config keys, figure axes, captions).
Remove `rtol`/`atol` language unless you switch to an adaptive solver. This
aligns Appendix A, Proposition 2, the reversibility figure, and the abstract.
