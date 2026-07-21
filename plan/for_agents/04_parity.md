# Order 04 — parity and the safety net

> **Implementation log (2026-07-21):** Landed the 48-golden × two-dtype
> safety net, four-case TorchScript cross-check, and remaining MHR coverage;
> CUDA follow-up suite: 228 passed with no skips. Important finding: strict
> CRBA SPD is impossible with the settled rank-119/125 rotation map; see
> `04_results.md`.

Turns the order-01 goldens into enforced tests and completes MHR's test
coverage. Depends on orders 01–03.

## Ground rules (binding)

- Read `docs/contracts.md` (§Parity, §Measured baselines) first.
- **No BetterRobot modifications.** Never modify `models/MHR/assets/` or
  `references/`.
- The safety net only grows — no existing test weakened, no tolerance bumped
  to land a change.
- Write `plan/for_agents/04_results.md`: delivered, deviations, exact test
  counts, measured parity maxima per LOD.

## 1. Parity tests (`tests/parity/test_mhr_parity.py`)

- Compare vertices AND native joint positions (and orientations —
  sign-agnostic quaternion compare) against every golden case, per LOD, both
  dtypes, converting the goldens' native units.
- **Tolerance policy — measure, then pin.** The oracle is **float32-only**
  (pymomentum stores floats), so the f64-vs-f64 ≤ 1e-6 standard is not
  achievable here. Measure the worst |Δ| over the grid (ours f64 vs golden
  f32, and ours f32 vs golden f32), pin at a small multiple of the measured
  worst (expect ~1e-5 m; the 166M-MAC corrective matmul dominates the noise),
  and record the measured maxima in the results file and the pinned values in
  the test.
- **Amend `docs/contracts.md`** §Parity with the f32-oracle case honestly, and
  add the MHR rows to §Measured baselines (pinned tolerance, measured worst).
- Skip cleanly with an explicit reason when `models/MHR/converted/` weights
  are absent (goldens themselves are committed).

## 2. Secondary oracle — TorchScript

`models/MHR/assets/mhr_model.pt` (LOD1) loads in **our** venv with plain
torch — no pymomentum. Cross-check a few grid cases against it; this validates
both our implementation and the pymomentum-derived goldens independently.
Record the TorchScript-vs-golden agreement too (it bounds the oracle's own
noise floor).

## 3. Remaining coverage

- **Round-trips**: `from_classic ∘ to_classic = id` and
  `to_classic ∘ from_classic = id` (the MHR design makes both exact; f64
  ≈ 1e-12, f32 measured + pinned). Include the root ×10/units factors and a
  spine-heavy pose (the stacked-sub-node construction is what's under test).
- **Gradchecks (float32 standard)**: `identity/expression/proportions →
  vertices` and `q → vertices` including through the MLP corrective and the
  scaled-LBS hand path.
- **Batching**: arbitrary leading batch axes through spawn → with_shape →
  vertices; broadcast of shape groups vs q.
- **Dtype/device**: preservation end-to-end; CUDA cases (RTX 6000 Ada,
  pin via `CUDA_VISIBLE_DEVICES`).
- **Partial flags**: `use_correctives=False` vs the correctives-off goldens;
  `use_expression=False` vs the expression-excluded goldens.
- **Mimic coupling**: posing each twist param moves its helper rows in the
  `.model` ratios; `expand_configuration(q)` equals the transform applied to
  the native params for random classic inputs.
- **Mass**: neutral-shape total mass reported; CRBA SPD check (min eigenvalue
  > 0) on the MHR tree.

## 4. Commit the goldens

LOD1 grid ≈ 30 × ~220 KB ≈ 7 MB + smoke cases (keep LOD0 to ≤ 3 cases at
~880 KB each) — same ballpark as the committed SMPL-X grid. Ensure no
gitignore rule shadows `tests/parity/goldens/mhr_*`; update
`tools/parity/mhr/README.md` to say the grid is committed and when to
regenerate.

## Acceptance

- Full suite green (report exact counts before/after).
- MHR parity pinned per LOD, documented in contracts; measured maxima in the
  results file.
- TorchScript cross-check agrees within the same tolerance class.
- All coverage in §3 present; no existing test touched except to extend.
