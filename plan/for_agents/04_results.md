# Order 04 results — parity and safety net

Completed on branch `dev` on 2026-07-21, with the CRBA acceptance deviation
described below. BetterRobot, `models/MHR/assets/`, and `references/` were not
modified.

## Delivered

- A memory-bounded primary parity test covering all 48 committed-grid cases,
  all seven LODs, and both float32 and float64 (96 oracle comparisons). It
  checks vertices, native joint positions, sign-invariant xyzw quaternions,
  and accumulated scales after the recorded centimeter-to-meter conversion.
- Exact golden case-set and metadata validation. Missing converted local
  weights skip explicitly; missing or malformed committed goldens fail.
- Separate feature-off comparisons for the two correctives-disabled cases and
  `use_expression=False` against `expression_none`.
- A one-load TorchScript secondary-oracle test over four representative LOD1
  cases, including correctives off and absent expression.
- Spine-heavy two-way classic round trips, full 172-entry mimic-transform
  equality, independent shape/q batching, two float32 gradchecks, dtype/device
  assertions, a visible CUDA test, and CRBA spectrum diagnostics.
- A fixed 375×172 corrective aggregation transform replaced duplicate-index
  `index_add`; this retains parity while removing that CUDA nondeterminism
  hazard.
- The float32-oracle contract, MHR measured baselines, and committed-grid
  regeneration policy are documented.

## Primary parity measurements

Pins applied independently to every LOD are V/P/Q/S
`5e-6 m / 1e-6 m / 1e-6 / 1e-7` for float32 and
`2e-6 m / 5e-7 m / 5e-7 / 1e-7` for float64. V is vertex position, P is native
joint position, Q is the sign-invariant quaternion coefficient error, and S
is accumulated joint scale.

| Dtype | LOD | V max | P max | Q max | S max |
|---|---:|---:|---:|---:|---:|
| float32 | 0 | 2.02655792236e-6 | 3.57627868652e-7 | 2.98023223877e-7 | 0 |
| float32 | 1 | 2.92062759399e-6 | 7.59959220886e-7 | 6.55651092529e-7 | 0 |
| float32 | 2 | 1.96695327759e-6 | 3.57627868652e-7 | 2.98023223877e-7 | 0 |
| float32 | 3 | 2.26497650146e-6 | 7.15255737305e-7 | 4.76837158203e-7 | 0 |
| float32 | 4 | 2.02655792236e-6 | 3.57627868652e-7 | 3.57627868652e-7 | 0 |
| float32 | 5 | 1.78813934326e-6 | 4.76837158203e-7 | 2.98023223877e-7 | 0 |
| float32 | 6 | 1.78813934326e-6 | 2.68220901489e-7 | 3.57627868652e-7 | 0 |
| float64 | 0 | 6.79632086609e-7 | 1.83319529667e-7 | 1.31397505598e-7 | 2.48013358739e-8 |
| float64 | 1 | 8.48869801895e-7 | 2.21570063519e-7 | 1.92913394426e-7 | 4.66293661461e-8 |
| float64 | 2 | 5.58113464333e-7 | 1.31546895821e-7 | 1.02205976682e-7 | 5.79166241632e-8 |
| float64 | 3 | 5.61791043685e-7 | 9.95070510434e-8 | 1.08196531512e-7 | 2.90327222263e-8 |
| float64 | 4 | 5.53013310478e-7 | 2.06189637142e-7 | 1.33384776585e-7 | 2.55662947479e-8 |
| float64 | 5 | 4.46300694623e-7 | 1.22673170511e-7 | 1.22209555822e-7 | 2.29538588137e-8 |
| float64 | 6 | 5.93537564431e-7 | 1.20235824852e-7 | 1.82994454256e-7 | 2.92798483237e-8 |

The worst primary result is the LOD1 float32 vertex error,
`2.92062759399e-6 m`. The 48 archives occupy 8.8215 MiB and no gitignore rule
matches them.

## Secondary oracle

The complete 30-case read-only TorchScript scan measured a worst vertex error
of `0.0001220703125 cm` (`1.220703125e-6 m`, `identity_00`); skeleton position,
quaternion, and scale were bit-exact. The enforced four-case test pins vertices
at `2e-6 m` and the skeleton at the primary float32 tolerance class.

## Verification

- Before Order 04: **199 passed, 4 skipped**.
- MHR body coverage: **19 passed, 1 skipped** (CUDA unavailable).
- Primary parity: **18 passed**.
- TorchScript secondary oracle: **1 passed**.
- Full suite: **223 passed, 5 skipped, 20 warnings** in `117.21 s`.
- The new MHR files pass `ruff check` and `ruff format --check`; full
  `ruff check .` and `git diff --check` pass.

## Deviations and findings

1. **Strict MHR CRBA SPD is mathematically impossible under the settled q
   contract.** The 375×125 physical rotation transform has rank 119: the 18
   public spine coordinates drive 12 commuting same-axis rotations, leaving
   six structural null directions. Even perfect physical inertia placement
   therefore bounds CRBA rank at `6 + 119 = 125 < nv=131`. The current
   19-group placement measures rank 50/131, 72 zero diagonal entries, and
   `λmin=-7.7016e-15` (roundoff) at neutral LOD1 float64. The test records PSD
   and these diagnostics; it does not claim or fake SPD. Redefining independent
   coordinates or adding nonphysical armature requires an owner decision.
2. **CUDA could not be executed.** PyTorch reports no available CUDA devices
   in this environment, despite the plan's RTX 6000 Ada note. The CUDA forward
   and backward test is present and skips visibly.
3. **Float32 gradcheck needs deterministic kernels.** Under the default CPU
   kernel selection, repeated analytical backward passes differ while still
   matching finite differences. The two tests save, enable, and restore
   PyTorch deterministic-algorithm state; no nondeterminism tolerance was
   added. Removing duplicate-index aggregation did not eliminate the remaining
   BLAS reduction variability.
## Primary commands

```bash
.venv/bin/pytest tests/bodies/test_mhr.py -q
.venv/bin/pytest tests/parity/test_mhr_parity.py -q
.venv/bin/pytest tests/parity/test_mhr_torchscript.py -q
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
git diff --check
```
