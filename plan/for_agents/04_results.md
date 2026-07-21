# Order 04 results

Completed on `dev` on 2026-07-20 after Order 03.

## Delivered

- A model-agnostic load-time reducer for terminal joint subtrees. It rejects
  root and non-terminal drops, transfers each dropped joint's weights to its
  nearest surviving ancestor, renormalizes rows, preserves dense/CSR joint
  regressors, and slices the exact non-root pose-corrective blocks.
- Named shape-basis slicing with strict `(V, 3, S)` shape, integer-width, and
  source-range validation. Reduction likewise validates all tensor ranks,
  joint/vertex dimensions, pose width, and positive finite weight-row sums.
- `SMPLX` and `SMPLXClassic` with native jaw/eyes-before-hands ordering,
  separate shape and expression groups, full and PCA hand conversion, hand
  means, `flat_hand_mean`, batching, dtype/device placement, and registry
  loading.
- `use_face` and `use_hands` constructor flags backed by the same generic
  reducer. Shared SMPL-family loading/building helpers live in
  `bodies/smpl_family.py`; no model fork or duplicate pipeline was added.
- A deterministic upstream-smplx dump tool and documented regeneration
  command. The 29 SMPL-X archives are ignored, generated on demand, and were
  removed after verification rather than committed.

## Dropped joint sets

`use_face=False` drops exactly:

- `jaw`, `left_eye_smplhf`, `right_eye_smplhf`

`use_hands=False` drops exactly:

- Left: `left_index1`, `left_index2`, `left_index3`, `left_middle1`,
  `left_middle2`, `left_middle3`, `left_pinky1`, `left_pinky2`,
  `left_pinky3`, `left_ring1`, `left_ring2`, `left_ring3`, `left_thumb1`,
  `left_thumb2`, `left_thumb3`.
- Right: `right_index1`, `right_index2`, `right_index3`, `right_middle1`,
  `right_middle2`, `right_middle3`, `right_pinky1`, `right_pinky2`,
  `right_pinky3`, `right_ring1`, `right_ring2`, `right_ring3`,
  `right_thumb1`, `right_thumb2`, `right_thumb3`.

The resulting native joint counts are 25 without hands, 52 without face,
and 22 without either.

## Full-model parity

Errors are maximum absolute metres over all vertices or native LBS joints.

| Case | float64 vertices | float64 joints | float32 vertices | float32 joints |
|---|---:|---:|---:|---:|
| `beta_minus_two` | 8.863e-09 | 2.303e-09 | 2.980e-07 | 2.924e-07 |
| `beta_plus_two` | 8.863e-09 | 2.303e-09 | 2.496e-07 | 2.454e-07 |
| `expression_minus_two` | 5.817e-10 | 1.238e-10 | 2.794e-07 | 2.384e-07 |
| `expression_plus_two` | 5.817e-10 | 1.238e-10 | 2.794e-07 | 2.384e-07 |
| `face_all` | 2.114e-10 | 1.238e-10 | 2.645e-07 | 2.384e-07 |
| `full_both` | 6.528e-10 | 4.076e-10 | 3.576e-07 | 2.384e-07 |
| `full_left` | 3.751e-10 | 1.878e-10 | 2.868e-07 | 2.384e-07 |
| `full_right` | 5.860e-10 | 3.743e-10 | 2.384e-07 | 2.980e-07 |
| `jaw` | 2.115e-10 | 1.238e-10 | 2.608e-07 | 2.384e-07 |
| `left_eye` | 2.115e-10 | 1.238e-10 | 2.794e-07 | 2.384e-07 |
| `pca_both` | 7.595e-10 | 4.111e-10 | 4.172e-07 | 2.384e-07 |
| `pca_left` | 4.289e-10 | 2.667e-10 | 2.980e-07 | 2.384e-07 |
| `pca_right` | 2.115e-10 | 1.238e-10 | 2.813e-07 | 2.384e-07 |
| `pca_twelve` | 3.773e-10 | 2.391e-10 | 3.204e-07 | 2.384e-07 |
| `random_00` | 2.507e-08 | 2.881e-09 | 4.768e-07 | 4.768e-07 |
| `random_01` | 3.006e-08 | 4.025e-09 | 5.364e-07 | 4.768e-07 |
| `random_02` | 1.031e-08 | **8.653926908408494e-09** | 5.960e-07 | 4.172e-07 |
| `random_03` | 1.302e-08 | 4.663e-09 | **6.556510925292969e-07** | **5.066394805908203e-07** |
| `random_04` | 1.979e-08 | 4.796e-09 | 4.470e-07 | 2.980e-07 |
| `random_05` | 1.438e-08 | 4.392e-09 | 4.470e-07 | 2.980e-07 |
| `random_06` | 1.119e-08 | 4.423e-09 | 3.576e-07 | 2.384e-07 |
| `random_07` | 2.119e-08 | 8.087e-09 | 4.768e-07 | 3.576e-07 |
| `reduced_shape_expression` | 3.970e-09 | 1.162e-09 | 2.775e-07 | 2.384e-07 |
| `right_eye` | 2.115e-10 | 1.238e-10 | 2.794e-07 | 2.384e-07 |
| `root_wart` | **5.590597473847936e-08** | 2.075e-09 | 3.576e-07 | 1.937e-07 |
| `zero_full_flat` | 4.441e-16 | 1.110e-16 | 1.788e-07 | 2.980e-07 |
| `zero_full_mean` | 2.115e-10 | 1.238e-10 | 2.794e-07 | 2.384e-07 |
| `zero_pca_flat` | 4.441e-16 | 1.110e-16 | 1.788e-07 | 2.980e-07 |
| `zero_pca_mean` | 2.115e-10 | 1.238e-10 | 2.794e-07 | 2.384e-07 |

The required float64 threshold is `1e-6`. Float32 is also pinned at `1e-6`:
the measured worst case leaves approximately 1.53x headroom. The reduced
five-beta/three-expression case directly verifies parameter-basis slicing
against the oracle.

## Partial-model property

These are maximum float32 vertex errors against a full body whose dropped
joint rotations are set to identity. For natural hand means, the full-model
classic deltas are `-hands_mean`; this is the actual rest pose, not a zero
classic delta.

| `flat_hand_mean` | `use_hands` | `use_face` | Maximum error |
|---|---|---|---:|
| `False` | `False` | `True` | 2.384185791015625e-07 |
| `False` | `True` | `False` | 3.5762786865234375e-07 |
| `False` | `False` | `False` | 3.5762786865234375e-07 |
| `True` | `False` | `True` | 2.384185791015625e-07 |
| `True` | `True` | `False` | 3.5762786865234375e-07 |
| `True` | `False` | `False` | 3.5762786865234375e-07 |

The pinned `5e-7` tolerance is the next single-significant-digit ceiling
above the measured worst case, retaining approximately 1.40x headroom for
the expected float32 summation-order difference.

## Differentiability, batching, and CUDA

- Float32 expression gradcheck used vertices `[0, 2618, 5237, 7856,
  10474]` and passed in 0.45 s.
- Shape/expression and configuration batching passed, including unbatched
  configuration broadcasting across a shape batch.
- Full and PCA hand round trips passed with natural and flat means. Because
  the shipped PCA component rows are scaled rather than orthonormal,
  `to_classic` correctly uses a pseudoinverse instead of a transpose.
- The CUDA FK plus vertices smoke passed on pinned RTX 6000 Ada GPU 2 in
  20.51 s. The ordinary sandbox gate reports the two SMPL/SMPL-X
  device-conditional tests as skipped.

## Verification

- `uv lock --check`: 76 root-project packages, current.
- `uv lock --check --project tools/parity/smplx`: 38 oracle packages,
  current.
- Full local oracle grid: 29 archives, 7,046,185 bytes; all 58
  dtype/case parity checks passed.
- `uv run pytest tests -q -rs`: **143 passed, 2 skipped, 6 warnings in
  43.19 s** with the full on-demand grid present. Warnings are the three
  required float32-gradcheck notices, PyTorch's CSR-beta notice, and the two
  previously recorded legacy SciPy-pickle deprecations.
- Targeted SMPL-X CUDA test: **1 passed in 20.51 s**.
- `uv run ruff check src tests tools/parity/smplx/dump_smpl.py
  tools/parity/smplx/dump_smplx.py`: passed.
- Independent review found no remaining basis-validation, reduction,
  hand-mean, converter, parity, or acceptance blocker.
- The combined `models/` and `references/` path/size/mtime manifest remained
  `1e12ec801fada88a15763f372770d3e065e4d1015a680a3f0e50917cb52e2258`.
- BetterRobot remained clean on `dev`; no BR-side changes were required.

## Deviations and findings

- **Plan deviations:** none.
- Upstream SMPL-X hand PCA rows are not orthonormal in the shipped fixture:
  the observed maximum error in `C C^T` versus identity is approximately
  `0.9987`. Expansion still uses the native component matrix, while inversion
  uses its pseudoinverse so classic coefficients round-trip correctly.
- The first reduction tests contained an internally inconsistent synthetic
  fixture (`J_regressor`, weights, and posedirs disagreed on `V`). Strict
  validation exposed it; the fixture was corrected rather than relaxing the
  production invariant.
