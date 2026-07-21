# Order 03 results

Completed on `dev` on 2026-07-20 after Order 02.

## Delivered

- Functionally immutable `BodyStructure`, `BodyValues`, and `ParametricBody`
  composition. Named shape groups bake pelvis-centered vertices, regressed
  rest joints, and batch-bearing BetterRobot joint placements.
- Shared batched tensor stages for linear shape blending, dense/CSR joint
  regression, normalized non-root `(R - I)` pose features, linear
  correctives, native-joint skinning transforms, and padded `[V, K]` LBS.
- Sign-preserving axis-angle/quaternion conversion and shape-dependent
  SMPL-family root-wart conversion. Negative-scalar unit quaternions map to
  the corresponding `(pi, 2*pi]` axis-angle branch, so converter round trips
  preserve the actual BetterRobot `q` tensor rather than only its rotation.
- `SMPL` with explicit/environment model resolution, dtype/device conversion,
  loaded kintree parents, four-influence padded skinning, pelvis-centered rest
  data, preserved native joint order, and typed `SMPLClassic` conversion.
- Decorator-backed `better_human.load("smpl", ...)` plus the small public API
  exports `SMPL`, `ParametricBody`, and `load`.
- An isolated 38-package uv lock pinned to upstream `smplx==0.1.28`, a
  deterministic dump script, and 13 committed neutral-gender golden archives.
  The archives contain 78 float64 input/output arrays and total 2,071,126
  bytes.

## Parity

Errors are maximum absolute metres over all vertices or native LBS joints.

| Case | float64 vertices | float64 joints | float32 vertices | float32 joints |
|---|---:|---:|---:|---:|
| `beta_minus_two` | 7.793e-08 | 4.736e-08 | 3.576e-07 | 1.490e-07 |
| `beta_plus_two` | 9.907e-08 | 5.048e-08 | 4.768e-07 | 2.384e-07 |
| `beta_zero` | 8.844e-08 | 4.892e-08 | 2.980e-07 | 1.788e-07 |
| `random_00` | 7.590e-08 | 4.014e-08 | 6.258e-07 | 4.768e-07 |
| `random_01` | 9.695e-08 | 4.749e-08 | 4.768e-07 | 2.980e-07 |
| `random_02` | 8.350e-08 | 4.397e-08 | 4.172e-07 | 2.198e-07 |
| `random_03` | 9.807e-08 | 4.982e-08 | 3.874e-07 | 2.682e-07 |
| `random_04` | 9.440e-08 | 5.156e-08 | 2.384e-07 | 1.192e-07 |
| `random_05` | 9.428e-08 | 5.401e-08 | 5.960e-07 | 2.980e-07 |
| `random_06` | **1.0412031070661243e-07** | 4.752e-08 | **7.078051567077637e-07** | 5.290e-07 |
| `random_07` | 7.633e-08 | 3.881e-08 | 2.980e-07 | 1.788e-07 |
| `root_wart` | 5.805e-08 | 3.492e-08 | 3.576e-07 | 1.192e-07 |
| `zero` | 8.915e-08 | **5.780450140768778e-08** | 1.192e-07 | 5.960e-08 |

The pinned float64 threshold is the required `1e-6`. The float32 threshold is
also pinned at `1e-6`: its measured worst case was
`7.078051567077637e-07`, so the next clean decimal ceiling retains roughly
1.41× headroom without hiding millimetre-scale regressions. The float64 floor
is consistent with upstream smplx 0.1.28 adding `1e-8` inside its Rodrigues
conversion.

The first dump used the local upstream checkout at commit
`1265df7ba545e8b00f72e7c557c766e15c71632f` (version 0.1.28). The isolated
locked PyPI environment was then materialized and regenerated the grid; all
78 arrays were exactly equal. The temporary environment and duplicate dump
were removed afterward.

## Differentiability, batching, and CUDA

- Float32 gradcheck used vertices `[0, 1723, 3445, 5167, 6889]`.
- `betas -> selected vertices`: passed in 0.57 s.
- `q -> selected vertices`: passed in 0.15 s.
- Shape batches, same-batch configurations, and unbatched-`q` broadcasting
  passed. A regression also proves scaled joint quaternions give identical FK
  and correctives after normalization.
- The CUDA FK plus vertices smoke passed on pinned RTX 6000 Ada GPU 1 in
  16.98 s. The default sandbox cannot see the driver, so the ordinary full
  suite reports that one device-conditional test as skipped; the escalated,
  pinned run passed separately.

## Verification

- `uv lock --check`: 76 root-project packages, current.
- `uv lock --check --project tools/parity/smplx`: 38 oracle packages,
  current.
- `uv sync --locked --offline --dry-run`: no changes.
- `uv run pytest tests -q -rs`: **62 passed, 1 skipped, 5 warnings in
  37.35 s**. Warnings are the required float32-gradcheck notice, PyTorch's
  CSR-beta notice, and the two legacy SciPy pickle deprecations already
  recorded in Order 02.
- Targeted CUDA test: **1 passed in 16.98 s**.
- `uv run ruff check src tests tools/parity/smplx/dump_smpl.py`: passed.
- Independent review found no remaining converter, quaternion-normalization,
  parity-collection, batching, or acceptance blocker after its regressions
  were added.
- The combined `models/` and `references/` path/size/mtime manifest remained
  `1e12ec801fada88a15763f372770d3e065e4d1015a680a3f0e50917cb52e2258`.
- BetterRobot remained clean on `dev`; no BR-side changes were required.

## Deviations and findings

- **Plan deviations:** none.
- BetterRobot intentionally exposes the free-flyer joint name as `root` while
  its associated body name is `pelvis`. Alignment tests therefore assert both
  `robot.joint_names[1:] == ("root", *native_names[1:])` and
  `robot.body_names[1:] == native_names`; the required `BR j+1 <-> native j`
  index binding is exact.
- Importing the local oracle initially created six bytecode files under the
  protected `references/` tree. They were immediately removed, and the
  protected manifest returned exactly to its pre-order digest above.
