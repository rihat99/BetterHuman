# Contracts

Rules every family, term and viewer layer must keep. Architecture is in `docs/architecture.md`.

## Units and frames

- Metres, seconds, kilograms, radians. Density in kg/m³.
- y-up world. Families whose native frame differs are converted by their loader. The baked BR
  model carries gravity against `FamilySpec.up_axis` (`model.gravity[:3] == (0, -9.81, 0)` for every
  shipped family), so `root_wrench` and BR dynamics need no root rotation; `terms.ground` defaults to
  `up=(0, 1, 0)` for the same reason.
- Joint order: `HumanState.joints`, `joint_poses`, `rest_joints` use the family's native skin-joint
  order. `q` and BR quantities use BR model order; `rig.native_joint_ids(spec, model)` maps between them.
- `pelvis` is the root offset from the native origin. `transl + pelvis` is the root position.

## State

- `q` = one free-flyer root `(x, y, z, qx, qy, qz, qw)` followed by one spherical
  joint (4 values) per non-root joint. `v` = 6 + 3 per spherical joint.
- SMPL: `nq = 99`, `nv = 75`. Read `identity.nq`, `identity.nv`, `identity.q_neutral`; do not hard-code.
- Move on the manifold with BR: `identity.model.integrate(q, dq)`.

## Batching

| quantity | shape |
|---|---|
| identity groups to `bake` | `(I..., n)`; omitted groups are zero |
| pose `q` | `(B..., nq)`, usually `(B, nq)` or `(B, T, nq)` |
| stream groups to `forward` | `(B..., n)`, same leading dims as `q` |
| outputs | `(broadcast(I..., B...), ...)`, e.g. `vertices() (..., V, 3)` |

`B...` broadcasts with `I...`: one identity can drive a whole clip, or a batch of identities can
take one pose each. Unknown group names raise `ValueError`.

## Differentiability

- `bake`, `forward`, `vertices`, `points`, `landmarks`, mass (`Inertia.from_mesh`) and BR
  kinematics/dynamics are differentiable in the identity values, `q` and streams.
- `HumanState` memoises `vertices()`; create a new state for new inputs, never mutate one.

## Classic conversion

- `<family>_from_classic(identity, classic) -> (q, stream)` and
  `<family>_to_classic(identity, q, stream=None) -> <Family>Classic` in `families/classic.py`.
  MHR: `mhr_from_classic(body, classic) -> (identity, q, stream)` and
  `mhr_to_classic(identity, q, stream=None)` in `families/mhr.py`.
- Hand PCA (SMPL-H, SMPL-X, MANO): `num_pca_comps` must be in `[1, 45]` or `None` (full axis-angle).
  `to_classic` raises `ValueError` when the hand pose leaves the requested PCA subspace.
- Dropped joints: a hand pose given to a body loaded with `hands=False` raises `ValueError`;
  `to_classic` on such a body returns `None` for that hand, so the round trip is closed.
- Round trips are exact up to rotation equivalence: an axis-angle with norm above π comes back
  as the same rotation with norm below π (vertices match to float precision).

## Weight-file layout

Files live under `$BETTERHUMAN_MODELS_DIR/<folder>/<file_pattern>`; an explicit `path` argument
wins. Patterns are in `families/tables.py`:

| family | folder | file pattern |
|---|---|---|
| SMPL | `smpl` | `SMPL_{gender}.npz` |
| SMPL-H | `smplh` | `smplh/{gender_lower}/model.npz` |
| SMPL-X | `smplx` | `SMPLX_{gender}.npz` |
| MANO | `mano` | `MANO_{side}.npz` |
| FLAME | `flame` | `{gender}_model.npz` |
| MHR | `MHR/converted` | `mhr_lod{0..6}.npz` (default LOD 1) |

## Parity gates (do not weaken)

| gate | test | tolerance | status |
|---|---|---|---|
| SMPL vs upstream goldens | `tests/parity/test_smpl_parity.py` | see test | pass |
| SMPL-X, 29 goldens, float32 and float64, expression streamed | `tests/parity/test_smplx_parity.py` | 1e-6 m vertices and joints | worst 5.1e-7 m |
| SMPL-X with `stream_joints=False` | same file | error in (1e-5, 2e-3) m | about 1.2 mm |
| MHR, LOD 0-6, float32 / float64 | `tests/parity/test_mhr_parity.py` | f32 `(5e-6, 1e-6, 1e-6, 1e-7)`, f64 `(2e-6, 5e-7, 5e-7, 1e-7)` | pass |
| MHR vs TorchScript `mhr_model.pt` | `tests/parity/test_mhr_torchscript.py` | vertices 2e-6 m, positions 1e-6 m, quaternions 1e-6 | pass |

Parity tests skip only when the licensed asset is absent.

## Measured baselines

`benchmarks/run.py`, float32, median ms. Ratio = ours / upstream (below 1 = ours faster).
Upstream = vendored `smplx` forward for SMPL and SMPL-X, TorchScript `mhr_model.pt` for MHR.
CPU: AMD EPYC 9654, 8 threads. CUDA: NVIDIA RTX PRO 6000 Blackwell.

| family | op | CPU B=1 | CPU B=1024 | CUDA B=1 | CUDA B=1024 |
|---|---|---:|---:|---:|---:|
| SMPL | vertices (ratio) | 1.21 (0.69) | 296 (0.95) | 1.19 (0.80) | 8.05 (0.99) |
| SMPL-X | vertices (ratio) | 3.02 (0.42) | 464 (0.74) | 1.25 (0.47) | 12.3 (0.82) |
| MHR | vertices (ratio) | 22.4 (0.59) | 1310 (0.25) | 2.50 (1.15) | 26.8 (1.02) |
| SMPL | bake with mass (B=1) | 11.2 | | 20.8 | |
| SMPL-X | bake with mass (B=1) | 23.5 | | 45.4 | |
| MHR | bake with mass (B=1) | 20.3 | | 21.8 | |
| SMPL | fk / rnea | 0.26 / 0.31 | 6.87 / 15.8 | 0.60 / 0.73 | 0.74 / 0.81 |
| SMPL-X | fk / rnea | 0.35 / 0.37 | 12.0 / 25.6 | 0.79 / 0.72 | 0.83 / 0.82 |
| MHR | fk / rnea | 0.94 / 0.55 | 41.2 / 45.2 | 1.61 / 0.77 | 2.08 / 1.55 |

Caveats:

- Inputs differ slightly. MHR "default" vertices include the expression stream; SMPL-X "default"
  passes no stream.
- SMPL bake without mass is 0.41 ms on CPU: mass integration dominates `bake`.
- Batch 8192 vertex cells (ours and upstream) were skipped where the estimate exceeded 8 GB
  (for example SMPL 10.8 GB, SMPL-X 16.5 GB, MHR TorchScript 29.0 GB). fk and rnea ran at 8192.
- Full tables: `benchmarks/results/p4_bodies_{cpu,cuda}.json`.
