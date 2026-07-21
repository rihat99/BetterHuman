# Order 01 results — MHR converter and parity goldens

Completed on branch `dev` on 2026-07-21. This order adds tools and oracle data
only: no BetterHuman library code and no BetterRobot code were modified.

## Delivered

- `tools/parity/mhr/environment.yml`: pinned Python 3.12.12, NumPy 2.3.5,
  PyTorch 2.8.0 CPU, PyMomentum 0.1.90 CPU, and trimesh 4.9.0.
- `tools/parity/mhr/README.md`: exact environment, conversion, and golden
  regeneration commands.
- `tools/convert/mhr/convert_mhr.py`: an all-LOD converter with source/schema,
  dimensions, units, topology, transform, skinning, bind, corrective, limit,
  tpose, segmentation, and cross-LOD assertions. It writes one lazy-loadable,
  self-contained, uncompressed NPZ per LOD under the gitignored
  `models/MHR/converted/` directory.
- `tools/parity/mhr/dump_mhr.py`: deterministic CPU/float32 reference runner.
  It writes every input, flag, vertex output, and `(127, 8)` skeleton state,
  and performs a bitwise reload/re-run check.
- 48 generated goldens under `tests/parity/goldens/mhr_lod{0..6}/`: exactly 30
  LOD1 cases and 3 smoke cases for each other LOD. LOD1 covers zero with
  correctives on/off, tpose, five random-pose cases, identity, expression and
  expression-`None`, proportions, both hand-scale limits, root motion, and
  seven combined cases. One nonzero combined case also disables correctives.

## Measured outputs

All archive sizes below are raw bytes from `stat`; MiB is the converter's
base-2 display.

| LOD | Vertices | Faces | Converted bytes | MiB | Golden cases | Golden bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 73,639 | 147,274 | 2,769,297,616 | 2,641.01 | 3 | 2,376,431 |
| 1 | 18,439 | 36,874 | 695,985,472 | 663.74 | 30 | 6,159,548 |
| 2 | 10,661 | 21,318 | 403,843,792 | 385.14 | 3 | 353,563 |
| 3 | 4,899 | 9,794 | 187,423,072 | 178.74 | 3 | 173,741 |
| 4 | 2,461 | 4,918 | 95,851,792 | 91.41 | 3 | 98,669 |
| 5 | 971 | 1,938 | 39,887,392 | 38.04 | 3 | 49,572 |
| 6 | 595 | 1,186 | 25,764,832 | 24.57 | 3 | 38,522 |

Golden total: **48 files, 9,250,046 bytes**.

Transform measurements, all asserted by the converter:

- Dense augmented shape: `(889, 321)`; columns `204:321` are exactly zero.
- `.model` assignment statements `(translation, rotation, scale)`:
  **(191, 157, 2) = 350**.
- Sparse matrix entries: **(191, 175, 2) = 368**.
- Unique driven rows: **(136, 157, 2) = 295**, across **113 joints**.
- Multi-source rows: **(55, 12, 0)**. The 12 rotation rows are the documented
  spine rows; translation also has 55 multi-source rows.
- Model-parameter column types: **(74, 128, 2)**. Excluding the three root
  translation and three root rotation parameters gives 71 proportions and 125
  pose parameters; the two scale parameters are `scale_l_hands` and
  `scale_r_hands`.
- Parsed model metadata: 8 parameter sets, 198 unique finite non-root limits,
  and 19 authored tpose entries. Both elbow-bend tpose values are `-0.5`.

Other raw validation metrics were identical for every LOD:

- Maximum pre-rotation norm error before normalization: `5.96046448e-08`.
- Maximum skin-weight row-sum error: `1.19209290e-07`; no row was
  renormalized.
- FBX mesh versus blendshape base maximum: `0 cm`.
- FBX bind versus zero-parameter FK maximum: `4.57763672e-05 cm`.
- FBX bind/inverse product maximum error: `1.52587891e-05` in the native-cm
  calculation.
- Corrective activation first-layer nnz: 53,136; output shape is
  `(3000, V*3)` for each LOD.
- FBX globals: up `+y`, front `+z`, coordinate/right `+x`, unit scale 1 cm.

## FK source verification

The executable C++ path in
`references/parametric_body/momentum/momentum/character/joint_state.cpp` and
the tensor path in
`references/parametric_body/momentum/pymomentum/tensor_momentum/tensor_skeleton_state.cpp`
agree. For joint `i`:

```text
world_i = world_parent
        · T(translationOffset_i + [tx, ty, tz])
        · Rpre_i · Rz(rz) · Ry(ry) · Rx(rx)
        · S(exp2(sc))
```

The fixed translation offset and parameter translation are summed before local
rotation, in the parent frame. The scale parameter is base-2 logarithmic. A
comment in `joint_state.cpp` says the resulting product is `Rx·Ry·Rz`, but the
actual descending `2,1,0` loop and the independent tensor implementation both
execute `Rz·Ry·Rx`; the converter records the executable order.

## Verification

- Final tools environment: NumPy 2.3.5; PyTorch 2.8.0
  `cpu_mkl_py312_hb1fc07b_102` (`torch.version.cuda is None`); PyMomentum
  0.1.90 `cpu_py312_h3e3b27a_0`; trimesh 4.9.0; Python 3.12.12.
- All seven conversions completed in one invocation, including exact
  cross-LOD skeleton, bind, and transform equality checks.
- Every golden was reloaded with `allow_pickle=False` and checked for its LOD,
  shapes, native-unit tag, and float32 input/output dtype.
- LOD6 `fully_random.npz` was re-evaluated after reload and matched vertices
  and skeleton state bit-for-bit.
- `ruff check`, `ruff format --check`, and `py_compile` pass for both Python
  tools.
- Full BetterHuman suite: **163 passed, 4 skipped, 0 failed**, 6 warnings.
- All 16 MHR asset SHA-256 values match the pre-conversion manifest. Nested
  `git status` is clean for both MHR and Momentum references, and no generated
  bytecode remains under the MHR reference tree.

Exact primary commands:

```bash
conda env update --name betterhuman-mhr-parity \
  --file tools/parity/mhr/environment.yml --prune
conda run --name betterhuman-mhr-parity python \
  tools/convert/mhr/convert_mhr.py \
  --assets-dir models/MHR/assets --output-dir models/MHR/converted
conda run --name betterhuman-mhr-parity python \
  tools/parity/mhr/dump_mhr.py \
  --assets-dir models/MHR/assets \
  --reference-dir references/parametric_body/MHR \
  --converted-dir models/MHR/converted \
  --output-dir tests/parity/goldens
.venv/bin/ruff check tools/convert/mhr/convert_mhr.py \
  tools/parity/mhr/dump_mhr.py
.venv/bin/ruff format --check tools/convert/mhr/convert_mhr.py \
  tools/parity/mhr/dump_mhr.py
.venv/bin/pytest tests/ -q
```

## Deviations and findings

1. **The plan's “350 transform rows” assertion is not the stored transform.**
   The number 350 is the count of textual assignment statements. Duplicate
   translation left-hand sides collapse, while multi-driver rotation
   expressions expand, producing 295 unique matrix rows and 368 nonzero
   entries. The converter records and asserts all three measurements instead
   of relabeling one as another.
2. **Native joint indexing differs from later plan wording.** Joint 0 is the
   fixed `body_world`; joint 1 is `root` and owns the six root channels.
   Correctives deliberately use joints `[2:127]`. BetterRobot universe 0 plus
   explicit free-flyer 1 already matches this layout; Order 03 must not make
   native joint 0 the free-flyer.
3. **The static FBX inverse bind is authoritative.** Recomputing it from
   zero-parameter FK changes the bind by `4.57763672e-05 cm`; recomputing it
   after a proportion bake would also cancel the intended proportional
   deformation. The archive preserves the FBX bind and inverse bind.
4. **PyMomentum does not expose original FBX axis globals or `.model` pose
   constraints.** The converter therefore parses those two read-only source
   formats directly, while comparing parsed limits against PyMomentum.
5. **Pixi was unavailable, so the documented conda fallback was used.** The
   first solve also showed trimesh 4.9.0 was unavailable from conda-forge; it is
   the sole pure-Python pip entry. PyMomentum remains conda-only. The final spec
   explicitly forces CPU PyTorch to avoid solver selection of a CUDA build.
6. **Reference imports initially emitted four ignored `.pyc` files.** They were
   removed, the empty cache directory was removed, and the dumper now sets
   `sys.dont_write_bytecode` before importing the reference package. A repeat
   oracle run left the reference tree unchanged.
7. **Large corrective storage must stay lazy in Order 02.** LOD0 is a 2.58 GiB
   monolithic NPZ primarily because its `(3000, V*3)` output layer is embedded
   for self-containment. The loader can satisfy `use_correctives=False` only by
   retaining `np.load` laziness and never indexing corrective arrays in that
   branch.

## BetterRobot gaps for later orders

No BetterRobot change was made. The verified constraints to carry forward are:

- BetterRobot poses are SE(3) `[translation, xyzw]` and do not carry MHR's
  accumulated `exp2(sc)` scale. BetterHuman must keep scale separately for
  scaled offsets and LBS.
- Build-time joint/frame origins are not automatically quaternion-normalized,
  and `preserve_joint_order=True` repairs ordering rather than asserting native
  order. The converter/BetterHuman checks are therefore required.
- Frames cannot own inertia; any undriven segmented part with nonzero mass must
  be represented by a fixed-joint body.
- Multi-source mimic declarations are not directly supported, but this is not
  an MHR blocker: same-axis stacked nodes can represent the 12 multi-driver
  rotation rows, and the remaining fan-outs are single-source.
- Full-joint Torch/Warp FK has serial per-joint work. This is a benchmark risk,
  not yet a measured failure; Order 05 must measure it before proposing a
  BetterRobot change.

Capabilities already present and not to be duplicated include the explicit
universe-to-free-flyer builder, fixed joints and frames, affine mimic expansion,
batched differentiable placements/inertias, and zero-mass intermediate bodies.
