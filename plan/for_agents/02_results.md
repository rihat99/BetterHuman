# Order 02 results — BodyConfig + landmark restoration + smpl_like re-home

Completed on `dev` (both working trees) on 2026-07-21. Two coupled halves landed
together so **both repos end green**: (A) BetterHuman gained a per-model
`BodyConfig` and stopped importing SMPL topology from BetterRobot; (B)
BetterRobot's SMPL/human specifics left its shipped API and its suite moved to a
generic, neutrally-named fixture.

## Test counts (as measured)

| Repo | Command | Result |
|---|---|---|
| BetterHuman | `CUDA_VISIBLE_DEVICES=2 uv run pytest tests/ -q` | **109 passed, 2 skipped** |
| BetterRobot | `CUDA_VISIBLE_DEVICES=2 uv run pytest tests/ -q` | **1583 passed, 0 skipped, 0 failed** |

- BetterHuman baseline before this order: 106 passed / 2 skipped. Net +3 = the
  three new landmark tests (2 SMPL, 1 SMPL-X). The 2 skips are pre-existing and
  visible: `tests/parity/test_smplx_parity.py` — "SMPL-X goldens are absent".
- BetterRobot: CUDA was active (GPU 2); the `tests/warp/*_cuda.py` float32
  gradchecks executed (not skipped). All parity/round-trip/gradcheck tolerances
  are unchanged — this order is a pure refactor + re-home, no numerics touched.

## Part A — BetterHuman

### `BodyConfig` (new type + two data modules)

`class BodyConfig` (frozen dataclass) lives in
`src/better_human/bodies/config_body.py` (49 LOC). Per-model instances:
`SMPL_CONFIG` in `bodies/config_smpl.py` (89 LOC), `SMPLX_CONFIG` in
`bodies/config_smplx.py` (124 LOC). Naming is category-first (`config_*`).

Fields (every one has a real consumer, except the three order-03 asset names —
see Findings #3):

| Field | Type | Consumer |
|---|---|---|
| `joint_names` | `tuple[str,...]` | loader count check, `BodyStructure`, robot build |
| `frame_names` | `tuple[str,...]` | landmark metadata (restored) |
| `frame_vertex_ids` | `tuple[int,...]` | `BodyStructure.landmarks` → `landmark_positions()` |
| `unit` | `str` | `BodyStructure.unit` (contracts-mandated) |
| `up_axis` | `str` | `BodyStructure.up_axis` |
| `forward_axis` | `str` | `BodyStructure.forward_axis` |
| `native_pose_format` | `str` | `BodyStructure.native_pose_format` |
| `segmentation_vertex_file` | `str` | order 03 (recorded now) |
| `density_table` | `str` | order 03 (recorded now) |
| `segmentation_closed_file` | `str \| None = None` | order 03 (recorded now) |
| `body_joints` | `tuple[int,...] = ()` | SMPL-X reduction + `from/to_classic` |
| `face_joints` | `tuple[int,...] = ()` | SMPL-X reduction + `from/to_classic` |
| `left_hand_joints` | `tuple[int,...] = ()` | SMPL-X `from/to_classic` |
| `right_hand_joints` | `tuple[int,...] = ()` | SMPL-X `from/to_classic` |

Parents are **not** in config — still derived from the npz `kintree_table` via
`parents_from_kintree`. SMPL leaves the four reduction-group tuples empty (`()`,
it has no reducible subtrees). SMPL-X `segmentation_closed_file` is `None` (no
authored closed-3D asset exists in git history — only `smpl_3d_segmentation.npy`).
Landmark `frame_names`/`frame_vertex_ids` were restored from
`git show main:src/better_human/smpl/config/{smpl,smplx}.json`.

### Body refactor + landmark consumer

- `bodies/smpl.py`: removed `from better_robot...smpl_like import JOINT_NAMES`;
  reads `SMPL_CONFIG`. LOC 174 → 179 (+5; the multi-line `landmarks=` dict).
- `bodies/smplx.py`: removed the 57-line inline `SMPLX_JOINT_NAMES` tuple and the
  `BODY_/FACE_/LEFT_HAND_/RIGHT_HAND_/HAND_JOINTS` constants; reads `SMPLX_CONFIG`.
  Dropped `SMPLX_JOINT_NAMES` from `__all__` (unused elsewhere). `SHAPE_SPACE_DIM`
  / `EXPRESSION_SPACE_DIM` / `FULL_HAND_DIM` kept as module constants (validation
  bounds coupled to converter logic; see Finding #4). **LOC 447 → 391 (−56).**
- `core/body.py`: `BodyStructure.landmarks` populated as
  `{"frame_vertex_ids": LongTensor(21)}` (device-placed) by both bodies;
  convention fields populated from config. Added public
  `ParametricBody.landmark_positions(vertices) -> (..., L, 3)` (gathers
  `vertices[..., frame_vertex_ids, :]`) — the live consumer that makes `landmarks`
  a real feature. No viz overlay added (handoff marked it optional; kept minimal).
- `tests/test_smoke.py`: dropped the `make_smpl_like_model` import; now builds a
  weight-free 24-body free-flyer + 23-spherical tree directly with
  `build_kinematic_tree_model` (nq=99, nv=75). No weights required.
- New tests: `test_landmark_positions_gather_matches_config` (SMPL + SMPL-X;
  asserts count/shape, exact gather equality, the `nose` keypoint vertex —
  SMPL 332 / SMPL-X 9120) and `test_landmark_positions_preserves_leading_batch`
  (SMPL, `(2, 21, 3)`).

`bodies/smpl.py` and `bodies/smplx.py` now import **no topology** from
`better_robot` (verified by grep). All parity, round-trip, and float32-gradcheck
tests still pass at unchanged tolerances.

## Part B — BetterRobot

`better_robot.io.builders.smpl_like` (`JOINT_NAMES`, `PARENTS`,
`_default_offsets_tensor`, `make_smpl_like_body`, `make_smpl_like_model`) was
**deleted**. The generic `build_kinematic_tree_body/model` stays. A neutral
test-support fixture `tests/support/branching_tree.py` (new) provides the same
24-joint free-flyer + 23-spherical topology and offset values with neutral names
`j0..j23` and no anatomical comments: `JOINT_NAMES`, `PARENTS`,
`default_offsets_tensor`, `make_branching_tree_body/model`.

### Acceptance grep

`grep -riE "smpl|pelvis|betas|human" .../BetterRobot/src` now returns only
unrelated matches: the `free_flyer.py` "humanoid" comment, the
`force_vectors.py` "human-scale" comment, and `residuals/human.py`
(`SwingTwistLimitResidual`) + its `__init__`/CLAUDE.md — a pre-existing, separate
spherical-joint rotation-limit feature, out of the authorized extraction scope
(see Finding #5). No SMPL/human topology or naming remains.

### Every BetterRobot change (the two authorized edits)

1. **Mesh tools (order 01, pre-existing in the working tree):**
   `src/better_robot/spatial/mesh.py`, `src/better_robot/spatial/__init__.py`,
   `tests/spatial/test_mesh.py` — **left untouched** by this order.
2. **smpl_like extraction (this order).**

Extraction infrastructure (6 files):
- `src/better_robot/io/builders/smpl_like.py` — **deleted**
- `src/better_robot/io/builders/__init__.py` — dropped the smpl_like exports
- `src/better_robot/io/builders/kinematic_tree.py` — docstring scrub (removed
  smpl_like mention + the stale `§6` anchor)
- `tests/support/branching_tree.py` — **new** neutral fixture
- `docs/reference/api/.../better_robot.io.builders.md` — dropped toctree entry
- `docs/reference/api/.../better_robot.io.builders.smpl_like.md` — **deleted**

Rewired callers — **17 files** (all now use the neutral fixture, or the generic
builder directly):

tests/ (14):
1. `tests/io/test_smpl_like.py` → **renamed** `tests/io/test_branching_tree.py`
   (converted to a fixture/builder test)
2. `tests/io/test_kinematic_tree.py` (imports + `test_branching_tree_*` renames)
3. `tests/io/test_preserve_order.py` (imports + DFS-order assertion → `j1,j4,j7,j10`)
4. `tests/data_model/test_integrate_vectorized.py` (+ label `"smpl_like"` → `"branching_tree"`)
5. `tests/data_model/test_model_structure.py`
6. `tests/optim/test_tangent_autograd.py`
7. `tests/viewer/test_scene_mode_toggling.py`
8. `tests/viewer/test_urdf_mode_availability.py`
9. `tests/warp/test_fk_cuda.py`
10. `tests/warp/test_fk_prototype.py`
11. `tests/warp/test_rnea_cuda.py`
12. `tests/bench/bench_integrate_difference.py`
13. `tests/bench/bench_trajopt_sparse.py` (import + call + `FRAME_NAMES` remap +
    repo-root bootstrap; see Finding #1/#2)
14. `tests/bench/test_trajopt_sparse_smoke.py` (docstring)

benchmarks/ (2 — **not in the handoff's estimate**, see Finding #1):
15. `benchmarks/baseline.py`
16. `benchmarks/m1_warp_fk_vs_compiled_torch.py`

examples/ (1):
17. `examples/04_smpl_like_body.py` → **renamed** `examples/04_branching_tree_body.py`
    (converted to a generic `build_kinematic_tree_model` example)

## Findings / deviations

1. **More callers than the "~13" estimate.** The smpl_like symbols were also
   imported by two `benchmarks/` modules (`baseline.py`,
   `m1_warp_fk_vs_compiled_torch.py`, loaded via importlib by
   `tests/bench/test_baseline.py` and `test_m1_warp_benchmark_schema.py`), and
   `bench_trajopt_sparse.py` referenced SMPL joint names by string
   (`FRAME_NAMES = body_head, body_left_wrist, …`). All rewired. `FRAME_NAMES`
   was remapped to the **same physical joints** under neutral names
   (head→`j15`, left_wrist→`j20`, right_wrist→`j21`, left_ankle→`j7`,
   right_ankle→`j8`), preserving the deterministic §11 benchmark:
   `active_envelope_coordinates == 127`, `route == banded`, `SUCCESS` — verified
   unchanged.
2. **Standalone-subprocess imports.** `bench_trajopt_sparse.py` runs a `--child`
   subprocess (no pytest conftest) and the two `benchmarks/*` are standalone
   CLIs; none can import `tests.support` off the bare script path. Added a small
   repo-root `sys.path` bootstrap (module-level in the bench, local at the call
   site in the two benchmarks). Model is byte-identical to before, so numerics
   are unchanged.
3. **Order-03 asset fields have no order-02 consumer.**
   `segmentation_closed_file`, `segmentation_vertex_file`, `density_table` are
   recorded but unused this order. Included per owner-settled decision 2 and the
   handoff's explicit "include them now". This is a deliberate exception to the
   "no dead fields" rule — flagged for owner awareness.
4. **Shape-space dims stayed in `smplx.py`** (handoff marked the move optional).
   They are SMPL-X-only validation bounds; putting them on the shared
   `BodyConfig` would create SMPL-dead fields.
5. **`residuals/human.py` still matches the grep.** `SwingTwistLimitResidual` is
   a pre-existing, separate spherical-joint rotation-limit feature (roadmap:
   "human residuals on the BR optimizer"), unrelated to SMPL body topology and
   **out of the authorized edit scope** (ground rule 4). Reported, not modified.
6. **Two committed labels retain SMPL text.**
   `tests/bench/baselines/torch_filtered_smpl_b1_rtx6000_ada.json` (a committed
   measurement artifact) and one description string in `benchmarks/baseline.py`
   (`"smpl": "make_smpl_like_model, …"`) keep their text as opaque identifiers
   matching that artifact. Not shipped API, not in the src acceptance grep; left
   unchanged to avoid diverging from the committed baseline.
7. **ruff:** all touched files are clean except pre-existing baseline dirt in
   **unmodified** regions — `tests/io/test_kinematic_tree.py` (E741 `I`, PLC0415
   in the com/inertia block) and `io/builders/kinematic_tree.py` (E741 `I` in
   `_normalize_inertia`); both present at `HEAD`, neither introduced here.
