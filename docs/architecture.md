# Architecture

BetterHuman hosts many parametric human body models behind one API, on top of
BetterRobot. This page defines the layers, the core objects, where every
tensor lives, and the seams future models plug into. The reference studies
that motivated each decision live in `references/design/` (local, untracked);
the short keys used below (SMX, STAR, SUPR, MHR, ANY, MOM, SMA, BRR, …) refer
to those files.

## Layering

```text
viz          (skinned-mesh render mode over the BetterRobot viewer)
 |
bodies       (SMPL, SMPLX, ... — one module per model + the registry)
 |
core         (ParametricBody, shared stage math, classic converters)
 |
io           (weight-file schemas, validated loaders, chumpy converter)
 |
better_robot (Model/Data, FK, Jacobians, dynamics, optim, viewer backend)
```

Imports point down. `io` knows file formats but no body math; `core` knows
the math but no file formats and no concrete model; each module in `bodies`
knows exactly one model's topology, schema, quirks, and converters; `viz`
consumes bodies and never the reverse. Namespaces `better_human.retarget`
and `better_human.measure` are reserved for the roadmap (human→robot
retargeting, body measurements) and intentionally do not exist yet — no
placeholder packages.

## The common mathematical core

Every studied vertex-based body model reduces to one seven-stage pipeline
(SMX, STAR, SUPR, MHR, ANY, MOM):

```text
M(β, θ, ψ) = W( T(θ, J(β)) · ( v_template + B_S(β) + B_E(ψ) + B_P(φ(θ, β)) ) )
             └───skinning──┘   └template──┘ └shape┘ └expr┘ └─pose corrective─┘
```

| Stage | Universal form | Who computes it |
|---|---|---|
| Template `v_template [V,3]` | fixed per model | loaded weights |
| Shape `B_S(β)` | linear blend of basis columns | `core`, at `with_shape` time |
| Expression `B_E(ψ)` | extra shape columns, same code path | `core`, at `with_shape` time |
| Joint regression `J(β)` | linear map of shaped verts (or absent) | `core`, at `with_shape` time |
| FK `T(θ, J)` | configuration `q` on the kinematic tree | **BetterRobot** |
| Pose corrective `B_P(φ(θ,β))` | callable of pose feature (and β for STAR) | `core`, per query |
| Skinning `W(·)` | LBS, padded `[V,K]` influences | `core`, per query |

The pipeline is written **once** in `core`; a model contributes *data* (weight
tensors, topology, schema) and *small policy choices* (pose-feature function,
converter quirks), never a copy of the pipeline. SMPL and STAR differ in one
matrix and one feature function (STAR); smplx's five classes with ten
copy-pasted `forward()`s are the anti-example this rule exists to prevent
(SMX §7).

## Core objects and where tensors live

BetterHuman mirrors BetterRobot's split — structure vs values vs query — and
composes with it rather than forking it:

- **`BodyStructure`** (static, BetterHuman side): vertex/face counts, faces,
  joint names and tree metadata, skinning sparsity pattern `[V,K]` (indices),
  body-part assignment and closed-part topology, parameter-group declarations
  (names, widths), landmark/keypoint index tables, classic-convention metadata
  (units, up-axis, native pose format).
- **`BodyValues`** (differentiable tensors, BetterHuman side): the loaded
  weight tensors (`v_template`, shape/expression basis, corrective basis,
  joint regressor, skinning weight values) **plus the shape-baked products**
  — `v_shaped`, rest joints `J(β)`, and packed body inertias — which are
  batch-bearing (`(B, V, 3)` = B different bodies), exactly like
  BetterRobot's batch-bearing `ModelValues` tables.
- **`better_robot.Model`** (the robot): free-flyer root + typed joints
  (spherical for SMPL-family bodies), built programmatically at spawn via
  `build_kinematic_tree_model` with **`preserve_joint_order=True`** — BR's
  default DFS topo-sort would reorder joints and silently misalign every
  native-order human tensor (`weights [V,K]`, `posedirs`, `J_regressor`).
  The standing index convention: BR joint `j+1` ↔ native joint `j`
  (BR's universe joint occupies index 0). Shape baking writes the
  β-dependent `joint_placements` and `body_inertias` into the BR
  `ModelValues` through `Model.with_values` —
  the path BR already tests end-to-end with gradients (BRR §3).
- **`better_robot.Data`** (per query): joint/frame poses, Jacobians,
  dynamics quantities. BetterHuman adds no parallel Data class in v1;
  vertex outputs are plain tensors returned by the body's methods. A
  BetterHuman-side data container appears only if lazy vertex/landmark
  caching earns it later.

**`ParametricBody`** is the composition — one class, configured, never a
family tree:

```python
class ParametricBody:
    structure: BodyStructure      # static human tables
    values:    BodyValues         # weights + shape-baked products (batch-bearing)
    robot:     better_robot.Model # BR structure + shape-baked ModelValues
```

Concrete models (`SMPL`, `SMPLX`) are thin subclasses that implement
loading (schema + spawn-parameter reduction), the pose-feature policy, and
`from_classic` / `to_classic`. They contain no pipeline math. Each reads its
static metadata — native joint names, the reduction index groups, landmark
keypoint tables, native axis conventions, and the segmentation/density asset
filenames — from a frozen `BodyConfig` in a per-model data module
(`bodies/config_smpl.py`, `bodies/config_smplx.py`), and shares data-binding
helpers through `bodies/smpl_family.py`; the SMPL topology no longer lives in
BetterRobot (which keeps only the generic kinematic-tree builder). Landmark
keypoints restored from that config are read back through
`body.landmark_positions(vertices)`.

## The user-facing flow

```python
import better_human as bh

body = bh.SMPLX(model_path=..., gender="neutral", num_betas=16,
                use_hands=False, use_face=False)      # spawn params, §partial
body = body.with_shape(betas=betas)                    # bake shape, differentiable
data  = body.fk(q)                                     # BetterRobot FK
verts = body.vertices(q)                               # correctives + LBS (FK inside)
q     = body.from_classic(classic)                     # native params -> q
classic = body.to_classic(q)                           # exact inverse
body2 = bh.load("smpl", model_path=...)                # registry factory
```

- `with_shape(**groups)` takes **named parameter groups** (`betas`,
  `expression`, …) — never one anonymous vector — because MHR splits identity
  across mesh blendshapes *and* bone scales, and Anny uses phenotype sliders
  (MHR, MOM, ANY). It recomputes `v_shaped`, `J(β)`, and the BR
  `ModelValues`, returns a new body (functional style, matching
  `Model.with_values`), and is differentiable end-to-end: betas can carry a
  leading batch dim, and gradients flow through FK/IK/dynamics into betas
  (proven in BR by `tests/optim/test_parametric_model_values.py`).
- Pose queries (`fk`, `vertices`, Jacobians, dynamics) never touch shape
  math. This is the central performance contract: **shape is baked once,
  pose inference is pure robotics FK + skinning** — the majority of
  reference implementations recompute shape every forward and that is their
  main self-inflicted cost (BEST_PRACTICES §12.5). Precision: this "once"
  applies to identity parameters. Expression is a shape group too (it moves
  `v_shaped` and the regressed joints), so per-frame facial animation
  re-bakes shape — amortized by batching all frames' expressions in one
  `with_shape` call, not by the pose path.
- `q` follows BetterRobot conventions exclusively (scalar-last quaternions,
  free-flyer root `[tx,ty,tz,qx,qy,qz,qw]`, `nq ≠ nv`,
  `Model.integrate`/`difference` instead of `q + v`). Axis-angle, Euler
  angles, and 4×4 matrices exist only inside converters.

## Classic conversion — per model, at the boundary

Each model owns a `from_classic` / `to_classic` pair mapping its native
parameterization to `q`. These are **exact inverses**, covered by round-trip
tests, and **per-model by design**: the SMPL family's root wart
(`global_orient` pivots about the β-dependent regressed pelvis `J₀`;
`transl` is a post-skinning world offset, so the pelvis sits at
`J₀(β) + transl`) is *not* universal — MHR, Momentum, SOMA, Anny have clean
roots (BEST_PRACTICES §4.2–4.3). Consequences:

- The SMPL-family converter re-adds/removes the `J₀(β)` offset; since the
  body has shape baked, it reads `J₀` from its own `BodyValues` — the
  conversion is shape-dependent and that is correct.
- **Pinned root-frame convention** (the precondition that makes the simple
  conversion valid): at shape-bake time the rest mesh and rest joints are
  **pelvis-centered** — shifted by `−J₀(β)` so the pelvis sits at the
  origin — and the free-flyer root placement is zero. Then `q[0:3]` is the
  pelvis world position, `q[3:7]` its orientation, and the classic map is
  exactly `transl = q[0:3] − J₀(β)` with no rotation-coupled term. Keeping
  the MPI-world template instead would make the root translation
  `J₀ + transl − R₀·J₀` — a silent parity breaker; do not do that.
- Hand PCA and the baked hand mean pose (SMPL-X) are converter-level
  reparameterizations of the spherical hand joints, not forward-pass state
  (SMX §3.2–3.3).
- Classic structs are small typed dataclasses per model (e.g.
  `SMPLXClassic` with `global_orient`, `body_pose`, `transl`, …), not
  positional tensors.

## Partial-model spawning — parameters, never subclasses

Spawn parameters (`num_betas`, `use_hands`, `use_face`, …) drive **one
generic reduction mechanism** at load time:

1. select shape/expression basis columns (a slice),
2. drop articulations: remove joints from the tree, reparent children,
   reassign+renormalize their skinning weights to the surviving parent,
3. slice the corrective basis columns belonging to dropped joints (their
   pose features are identically rest, so their contribution is exactly
   zero),
4. build the reduced BR tree.

The result is a smaller `ParametricBody` with full fidelity on surviving
DOFs. Precedents: Anny's `bones_to_remove` reparent+reweight, Momentum's
`simplify`/`subsetParameterTransform`, SUPR's count-agnostic loader
(BEST_PRACTICES §6). Semantic flags per model map to joint sets; the
mechanism is shared. The parameter surface must be stabilized before any
release — SOMA's god constructor with alias debt is the warning (SMA §7).

## Pluggability policy — seams now, code when a second caller exists

The stages are the extension seams for future models (MHR's authored
skeleton + MLP correctives + parameter-reduction transform; Anny's phenotype
shape space; SUPR's typed constrained tree). Per the design principles, an
abstraction ships only with ≥ 2 in-tree callers, so **v1 implements exactly
what SMPL + SMPL-X exercise**: linear PCA shape/expression, a dense joint
regressor, linear `(R−I)` correctives, padded-`[V,K]`
LBS. The documented seams — recorded here so later models slot in without
redesign — are:

- `ShapeModel`: coefficient groups → `(Δverts, Δjoints)` (PCA now;
  phenotype interpolation later).
- `JointRegressor`: shaped verts → rest joints, **nullable** (MHR/Momentum
  have authored skeletons; SUPR uses an affine-with-bias variant).
- `PoseCorrective`: callable `f(pose_feature, shape)` (linear now; STAR/SUPR
  quaternion features and MHR's MLP later).
- `ParameterTransform`: compact user DOFs → joint DOFs (identity now;
  Momentum/MHR sparse maps later).

When the third model family lands, these become explicit protocols; until
then they are well-named functions/dataclasses in `core` with the interface
shape above.

## Weights: loading, resolution, licensing

- **Resolution order**: explicit `model_path` argument → else
  `$BETTERHUMAN_MODELS_DIR/<family>/<canonical-file>` using a documented
  per-model layout → else an actionable error naming both options. Nothing
  is ever downloaded; most weights are license-gated and non-redistributable
  (SMX, STAR, SUPR licenses), so the user always supplies them.
- **One clean, self-describing `.npz` per model** with a validated field
  schema (field tables from the reference studies are the schema). Loading
  validates names, shapes, and dtypes up front — no late `AttributeError`,
  no silent clamping with `print` (SMX §7 anti-patterns).
- **Chumpy dies at load.** The converter CLI turns original chumpy-era
  `.pkl`s into our `.npz` using a pickle-stub shim (fake `chumpy` module
  whose objects unpickle to arrays, SMA §5.4) — expected to work in the
  modern venv with no museum NumPy. If the shim proves insufficient for
  some file vintage, the fallback is a documented pinned converter venv
  (uv makes that cheap); the shim is attempted first.
- Per-asset licenses stay verbatim next to any asset we do ship (mink/GMR
  pattern); we ship no gated weights.

## Mass and dynamics

BetterRobot owns the algorithms and the mesh integrator
(`Inertia.from_mesh`: differentiable, batched, torch-only — BRR §4).
BetterHuman contributes the human glue: it partitions the mesh into closed
per-part solids and integrates each. **Two segmentation sources feed one closed
`PartMeshTopology`** (`core/mass.py`), and `docs/segmentation.md` shows both:

- **Authored closed-3D** (default for SMPL-family models that ship the asset).
  Vetted per-part face tables index `[shaped_vertices; rest_joints]`; each part's
  open boundary is closed by fanning to a **rest joint used as a cap vertex**.
  Winding is fixed once from rest geometry with BetterRobot's non-strict
  `orient_faces_by_component` (which replaced the historical
  `trimesh.fix_normals` and tolerates the three open, non-manifold shoulder-seam
  parts). No runtime boundary tracing.
- **Argmax fallback** (generic; SMPL-X). Each vertex takes its largest skinning
  influence; a triangle belongs to the majority part among its three vertices;
  each open boundary cycle is closed by reverse-wound triangles meeting at the
  cycle's centroid.

Static ownership and cap topology are computed at spawn, while cap positions and
inertias are recomputed from `v_shaped` during shape baking. Each integrated
inertia is translated into its native joint frame before the packed rows are
written through `Model.with_values`; the massless BetterRobot universe row is
prepended. RNEA, CRBA, ABA, and centroidal algorithms then work unchanged and
remain differentiable with respect to shape.

At density 1000 kg/m³ the neutral SMPL fixture integrates to 74.76 kg via the
authored closed-3D parts — within +0.0055 % of the 74.76 kg whole-mesh volume,
with aggregate COM 0.02 mm from the whole-mesh result. The argmax fallback on
the same fixture gives 73.43 kg (a +1.78 % volume gap, COM 2.47 mm), so the
authored partition is preferred wherever its asset exists.

Mass baking adds nontrivial cost to `with_shape`, so it is enabled by default
but may be disabled at spawn with `compute_mass=False` for deformation-only
workloads.

## Joint limits

Hard `q` limits on spherical joints are vacuous in BR (quaternion-component
boxes); anatomical limits are expressed as **soft swing/twist cone
residuals** — BR already ships `SwingTwistLimitResidual` (BRR §2).
BetterHuman's contribution is per-joint anatomical swing/twist tables per
model, provided as data on `BodyStructure`.

## Visualization

The BR viewer has no deforming-mesh mode; its viser backend can upload
arbitrary meshes (`add_mesh`). `viz` adds a skinned-mesh render mode that
re-uploads LBS-posed vertices per frame, next to BR's skeleton mode. No
BetterHuman rendering stack.

## Package layout (v1)

```text
src/better_human/
├── __init__.py        # SMPL, SMPLX, load, ParametricBody — small public API
├── core/
│   ├── body.py        # ParametricBody, BodyStructure, BodyValues
│   ├── pipeline.py    # blend shapes, joint regression, correctives, LBS
│   ├── reduction.py   # generic partial-model reduction (joints/columns)
│   ├── mass.py        # closed part meshes and shape-dependent inertias
│   └── classic.py     # shared converter algebra (pelvis-offset, axis-angle<->quat)
├── bodies/
│   ├── registry.py    # load("smplx", ...) — decorator-registered factories
│   ├── config_body.py   # BodyConfig dataclass (per-model static metadata)
│   ├── config_smpl.py   # SMPL joint names, landmarks, asset filenames
│   ├── config_smplx.py  # SMPL-X names, reduction groups, landmarks, assets
│   ├── smpl_family.py    # shared data binding (values, robot, mass, density)
│   ├── smpl.py          # schema binding, converters, spawn flags
│   ├── smplx.py
│   └── assets/          # authored segmentation + density tables + loaders
├── io/
│   ├── schema.py      # per-model npz field schemas, validation
│   ├── load.py        # path resolution (arg > env root), validated loading
│   └── convert_chumpy.py  # CLI: original .pkl -> clean .npz (pickle-stub shim)
└── viz/
    └── skinned_mesh.py    # BR viewer render mode for deforming vertices
```

## Verification

`smplx` (the MPI package) is the correctness oracle for the SMPL family.
Parity is enforced through **golden files**: a pinned uv venv under
`tools/parity/` runs the original implementation once and dumps
vertices/joints for a fixed grid of shapes/poses (including the known traps:
hand mean pose, SMPL-X jaw+eyes-before-hands ordering, expression columns);
our tests compare against the goldens at documented tolerances and skip
cleanly when local weights are absent. Converter round-trips
(`from_classic ∘ to_classic = id`) and float32 gradchecks through
`betas → vertices` and `q → vertices` complete the safety net. These tests
are sacred in the BetterRobot sense: never weakened to land a change.
