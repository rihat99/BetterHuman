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
| Skinning `W(·)` | LBS, spawn-built dense `[V,J]` matrix; padded `[V,K]` archive storage retained | `core`, per query |

At spawn, the padded sparse `(V, K)` weights are scattered into one dense
`(V, J)` matrix. At query time each joint transform becomes a 3×4 affine,
the affines are blended once, and one blended affine is applied per vertex.
Scaled joints fold their scale and bind pivot into the joint affine first.
This matrix-blend formulation replaces the former per-vertex, per-influence
transform gather while preserving the archive representation for validation
and segmentation.

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
  joint regressor, padded skinning values, and the spawn-built dense skinning
  matrix) **plus the shape-baked products** — `v_shaped`, rest joints `J(β)`,
  and packed body inertias — which are batch-bearing (`(B, V, 3)` = B
  different bodies), exactly like BetterRobot's batch-bearing `ModelValues`
  tables.
- **`better_robot.Model`** (the robot): free-flyer root + typed joints
  (spherical for SMPL-family bodies; revolute/mimic stacks for MHR), built
  programmatically at spawn via `build_kinematic_tree_model` with
  **`preserve_joint_order=True`** — BR's
  default DFS topo-sort would reorder joints and silently misalign every
  native-order human tensor (`weights [V,K]`, `posedirs`, `J_regressor`).
  For the SMPL family, BR joint `j+1` ↔ native joint `j` because BR's universe
  occupies index 0. MHR instead gathers its 127 native poses from a 203-joint
  expanded BR tree. Shape baking writes the
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

Concrete models (`SMPL`, `SMPLX`, `MHR`) are thin subclasses that implement
loading (schema + spawn controls), the pose-feature policy, and
`from_classic` / `to_classic`. They contain no pipeline math. Each reads its
static metadata — native joint names, the reduction index groups, landmark
keypoint tables, native axis conventions, and the segmentation/density asset
filenames — from a frozen `BodyConfig` in a per-model data module
(`bodies/smpl_family/config_smpl.py`, `bodies/smpl_family/config_smplx.py`,
`bodies/mhr/config.py`). The SMPL classes share data-binding helpers through
`bodies/smpl_family/shared.py`; the SMPL topology no longer lives in BetterRobot (which keeps only
the generic kinematic-tree builder). Landmark keypoints restored from that config are read back
through `body.landmark_positions(vertices)`.

### Design decision: composition, not inheritance

`ParametricBody` contains a `better_robot.Model`; it does not inherit from one. This boundary
is deliberate:

- BetterRobot already separates and composes `ModelStructure`, `ModelValues`, and `Model`.
  BetterHuman follows that idiom by pairing its mesh-specific structure and values with the
  composed robot.
- BetterRobot registers the exact `ModelValues` type as a PyTorch pytree and reconstructs that
  base type. A subclass can instead become a leaf or lose its subclass on reconstruction, so
  inheritance would make transforms such as `vmap` and `compile` depend on registration details.
- The schemas have different ownership and update lifecycles. `ModelValues` does support leading
  batch dimensions for its batch-bearing tables, but body bases, vertices, inverse binds, and
  shape-baked products are not robot values. Folding them together would also couple BetterHuman
  to BetterRobot's evolving alpha schemas—currently 78 `ModelStructure` fields and 15
  `ModelValues` fields—rather than to a stable composition seam.
- Native human-model vocabulary is not the robot execution vocabulary. MHR, for example, uses
  `joint_names` and `parents` for 127 authored skin joints, while the same names on its BetterRobot
  model describe a 203-joint expanded tree. One inherited field could not carry both meanings.
- Mesh representation will evolve independently of robot topology: future bodies may expose
  multiple meshes, materials, collision surfaces, and several LODs for one robot. Composition lets
  those assets change without multiplying or redefining BetterRobot model types.

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

For a one-off shape-and-pose query, the callable form is exact shorthand for the two-step flow:
`verts = body(q, betas=betas)` is equivalent to
`body.with_shape(betas=betas).vertices(q)`. When many poses share one identity, keep the
fixed-shape fast path: bake once with `shaped = body.with_shape(betas=betas)`, then call
`shaped.vertices(q)` for each pose. The callable adds no cache and returns only vertices.

For SMPL and SMPL-X, classic inputs contain pose state, so `from_classic`
returns `q`. MHR's native input also contains identity, expression, and bone
proportions. Its shape-aware boundary therefore returns
`shaped_body, q = body.from_classic(classic)`; the inverse is
`shaped_body.to_classic(q)`.

- `with_shape(**groups)` takes **named parameter groups** (`betas`,
  `expression`, …) — never one anonymous vector — because MHR separates mesh
  identity from bone-length and hand-scale proportions, and Anny uses
  phenotype sliders (MHR, MOM, ANY). It recomputes model-specific shaped vertices and rest
  placements plus the BR `ModelValues`, then returns a new body (functional
  style, matching `Model.with_values`), and is differentiable end-to-end:
  betas can carry a
  leading batch dim, and gradients flow through FK/IK/dynamics into betas
  (proven in BR by `tests/optim/test_parametric_model_values.py`).
- Pose queries (`fk`, `vertices`, Jacobians, dynamics) never touch shape
  math. This is the central performance contract: **shape is baked once;
  vertex inference is FK + pose correctives + LBS** — the majority of
  reference implementations recompute shape every forward and that is their
  main self-inflicted cost (BEST_PRACTICES §12.5). Precision: this "once"
  applies to identity parameters. Expression is a shape group too (it moves
  `v_shaped`, and the regressed joints where a model has a regressor), so
  per-frame facial animation re-bakes shape — amortized by batching all
  frames' expressions in one `with_shape` call, not by the pose path.
- `q` follows BetterRobot conventions exclusively (scalar-last quaternions,
  free-flyer root `[tx,ty,tz,qx,qy,qz,qw]`, `nq ≠ nv`,
  `Model.integrate`/`difference` instead of `q + v`). Axis-angle, Euler
  angles, and 4×4 matrices exist only as model-private converter, loader, or
  pose-feature intermediates.

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
- MHR classic parameters include shape as well as pose. Its `from_classic`
  applies `with_shape` and returns the resulting body together with `q`, so no
  shape information is silently discarded at the boundary.
- Classic structs are small typed dataclasses per model (e.g.
  `SMPLXClassic` with `global_orient`, `body_pose`, `transl`, …), not
  positional tensors.

## MHR's authored-rig path

MHR has no joint regressor. Native joint 0 (`body_world`) maps to the
BetterRobot universe, while native joint 1 (`root`) owns the free-flyer. The
public configuration is that root plus 125 archive-named revolutes
(`nq=132`, `nv=131`). Forty-seven additional single-source mimic nodes fan a
public coordinate into other authored rotation rows; same-axis stacked nodes
represent the multi-driver spine rows. The latter preserve exact classic
parameter-vector round trips but create six physical null directions, which is
why MHR's CRBA is singular.

Its shape state is 45 identity coefficients, optional 72-dimensional
expression, and 73 proportions. Identity and expression are mesh-only. Of the
proportions, 71 change authored joint translations and two encode right/left
hand log-scale; accumulated `exp2` scale enters LBS about the authored bind
pivots. Static inverse binds from the release remain authoritative.

For each pose query, BetterRobot expands the revolute/mimic configuration.
The resulting 125 native Euler triples become 750 rotation features for the
`750 → 3000 → V*3` MLP corrective, then inverse-bind transforms and scaled LBS
produce vertices. Seven LOD archives share this rig (LOD0–6, default LOD1);
`use_expression` and `use_correctives` omit their large archive payloads at
load time, while `compute_mass=False` avoids topology construction and inertia
integration.

The LOD0 archive is 2,769,297,616 bytes (2,641.01 MiB, 2.579 GiB); its stored
float32 `corrective_output_weights` tensor alone is 2,651,004,000 bytes
(2.469 GiB). Consequently, a correctives-off load must neither require nor
inspect that optional payload.

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

## Pluggability policy — concrete seams, minimal abstractions

MHR exercises the non-SMPL paths without adding speculative public protocols.
`ParametricBody` has callable shape-bake and pose-corrective policies used by
multiple families; `BodyValues` permits a nullable joint regressor, authored
inverse binds, bind pivots, and accumulated joint scales. Shared `core`
functions implement blend shapes, both pose-feature families, dense MLP
correctives, inverse-bind transforms, and scale-aware LBS.

MHR's sparse compact-parameter map remains loader-specific tree construction;
there is no generic public `ParameterTransform` interface with only one caller.
Future phenotype or quaternion-corrective models should reuse the concrete
stage seams first and add a new abstraction only when a second in-tree caller
proves it necessary.

## Weights: loading, resolution, licensing

- **Resolution order**: explicit `model_path` argument → else
  `$BETTERHUMAN_MODELS_DIR/<family>/<canonical-file>` using a documented
  per-model layout → else an actionable error naming both options. Nothing
  is ever downloaded; most weights are license-gated and non-redistributable
  (SMX, STAR, SUPR licenses), so the user always supplies them.
- **One clean, self-describing `.npz` per model variant or LOD** with a
  validated field schema (field tables from the reference studies are the
  schema). Loading validates names, shapes, and dtypes up front — no late
  `AttributeError`, no silent clamping with `print` (SMX §7 anti-patterns).
- **Chumpy dies at load.** The converter CLI turns original chumpy-era
  `.pkl`s into our `.npz` using a pickle-stub shim (fake `chumpy` module
  whose objects unpickle to arrays, SMA §5.4) — expected to work in the
  modern venv with no museum NumPy. If the shim proves insufficient for
  some file vintage, the fallback is a documented pinned converter venv
  (uv makes that cheap); the shim is attempted first.
- Per-asset licenses stay verbatim next to any asset we do ship (mink/GMR
  pattern); we ship no gated weights.
  The MHR repository states Apache-2.0, but users must verify the terms inside
  the exact release archive before redistributing its assets.

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
- **Argmax fallback** (generic; SMPL-X and MHR). Each vertex takes its largest
  skinning influence; a triangle belongs to the majority part among its three
  vertices; each open boundary cycle is closed by reverse-wound triangles
  meeting at the cycle's centroid. MHR first maps native winning joints to its
  19 reference segmentation groups.

Static ownership and cap topology are computed at spawn, while cap positions and
inertias are recomputed from `v_shaped` during shape baking. Each integrated
inertia is translated into its native joint frame before the packed rows are
written through `Model.with_values`; the massless BetterRobot universe row is
prepended. BetterRobot's RNEA, CRBA, ABA, and centroidal algorithms consume
these values unchanged and remain differentiable with respect to shape;
solve-based use still requires independent coordinates and nonsingular mass.

At density 1000 kg/m³ the neutral SMPL fixture integrates to 74.76 kg via the
authored closed-3D parts — within +0.0055 % of the 74.76 kg whole-mesh volume,
with aggregate COM 0.02 mm from the whole-mesh result. The argmax fallback on
the same fixture gives 73.43 kg (a +1.78 % volume gap, COM 2.47 mm), so the
authored partition is preferred wherever its asset exists.

Mass baking adds nontrivial cost to `with_shape`, so it is enabled by default
but may be disabled at spawn with `compute_mass=False` for deformation-only
workloads.

At density 1000 kg/m³, neutral MHR LOD1 has 19 positive physical bodies with
a total mass of 81.481849670 kg. LOD1 is the default measured baseline;
centroid-capped totals on coarser LOD surfaces are approximate.

MHR's current grouped placement does not provide invertible dynamics: neutral
LOD1 CRBA has rank 50/131, while the overcomplete pose map alone imposes a
theoretical rank ceiling of 125/131. The exact PSD diagnostics are recorded in
the contracts; solve-based MHR dynamics must not be presented as supported
until its independent-coordinate design is settled.

## Joint limits

Hard component limits on spherical joints are not anatomical constraints;
BetterRobot provides `SwingTwistLimitResidual` for callers that supply cone
limits, but BetterHuman does not yet ship per-model anatomical tables. MHR's
scalar revolute coordinates do receive their sign-aware archive limits.

## SMPL and SMPL-X performance

`benchmarks/benchmark_smpl.py` compares the neutral float32 SMPL and SMPL-X
models with the vendored upstream `smplx` 0.1.28 package using the same NPZ
parity fixtures. SMPL uses 10 betas. SMPL-X uses 16 betas, 10 expressions,
full hands and face, natural hand means, no hand PCA, and no face contour.
The upstream SMPL-X call disables its optional duplicate shaped-vertex output;
its static landmark and extra-joint work remains part of the full forward,
with a reference instance constructed at each matching batch size.

The upstream row recomputes shape and pose. BetterHuman's **pose-only** row
uses one scalar shape baked before timing; its **shape+pose** row performs
`with_shape` plus `vertices(q)` every call and is the fair end-to-end
comparison. Shape rows are identical within each batch while pose rows vary.
Before timing, both BetterHuman paths were compared directly with upstream at
every model, batch, device, and FK lane; the worst maximum absolute vertex
difference was `5.9604644775390625e-7 m`, below the `1e-6 m` parity pin.

CPU measurements used five warmups and 30 calls, 112 intra/inter-op threads,
PyTorch 2.13.0+cu126, and BetterRobot 0.2.0 on the shared dual-socket Intel
Xeon Platinum 8570 host:

| Model | Batch | Original shape+pose (ms) | BetterHuman pose-only (ms / ×ref) | BetterHuman shape+pose (ms / ×ref) |
|---|---:|---:|---:|---:|
| SMPL | 1 | 5.982045 | 6.294569 / 1.052244× | 21.256035 / 3.553306× |
| SMPL | 16 | 2.421118 | 11.292286 / 4.664079× | 11.746825 / 4.851819× |
| SMPL | 256 | 23.291602 | 28.108368 / 1.206803× | 31.323141 / 1.344826× |
| SMPL-X | 1 | 2.072404 | 13.443112 / 6.486724× | 9.304661 / 4.489791× |
| SMPL-X | 16 | 2.315695 | 10.618555 / 4.585471× | 10.957764 / 4.731954× |
| SMPL-X | 256 | 40.569139 | 44.174468 / 1.088869× | 54.745655 / 1.349441× |

The small-batch CPU medians are visibly affected by shared-host scheduling;
the raw values are retained rather than selectively rerun. At batch 256,
both paths are within 1.35× of upstream for both models.

CUDA used explicit synchronization and an otherwise idle NVIDIA RTX 6000 Ada
Generation (48 GiB, compute capability 8.9, driver 560.35.03). Native Torch
FK measured:

| Model | Batch | Original shape+pose (ms) | BetterHuman pose-only (ms / ×ref) | BetterHuman shape+pose (ms / ×ref) |
|---|---:|---:|---:|---:|
| SMPL | 1 | 1.664994 | 17.377573 / 10.437015× | 17.905309 / 10.753975× |
| SMPL | 16 | 1.707768 | 18.073975 / 10.583390× | 18.894050 / 11.063593× |
| SMPL | 256 | 3.010079 | 22.172007 / 7.365921× | 23.031094 / 7.651324× |
| SMPL-X | 1 | 3.183152 | 41.392021 / 13.003472× | 44.985560 / 14.132397× |
| SMPL-X | 16 | 3.414323 | 45.384021 / 13.292246× | 46.294748 / 13.558983× |
| SMPL-X | 256 | 5.671885 | 30.813797 / 5.432726× | 26.117499 / 4.604730× |

Every native CUDA batch-16/256 result is more than 2× upstream. The pose-only
stage split identifies Torch FK as the bottleneck: FK measures
16.791555/19.648005 ms for SMPL and 43.724548/22.255063 ms for SMPL-X at
batch 16/256, while correctives are at most 0.494661 ms and transforms plus
LBS at most 3.056215 ms. Isolated stage medians are diagnostic and not
additive.

BetterRobot's optional Warp 1.15.0 lane changes only the pre-baked pose-only
FK. Its matching-invocation upstream medians and stage split were:

| Model | Batch | Original (ms) | BetterHuman pose-only (ms / ×ref) | Warp FK (ms) | Corrective (ms) | Transforms + LBS (ms) |
|---|---:|---:|---:|---:|---:|---:|
| SMPL | 1 | 1.855876 | 2.668455 / 1.437841× | 0.438639 | 0.316354 | 0.779355 |
| SMPL | 16 | 1.237468 | 1.810249 / 1.462866× | 0.391695 | 0.298544 | 0.784786 |
| SMPL | 256 | 3.149287 | 4.346257 / 1.380076× | 0.767779 | 0.486483 | 2.597154 |
| SMPL-X | 1 | 3.152521 | 2.488716 / 0.789437× | 0.678249 | 0.431269 | 1.052793 |
| SMPL-X | 16 | 3.401570 | 2.721118 / 0.799960× | 0.697566 | 0.451804 | 1.161303 |
| SMPL-X | 256 | 5.668683 | 5.212826 / 0.919583× | 0.631048 | 0.457872 | 3.086571 |

Warp pose-only is within 1.47× of upstream throughout and faster for all
SMPL-X batches in this run. The fair shape+pose path retains Torch FK because
its joint placements carry the shape batch; it remains 7.41–13.17× upstream
at CUDA batch 16/256. This is a reported BetterRobot seam, not a reason to
duplicate FK in BetterHuman. Full tables and agreement values are in
`plan/for_agents/03_results.md`; the reproducible harness is
`benchmarks/benchmark_smpl.py`.

## MHR performance

The LOD1 reference TorchScript recomputes shape on every call. BetterHuman is
therefore reported both as **pose-only** (one scalar shape baked before timing,
then batched `q`) and **shape+pose** (`with_shape` plus `vertices(q)` each
call). The latter is the fair end-to-end comparison.

These are CPU float32 medians over 30 calls after five warmups on a dual-socket
Intel Xeon Platinum 8570 with PyTorch 2.13.0, BetterRobot 0.2.0, and 112
intra/inter-op threads:

| Batch | Correctives | TorchScript shape+pose (ms) | BetterHuman pose-only (ms / ×ref) | BetterHuman shape+pose (ms / ×ref) |
|---:|:---:|---:|---:|---:|
| 1 | on | 19.505918 | 104.505661 / 5.357639× | 175.038120 / 8.973590× |
| 1 | off | 10.832997 | 83.003433 / 7.662093× | 156.004285 / 14.400843× |
| 16 | on | 43.595248 | 32.781757 / 0.751957× | 61.089930 / 1.401298× |
| 16 | off | 38.767081 | 51.125402 / 1.318784× | 101.672627 / 2.622654× |
| 256 | on | 767.727395 | 178.638341 / 0.232685× | 270.257807 / 0.352023× |
| 256 | off | 860.573079 | 141.988578 / 0.164993× | 247.239801 / 0.287297× |

At batch 256, inverse-bind construction plus matrix-blend LBS measures
85.244686/84.558726 ms with correctives on/off, down from the prior
1427.753313/1346.318814 ms formulation. FK is 56.038709/55.463856 ms and the
corrective stage is 24.450985/0.048177 ms. Isolated stage medians are diagnostic
and not additive. At batch 1, the roughly 42 ms BetterRobot Python/topological
FK traversal remains material. Shared-host load made the small-batch stage
medians visibly noisy, so these exact values are the recorded run rather than a
general hardware claim.

CUDA was measured on an otherwise idle NVIDIA RTX 6000 Ada Generation
(48 GiB, compute capability 8.9, driver 560.35.03) with the same warmup and
sample counts, explicit synchronization, and PyTorch 2.13.0+cu126. The native
Torch FK lane measured:

| Batch | Correctives | TorchScript shape+pose (ms) | BetterHuman pose-only (ms / ×ref) | BetterHuman shape+pose (ms / ×ref) |
|---:|:---:|---:|---:|---:|
| 1 | on | 3.549979 | 95.193920 / 26.815347× | 190.865683 / 53.765299× |
| 1 | off | 2.275135 | 70.019388 / 30.775927× | 140.907083 / 61.933504× |
| 16 | on | 3.951922 | 73.523560 / 18.604507× | 146.560227 / 37.085810× |
| 16 | off | 2.384344 | 72.506513 / 30.409418× | 145.687805 / 61.101840× |
| 256 | on | 12.171629 | 79.546592 / 6.535411× | 155.047731 / 12.738454× |
| 256 | off | 9.428837 | 78.125182 / 8.285771× | 153.348507 / 16.263777× |

At batch 256, native Torch FK still accounts for 73.015869/72.660426 ms,
while inverse-bind construction plus matrix-blend LBS is only
5.103035/5.102114 ms. The native lane therefore remains FK-bound.

BetterRobot's optional Warp 1.15.0 lane fuses pose-only FK. Its within-run
comparison and isolated split were:

| Batch | Correctives | TorchScript (ms) | BetterHuman pose-only (ms / ×ref) | Warp FK (ms) | Corrective (ms) | Transforms + LBS (ms) |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | on | 2.191199 | 2.565200 / 1.170683× | 0.632403 | 0.945547 | 0.578162 |
| 1 | off | 1.120560 | 1.062819 / 0.948471× | 0.615940 | 0.010177 | 0.565254 |
| 16 | on | 2.516612 | 2.263239 / 0.899320× | 0.658878 | 0.977999 | 0.771719 |
| 16 | off | 1.236523 | 1.306156 / 1.056314× | 0.642872 | 0.009840 | 0.749222 |
| 256 | on | 11.200383 | 8.333692 / 0.744054× | 1.688651 | 1.925943 | 5.245006 |
| 256 | off | 9.071364 | 6.679759 / 0.736357× | 1.596782 | 0.009864 | 5.182707 |

The Warp pose-only path now meets the `≤2×` target in both corrective modes at
batch 16 and batch 256. At batch 256 it is faster than the reference in this
run, with matrix-blend LBS measuring 5.245006/5.182707 ms instead of the prior
30.067871/27.737855 ms. Batched shape baking cannot currently use Warp because
BetterRobot requires matching batched joint and frame tables; the
apples-to-apples shape+pose path therefore remains on Torch FK and misses the
target at both CUDA batches. This is a reported BetterRobot seam, not a reason
to duplicate FK inside BetterHuman. The reproducible harness is
`benchmarks/benchmark_mhr.py`.

## Visualization

The BR viewer has no deforming-mesh mode; its viser backend can upload
arbitrary meshes (`add_mesh`). `viz` adds a skinned-mesh render mode that
re-uploads LBS-posed vertices per frame, next to BR's skeleton mode. No
BetterHuman rendering stack.

## Package layout (v1)

```text
src/better_human/
├── __init__.py        # SMPL, SMPLX, MHR, load, ParametricBody
├── core/
│   ├── body.py        # ParametricBody, BodyStructure, BodyValues
│   ├── pipeline.py    # blend shapes, joint regression, correctives, LBS
│   ├── reduction.py   # generic partial-model reduction (joints/columns)
│   ├── mass.py        # closed part meshes and shape-dependent inertias
│   └── classic.py     # shared converter algebra (pelvis-offset, axis-angle<->quat)
├── bodies/
│   ├── __init__.py      # stable public body and classic-value exports
│   ├── registry.py      # load("smplx", ...) — decorator-registered factories
│   ├── config_body.py   # BodyConfig dataclass (per-model static metadata)
│   ├── assets/          # authored segmentation + density tables + loaders
│   ├── smpl_family/
│   │   ├── __init__.py      # family exports
│   │   ├── config_smpl.py   # SMPL names, landmarks, asset filenames
│   │   ├── config_smplx.py  # SMPL-X names, reduction groups, assets
│   │   ├── shared.py        # shared values, robot, mass, and density binding
│   │   ├── smpl.py          # SMPL schema, converters, and spawn flags
│   │   └── smplx.py         # SMPL-X schema, converters, and spawn flags
│   └── mhr/
│       ├── __init__.py      # family exports
│       ├── config.py        # LODs, units, axes, and archive pattern
│       ├── archive.py       # archive tensors, metadata, bases, and segmentation
│       ├── tree.py          # authored-to-BetterRobot kinematic-tree binding
│       ├── deformation.py   # shape bake and MLP pose correctives
│       └── body.py          # MHR loader and classic boundary
├── io/
│   ├── schema.py      # per-model npz field schemas, validation
│   ├── load.py        # path resolution (arg > env root), validated loading
│   └── convert_chumpy.py  # CLI: original .pkl -> clean .npz (pickle-stub shim)
└── viz/
    └── skinned_mesh.py    # BR viewer render mode for deforming vertices
```

## Verification

`smplx` (the MPI package) is the correctness oracle for the SMPL family. MHR
uses a pinned `pymomentum` tools environment, with the released LOD1
TorchScript model as an independent secondary oracle. Parity is enforced
through **golden files**: a pinned environment under `tools/parity/` runs the
original implementation once and dumps vertices/joints for a fixed grid of
shapes/poses (including the known traps:
hand mean pose, SMPL-X jaw+eyes-before-hands ordering, expression columns);
our tests compare against the goldens at documented tolerances and skip
cleanly when local weights are absent. Converter round-trips
(`from_classic ∘ to_classic = id`) and float32 gradchecks through
shape groups and `q` to vertices complete the safety net. These tests are
sacred in the BetterRobot sense: never weakened to land a change.
