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

`benchmarks/benchmark_smpl.py` separates two uses that were previously mixed in one
ratio. **Workload A** recomputes shape and pose from supplied parameters on every call.
Both implementations therefore perform a full forward, so its ratio compares equivalent
computations. **Workload B** bakes shape once and streams poses. Upstream `smplx` 0.1.28
has no shape-amortized API, so its per-frame cost remains a full forward; this ratio
compares what each library costs for the same user workload, not matched FLOPs.

SMPL uses 10 betas. SMPL-X uses 16 betas, 10 expressions, full hands and face, natural
hand means, no hand PCA, and no face contour. The upstream SMPL-X call disables its
optional duplicate shaped-vertex output; static landmark and extra-joint work remains in
the full forward, with a reference instance constructed at each matching batch size.

The benchmark uses neutral float32 SMPL and SMPL-X fixtures, five warmups, 30 timed calls,
and synchronized CUDA timings. CPU used 112 intra/inter-op threads on the shared dual-socket
Intel Xeon Platinum 8570 host. CUDA used an NVIDIA RTX 6000 Ada Generation, PyTorch
2.13.0+cu126, PyTorch CUDA runtime 12.6, driver 560.35.03, BetterRobot 0.2.0, and
optionally Warp 1.15.0 (whose startup reported CUDA Toolkit 12.9). Shared-host scheduling
and concurrent load can affect every observational median, particularly small-batch and
launch-bound rows; the tables retain the recorded runs without selective reruns.

Every SMPL-family model, batch, device, and workload passes a pre-timing maximum-absolute
vertex-error gate of `1e-6 m`. The worst recorded error is
`5.9604644775390625e-7 m`.

### Forward: CPU

Workload A compares two full shape-plus-pose forwards:

| Model | Batch | Upstream (ms) | BetterHuman (ms) | × upstream |
|---|---:|---:|---:|---:|
| SMPL | 1 | 5.747977 | 12.415018 | 2.159893 |
| SMPL | 16 | 1.770128 | 8.117990 | 4.586105 |
| SMPL | 256 | 24.879647 | 32.571483 | 1.309162 |
| SMPL-X | 1 | 1.753289 | 13.675982 | 7.800185 |
| SMPL-X | 16 | 3.085485 | 10.912850 | 3.536834 |
| SMPL-X | 256 | 41.908221 | 59.441630 | 1.418376 |

Workload B compares upstream's full per-frame forward with BetterHuman's shape-baked
`vertices(q)`:

| Model | Batch | Upstream (ms) | BetterHuman (ms) | × upstream |
|---|---:|---:|---:|---:|
| SMPL | 1 | 5.747977 | 6.349140 | 1.104587 |
| SMPL | 16 | 1.770128 | 7.358080 | 4.156808 |
| SMPL | 256 | 24.879647 | 25.725491 | 1.033997 |
| SMPL-X | 1 | 1.753289 | 13.128313 | 7.487818 |
| SMPL-X | 16 | 3.085485 | 11.918368 | 3.862721 |
| SMPL-X | 256 | 41.908221 | 55.133575 | 1.315579 |

### Forward: native CUDA

Workload A keeps the differentiable native Torch FK lane:

| Model | Batch | Upstream (ms) | BetterHuman (ms) | × upstream |
|---|---:|---:|---:|---:|
| SMPL | 1 | 1.999315 | 21.644656 | 10.826039 |
| SMPL | 16 | 2.123617 | 23.946178 | 11.276126 |
| SMPL | 256 | 3.261280 | 25.821618 | 7.917632 |
| SMPL-X | 1 | 1.991998 | 23.727799 | 11.911555 |
| SMPL-X | 16 | 1.909512 | 24.667794 | 12.918376 |
| SMPL-X | 256 | 4.279208 | 28.300942 | 6.613593 |

Workload B also uses native Torch FK here:

| Model | Batch | Upstream (ms) | BetterHuman (ms) | × upstream |
|---|---:|---:|---:|---:|
| SMPL | 1 | 1.999315 | 21.990452 | 10.998996 |
| SMPL | 16 | 2.123617 | 23.083747 | 10.870011 |
| SMPL | 256 | 3.261280 | 24.895283 | 7.633592 |
| SMPL-X | 1 | 1.991998 | 23.325769 | 11.709732 |
| SMPL-X | 16 | 1.909512 | 24.214457 | 12.680966 |
| SMPL-X | 256 | 4.279208 | 27.656212 | 6.462927 |

### Forward: CUDA with Warp requested

Shape-batched joint placements keep Workload A on Torch FK even when Warp is requested:

| Model | Batch | Upstream (ms) | BetterHuman (ms) | × upstream |
|---|---:|---:|---:|---:|
| SMPL | 1 | 2.004711 | 18.611841 | 9.284052 |
| SMPL | 16 | 1.736296 | 19.081667 | 10.989867 |
| SMPL | 256 | 2.976521 | 20.883783 | 7.016172 |
| SMPL-X | 1 | 3.031563 | 40.573462 | 13.383678 |
| SMPL-X | 16 | 3.127746 | 26.067906 | 8.334406 |
| SMPL-X | 256 | 5.826313 | 55.559674 | 9.535991 |

Workload B can use Warp because shape is already baked:

| Model | Batch | Upstream (ms) | BetterHuman Warp (ms) | × upstream |
|---|---:|---:|---:|---:|
| SMPL | 1 | 2.004711 | 2.371145 | 1.182787 |
| SMPL | 16 | 1.736296 | 2.447046 | 1.409348 |
| SMPL | 256 | 2.976521 | 3.984193 | 1.338540 |
| SMPL-X | 1 | 3.031563 | 2.544034 | 0.839182 |
| SMPL-X | 16 | 3.127746 | 2.460619 | 0.786707 |
| SMPL-X | 256 | 5.826313 | 5.537430 | 0.950417 |

The batch-256 streaming stage medians show why the lanes differ. These isolated stage
medians are diagnostic and are not additive:

| Model | Lane | FK (ms) | Corrective (ms) | Transforms + LBS (ms) |
|---|---|---:|---:|---:|
| SMPL | CPU | 7.015952 | 0.978832 | 16.607948 |
| SMPL | native CUDA | 21.399667 | 0.529627 | 2.773611 |
| SMPL | Warp | 0.749190 | 0.442141 | 2.546454 |
| SMPL-X | CPU | 11.567204 | 1.961538 | 36.035786 |
| SMPL-X | native CUDA | 26.454930 | 0.449537 | 3.097895 |
| SMPL-X | Warp | 0.962954 | 0.705014 | 3.697341 |

### Gradients

The loss is `vertices.sum()`. In Workload A, BetterHuman differentiates `q` and every
shape group while upstream differentiates every supplied pose and shape tensor. In
Workload B, BetterHuman differentiates `q` after shape baking; upstream differentiates pose
inputs through its full forward. `F+B / F` means total forward-plus-backward time divided by
the separately measured forward time. It is not a backward-only median. Gradient forward
medians belong to the autograd timing protocol and need not equal the inference tables.

CPU Workload A:

| Model | Batch | Up F | Up F+B | Up F+B/F | BH F | BH F+B | BH F+B/F | BH F+B × up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMPL | 1 | 28.071696 | 64.525559 | 2.298598 | 18.242398 | 34.155343 | 1.872306 | 0.529330 |
| SMPL | 16 | 2.731242 | 8.996616 | 3.293965 | 16.611343 | 45.854965 | 2.760461 | 5.096913 |
| SMPL | 256 | 20.872322 | 51.779978 | 2.480796 | 33.466383 | 106.526944 | 3.183103 | 2.057300 |
| SMPL-X | 1 | 2.117856 | 5.734379 | 2.707633 | 15.242779 | 45.947761 | 3.014395 | 8.012684 |
| SMPL-X | 16 | 3.220175 | 10.879412 | 3.378515 | 19.508656 | 54.005480 | 2.768283 | 4.964007 |
| SMPL-X | 256 | 41.348561 | 112.348765 | 2.717114 | 73.301568 | 221.165732 | 3.017203 | 1.968564 |

CPU Workload B:

| Model | Batch | Up F | Up F+B | Up F+B/F | BH F | BH F+B | BH F+B/F | BH F+B × up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMPL | 1 | 4.242632 | 6.534811 | 1.540273 | 11.472444 | 36.634596 | 3.193269 | 5.606068 |
| SMPL | 16 | 2.788969 | 8.672000 | 3.109393 | 8.107002 | 25.599815 | 3.157741 | 2.952008 |
| SMPL | 256 | 20.357383 | 49.028289 | 2.408379 | 14.992769 | 66.829707 | 4.457463 | 1.363085 |
| SMPL-X | 1 | 2.623462 | 6.742190 | 2.569959 | 12.706206 | 40.014697 | 3.149225 | 5.934970 |
| SMPL-X | 16 | 3.843130 | 12.407327 | 3.228443 | 15.421431 | 46.431064 | 3.010814 | 3.742229 |
| SMPL-X | 256 | 51.169806 | 132.857832 | 2.596411 | 59.711080 | 213.446547 | 3.574656 | 1.606579 |

Native-CUDA Workload A:

| Model | Batch | Up F | Up F+B | Up F+B/F | BH F | BH F+B | BH F+B/F | BH F+B × up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMPL | 1 | 2.016026 | 7.362839 | 3.652154 | 21.694166 | 80.218205 | 3.697686 | 10.895010 |
| SMPL | 16 | 1.701471 | 6.466421 | 3.800489 | 22.375858 | 85.357652 | 3.814721 | 13.200138 |
| SMPL | 256 | 3.025510 | 6.961112 | 2.300807 | 22.398428 | 86.299966 | 3.852947 | 12.397440 |
| SMPL-X | 1 | 7.191468 | 17.394033 | 2.418704 | 65.165043 | 199.382611 | 3.059656 | 11.462702 |
| SMPL-X | 16 | 7.867332 | 19.639098 | 2.496284 | 68.585478 | 210.025333 | 3.062242 | 10.694245 |
| SMPL-X | 256 | 9.423600 | 20.218570 | 2.145525 | 70.283626 | 213.747098 | 3.041208 | 10.571821 |

Native-CUDA Workload B:

| Model | Batch | Up F | Up F+B | Up F+B/F | BH F | BH F+B | BH F+B/F | BH F+B × up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMPL | 1 | 2.278359 | 7.392388 | 3.244610 | 21.974447 | 70.566171 | 3.211283 | 9.545788 |
| SMPL | 16 | 2.131151 | 6.941037 | 3.256942 | 20.352127 | 69.597196 | 3.419652 | 10.026916 |
| SMPL | 256 | 3.747643 | 8.268234 | 2.206249 | 20.909514 | 70.088864 | 3.352008 | 8.476885 |
| SMPL-X | 1 | 7.271823 | 18.171266 | 2.498860 | 64.120355 | 177.104108 | 2.762057 | 9.746382 |
| SMPL-X | 16 | 7.834374 | 18.430926 | 2.352571 | 66.612173 | 180.349366 | 2.707454 | 9.785150 |
| SMPL-X | 256 | 9.238328 | 18.957483 | 2.052047 | 68.362986 | 185.992442 | 2.720660 | 9.811030 |

Native-CUDA forward and backward both inherit BetterRobot's launch-bound Torch FK graph;
the backward gap is therefore not a separate pathology. The component attribution and
launch analysis live in [BR_wishlist.md §2](../BR_wishlist.md). Complete commands,
agreement rows, and raw tables are recorded in
[Order 02 results](../plan/for_agents/02_results.md).

## MHR performance

The released LOD1 TorchScript model is MHR's performance reference. It recomputes shape on
every call, so the same two workloads apply: Workload A is a full shape-plus-pose forward in
both implementations; Workload B compares its unavoidable full per-frame forward with
BetterHuman's shape-baked pose stream. Corrective-on and corrective-off rows remain separate.

MHR keeps its established, visible pre-timing gate at `5e-6 m` rather than tightening it to
one favorable run. The worst recorded CPU error is `2.7418136596679688e-6 m`; the worst
native or Warp CUDA error is `1.5497207641601562e-6 m`. The protocol and shared-host caveat
are the same as for the SMPL family.

### Forward: CPU

Workload A:

| Batch | Correctives | TorchScript (ms) | BetterHuman (ms) | × reference |
|---:|:---:|---:|---:|---:|
| 1 | on | 23.336780 | 165.512548 | 7.092347 |
| 1 | off | 10.762407 | 154.999911 | 14.401974 |
| 16 | on | 30.333932 | 69.517568 | 2.291743 |
| 16 | off | 26.158406 | 62.134325 | 2.375310 |
| 256 | on | 911.843281 | 206.352065 | 0.226302 |
| 256 | off | 787.022646 | 249.103795 | 0.316514 |

Workload B:

| Batch | Correctives | TorchScript (ms) | BetterHuman (ms) | × reference |
|---:|:---:|---:|---:|---:|
| 1 | on | 23.336780 | 60.168659 | 2.578276 |
| 1 | off | 10.762407 | 82.997890 | 7.711833 |
| 16 | on | 30.333932 | 37.420636 | 1.233623 |
| 16 | off | 26.158406 | 51.659231 | 1.974862 |
| 256 | on | 911.843281 | 136.900657 | 0.150136 |
| 256 | off | 787.022646 | 148.225082 | 0.188336 |

### Forward: native CUDA

Workload A:

| Batch | Correctives | TorchScript (ms) | BetterHuman (ms) | × reference |
|---:|:---:|---:|---:|---:|
| 1 | on | 2.252701 | 150.867692 | 66.971912 |
| 1 | off | 1.160139 | 150.326115 | 129.575952 |
| 16 | on | 2.574029 | 155.786447 | 60.522402 |
| 16 | off | 1.305433 | 313.584503 | 240.214843 |
| 256 | on | 11.318191 | 169.116997 | 14.942053 |
| 256 | off | 9.160384 | 161.538835 | 17.634505 |

Workload B:

| Batch | Correctives | TorchScript (ms) | BetterHuman (ms) | × reference |
|---:|:---:|---:|---:|---:|
| 1 | on | 2.252701 | 76.708944 | 34.051986 |
| 1 | off | 1.160139 | 74.335215 | 64.074404 |
| 16 | on | 2.574029 | 77.473613 | 30.098184 |
| 16 | off | 1.305433 | 135.909907 | 104.110939 |
| 256 | on | 11.318191 | 84.796631 | 7.492066 |
| 256 | off | 9.160384 | 164.078545 | 17.911754 |

### Forward: CUDA with Warp requested

Workload A remains on Torch FK:

| Batch | Correctives | TorchScript (ms) | BetterHuman (ms) | × reference |
|---:|:---:|---:|---:|---:|
| 1 | on | 3.647041 | 193.928811 | 53.174296 |
| 1 | off | 2.357448 | 151.432239 | 64.235651 |
| 16 | on | 4.051382 | 156.430489 | 38.611632 |
| 16 | off | 2.483768 | 155.622264 | 62.655717 |
| 256 | on | 12.988791 | 164.781857 | 12.686466 |
| 256 | off | 10.341867 | 163.516999 | 15.811168 |

Workload B uses Warp FK:

| Batch | Correctives | TorchScript (ms) | BetterHuman Warp (ms) | × reference |
|---:|:---:|---:|---:|---:|
| 1 | on | 3.647041 | 2.369919 | 0.649820 |
| 1 | off | 2.357448 | 1.010115 | 0.428478 |
| 16 | on | 4.051382 | 2.230097 | 0.550453 |
| 16 | off | 2.483768 | 1.210952 | 0.487546 |
| 256 | on | 12.988791 | 8.292859 | 0.638463 |
| 256 | off | 10.341867 | 6.657831 | 0.643775 |

The batch-256 Workload B stage split shows Warp removing the FK launch cost:

| Correctives | Lane | FK (ms) | Corrective (ms) | Transforms + LBS (ms) |
|:---:|---|---:|---:|---:|
| on | CPU | 59.337605 | 24.533667 | 81.591606 |
| off | CPU | 57.114863 | 0.039530 | 84.504691 |
| on | native CUDA | 80.182771 | 1.629461 | 5.207095 |
| off | native CUDA | 77.646215 | 0.009222 | 5.125349 |
| on | Warp | 1.551654 | 1.638420 | 5.136963 |
| off | Warp | 1.545492 | 0.009011 | 5.137304 |

### Gradients

MHR has no upstream gradient row. Workload A differentiates `q` and all shape groups;
Workload B differentiates `q` with shape baked outside timing. Both use Torch FK and Torch
autograd. As above, `F+B / F` is total forward-plus-backward divided by forward-only, not a
backward-only measurement.

CPU:

| Batch | Correctives | Workload | Forward (ms) | F+B (ms) | F+B / F |
|---:|:---:|:---:|---:|---:|---:|
| 1 | on | A | 212.369610 | 446.482271 | 2.102383 |
| 1 | on | B | 130.829949 | 336.132665 | 2.569233 |
| 16 | on | A | 124.484499 | 298.137320 | 2.394975 |
| 16 | on | B | 62.761629 | 169.143155 | 2.695009 |
| 256 | on | A | 255.153583 | 656.606512 | 2.573378 |
| 256 | on | B | 158.601529 | 490.848457 | 3.094853 |
| 1 | off | A | 189.611799 | 435.343932 | 2.295975 |
| 1 | off | B | 101.297063 | 261.008742 | 2.576666 |
| 16 | off | A | 174.503453 | 393.588206 | 2.255475 |
| 16 | off | B | 51.644019 | 147.767241 | 2.861265 |
| 256 | off | A | 329.645655 | 722.676883 | 2.192284 |
| 256 | off | B | 184.607942 | 510.086399 | 2.763079 |

Native CUDA:

| Batch | Correctives | Workload | Forward (ms) | F+B (ms) | F+B / F |
|---:|:---:|:---:|---:|---:|---:|
| 1 | on | A | 207.695002 | 581.920783 | 2.801804 |
| 1 | on | B | 102.178812 | 384.318314 | 3.761233 |
| 16 | on | A | 310.976191 | 671.553823 | 2.159502 |
| 16 | on | B | 208.497832 | 545.357878 | 2.615653 |
| 256 | on | A | 376.295924 | 617.513200 | 1.641031 |
| 256 | on | B | 181.165978 | 363.791832 | 2.008058 |
| 1 | off | A | 349.414554 | 584.243813 | 1.672065 |
| 1 | off | B | 172.510312 | 363.918961 | 2.109549 |
| 16 | off | A | 354.854490 | 738.743698 | 2.081821 |
| 16 | off | B | 105.754499 | 393.191090 | 3.717961 |
| 256 | off | A | 218.495971 | 603.712264 | 2.763036 |
| 256 | off | B | 107.715235 | 398.367353 | 3.698338 |

The MHR backward multipliers do not reveal a separate backward-only anomaly; they inherit
the same launch-bound FK graph described in [BR_wishlist.md §2](../BR_wishlist.md). The
official Warp run is forward-only. Combining `--use-warp --grad` remains supported as an
explicitly labeled hybrid: Warp forward with a Torch-recompute VJP. Complete commands,
agreement rows, and raw tables are in
[Order 02 results](../plan/for_agents/02_results.md).

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
