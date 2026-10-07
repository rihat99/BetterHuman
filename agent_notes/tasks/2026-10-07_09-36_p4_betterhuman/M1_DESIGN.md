# M1 design — families as data, identity baked once, streams per frame

Written 2026-10-07 after M0. This is the contract every M1–M5 agent codes against.

## M0 facts

- BH tests vs BR dev (`CUDA_VISIBLE_DEVICES="" uv run pytest tests -q`): every `tests/viz/*` module fails at
  import (`better_robot.viewer.RenderContext` is gone; `viz/__init__.py` imports `skinned_mesh.py`). Outside
  `viz`: 238 passed, 1 failed (`tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`,
  exact-equality `atol=0` on a float32 matmul; 7.6e-6 off since the torch relock `793d716`, not P1–P3).
- Assets are in the main checkout's gitignored `models/`; the worktree has a symlink `models ->
  /home/rikhat.akizhanov/better/BetterHuman/models` (gitignored). Present: SMPL (npz, 3 genders + pkl), SMPL-X
  (npz + pkl, 3 genders, + lockedhead), SMPL-H (`smplh/smplh/{neutral,male}/model.npz` 52 joints, shapedirs 16;
  female only as pkl), MANO (`MANO_{LEFT,RIGHT}.npz`, 16 joints, posedirs 135, hands_components 45x45),
  MHR (`MHR/converted/mhr_lod{0..6}.npz`, torchscript oracle `MHR/assets/mhr_model.pt`), STAR, DMPL. **No FLAME**.
- BH imports of BR privates: `io.builders.kinematic_tree.build_kinematic_tree_model`, `io.ir.IRFrame`
  (smpl_family/shared.py), `data_model.reduced_coordinates.expand_configuration` (mhr/deformation.py),
  `data_model.model_structure.JOINT_KIND_CODES` (studio/posing.py), `viewer.RenderContext/ViserBackend/
  renderers.testing.MockBackend` (viz, examples), `tasks.solve_ik` (example 03), `Data` (everywhere).
  Public replacements: `ModelBuilder` + `io.build_model(ir, preserve_joint_order=True)`, `Model.q_expansion/
  q_offset` (a public `Model.expand_configuration(q)` is added to BR), `joint_models[j].kind`, the new viewer,
  `terms.solve_ik`, `kinematics.forward_kinematics_state`.

## Package layout after M1

```
better_human/
  __init__.py        Body, Identity, HumanState, FamilySpec, load(name, ...), families
  core/
    spec.py          FamilySpec, Corrective (data only), drop_joints()
    rig.py           Rig protocol; RegressedRig (SMPL family); AuthoredRig lives in families/mhr_rig.py
    body.py          Body, Identity, HumanState
    pipeline.py      pure tensor ops (kept: blend_shapes, regress_joints, features, lbs, ...)
    classic.py       axis-angle/quaternion/root converters (kept)
    reduction.py     reduce_joint_data, slice_parameter_basis (kept)
    mass.py, contact_frames.py (kept)
  families/
    __init__.py      smpl, smplh, smplx, mano, flame, mhr loaders + registry `load(name, ...)`
    smpl_family.py   one loader for SMPL/SMPL-H/SMPL-X/MANO/FLAME driven by a FamilyTable
    tables.py        FamilyTable per family (joint names, landmark ids, subtree index groups, asset names)
    classic_smpl.py  SMPLClassic/SMPLHClassic/SMPLXClassic/MANOClassic/FLAMEClassic + from_classic/to_classic
    mhr.py           MHR loader (archive + tree + authored rig), MHRClassic converters
  io/                unchanged (schema, load, convert_chumpy) + FLAME/MANO/SMPL-H schemas
  viz/               M4/M5
```

`bodies/` is trashed at the end of M1. `better_human.terms` (M3), `viz.human_layers` (M4), studio (M5).

## Data

```python
@dataclass(frozen=True)
class Corrective:                     # pose-dependent vertex offsets, data only
    kind: Literal["none", "linear", "mlp"]
    posedirs: Tensor | None           # linear: (P, V*3)
    first_layer: Tensor | None        # mlp: (H, P) dense
    output: Tensor | None             # mlp: (H, V*3)

@dataclass(frozen=True)
class FamilySpec:
    name: str                         # "smpl" | "smplh" | "smplx" | "mano" | "flame" | "mhr"
    vertices: Tensor                  # (V, 3) template
    faces: Tensor                     # (F, 3) long
    joint_names: tuple[str, ...]      # J native skin joints, parent-before-child
    parents: tuple[int, ...]          # -1 for the root
    skin_weights: Tensor              # (V, K) top-K, rows sum to one
    skin_indices: Tensor              # (V, K) long
    identity: Mapping[str, Tensor]    # bake-time vertex groups, (V, 3, n) each: {"betas": ...} / {"identity": ...}
    streams: Mapping[str, Tensor]     # per-frame vertex groups, (V, 3, n): {"expression": ...}
    corrective: Corrective
    rig: Rig                          # how identity -> joint placements and q -> corrective features
    landmarks: Mapping[str, Tensor]   # name -> (L,) long vertex ids ("keypoints" for OpenPose-style sets)
    unit: str; up_axis: str; forward_axis: str
    meta: Mapping[str, Any]           # family extras (hand PCA components/means, MHR bindings); loaders own it
```

`Rig` is the only behaviour in the spec and has two implementations:

```python
class Rig(Protocol):
    def build_model(self, spec, *, frames, dtype, device) -> Model          # unbaked BR model, joint order = spec order
    def bake(self, spec, model, identity_values, v_shaped) -> RigBake       # placements (..., nj, 7), rest_joints (..., J, 3),
                                                                            # pelvis (..., 3), joint_scales (..., J) | None
    def stream_joint_basis(self, spec, name) -> Tensor | None               # (J, 3, n): joint motion of a stream group
    def native_joint_ids(self, model) -> Tensor                             # (J,) BR joint ids in native order
    def corrective_features(self, model, q) -> Tensor                       # (..., P)
```

`RegressedRig(joint_regressor (J, V))`: free-flyer root + spherical joints, pelvis-centred, `stream_joint_basis =
J_regressor @ basis`, features `(R_j - I)` of the non-root spherical joints (`q[..., 7:]`).
`AuthoredRig` (MHR): the existing tree builder (mimic spine rows), proportions -> scales and offsets, pre-rotations,
inverse binds (`meta["inverse_bind_poses"]`, `meta["bind_joint_positions"]`), features from `q_full` Euler channels.

## Objects

```python
class Body:                            # one family on one dtype/device, with optional mass and contact frames
    Body(spec, *, dtype=float32, device=None, mass=True, density=1000.0 | (J,) | "table-name",
         contact_frames=None, stream_joints=True)
    spec, model (unbaked br.Model), faces, num_vertices, joint_names, parents, part_ids (V,), contact_frames
    bake(**identity: (..., n)) -> Identity        # omitted groups are zero; leading dims broadcast
    identity_groups: Mapping[str, int]; stream_groups: Mapping[str, int]

@dataclass(frozen=True)
class Identity:                        # everything that depends on identity only; differentiable w.r.t. params
    body, params: Mapping[str, Tensor], v_shaped (..., V, 3), rest_joints (..., J, 3), pelvis (..., 3),
    model: br.Model (values baked: placements, inertias, contact-frame placements), joint_scales | None,
    stream_joint_basis: Mapping[str, Tensor (J, 3, n)]   # only streams that move joints
    forward(q, stream=None) -> HumanState
    nq, nv (from model)

@dataclass(frozen=True)
class HumanState:                      # one posed batch; lazy outputs, each computed at most once
    identity, q (..., nq), stream: Mapping[str, Tensor (..., n)], kinematics: KinematicState (BR), 
    rest_joints (..., J, 3)   # identity rest joints shifted by the stream joint deltas
    joint_poses -> (..., J, 7) world, native order
    joints -> (..., J, 3)
    skin_transforms -> (..., J, 7)  (T_world ∘ T_rest^-1)
    rest_vertices -> (..., V, 3)   v_shaped + streams + corrective
    vertices() -> (..., V, 3)
    points(vertex_ids (M,)) -> (..., M, 3)       # skins only M vertices (corrective sliced too)
    landmarks(name="keypoints") -> (..., L, 3)
```

Streams: `forward(q, stream={"expression": e (..., n)})`. Vertex effect: `basis @ e`. Joint effect (exact, Q8):
`delta = stream_joint_basis @ e` `(..., J, 3)`; `rest_joints_t = rest_joints + delta - delta[0]`; placements are
rebuilt from `rest_joints_t` (batched over the stream batch, BR FK takes batched placements); `pelvis_t = pelvis +
delta[0]` is what the classic root conversion uses. `Body(stream_joints=False)` skips the joint effect (fast,
approximate). Identity values (inertias, contact frames, mass) never change per frame.

Batching: `identity` leading dims `I...`, `q` leading dims `B...`; they broadcast right-aligned (BR execution
batch). A clip of one subject is `bake(betas (16,))` + `forward(q (T, nq), stream={"expression": (T, 10)})`.

## Classic parameters

Per family module: `XClassic` dataclass (unchanged fields), `from_classic(identity, classic) -> (q, stream)` and
`to_classic(identity, q, stream=None) -> XClassic`. MHR: classic carries shape, so `mhr.from_classic(body, classic)
-> (identity, q, stream)`. Expression is no longer "shape state": it goes into `stream`.

## Parity gate (unchanged thresholds)

SMPL-X goldens (`tests/parity/goldens/smplx_neutral/*.npz`, upstream smplx) 1e-6 float64/float32 on vertices and
joints; SMPL goldens; MHR goldens (pymomentum, 1.4e-6 m). The tests are rewritten to the new API; the numbers stay.

## Loader signatures

```python
families.smpl(path=None, *, gender="neutral", num_betas=10, dtype, device) -> FamilySpec
families.smplh(path=None, *, gender="neutral", num_betas=16, hands=True)        # 52 joints, no PCA in npz
families.smplx(path=None, *, gender="neutral", num_betas=16, num_expression=10, hands=True, face=True,
               flat_hand_mean=False)
families.mano(path=None, *, side="right", num_betas=10, flat_hand_mean=False)   # 16 joints
families.flame(path=None, *, gender="generic", num_betas=100, num_expression=50)  # 5 joints; asset absent here
families.mhr(path=None, *, lod=1, expression=True, correctives=True)
load(name, **options) -> Body   # convenience: families.<name>(**loader options) + Body(**body options)
```

`path=None` resolves `$BETTERHUMAN_MODELS_DIR/<family>/<file>`. Partial bodies (`hands=False`, `face=False`) drop
terminal subtrees at load (`core.spec.drop_joints`).
