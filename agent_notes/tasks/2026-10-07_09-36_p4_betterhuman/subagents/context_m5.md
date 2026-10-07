# Context for M5 subagents: HumanStudio on the BetterRobot viewer

Repo: `/home/rikhat.akizhanov/better/dev/BetterHuman` (git worktree, branch `dev`). Run everything from that
directory with `CUDA_VISIBLE_DEVICES="" uv run ...` (CPU only). BetterRobot is the editable dependency
`/home/rikhat.akizhanov/better/dev/BetterRobot` (its `dev` worktree): read it, never edit it. No new dependencies
(viser is installed). Never `rm`: `python3 ~/.claude/skills/autoresearch/scripts/trash.py <paths>`. Do not commit.
Never bind a server to `0.0.0.0` (tests use a fake server; the one smoke test may use `127.0.0.1`). Never skip,
xfail or loosen a test to get green; report what fails. Models: set `BETTERHUMAN_MODELS_DIR=$PWD/models`
(SMPL `models/smpl/SMPL_NEUTRAL.npz` etc., SMPL-X, MHR `models/MHR/converted/mhr_lod{0..6}.npz`).

## What exists

- New core (read `agent_notes/tasks/2026-10-07_09-36_p4_betterhuman/M1_DESIGN.md` §API first):
  `bh.load(family, path=None, *, mass=True, density=1000.0, contact_frames=None, stream_joints=True) -> Body`;
  `body.bake(**identity) -> Identity`; `identity.forward(q, stream=None) -> HumanState` with `vertices()`,
  `joints (J,3)`, `joint_poses (J,7)` (BR pose order `[x y z qx qy qz qw]`), `frame_poses()`, `kinematics`,
  `model` (the BR `Model`). `body.identity_groups`/`stream_groups` (name -> width), `body.faces`, `body.parents`,
  `body.joint_names`, `body.skin_matrix (V,J)`, `body.spec` (`FamilySpec`: `skin_weights/skin_indices (V,K)`,
  `parts`, `meta`), `body.contact_frames` (`ContactFrames` or None), `body.part_body_ids`, `body.native_joint_ids`.
  Code: `src/better_human/core/{body,spec,rig,contact_frames,mass}.py`, `src/better_human/__init__.py`.
- Mass helpers kept from the old core: `core/mass.py` (`segment_vertices`, `build_part_topology`,
  `meshes_from_topology`, `body_inertias_from_shape`, `_majority_face_parts`), `better_robot.spatial.Inertia`.
- Human layers on the BR viewer: `src/better_human/viz/human_layers.py` (`mesh_layer`, `skeleton_layer`,
  `frames_layer`, `human_layers`, `HumanView` with `update(state)` + `viewer.show()`).
- BR viewer (`better_robot.viewer`): `Viewer(host="127.0.0.1", port, theme, server=None)`; `viewer.add(layer,
  group)`, `viewer.show()`, `viewer.set_visible(layer, bool)`, `viewer.remove_group(group)`, `viewer.server`
  (the viser server: GUI via `viewer.server.gui`, scene via `viewer.server.scene`). `Layer` base class
  (`build(scene, root)`, `set_frame(k)`, `_track(handle)`, `set_visible`, `dispose`). Built-ins: `MeshLayer`
  (one colour), `SkeletonLayer` (per-joint `colors (J,3)` and radius), `FramesLayer`, `PointsLayer`,
  `SpheresLayer`. Theme: `replace(DARK, up="+y")` for y-up humans. Tools: `better_robot.viewer.tools`
  (`PoseGizmo(viewer, name, pose, on_change=, scale=, **transform_controls_options)` — `disable_sliders=True,
  disable_axes=True` gives a rotation-only handle; picking `CameraFrustum.from_camera(camera)`, `ray_hit`,
  `vertices_in_sphere`, `vertices_in_rect`, `project`, all numpy). Read `viewer/CLAUDE.md` and
  `docs/concepts/viewer.md` in BR. Test fake server: `/home/rikhat.akizhanov/better/dev/BetterRobot/tests/viewer/fake_server.py`
  (copy what you need into `tests/viz/fake_server.py` in BH; BH tests cannot import BR tests).
- Old studio (to be replaced, all under `src/better_human/viz/studio/`): `state.py` (`StudioSession`, pure
  state hub with subscribe/notify), `posing.py` (`JointHandle`, `joint_handles`, `apply_local_rotation` — already
  on the public BR API; keep the maths), `gizmo.py` (`GizmoController` on raw viser transform controls),
  `picking.py` (torch copies of the BR picking tools), `layers/` (own viser layers: mesh, skeleton, frames,
  parts), `segment_editor.py`, `frames_editor.py`, `persistence.py`, `app.py` (`HumanStudio`, `launch`),
  `__main__.py`. It imports the trashed `better_human.bodies` / `ParametricBody`, so nothing in it imports today.
  Feature list to preserve, verbatim: `docs/studio.md`. Old tests: `tests/viz/test_studio_*.py` (65 tests, same
  import failure).

## Target design (decided; do not reopen)

- The studio is a client of the BR `Viewer`: it creates `Viewer(theme=replace(DARK, up="+y"), port=, server=)`,
  adds layers through `viewer.add`, pushes with `viewer.show()`, and builds its sidebar tabs with
  `viewer.server.gui`. Scene geometry comes only from BR `Layer` subclasses: the BH `HumanView` layers (mesh,
  skeleton, frames) plus two studio layers in `viz/studio/layers.py`: `SegmentationLayer` (one submesh per
  part + unassigned, `rebuild()` when the segmentation changes, in-place vertex update on pose change) and
  `PartsLayer` (closed part meshes posed rigidly by their dominant joint, explode slider). Trash
  `viz/studio/layers/` and `viz/studio/picking.py` (use BR picking).
- `StudioSession` (keep `state.py`, same responsibilities) holds `body: Body`, `identity: Identity`, `q`,
  identity-group values, segmentation (`part_names`, `part_ids (V,)`), frame set (`dict[str, FrameEntry]`), and
  caches `state: HumanState`, `vertices`, `joint_poses`. No viser import.
- Posing: `PoseGizmo` at the selected joint's world pose; rotation-only for non-root joints, 6-DoF for the
  root; `on_change` -> `session.rotate_joint` / `move_root` (existing maths). Joint selection by clicking: one
  scene-level `viewer.server.scene.on_pointer_event(event_type="click")` handler; its ray is tested against
  the joint spheres first (nearest joint within the sphere radius along the ray), then the mesh (`ray_hit`).
  Rectangle marquee: `on_pointer_event(event_type="rect-select")` + `vertices_in_rect`. Sphere brush:
  mesh hit + `vertices_in_sphere`. Selected joint highlight: `SkeletonLayer` per-joint colours rewritten in
  place.
- Loading: `bh.load(family, path, mass=True, density=..., contact_frames=...)`; topology key via
  `persistence.model_key`. Density table for SMPL from `families/assets` (`load_density_table`). Total mass
  from the baked model's inertias (`identity.model.values.inertias` -> `Inertia.mass`; check the BR API).
- Persistence unchanged in behaviour; "Export as config" renders a `contact_frames={...}` mapping snippet for
  `bh.load(...)` instead of the old `BodyConfig` tuples.
- Tests: port the 65 old tests to the new API on the fake server; same coverage areas (state, posing, gizmo,
  layers, persistence, app headless). Keep the count close; drop a test only when the thing it tested no
  longer exists (say which).
- Style: ruff (line length 120), `ruff format`, type hints, docstrings with shapes, no narrating comments.
  `better_human` must not import viser outside `viz/studio` construction paths (contract test
  `tests/contract/test_br_imports.py` also forbids private BR imports: anything under `better_robot.*._*`,
  `better_robot.io.builders`, `better_robot.io.ir`, `better_robot.data_model.*` except via `better_robot`).
