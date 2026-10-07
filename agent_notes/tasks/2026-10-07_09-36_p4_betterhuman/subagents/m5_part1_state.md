# M5 part 1 hand-off: state / posing / gizmo / layers (PARTIAL: tests only partly ported)

## Public API (src/better_human/viz/studio/)
- state.py: `StudioSession(body: Body, model_key: str, frame_set=None)`; attrs `body, model_key, identity: Identity,
  shape: dict[str,(n,)], handles: tuple[JointHandle], q (nq,), state: HumanState, vertices (V,3), joint_poses (J,7)
  [x y z qx qy qz qw], vertex_native_joints (V,), part_ids (V,), part_names, frame_set: dict[str, FrameEntry]`.
  Methods: `load_body(body, model_key, frame_set=None, *, notify=True)`, `set_q(q)`, `rotate_joint(handle, world_old, world_new)`,
  `move_root(handle, world_old, world_new)`, `reset_pose()`, `set_shape(**groups)` (re-bakes `identity`),
  `shape_tensor(group, values)`, `total_mass` (property; None when loaded with mass=False),
  `set_segmentation(names, ids)`, `reset_segmentation()`, `set_frame_set(d)`, `set_frame_joint(name, joint|None)`,
  `resolved_joint(entry)`, `subscribe(cb)`. Change kinds: pose|shape|body|segmentation|frames.
  `FrameEntry`, `compact_segmentation` unchanged. RENAMED: old `native_poses` -> `joint_poses`; old `data` -> `state`.
- posing.py: `joint_handles(identity: Identity)` (was `body`); `JointHandle`, `apply_local_rotation` unchanged.
  SMPL path chosen when `body.native_joint_ids == arange(1, J+1)`, else MHR channel walk (MHR NOT tested).
- gizmo.py: `GizmoController(viewer, session, *, on_status=None, scale=0.15)`; `select(joint|None)`, `clear()`,
  `handle_for(joint)`; attrs `joint`, `gizmo` (PoseGizmo or None). Non-root: rotation-only; root free_flyer: 6-DoF.
  No handle -> `on_status("joint 'x' has no degree of freedom")`. Subscribes to session (pose/shape -> moves gizmo; body -> clear).
- layers.py: `SegmentationLayer(name, session, *, opacity)` (`rebuild()`, `vertices (1,V,3)` in place, `face_groups()`),
  `PartsLayer(name, session, *, explode)` (per-part open submesh of the posed mesh's faces, rigid by dominant joint;
  NOT closed meshes from build_part_topology), `part_colors(P)`, `GROUP="human"`,
  `StudioLayers(viewer, session, *, theme=DARK)`: attrs `view` (HumanView: mesh + frames if contact frames; skeleton=False),
  `skeleton` (own SkeletonLayer, (1,J,3) colours), `segmentation`, `parts`, `edit` (PointsLayer), `layers: dict[name, Layer]`.
  Methods: `set_visible(name, bool)`, `set_highlight(joint|None)`, `set_opacity(x)`, `set_radius(r)`, `set_explode(x)`,
  `set_edit_vertices(ids)`, `rebuild()`. Default visibility: mesh/skeleton/frames on, segmentation/parts/edit off.
  Subscribes: pose/shape -> in-place arrays + viewer.show(); segmentation -> SegmentationLayer.rebuild + parts rebuild;
  body -> remove group "human" and re-add.

## What part 2 must do
- `viewer = Viewer(theme=replace(DARK, up="+y"), port=..., server=...)`; `session = StudioSession(bh.load(...), key)`;
  `layers = StudioLayers(viewer, session, theme=viewer theme)`; `gizmo = GizmoController(viewer, session, on_status=...)`.
- Click handler (not written here): `viewer.server.scene.on_pointer_event(event_type="click")`; ray vs
  `session.state.joints` spheres (radius `layers.radius`) first, then mesh `ray_hit`; then `gizmo.select(j)` +
  `layers.set_highlight(j)`. Subscribe order: StudioLayers before GizmoController (both already subscribe in ctor).

## Tests
- tests/viz/test_studio_state.py rewritten: 2 tests (session pose/shape/notify/frames; gizmo+layers on FakeServer) PASS.
- tests/viz/fake_server.py copied verbatim from BR (FakeHandle does not expose ctor kwargs as attrs: `gizmo.pose` raises).
- NOT DONE (budget): tests/viz/test_studio_posing.py, test_studio_gizmo.py, test_studio_layers.py still hold the OLD
  API and fail at import; port them (MHR handles, root move, opacity/radius/explode, segmentation rebuild, rotation-only options).
- ruff check + format clean on touched files; tests/contract 2 passed. `import ...studio.state` does not import app.py
  (`__init__.py` already lazy, untouched).
- Trashed: viz/studio/layers/, viz/studio/picking.py, tests/viz/test_studio_picking.py (-> repo trash/).

## Open issues
- `set_highlight` writes BR `SkeletonLayer._colors` (private attribute; no public setter in BR).
- PartsLayer explode/visibility untested visually; opacity/radius changes rebuild the whole group (simple, slower).
