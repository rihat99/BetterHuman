# 0020 The public names the `viz` and studio rewrites added

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
The `viz` rewrite (0010) and the studio move (0009) made names public that no accepted record lists: a reviewer
found them with an API diff of the two work packages and asked the owner to accept them or park them (open question
1 of the H5/H6 review). The names exist and are used: the studio reads files only through `io.header` and
`io.upgrade`, draws contact frames through `HumanState.contact_frame_poses`, and colours parts through
`PartTopology.face_parts`; the examples and BetterVideoReconstruction use the `viz` layers.

## Decision
We will keep these names public, unchanged:

- `better_human.HumanState.contact_frame_poses(frames)`: `(..., N, 7)` world poses of a frame set the model was not
  built with, equal to the frames `HumanModel.from_spec(contact_frames=frames)` attaches.
- `better_human.io.header(format)`: the `format` / `format_version` fields a writer writes;
  `better_human.io.upgrade(fields, *, format, source)`: a file's fields migrated in memory to the latest version.
- `better_human.mass.PartTopology.face_parts(faces, part_ids)`: the part of every face (the part two of its vertices
  hold, else its first vertex's).
- `better_human.viz`: `HUMAN_THEME`, `HumanView`, `human_layers`, `mesh_layer`, `skinned_layer`, `skeleton_layer`,
  `frames_layer`, `forces_layer` (the module `better_human.viz.layers`).
- `better_human.studio.launch(*, data_root, generator, family, gender, lod, port, host, density)` and
  `better_human.studio.HumanStudio`, with its keyword-only constructor (the same arguments and `server`), its methods
  `run`, `locked`, `set_status`, `set_mode`, `run_action`, `load_model`, `reload`, `apply_shape`, `reset_shape`,
  `random_shape`, `chosen_density`, `save_pose`, `load_pose`, `on_click`, `start_marquee`, `pick_joint`,
  `select_joint`, `pick_frame`, `deselect_frame`, `set_opacity`, and its attributes `session`, `viewer`, `layers`,
  `gizmo`, `segment`, `frames`, `paths`, `generator`, `family`, `gender`, `lod`, `density`, `opacity`, `mode`,
  `status`, `actions`, `tabs` and the sidebar controls (`family_input`, `gender_input`, `lod_input`,
  `density_input`, `shape_sliders`, `pose_name`, `saved_poses`, `status_text`, `selection_text`, `mass_text`).

## Example
```python
import better_human as bh
from better_human.io import header, upgrade
from better_human.studio import launch

poses = state.contact_frame_poses({"heel": (3387, None)})                 # (..., 1, 7)
fields = upgrade(json.loads(path.read_text()), format="better_human.frame_set", source=str(path))
assert header("better_human.frame_set") == {"format": "better_human.frame_set", "format_version": 1}
launch(data_root="~/studio_data", generator=numpy.random.default_rng(0), family="smplx")
```

## Consequences
The API snapshot records these names, and changing one is a breaking change with a changelog entry. `HumanStudio`
is an application object: its methods are what its GUI controls call, and its attributes are its parts and its
sidebar controls, which its smoke test fires; a user scripts the studio through them, but they follow the GUI.

## Alternatives
Park them: make `contact_frame_poses`, `header`, `upgrade` and `face_parts` private and keep `HumanStudio`'s surface
to `run` plus the constructor. The studio, which may use public names only (0009), would then need a second, private
path to the same functions, and its smoke test could no longer reach the controls it fires.

## Compatibility
None: every name exists today; nothing is removed or renamed.

## Enforcement
`tests/contract/public_api.json` with `tests/contract/test_api_snapshot.py` (the snapshot of the `__all__` tree);
`tests/contract/test_studio_boundary.py` (the studio imports only names of a package `__all__`);
`tests/studio/test_app.py` (the smoke test fires every control); `tests/contract/test_keyword_only.py`.

## Open questions
Whether `HumanStudio`'s sidebar controls stay public attributes or move behind the methods that use them.
