# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each release groups its entries by change type;
every removal names its replacement. The [migration page](migration-0.2.0.md) maps every public name of the 0.1
line that 0.2.0 renames or removes, one table row each.

## Unreleased

## v0.2.0 (2026-11-10)

### Added
- `HumanModel.from_spec(spec, *, density="Dempster", contact_frames=None, stream_joints=True)`: a frozen body
  model built once from a `FamilySpec`, in the dtype and on the device its family loader chose.
- `from_classic(human, classic)` and `to_classic(identity, q, *, streams=None)` at the root: one conversion
  between native parameters and BetterHuman's, dispatched on the family's `FamilySpec.classic` (a
  `core.ClassicFormat`: `families.SMPLFamilyFormat`, `families.MHRFormat`, `families.SOMAFormat`).
- `core.Landmarks(names, vertex_ids)`, `core.LinearCorrective`, `core.MLPCorrective`, and the skinning functions
  `core.blend_shapes`, `core.linear_blend_skinning` and `core.skin_affines` exported from `core`.
- `FamilySpec.file_joint_ids`, `FamilySpec.classic`, `FamilySpec.provenance`; `Parts.density_categories`,
  `Parts.density_tables` and `Parts.closed_faces`.
- `Rig.without_joints`, the hook `families.drop_joints` asks a rig to reduce its data with.
- `families` exports `RegressedRig`, `MHRRig`, `SOMARig`, `drop_joints` and the seven `*Classic` records;
  `families.smplx` and `families.mano` take `num_pca_comps`.
- The `mass` package: `mass.PartTopology` (`from_closed_faces`, `from_part_ids`, `meshes`, `face_parts`),
  `mass.part_inertias`, `mass.part_densities` and `mass.segment_vertices`.
- `io.weight_path`, `io.models_dir`, `io.check_format`, `io.header`, `io.upgrade`, `io.read_frame_set`,
  `io.write_frame_set`, `io.shipped_frame_set` and the errors `io.FormatError` and `io.AssetNotFoundError`.
- Seven versioned file formats with their spec pages under `reference/formats`: `better_human.mhr` and
  `better_human.soma` (version 2), `better_human.frame_set`, `better_human.segmentation`,
  `better_human.density_tables`, `better_human.studio.segmentation` and `better_human.studio.pose` (version 1).
- `HumanState.contact_frame_poses(frames)`: the poses `from_spec(contact_frames=frames)` would give, without
  building that model.
- `terms.baked_identity(human, *groups)` and `terms.reprojection(..., joints=True)`.
- `viz.HUMAN_THEME`, BetterRobot's light theme with `+y` up.
- The `better_human.studio` package: `studio.HumanStudio`, `studio.launch` and
  `python -m better_human.studio --data-root DIR`, installed by the `studio` extra.
- Random-weight SMPL, SMPL-H and SMPL-X archives and their goldens in the repository, so the full test tier and the
  examples run without licensed weights.
- `tools/api_snapshot.py` and the tracked public API snapshot it checks.
- `HumanModel.part_joint_ids` `(P,)` and `RegressedRig.parents` `(J - 1,)`: index tensors built once on the spec's
  device, so an SMPL-family bake on CUDA copies no index from the host and fills no tensor from a Python scalar.
- [Release a version](../guides/release.md): the release checklist, from the BetterRobot pin to the tag.

### Changed
- `Identity` fields: `human`, `values`, `rest_vertices`, `rest_joint_translations`, `inverse_bind_affines`
  `(I..., J, 3, 4)` and `root_offset`; `Identity.forward(q, streams)`.
- `HumanState` fields: `streams`, `rest_joint_translations`, `root_offset`, `stream_offsets`; the joint positions
  are `HumanState.joint_translations`; `HumanState.landmarks()` takes no name.
- `FamilySpec` fields: `template_vertices`, `identity_bases`, `stream_bases`, `landmarks` (a `core.Landmarks` or
  `None`); `up_axis` is a literal `"+x"` to `"-z"`.
- `RigBake` fields: `rest_vertices`, `rest_joint_translations`, `inverse_bind_affines` and `root_offset`, all
  required.
- `core.Rig` is an abstract base and `core.Corrective` an abstract base of `core.LinearCorrective` and
  `core.MLPCorrective`; the rig hooks take their arguments by keyword; `Rig.build_model` builds in the spec's dtype
  and device. `HumanModel`, `Identity` and `HumanState` are frozen, compare by identity and print one line.
- `load(family, path=None, *, dtype, device, density, contact_frames, stream_joints)`: family options go through
  the family loader; the family name is exact, and an unknown one raises `UnknownNameError`.
- The root's `__all__` lists every subpackage, `core`, `families`, `io`, `mass`, `terms`, `viz` and `studio`, then
  its first-level names; `import better_human` imports the subpackages, and the studio loads viser only when
  `studio.HumanStudio` or `studio.launch` is read.
- The default density is the `"Dempster"` table, each part at its category's density (was 1000 kg/m³ for every
  part); `density=None` builds no inertias.
- The family loaders accept float32 and float64 only and raise `DtypeMismatchError` for other dtypes.
- The classic records are keyword-only with every field optional; the SMPL family's gained `betas`.
- Errors are BetterRobot's: `UnknownNameError` for an unknown family, identity group or stream group, and
  `ArgumentError`, `ShapeError`, `DtypeMismatchError` or `DeviceMismatchError` where a plain `ValueError`, a
  torch `RuntimeError` or a silent float32 promotion came before.
- `terms.joints(..., joint_names=)`, `terms.landmarks(*, target, state)`, `terms.human_state(..., streams=)`;
  `terms.ground(up=None)` points up along the family's up axis.
- `viz.skinned_layer(state, *, name, color, opacity)` derives its binds from `inverse_bind_affines` and raises
  for binds that scale or shear; `viz.forces_layer(state, frame_names, *, forces, ...)` takes frame names;
  `viz.HumanView` is final.
- `viz.HumanView.update` raises `ShapeError`, and writes nothing, for a clip or the state of a body of another size
  (numpy's `ValueError` came before, after some layers were written); `viz.forces_layer` raises `TypeError` for
  forces that are not a tensor.
- `studio.launch(*, data_root, generator, family, gender, lod, port, host, density)` takes `studio.HumanStudio`'s
  keyword arguments, typed (was `launch(**kwargs)`), and prints the URL viser serves.
- The studio replaces each segmentation and pose file it saves atomically, as `io.write_frame_set` does.
- MHR and SOMA weight archives are format version 2 with `format` and `provenance`; version 1 files are migrated in
  memory on load. Frame sets are versioned JSON; the shipped segmentation and density tables are versioned `.npz`
  archives.
- `$BETTERHUMAN_MODELS_DIR` is read only through `io.models_dir()`; the studio no longer sets it.
- The licensed parity goldens live in `$BETTERHUMAN_MODELS_DIR/goldens` and run in the release tier.
- Licence Apache-2.0 (was MIT); Python 3.12 or newer; `scipy` moved to the `dev` group; the `studio` extra is
  BetterRobot's `viewer` extra.
- `benchmarks/run.py` is the one benchmark runner: the full tier fails when a graph break, recompile, operator or
  host sync of its counter rows (SMPL-X vertices, the SMPL bake and vertices with and without part inertias, on the
  random-weight archives) exceeds `benchmarks/thresholds/torch-<version>.json`
  (`tests/contract/test_compile_counters.py`); `--licensed` adds the sweep of SMPL, SMPL-X, MHR and SOMA, MHR's
  TorchScript oracle and the load rows; wall time is recorded with the load average and page faults, never gated.
- `better_human.__version__` is read from the installed distribution's metadata, so it always equals the version
  in `pyproject.toml`; it was a literal in `__init__.py`.
- The hooks in `.pre-commit-config.yaml` are BetterRobot's but for the tier marker expressions: ruff check, ruff
  format --check and the fast tier (`-m "not slow and not cuda and not licensed"`) before a commit, the full tier
  before a push, all through `uv run --locked`. The release tier is one command, `bash tools/release_tier.sh --gpu
  <index>`, and runs the full tier a second time with BetterRobot's internal checks on. Ruff selects BetterRobot's
  rules; no code changes process-wide torch settings (C-DEP-3) but the benchmark runner and the oracle dumps, each
  with its reason.
- BetterHuman depends on BetterRobot 0.3.0 at one git revision, which `[tool.uv.sources]` in `pyproject.toml` pins;
  `uv sync` fetches it, so no BetterRobot checkout beside BetterHuman is needed.

### Removed
- `Body`: use `HumanModel` and `HumanModel.from_spec`.
- `Body(mass=False)`, `Body(contact_frames=path)`: use `density=None` and `contact_frames=io.read_frame_set(path)`
  on `HumanModel.from_spec`.
- `Body.faces`, `Body.joint_names`, `Body.parents`, `Body.num_vertices`, `Body.identity_groups`,
  `Body.stream_groups`, `Body.dtype`, `Body.device`: use the same names on `HumanModel.spec`.
- `Body.native_joint_ids`, `Body.topology`: use `HumanModel.joint_ids` and `HumanModel.part_topology`.
- `Identity.body`, `Identity.params`, `Identity.v_shaped`, `Identity.rest_joints`, `Identity.pelvis`: use
  `Identity.human`, `Identity.values`, `Identity.rest_vertices`, `Identity.rest_joint_translations` and
  `Identity.root_offset`.
- `Identity.inverse_bind_poses`, `Identity.joint_scales`: use `Identity.inverse_bind_affines`.
- `Identity.stream_joint_basis`: use `HumanModel.stream_joint_bases`.
- `Identity.spec`, `Identity.nq`, `Identity.nv`, `Identity.q_neutral`: use `Identity.human` and `Identity.model`.
- `HumanState.stream`, `HumanState.rest_joints`, `HumanState.pelvis`, `HumanState.rest_offset`,
  `HumanState.joints`: use `HumanState.streams`, `HumanState.rest_joint_translations`, `HumanState.root_offset`,
  `HumanState.stream_offsets` and `HumanState.joint_translations`.
- `HumanState.body`, `HumanState.spec`: use `HumanState.identity`.
- `FamilySpec.vertices`, `FamilySpec.identity`, `FamilySpec.streams`: use `FamilySpec.template_vertices`,
  `FamilySpec.identity_bases` and `FamilySpec.stream_bases`.
- `FamilySpec.meta`: use `FamilySpec.file_joint_ids`, `Parts.closed_faces` and `FamilySpec.classic`.
- `FamilySpec.unit`, `FamilySpec.forward_axis`: no replacement; nothing read them.
- `FamilySpec.to`, `Corrective.to`, `Rig.to`: no replacement; dtype and device are chosen at load, so load again.
- `Corrective.kind`, `Corrective.posedirs`, `Corrective.first_layer`, `Corrective.output`: use
  `core.LinearCorrective` and `core.MLPCorrective`.
- `Rig.native_joint_ids`: use `Rig.joint_ids`.
- `core.RegressedRig`, `core.drop_joints`: use `families.RegressedRig` and `families.drop_joints`.
- `MhrRig`, `SomaRig`: use `families.MHRRig` and `families.SOMARig`.
- `RigBake.rest_joints`, `RigBake.pelvis`, `RigBake.v_shaped`, `RigBake.inverse_bind_poses`: use
  `RigBake.rest_joint_translations`, `RigBake.root_offset`, `RigBake.rest_vertices` and
  `RigBake.inverse_bind_affines`.
- `RigBake.rest_poses`, `RigBake.joint_scales`, `RigBake.neutral_vertices`: no replacement; the joint scales are
  folded into `inverse_bind_affines`.
- `lbs_affines`: use `core.linear_blend_skinning`; the old `skin_affines` keywords: use
  `core.skin_affines(joint_transforms, inverse_bind_affines)`.
- `lbs`, `linear_corrective`, `mlp_corrective`, `dense_skinning_weights`, `skinning_transforms`,
  `skinning_transforms_from_inverse`: use `core.skin_affines`, `Corrective.apply` and `FamilySpec.skin_matrix`.
- `regress_joints`, `pose_feature_rotmat_minus_identity`, `pose_feature_6d_from_euler`, `dense_weight_from_sparse`:
  no replacement; private to the rig that uses each.
- `axis_angle_to_quaternion`: use `better_robot.lie.so3.exp`.
- `quaternion_to_axis_angle`, `classic_root_to_free_flyer`, `free_flyer_to_classic_root`, `classic_tensor`,
  `broadcast_configuration`, `reduce_joint_data`, `slice_parameter_basis`, `frame_local_placements`,
  `resolve_parents`, `vertex_normals`: no replacement; internal to `from_classic`, `to_classic`,
  `families.drop_joints` and `HumanModel.from_spec`.
- `parse_frame_set`: use `io.read_frame_set`.
- `smpl_from_classic`, `smplh_from_classic`, `smplx_from_classic`, `mano_from_classic`, `flame_from_classic`,
  `mhr_from_classic`, `soma_from_classic`: use `from_classic`.
- `smpl_to_classic`, `smplh_to_classic`, `smplx_to_classic`, `mano_to_classic`, `flame_to_classic`,
  `mhr_to_classic`, `soma_to_classic`: use `to_classic`.
- `num_pca_comps` on `SMPLXClassic`, `MANOClassic` and the converters: use `families.smplx(..., num_pca_comps=)`
  and `families.mano(..., num_pca_comps=)`.
- `load_family`, `parents_from_kintree`, `KEYPOINT_NAMES`, `SMPL`, `SMPLH`, `SMPLX`, `MANO`, `FLAME`,
  `FamilyTable`: use the family loaders and the `FamilySpec.joint_names` and `FamilySpec.landmarks` they fill.
- `PartMeshTopology`, `build_authored_topology`, `build_part_topology`, `meshes_from_topology`: use
  `mass.PartTopology` and its `from_closed_faces`, `from_part_ids` and `meshes`.
- `body_inertias_from_shape`, `segment_vertices`: use `mass.part_inertias` and `mass.segment_vertices`.
- `load_density_table`, `load_closed_segmentation`, `SMPL_CATEGORY_NAMES`, `density_for_parts`,
  `part_categories`: use `HumanModel.from_spec(spec, density="Dempster")` and `mass.part_densities`.
- `resolve_model_path`, `MODELS_ENV`: use `io.weight_path` and `io.models_dir`.
- The archive schemas (`SMPL_SCHEMA` to `SOMA_SCHEMA`, `MHR_CORRECTIVE_FIELDS`, `SOMA_CORRECTIVE_FIELDS`),
  `validate`, `load_model_npz`: no replacement; the readers are internal, and `io.check_format` checks a header.
- `io.convert_chumpy`: use the script `tools/convert/smpl/convert_chumpy.py`.
- `terms.shaped`: use `terms.baked_identity`.
- `terms.contact`: use `terms.points`.
- `terms.project`, `terms.contact_wrenches`: no replacement; internal to `terms.reprojection` and
  `terms.root_wrench`.
- `viz.human_layers` as a module: use the names in `viz`.
- `viz.studio`: use `better_human.studio`; its 15 internal names (`StudioSession`, `GizmoController`, ...) and
  `DEFAULT_ROOT`: no replacement; they are the application's internals, and the data root is `--data-root`.
- The `viewer` extra: use the `studio` extra.
- The benchmark report script and the tracked benchmark results: no replacement; results are not tracked, and the
  runner's JSON carries its own context.

### Fixed
- `viz.skinned_layer` draws SOMA bodies where `HumanState.vertices` puts them: it inverts the identity's
  `inverse_bind_affines` instead of assuming SMPL's translation-only binds (the SOMA mesh was off by up to 0.29 m).
- `viz.skinned_layer` inverts the state's `inverse_bind_affines`, the binds its bones were posed with, so streams
  that move joints (SMPL-X expression) no longer shift the mesh (1.17 mm); a state whose binds change over the clip
  raises `ShapeError`.
- `viz.human_layers` draws forces of shapes `(T, 3)` and `(3,)` together, as documented (torch's `RuntimeError`
  came before), and raises `ShapeError` for any other shape.

### Security
- The studio's Save buttons refuse a name that is not one file name (empty, `.`, `..`, an absolute path, or a name
  holding `/` or `\`) with an `ArgumentError` on the status line and write nothing: a typed name could place a file
  anywhere the user can write.
