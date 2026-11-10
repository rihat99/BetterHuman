# Migrating to 0.2.0

Replace each name in the left column by the one in the middle column. Version 0.2.0 builds a body once as a frozen
`HumanModel`, chooses dtype and device at the family loader, gives every family a typed `FamilySpec`, converts
native parameters through one `from_classic` / `to_classic` pair, and moves the studio into its own package. Where
the middle column is `—`, nothing replaces the name and the note says why and what to call instead.

Names are written relative to `better_human`: `core.Rig` is `better_human.core.Rig`. A name without a module
(`Body`, `Identity.params`) is gone from every module that exported it, and a module in the left column takes
everything it held with it. `tests/contract/test_changelog.py` checks that every public name of the 0.1 line that
0.2.0 lacks is on this page, and that every name in the middle column exists. The [changelog](changelog.md) lists
what 0.2.0 adds.

## Root `better_human`

The root exports `load`, `HumanModel`, `FamilySpec`, `Identity`, `HumanState`, `from_classic`, `to_classic` and
`__version__`, and the packages `families` and `io`; import `core`, `mass`, `terms`, `viz` and `studio` by name
(`import better_human.terms`).

| 0.1 | 0.2.0 | note |
|---|---|---|
| `Body` | `HumanModel` | it pairs with `HumanState`; "body" stays BetterRobot's rigid body |
| `Body(spec, *, dtype, device, mass=True, density=1000.0, contact_frames, stream_joints)` | `HumanModel.from_spec(spec, *, density="Dempster", contact_frames=None, stream_joints=True)` | dtype and device are chosen at the family loader: `families.smplx(path, dtype=torch.float64)`; the model is frozen and built once |
| `load(family, path=None, **options)` | `load(family, path=None, *, dtype, device, density, contact_frames, stream_joints)` | family options go through the loader: `HumanModel.from_spec(families.smplx(path, gender="female", num_betas=16))` |

## `HumanModel` (was `Body`)

| 0.1 | 0.2.0 | note |
|---|---|---|
| `Body.faces`, `Body.joint_names`, `Body.parents`, `Body.num_vertices`, `Body.identity_groups`, `Body.stream_groups`, `Body.dtype`, `Body.device` | `HumanModel.spec` | read them from the family data: `human.spec.faces`, `human.spec.joint_names`, ... |
| `Body.native_joint_ids` | `HumanModel.joint_ids` | the BetterRobot joint id of each native joint |
| `Body.topology` | `HumanModel.part_topology` | a `mass.PartTopology` |
| `Body.spec`, `Body.model`, `Body.skin_matrix`, `Body.part_body_ids`, `Body.contact_frames`, `Body.density`, `Body.bake` | `HumanModel.spec`, `HumanModel.model`, `HumanModel.skin_matrix`, `HumanModel.part_body_ids`, `HumanModel.contact_frames`, `HumanModel.density`, `HumanModel.bake` | same meaning; `human.model` is the zero-identity BetterRobot model |
| `Body(mass=False)` | `HumanModel.from_spec` | `density=None`: no inertias |
| `Body(density=1000.0)` (the default) | `HumanModel.from_spec` | `density=1000.0` keeps the uniform density; the default is now `"Dempster"`, each part at its category's density |
| `Body(contact_frames=path)` | `io.read_frame_set` | `contact_frames=` takes a mapping only: `contact_frames=io.read_frame_set(path)` or `io.shipped_frame_set("smplx_contacts")` |
| `Identity.stream_joint_basis` | `HumanModel.stream_joint_bases` | static per model |

## `Identity` and `HumanState`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `Identity.body` | `Identity.human` | |
| `Identity.params` | `Identity.values` | identity group to coefficients |
| `Identity.v_shaped` | `Identity.rest_vertices` | |
| `Identity.rest_joints` | `Identity.rest_joint_translations` | |
| `Identity.pelvis` | `Identity.root_offset` | |
| `Identity.inverse_bind_poses`, `Identity.joint_scales` | `Identity.inverse_bind_affines` | `(I..., J, 3, 4)` affines; MHR's joint scales are folded in at bake |
| `Identity.spec`, `Identity.nq`, `Identity.nv`, `Identity.q_neutral` | `Identity.human`, `Identity.model` | `identity.human.spec`, `identity.model.nq`, `identity.model.nv`, `identity.model.q_neutral` |
| `Identity.forward(q, stream)` | `Identity.forward(q, streams)` | |
| `HumanState.stream` | `HumanState.streams` | |
| `HumanState.rest_joints` | `HumanState.rest_joint_translations` | |
| `HumanState.pelvis` | `HumanState.root_offset` | |
| `HumanState.rest_offset` | `HumanState.stream_offsets` | `(..., V, 3)` vertex offsets of the stream groups |
| `HumanState.joints` | `HumanState.joint_translations` | native joint order |
| `HumanState.body`, `HumanState.spec` | `HumanState.identity` | `state.identity.human.spec` |
| `HumanState.landmarks(name="keypoints")` | `HumanState.landmarks()` | one landmark set per family, `FamilySpec.landmarks`; MHR and SOMA have none |

## `FamilySpec`, `Corrective` and `Parts`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `FamilySpec.vertices` | `FamilySpec.template_vertices` | `HumanState.vertices` is the posed mesh |
| `FamilySpec.identity`, `FamilySpec.streams` | `FamilySpec.identity_bases`, `FamilySpec.stream_bases` | |
| `FamilySpec.landmarks` (a mapping of named vertex sets) | `FamilySpec.landmarks`, `core.Landmarks` | a `Landmarks(names, vertex_ids)` record, or `None` |
| `FamilySpec.meta` | `FamilySpec.file_joint_ids`, `Parts.closed_faces`, `FamilySpec.classic` | `meta["native_joint_ids"]` is `spec.file_joint_ids`; `meta["closed_segmentation"]` is `spec.parts.closed_faces`; the hand keys (`hand_mean`, `hand_components`, `hand_joints`, `side`) live in the family's `families.SMPLFamilyFormat`; MHR's `bind_joint_positions` is folded into the inverse bind affines |
| `FamilySpec.unit`, `FamilySpec.forward_axis` | — | never read; units are metres and `FamilySpec.up_axis` sets gravity |
| `FamilySpec.to`, `Corrective.to`, `Rig.to` | — | dtype and device are chosen at load: load again with `families.smplx(path, dtype=..., device=...)`; BetterRobot keeps `Model.to` |
| `Corrective(kind, posedirs, first_layer, output)` | `core.LinearCorrective`, `core.MLPCorrective` | `core.Corrective` is their abstract base; a family without correctives has `FamilySpec.corrective = None` (was `kind="none"`) |
| `Corrective.kind`, `Corrective.posedirs`, `Corrective.first_layer`, `Corrective.output` | `LinearCorrective.posedirs`, `MLPCorrective.first_layer`, `MLPCorrective.output` | the class says the kind |
| `Parts(ids, joints)` | `core.Parts` | `Parts(ids, joints, density_categories, density_tables, closed_faces=None)` |

## `Rig` and `RigBake`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `Rig` (a `Protocol`) | `core.Rig` | an abstract base: a rig subclasses it and implements `build_model`, `bake`, `joint_ids` and `corrective_features`; the other hooks have defaults |
| `Rig.native_joint_ids` | `Rig.joint_ids` | |
| `Rig.build_model(spec, *, frames, dtype, device)`, `FrameSpec` (in `core.rig`) | `Rig.build_model(spec, *, frames)` | builds in the spec's dtype and device; `frames` are `(name, parent joint name, placement (7,))` tuples |
| `Rig.bake(spec, model, v_shaped, identity)` | `Rig.bake(spec, model, *, shaped_vertices, identity)` | keyword-only |
| `Rig.corrective_features(spec, model, q, *, identity=None)`, `Rig.placements_from_rest(spec, model, rest_joints)` | `Rig.corrective_features(spec, model, *, q, identity)`, `Rig.placements_from_rest(spec, model, *, rest_joint_translations)` | keyword-only |
| `core.RegressedRig` | `families.RegressedRig` | final; the SMPL family's rig, which centres the pelvis in its `bake` |
| `MhrRig`, `SomaRig` | `families.MHRRig`, `families.SOMARig` | final; acronyms in capitals |
| `RigBake.rest_joints`, `RigBake.pelvis`, `RigBake.v_shaped`, `RigBake.inverse_bind_poses` | `RigBake.rest_joint_translations`, `RigBake.root_offset`, `RigBake.rest_vertices`, `RigBake.inverse_bind_affines` | the words of `Identity` |
| `RigBake.rest_poses`, `RigBake.joint_scales`, `RigBake.neutral_vertices` | — | no optional fields left: joint scales are folded into `inverse_bind_affines` |
| `core.drop_joints` | `families.drop_joints` | the SMPL family only |

## `core` skinning and helpers

| 0.1 | 0.2.0 | note |
|---|---|---|
| `lbs_affines` (in `core.pipeline`) | `core.linear_blend_skinning` | `linear_blend_skinning(rest_vertices, affines, *, weights)` |
| `skin_affines(joint_transforms, *, rest_joints, inverse_bind_poses, joint_scales, bind_joint_positions)` | `core.skin_affines(joint_transforms, inverse_bind_affines)` | pass `identity.inverse_bind_affines`; the SMPL family's is the identity rotation with `-rest_joint_translations` |
| `blend_shapes` (in `core.pipeline`) | `core.blend_shapes` | |
| `lbs`, `linear_corrective`, `mlp_corrective`, `dense_skinning_weights`, `skinning_transforms`, `skinning_transforms_from_inverse` | `core.skin_affines`, `Corrective.apply`, `FamilySpec.skin_matrix` | duplicates of these; one skinning path |
| `regress_joints`, `pose_feature_rotmat_minus_identity`, `pose_feature_6d_from_euler`, `dense_weight_from_sparse` | `HumanState.corrective_feature` | private to the rig that uses each |
| `axis_angle_to_quaternion` | `better_robot.lie.so3.exp` | the same function |
| `quaternion_to_axis_angle`, `classic_root_to_free_flyer`, `free_flyer_to_classic_root`, `classic_tensor`, `broadcast_configuration` (in `core.classic`) | `from_classic`, `to_classic` | the one conversion path |
| `reduce_joint_data`, `slice_parameter_basis` (in `core.reduction`) | `families.drop_joints` | internal to it and the loaders |
| `frame_local_placements`, `resolve_parents`, `vertex_normals` | `HumanModel.from_spec`, `HumanState.contact_frame_poses` | internal to attaching contact frames |
| `parse_frame_set(path, num_vertices)` | `io.read_frame_set(path)` | vertex ids are checked when the frames are attached |

## Classic conversion

| 0.1 | 0.2.0 | note |
|---|---|---|
| `smpl_from_classic`, `smplh_from_classic`, `smplx_from_classic`, `mano_from_classic`, `flame_from_classic`, `mhr_from_classic`, `soma_from_classic` | `from_classic` | `identity, q, streams = from_classic(human, classic)`, dispatched on `human.spec.classic` |
| `smpl_to_classic`, `smplh_to_classic`, `smplx_to_classic`, `mano_to_classic`, `flame_to_classic`, `mhr_to_classic`, `soma_to_classic` | `to_classic` | `to_classic(identity, q, *, streams=None)`; SOMA's gains `streams` |
| `SMPLClassic`, `SMPLHClassic`, `SMPLXClassic`, `MANOClassic`, `FLAMEClassic` (in `families.classic`), `MHRClassic` (in `families.mhr`), `SOMAClassic` (in `families.soma`) | `families.SMPLClassic`, `families.SMPLHClassic`, `families.SMPLXClassic`, `families.MANOClassic`, `families.FLAMEClassic`, `families.MHRClassic`, `families.SOMAClassic` | fields keyword-only, each optional (an omitted field is zero); the SMPL family's gained `betas` |
| `SMPLXClassic.num_pca_comps`, `MANOClassic.num_pca_comps`, `smplx_to_classic(..., num_pca_comps=)` | `families.smplx`, `families.mano` | fixed at load: `families.smplx(path, num_pca_comps=6)`; `None` (the default) keeps 45-D axis-angle hands |

## `families` and `mass`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `mhr`, `soma` (in the modules `families.mhr`, `families.soma`) | `families.mhr`, `families.soma` | the loaders are the package's functions; their modules are internal |
| `load_family`, `parents_from_kintree` (in `families.smpl_family`) | `families.smpl`, `families.smplh`, `families.smplx`, `families.mano`, `families.flame` | internal to the loaders |
| `KEYPOINT_NAMES` (in `families.tables`) | `Landmarks.names` | `spec.landmarks.names` |
| `SMPL`, `SMPLH`, `SMPLX`, `MANO`, `FLAME`, `FamilyTable` (in `families.tables`) | `FamilySpec.joint_names`, `FamilySpec.landmarks` | the family tables are internal; a loaded spec carries its joint names and landmarks |
| `PartMeshTopology` (in `core.mass`) | `mass.PartTopology` | |
| `build_authored_topology`, `build_part_topology`, `meshes_from_topology` | `PartTopology.from_closed_faces`, `PartTopology.from_part_ids`, `PartTopology.meshes` | |
| `body_inertias_from_shape` | `mass.part_inertias` | |
| `segment_vertices` | `mass.segment_vertices` | |
| `load_density_table(filename, table, joint_names, exemplar)` (in `families.assets`) | `HumanModel.from_spec`, `mass.part_densities` | `from_spec(spec, density="Dempster")` reads the shipped table; `part_densities(tables, name, *, categories, part_names, dtype, device)` gives one density per part |
| `load_closed_segmentation`, `SMPL_CATEGORY_NAMES`, `density_for_parts`, `part_categories` | `Parts.closed_faces`, `Parts.density_categories` | internal; a loaded spec carries them |

## `io`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `resolve_model_path(family, filename, model_path=None)` | `io.weight_path(folder, filename, *, path=None)` | a missing file raises `io.AssetNotFoundError` naming the fix |
| `MODELS_ENV` (in `io.load`) | `io.models_dir` | `models_dir()` reads `$BETTERHUMAN_MODELS_DIR` |
| `SMPL_SCHEMA`, `SMPLH_SCHEMA`, `SMPLX_SCHEMA`, `MANO_SCHEMA`, `FLAME_SCHEMA`, `MHR_SCHEMA`, `SOMA_SCHEMA`, `MHR_CORRECTIVE_FIELDS`, `SOMA_CORRECTIVE_FIELDS`, `validate`, `load_model_npz` | `io.check_format` | the archive readers are internal; `check_format(data, *, format, version)` checks a file's header |

## `terms`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `shaped(body, *groups)` | `terms.baked_identity(human, *groups)` | |
| `contact` | `terms.points` | it was an alias |
| `project`, `contact_wrenches` | — | internal to `terms.reprojection` and `terms.root_wrench` |
| `joints(*, target, state, joint_ids=)` | `terms.joints(*, target, state, joint_names=)` | joints by name |
| `landmarks(*, target, name="keypoints", state)` | `terms.landmarks(*, target, state)` | the family's one landmark set |
| `reprojection(..., landmark_set=None)` | `terms.reprojection(..., joints=True)` | the landmarks stay the default |
| `human_state(..., stream=)` | `terms.human_state(..., streams=)` | |
| `ground(..., up=(0.0, 1.0, 0.0))` | `terms.ground(..., up=None)` | `None`, the default, points up along the family's `FamilySpec.up_axis` |

## `viz`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `skinned_layer(identity, state, *, name, color, opacity)` | `viz.skinned_layer(state, *, name, color, opacity)` | the binds come from `inverse_bind_affines`; binds that scale or shear raise |
| `forces_layer(state, frame_ids, forces, ...)` | `viz.forces_layer(state, frame_names, *, forces, ...)` | frames by name |
| `HumanView`, `human_layers`, `mesh_layer`, `skeleton_layer`, `frames_layer` (in the module `viz.human_layers`) | `viz.HumanView`, `viz.human_layers`, `viz.mesh_layer`, `viz.skeleton_layer`, `viz.frames_layer` | import them from `viz` |
| `replace(LIGHT, up="+y")` in the examples | `viz.HUMAN_THEME` | BetterRobot's light theme with `+y` up, the layers' default |

## `studio`

| 0.1 | 0.2.0 | note |
|---|---|---|
| `HumanStudio`, `launch` (in `viz.studio`) | `studio.HumanStudio`, `studio.launch` | the studio is its own package; install `better-human[studio]` |
| `HumanStudio(*, family, gender, lod, port, host, models_dir, data_dir, density, server)` | `studio.HumanStudio(*, data_root, family, gender, lod, port, host, generator, density, server)` | `data_root` and `generator` (a `numpy.random.Generator`) are required; weights come from `io.models_dir()` |
| `python -m better_human.viz.studio [--data-dir DIR] [--models-dir DIR]` | `python -m better_human.studio --data-root DIR [--seed 0]` | `--data-root` is required |
| `GizmoController`, `GROUP`, `BodyMeshLayer`, `MeshMode`, `PartsLayer`, `SegmentationLayer`, `StudioLayers`, `part_colors`, `frame_poses`, `JointHandle`, `apply_local_rotation`, `joint_handles`, `ChangeKind`, `FrameEntry`, `StudioSession` | — | internals of the studio application |
| `DEFAULT_ROOT` (the checkout's `studio_data`) | — | a wheel has no checkout: pass `--data-root` |

## Module paths

Import from the package; the modules under it are internal.

| 0.1 | 0.2.0 | note |
|---|---|---|
| `core.body`, `core.spec`, `core.rig`, `core.contact_frames`, `core.pipeline` | `core` | |
| `core.classic` | `from_classic`, `to_classic` | |
| `core.reduction` | `families.drop_joints` | |
| `core.mass` | `mass` | the mass package with its data |
| `families.classic`, `families.mhr_rig`, `families.soma_rig`, `families.smpl_family`, `families.tables` | `families` | |
| `families.assets`, `families.assets.densities` | `mass`, `HumanModel.from_spec` | the shipped tables are read through `from_spec(density=...)` |
| `io.load`, `io.schema` | `io` | |
| `io.convert_chumpy` | — | a script now, `uv run tools/convert/smpl/convert_chumpy.py`: it patched `inspect` and numpy for the whole process |
| `viz.studio` | `studio` | |

## Behaviour under the same names

| 0.1 | 0.2.0 | note |
|---|---|---|
| `load("SMPLX")`, any case | `load("smplx")` | the family name is exact |
| `ValueError` for an unknown family in `load`, or an unknown identity or stream group in `Body.bake` and `Identity.forward` | `better_robot.exceptions.UnknownNameError` | a `KeyError`: `except ValueError` no longer catches it |
| a float32 `transl`, `global_orient` or pose on a float64 model, promoted silently by the classic converters | `better_robot.exceptions.DtypeMismatchError` | pass the model's dtype |
| a wrong-shape classic field: torch's `RuntimeError` | `better_robot.exceptions.ShapeError` | |
| plain `ValueError` from the classic converters and the `viz` layers | `better_robot.exceptions.ArgumentError`, `better_robot.exceptions.ShapeError`, `better_robot.exceptions.DtypeMismatchError` | `ValueError` subclasses |
| a family loaded in any floating dtype | `better_robot.exceptions.DtypeMismatchError` | float32 and float64 only, the dtypes BetterRobot computes in |
| the default density: 1000 kg/m³ for every part | `HumanModel.from_spec` | `density="Dempster"`; masses and inertias change for code that relied on the default |
| `Identity == other` and `HumanState == other` raised on their tensors; reprs printed every tensor | `HumanModel`, `Identity`, `HumanState` | frozen, compared by identity, a one-line repr |

## Files, formats and environment

| 0.1 | 0.2.0 | note |
|---|---|---|
| `mhr_lod{N}.npz`, `soma_mid.npz` with `format_version=1` and no `format` | `format="better_human.mhr"` or `"better_human.soma"`, `format_version=2`, `provenance` | version 1 files still load: `io.upgrade` migrates them in memory |
| frame-set JSON `{name: {vertex, joint}}` | `{"format": "better_human.frame_set", "format_version": 1, "frames": {...}}` | `io.read_frame_set` migrates the old form on read |
| `better_contacts.json` in the checkout's `studio_data` | `io.shipped_frame_set("smplx_contacts")` | package data |
| `smpl_3d_segmentation.npy` (a pickle), `body_densities.json` keyed `"0"` to `"23"` (in `families.assets`) | `"better_human.segmentation"`, `"better_human.density_tables"` archives | versioned `.npz` files read by the loaders, no pickle; densities keyed by category name |
| `smpl_vert_segmentation.json`, `smplx_vert_segmentation.json` (in `families.assets`) | `studio` | they move with the studio, format `"better_human.studio.segmentation"` version 1; a segmentation the 0.1 studio saved (`{part name: vertex ids}`) migrates on read |
| studio pose files holding `q` only | format `"better_human.studio.pose"` version 1 | a pose stores its `joint_names`; a 0.1 pose lacks them and is refused, naming the fix: pose the model again in the studio and save it; a pose of other joints is refused |
| parity goldens in `tests/parity/goldens` | `$BETTERHUMAN_MODELS_DIR/goldens` | release tier, provenance stamped |
| `BETTERHUMAN_MODELS_DIR` read by `resolve_model_path`; the studio set it from `--models-dir` | `io.models_dir` | the one reader; `io.AssetNotFoundError`, a `FileNotFoundError`, names the fix |
| `better-human[viewer]`, `better-human[studio]` with `viser>=1.0` | `better-human[studio]` | one extra, BetterRobot's `viewer`, for `viz` and the studio |
| the `scipy` dependency | the dev dependency group | only the chumpy converter, its tests and the benchmarks use it |
| MIT licence | Apache-2.0 | |
| Python 3.10 and 3.11 | Python 3.12 or newer | |
