# Glossary

BetterHuman's own words. Every other word of a BetterHuman name is BetterRobot's and means what BetterRobot's
glossary, `BetterRobot/docs/reference/glossary.md`, says; its rules for building names hold here too (plural for an
axis over named things, `_ids` for integer indices, transform and affine, `forward` against `forward_kinematics`, no
`to` in BetterHuman, acronyms in capitals). This page restates none of its rows.

Every word of every public name, method, field and parameter of `better_human` is a term (the bold first cell of a
row) or an [everyday word](#everyday-words) of this page or of BetterRobot's. A word on both pages keeps
BetterRobot's meaning, except in the first table below.
`tests/contract/test_vocabulary.py::test_glossary_words` checks it against the page the installed BetterRobot ships
(`better_robot.testing.glossary.read()`); a change that adds a word adds it here, unless BetterRobot's page has it.

## Words BetterRobot uses with another meaning

Each row gives the BetterHuman meaning and names BetterRobot's. A BetterRobot word defined anywhere else on this
page fails `tests/contract/test_vocabulary.py`.

| Term | Meaning |
|---|---|
| **identity, `Identity`** | One subject's body shape, baked once: `HumanModel.bake(**identity)` turns the values of the identity groups into an `Identity` (or an identity batch `I...`) holding the rest mesh, the rest joints and a BetterRobot `Model` with the subject's placements and inertias. The studio and `terms.shape_prior` call it *shape* (`HumanStudio.random_shape`); MHR names one identity group `identity`. In BetterRobot, *identity* is an everyday word: the element that changes nothing (`se3.identity()`, an identity placement). |
| **rest** | The unposed body of one identity, in the skinning rest frame that the binds are measured in, root at the origin: `Identity.rest_vertices (I..., V, 3)`, `rest_joint_translations (I..., J, 3)`, `Rig.placements_from_rest`. `HumanState.rest_vertices()` adds the streams and the corrective, before skinning. BetterRobot's *rest configuration* (`terms.rest`, `ik_problem(rest_weight=)`) is the value of `q` a regularizing term pulls towards. |

## Families and loading

| Term | Meaning |
|---|---|
| **BetterHuman, `better_human`** | This library: parametric human bodies as BetterRobot models, batched and differentiable in PyTorch. |
| **family** | One parametric body model and its loader: SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR or SOMA (`families.smplx`, `load("smplx")`). |
| **family spec, `FamilySpec`** | One family as data on one dtype and device: template mesh, faces, native joints, skin weights, blend-shape bases, corrective, parts, landmarks, classic format and the rig, its one behavioural part. A family loader returns it. |
| **human model, `HumanModel`** | A family ready to bake subjects (`HumanModel.from_spec(spec)`, `load(family)`): the spec, a BetterRobot `Model` at zero identity (`HumanModel.model`), dense skin weights, mass and contact frames. |
| **SMPL, SMPL-H, SMPL-X** | The SMPL family (Skinned Multi-Person Linear model): the body; SMPL-H adds MANO's hands, SMPL-X adds hands and a face with expressions. One regressed rig and one classic format serve them, MANO and FLAME. |
| **MANO** | The SMPL family's hand, one side per load (`families.mano(side=)`). |
| **FLAME** | The SMPL family's head: neck, jaw, eyes and an expression group (`families.flame`). |
| **MHR** | Meta's Momentum Human Rig: an authored skeleton whose native joints are chains of revolutes, the identity groups `identity` and `proportions`, an MLP corrective and seven levels of detail (`families.mhr`, `MHRRig`). |
| **SOMA** | NVIDIA's SOMA-X body with its own shape space (identity group `shape`): 78 public joints, 32 twist helpers and a skeleton fitted to each shape at bake (`families.soma`, `SOMARig`). |
| **asset** | A data file BetterHuman reads and does not compute: in `io.AssetNotFoundError`, a licensed weight file or the weights directory under `$BETTERHUMAN_MODELS_DIR`; in the source tree, a file shipped in the package (`io/assets`, `studio/assets`). |
| **core** | `better_human.core`: the family data, rigs, identities, states and skinning that the other packages build on. |

## Identity, streams and skinning

| Term | Meaning |
|---|---|
| **identity group** | A named blend-shape group baked once per subject (`FamilySpec.identity_groups`, `identity_bases`): `betas` for the SMPL family, `identity` and `proportions` for MHR, `shape` for SOMA. `Identity.values` holds them, `{name: (I..., n)}`; an omitted group is zero. |
| **stream group, stream** | A named blend-shape group passed with each pose (`forward(q, streams={"expression": ...})`, `FamilySpec.stream_groups`, `stream_bases`): vertex offsets `stream_offsets (..., V, 3)` and, for a regressed rig loaded with `stream_joints=True`, joint motion `stream_joint_bases {name: (J, 3, n)}`. |
| **betas** | The SMPL family's identity group and classic field: shape coefficients `(..., n)` (upstream's name); `num_betas` is its width. |
| **proportions, proportion** | MHR's second identity group: the joint offsets and scales of a subject (`MHRRig.joint_scales(spec, proportions)`, `proportion_columns`). |
| **expression** | The face's stream group (SMPL-X, FLAME, MHR), coefficients `(..., n)`; also the classic field of that name, and `num_expression`, its width. |
| **bake, baked** | Compute once per subject what identity changes: `HumanModel.bake(**identity) -> Identity`, and `Rig.bake -> RigBake` (joint placements, rest mesh, rest joints, inverse binds, root offset). `terms.baked_identity` makes the bake a cached value, so identity groups can be variables. |
| **template** | A family's mesh at zero identity, `FamilySpec.template_vertices (V, 3)`; the identity blend shapes shape it (`shaped_vertices`). |
| **blend shape** | A linear vertex displacement basis `(V, 3, n)`: `blend_shapes(coefficients, basis)` returns the offsets `(..., V, 3)`. Identity and stream groups are blend shapes. |
| **basis, bases** | The columns of a linear map from coefficients to offsets: a blend-shape group `(V, 3, n)` (`identity_bases`, `stream_bases`), a stream group's joint motion `(J, 3, n)` (`stream_joint_bases`), MHR's `translation_basis (J, 3, P)` and `scale_basis (J, P)`. |
| **forward** | `Identity.forward(q, streams=)`: kinematics plus skinning, returning a `HumanState`. BetterRobot's `forward_kinematics(model, q)` computes kinematics only (BetterRobot decision record 0027). |
| **human state, `HumanState`** | One posed batch: BetterRobot's `KinematicState` (`kinematics`) and the skinned body (`vertices`, `points`, `landmarks`), each computed on first use. Joint fields are in native joint order. `terms.human_state` caches one in an optimization problem. |
| **bind, inverse bind** | A joint's frame in the rest pose, the one its skin weights act from. The inverse bind maps the rest frame into each joint's frame, `inverse_bind_affines (I..., J, 3, 4)`, and skinning composes the world joint transforms with it (`skin_affines`): the SMPL family binds `[I | -rest]`, MHR folds its joint scales in, SOMA uses its authored binds (`bind_rotations`). |
| **linear blend skinning** | `core.linear_blend_skinning(rest_vertices, affines, *, weights)`: BetterRobot's *skinning*, the one path every rig skins through (decision record 0003). |
| **top-K, K** | The `K` strongest skin weights of each vertex, `skin_weights` and `skin_indices (V, K)` padded with zero weights (`top_k_skin(weights, k)`). |
| **corrective** | Pose-dependent vertex offsets from the rig's corrective features (`Corrective.apply`): `LinearCorrective` for the SMPL family, `MLPCorrective` for MHR and SOMA. `mhr(correctives=False)` and `soma(correctives=False)` load without one. |
| **corrective feature, feature** | The vector `(..., P)` a corrective reads, computed from `q` by `Rig.corrective_features` (`HumanState.corrective_feature`); for the SMPL family, the flattened `R - I` of every joint but the root. |
| **`posedirs`** | Pose directions: the SMPL family's linear corrective matrix `(P, V * 3)` (upstream's name, `LinearCorrective(posedirs=)`). |

See [Architecture](../concepts/architecture.md) for the four objects and the one skinning path.

## Rigs

| Term | Meaning |
|---|---|
| **rig, `Rig`** | The one behavioural part of a family (decision record 0002): how identity becomes joint placements, how `q` feeds the corrective and which BetterRobot joint each native joint is. Three exist: `RegressedRig`, `MHRRig` and `SOMARig`. |
| **native joint, native** | A joint of the family's own skeleton, in the weight file's order: `J` of them, `FamilySpec.joint_names`. `HumanModel.joint_ids (J,)` holds the BetterRobot joint of each; MHR builds a chain of BetterRobot joints per native joint (`MHRRig(native_joints=)`). |
| **regressed rig, `RegressedRig`** | The SMPL family's rig: joints regressed from the shaped mesh, a free-flyer root and one spherical joint per other native joint. The one rig that drops joints (`families.drop_joints`). |
| **joint regressor, regressor** | A matrix `(J, V)` whose product with the vertices gives joint positions (`RegressedRig(joint_regressor=)`; SOMA's for its public joints). |
| **pre-rotation** | MHR's authored rotation of each joint before its own, `MHRRig(pre_rotations=) (J, 4)`. |
| **public joint** | One of SOMA's 78 joints that its poses address (`SOMARig(public_ids=, public_parents=, public_weights=)`); its other 32 native joints are twist helpers. |
| **twist helper, helper** | One of SOMA's 32 native joints that turn with the twist of their limb segment (`SOMARig(helper_ids=, local_twist=, inherited_twist=)`). There, *twist* is the rotation about a segment's own axis, not a BetterRobot twist. |
| **T-pose** | Arms stretched out sideways; SOMA's classic poses are relative to it (`SOMARig(t_pose_rotations=)`). |
| **repose** | Re-pose: SOMA's bake fits the skeleton to the shaped mesh, then poses mesh and skeleton back to the template bind (`SOMARig(repose_mask=)`). |

## Classic parameters

| Term | Meaning |
|---|---|
| **classic parameters, classic** | A family's inputs as its upstream library names them (`betas`, `global_orient`, `body_pose`, `transl`, MHR's `model_parameters`, SOMA's `poses`), kept as upstream writes them (principle 3.6). They enter and leave only through `from_classic` and `to_classic`, one `<Family>Classic` dataclass per family (decision record 0004). A classic `*_pose` field holds axis-angles of native joints (or hand PCA coefficients), not BetterRobot poses. |
| **classic format** | `ClassicFormat`, a family's map between its classic dataclass and `(identity, q, streams)` (`FamilySpec.classic`): `SMPLFamilyFormat`, `MHRFormat`, `SOMAFormat`. |

## Mass and parts

| Term | Meaning |
|---|---|
| **part, `Parts`** | A rigid piece of the segmented mesh, carried by one native joint's body, for mass properties: `Parts.ids (V,)` labels every vertex, `HumanModel.part_body_ids (P,)` holds each part's BetterRobot body (decision record 0007). |
| **part topology, topology** | `PartTopology`: the vertices, caps and triangles that close each part into a mesh, built once per `HumanModel` (`part_topology`) and applied to every identity. |
| **cap** | The points that close a part's open boundary: rest joints (`cap_joint_ids`, an authored segmentation) or centroids of its boundary loops (`cap_centroid_ids`). |
| **density table, table** | A named mapping from density category to kg/m³ (`Parts.density_tables`); `density="Dempster"` reads one. |
| **density category, category** | A part's key into the density tables (`Parts.density_categories`, `part_densities(categories=)`), `None` when no category covers the part. |

## Contact frames and fitting

| Term | Meaning |
|---|---|
| **contact frame** | A mesh vertex made a BetterRobot frame on a parent joint, origin on the vertex and z along the outward normal, placed again at every bake (`contact_frames`, `ContactFrames`, `HumanState.contact_frame_poses`). |
| **frame set** | Contact frames as data, `name -> (vertex_id, joint_name | None)`, in a file or shipped (`io.read_frame_set`, `io.write_frame_set`, `io.shipped_frame_set("smplx_contacts")`). |
| **landmark, `Landmarks`** | A named mesh vertex a family ships for fitting (`FamilySpec.landmarks`, `HumanState.landmarks() (..., L, 3)`, `terms.landmarks`). |
| **prior** | A term that pulls a variable towards its typical value, a Gaussian with unit precision: `terms.pose_prior` (joint rotations towards a mean, by default the neutral configuration) and `terms.shape_prior` (an identity group towards zero). |
| **reprojection** | Project posed points through a camera `(..., 3, 4)` and compare them with pixels `(..., M, 2)` (`terms.reprojection`). |

## Viewer and studio

| Term | Meaning |
|---|---|
| **human view, `HumanView`** | The viewer layers of one `HumanState` (mesh, skeleton, frames), updated in place by `HumanView.update(state)`. |
| **skinned** | Skinned in the browser: `viz.skinned_layer` is BetterRobot's `SkinnedMeshLayer` of a state (the rest mesh once, joint poses per step, no correctives or stream offsets); `HumanView(mesh="skinned")` draws it instead of the exact mesh. |
| **studio, `HumanStudio`** | The optional browser app `better_human.studio` (the `studio` extra): pose, shape, segment and place contact frames on a body (decision record 0009). |

## Abbreviations and symbols

| Term | Meaning |
|---|---|
| **`coeffs`** | Coefficients (`MHRClassic(identity_coeffs=)`, `SOMAClassic(identity_coeffs=)`; upstream's name). |
| **`comps`** | Components: `num_pca_comps`, the width of the hand PCA basis (upstream's name). |
| **`dir`** | Directory: `io.models_dir()`, the weights directory `$BETTERHUMAN_MODELS_DIR` names. |
| **`leye`, `reye`** | Left eye and right eye: the eye-joint classic fields of FLAME and SMPL-X (upstream's names). |
| **LOD** | Level of detail: one of MHR's seven meshes, `0` to `6` (`families.mhr(lod=)`, `HumanStudio(lod=)`). |
| **MLP** | Multilayer perceptron: `MLPCorrective`, one hidden ReLU layer `first_layer (H, P)` and an `output (H, V * 3)`. |
| **`multi`** | Several: SOMA's joints with more than one child (`SOMARig(multi_children=, multi_parents=, multi_first=)`). |
| **`num`** | Number of: `FamilySpec.num_joints`, `num_vertices`, `PartTopology.num_parts`, and the loaders' `num_betas`, `num_expression`, `num_pca_comps`. BetterRobot writes `n` (`njoints`). |
| **PCA** | Principal component analysis: SMPL-X and MANO hand poses as coefficients of a hand basis (`num_pca_comps`, `SMPLFamilyFormat(hand_components=)`). |
| **`transl`** | Translation: the classic field that places the root (upstream's name). |
| **viz** | Visualization: `better_human.viz`, the viewer layers of a `HumanState`. |

## Everyday words

These words keep their ordinary English meaning in public names, and BetterRobot's page does not list them.

action, ancestor, array, centroid, children, chosen, click, closed, coefficient, confidence, count, deselect,
direction, drop, dropped, event, expansion, file, filename, first, fit, fitted, found, gender, global, hand, header,
indices, inherited, jaw, launch, left, linear, locked, marquee, mask, mean, message, mode, neck, not, only, parameter,
pick, provenance, reload, reset, right, row, select, shaped, shipped, side, single, status, turned, type, update,
upgrade, version, vertex, without, write
