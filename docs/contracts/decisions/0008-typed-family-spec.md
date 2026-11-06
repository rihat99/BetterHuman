# 0008 `FamilySpec` has typed fields and no `meta`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`FamilySpec.meta: Mapping[str, Any]` carries seven behaviour keys pinned by `META_KEYS`. `meta["native_joint_ids"]`
(file numbering) and `Rig.native_joint_ids` (BetterRobot ids) are one name for two concepts. `meta["keypoint_names"]`
is written and never read, while apps import `families.tables.KEYPOINT_NAMES`. `unit` and `forward_axis` are never
read. `Corrective(kind=...)` is a hand-written tagged union.

## Decision
We will give every behaviour key a typed home: `FamilySpec.file_joint_ids`; hand PCA fields on the SMPL-family classic
format (0004); `Parts.closed_faces`; MHR bind positions folded into `MHRRig` (0003); `Landmarks(names, vertex_ids)`;
`identity_bases`, `stream_bases` and `template_vertices`; an abstract `Corrective` with final `LinearCorrective` and
`MLPCorrective`; `up_axis: Literal[...]`; remove `unit` and `forward_axis`; and keep `provenance: Mapping[str, str]`
as the only free-form field.

## Example
```python
spec = families.smplx(path)
spec.landmarks.names, spec.landmarks.vertex_ids     # was families.tables.KEYPOINT_NAMES
spec.file_joint_ids                                 # file numbering; Rig.joint_ids is the BetterRobot numbering
spec.provenance                                     # Mapping[str, str]: the only free-form field
```

## Consequences
Each key has one owner and one reader; a dict lookup no longer hides who uses it. `META_KEYS` and
its test go. The decision matches `Model` keeping no `meta` (BetterRobot 0016).

## Alternatives
Keep `meta` typed as a `TypedDict`: a dict lookup still hides the owner of each key. The smplx library passes loose
`**kwargs` and loader `Struct` attributes between models; momentum's `Character` holds typed members.

## Compatibility
Breaks `meta`, `META_KEYS`, `identity`/`streams`/`vertices` field names, `unit` and `forward_axis`. Apps replace their
`KEYPOINT_NAMES` imports by `spec.landmarks`. Not reversible after the apps migrate.

## Enforcement
New `tests/core/test_spec_fields.py`: `FamilySpec` has no `Mapping[str, Any]` field and `provenance` is the only
`Mapping[str, str]`; `tests/contract/test_abstract_or_final.py` (0002) covers `Corrective`.

## Open questions
None.
