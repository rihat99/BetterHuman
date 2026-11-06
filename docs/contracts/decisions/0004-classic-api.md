# 0004 One classic API: `bh.from_classic` and `bh.to_classic`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
The SMPL family converts `(identity, classic) -> (q, stream)`, MHR and SOMA convert `(body, classic) -> (identity, q,
stream)`, `soma_to_classic` has no `stream`, and two converters add `num_pca_comps`. There are 7 classic dataclasses
and 14 functions plus private helpers repeated per family. Apps import `families.classic.SMPLXClassic` and
`smplx_from_classic` past the public surface.

## Decision
We will provide `bh.from_classic(human, classic)` and `bh.to_classic(identity, q, *, streams)` over a
`FamilySpec.classic: ClassicFormat` (abstract in `core`, finals in `families`; one data-driven `SMPLFamilyFormat`
replaces five hand-written pairs), make every `*Classic` carry its identity inputs (`betas` for the SMPL family), and
fix `num_pca_comps` at load.

## Example
```python
human = bh.HumanModel.from_spec(families.smplx(path, num_pca_comps=12))   # PCA width fixed at load
identity, q, streams = bh.from_classic(human, SMPLXClassic(betas=betas, ...))
classic = bh.to_classic(identity, q, streams=streams)      # the family's own *Classic type
```

## Consequences
One entry point for every family; the family difference lives in data, and one round-trip checker runs over every
format. Cost: SMPL callers that reused one baked identity re-bake per call (0.4 ms without mass, 11 to 23 ms with),
and tests load one model per PCA width.

## Alternatives
Per-family functions with one shared signature: the smallest change, but a generic pipeline needs its own dispatch
table. Name-keyed mappings instead of dataclasses: fewest types, no typed fields. smplx fixes `use_pca` and
`num_pca_comps` at construction and takes the shape coefficients with the pose in `forward`, as this record does.

## Compatibility
Breaks the 10 SMPL-family functions and the MHR and SOMA ones; `SMPLXClassic` gains `betas`. Migration:
BetterVideoReconstruction's `SMPLXClassic` sites gain `betas`; BetterRetarget's imports move to `families`. Not
reversible after the apps migrate.

## Enforcement
New `tests/families/test_classic_formats.py`: one round-trip property checker run over every `ClassicFormat`, and
`bh.from_classic` raises with the expected type when `classic` is the wrong `*Classic`; the SMPL-X goldens keep their
1e-6 m gate.

## Open questions
None.
