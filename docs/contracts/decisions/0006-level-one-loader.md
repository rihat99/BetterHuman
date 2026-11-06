# 0006 `bh.load` has an explicit keyword-only signature

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`bh.load(family, path=None, **options)` routes options through the hidden `_BODY_OPTIONS` set and builds its
dispatch table by reflection over `families.__all__`. A common call is `bh.load("smplx", path, num_betas=16,
mass=False)`; the signature shows none of its options.

## Decision
We will define `bh.load(family, path=None, *, dtype, device, density, contact_frames, stream_joints)` with family
defaults, route family options (gender, `num_betas`, PCA width) through `HumanModel.from_spec(families.smplx(...),
...)`, and make the dispatch table a literal dict.

## Example
```python
human = bh.load("smplx", path, dtype=torch.float64, density="Dempster")
human = bh.HumanModel.from_spec(
    families.smplx(path, gender="neutral", num_betas=16), density=None, contact_frames=frames, stream_joints=True)
```

## Consequences
The signature shows every option, and a misspelled option is a `TypeError`. The family loaders stay unchanged:
`load` is a thin literal dispatch. Callers that passed family options to `load` use two lines.

## Alternatives
Keep `**options`, validated against the loader signature: the signature stays hidden (2.3). Each family loader returns
a `HumanModel` and takes the body options: three options repeated in seven signatures. smplx's `create(model_path,
model_type, gender, num_betas, ...)` takes the family as a string and forwards `**kwargs`, which is the shape being
replaced.

## Compatibility
Breaks `load(..., num_betas=...)`, `mass=` and every other family option on `load`. Migration:
BetterVideoReconstruction calls that pass family options move to `from_spec`; the BetterHuman README uses the new
signature. Reversible: adding keyword-only parameters is compatible.

## Enforcement
`tests/contract/test_keyword_only.py`, extended: `bh.load` has no `**kwargs` parameter and its dispatch table equals
the set of loaders in `families.__all__`.

## Open questions
None.
