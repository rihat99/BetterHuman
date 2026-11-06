# 0007 One `density` parameter with the shipped table as default

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`Body(mass=True, density=1000.0)` ignores the shipped density table, which the studio and BetterVideoReconstruction
load by hand (`families.assets.load_density_table("body_densities.json", ...)`, whose first argument has one legal
value). The keyword map from part names to categories silently falls back to the table mean for 20 of 110 SOMA names,
16 of 16 MANO and 1 of 5 FLAME. A float32 density on a float64 model is cast silently. An SMPL bake takes 11.2 ms with
mass and 0.41 ms without, because of a Python loop over parts.

## Decision
We will replace `mass` and `density` by one `density: str | float | Tensor | None` (a table name, a uniform kg/m3, a
`(P,)` tensor, or no inertias), default `"Dempster"`, make each family declare its part categories as data
(`Parts.density_categories`), and raise at load on an unknown category or a dtype mismatch.

## Example
```python
human = bh.load("smplx")                          # density="Dempster": the shipped table, categories declared by the family
human = bh.load("smplx", density=1000.0)          # uniform, as before
human = bh.load("smplx", density=None)            # no inertias: the 0.41 ms bake
```

## Consequences
Dynamics users get physically graded inertias by default. The table mean fallback is gone; each family ships category
tables for SOMA, MANO and FLAME. Vectorising the per-part loop waits for a profile (5.6) and is not part of this change.

## Alternatives
Uniform 1000 kg/m3 as default: physically rougher. `None` as default: the fastest bake, but dynamics users get no
inertias unless they ask. Drake's `SpatialInertia::SolidBoxWithDensity` and its siblings take the density from the caller and default
nothing; here the parts come from a mesh, so a named table is the default for that argument.

## Compatibility
Breaks `mass=` and the old `density=` meaning, and the hand-loaded `load_density_table` call (apps pass
`density="Dempster"` instead). The default changes inertia values for callers who relied on 1000 kg/m3; reversible by
passing `density=1000.0`.

## Enforcement
New `tests/families/test_density_categories.py` (every family's `Parts.density_categories` covers every part; the
probe of unmapped names becomes 0); new `tests/mass/test_densities.py` (an unknown category and a dtype mismatch raise
at load); the bake rows of `benchmarks/run.py` record the with-mass and without-mass times.

## Open questions
None.
