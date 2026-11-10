# 0019 `viz` imports `core` only: the `viz -> io` edge goes

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record 0011's note of 2026-11-06 made `viz -> io` a permanent allowed edge because the studio, then
`better_human.viz.studio`, persisted frame sets through `better_human.io`. Record 0009 moved the studio to
`better_human.studio`, whose own row allows `io`; `viz` is now one `layers` module (0010) that imports `core` and
BetterRobot's viewer and nothing of `io`. The allowance stayed in `tests/contract/test_layer_dependencies.py` with
its old reason, so a new `viz -> io` import would pass unnoticed.

## Decision
We will drop `io` from the `viz` row: `viz` imports `core` only, among BetterHuman's units. The `BR_ALLOWED` table
is unchanged. This amends the `viz` row of 0011's `ALLOWED` table as its note of 2026-11-06 left it, and replaces
that note.

## Example
```python
ALLOWED = {..., "viz": {"core"}, "studio": {ROOT, "core", "families", "io", "mass", "viz"}, ...}
```

## Consequences
The viewer layers depend on the body model alone; reading or writing a file is the caller's (or the studio's) job.
An import of `better_human.io` from `viz` fails `test_layer_dependencies`.

## Alternatives
Keep the allowance as a harmless widening: the table would then no longer describe the code, and the edge's stated
reason is false.

## Compatibility
None: no `viz` module imports `io` today.

## Enforcement
`tests/contract/test_layer_dependencies.py` (`ALLOWED["viz"] == {"core"}`); `docs/contracts/rules.md`
(`H-LAYER-1`) and `docs/concepts/architecture.md` show the same row. On acceptance, 0011's Status line names this
record as superseding its `viz` row.

## Open questions
None.
