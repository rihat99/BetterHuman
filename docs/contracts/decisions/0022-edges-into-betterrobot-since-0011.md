# 0022 BetterHuman's edges into BetterRobot since record 0011

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0011](0011-human-package-layout.md) fixed the `BR_ALLOWED` table, the BetterRobot packages each BetterHuman
unit imports. Three changes to it landed in `tests/contract/test_layer_dependencies.py` without a record, which an
accepted record's frozen text requires (ruling R-3):

- every row imports `better_robot.exceptions` (commits `20a4703`, `fbb0efd`): BetterHuman raises BetterRobot's
  error classes and roots its own, `io.FormatError` and `io.AssetNotFoundError`, under `BetterRobotError`
  (BetterRobot 0031); the root raises `UnknownNameError` from `bh.load`;
- `viz` imports `better_robot.lie` (commit `6aafc23`): `viz.skinned_layer` turns the inverse bind affines into bind
  poses with `so3.from_matrix`, since record [0003](0003-one-skinning-path.md) derives every bind from them;
- the studio row lost its `better_robot.viewer.tools` entry: the import walker (`better_robot.testing`, BetterRobot
  0035) reports a BetterRobot import as its first-level package, so `better_robot.viewer` covers that module and the
  entry could never be checked.

## Decision
We will write `BR_ALLOWED` as the first-level BetterRobot packages each BetterHuman unit imports, give every row, the
root's included, `better_robot.exceptions`, and add `better_robot.lie` to the `viz` row. This amends 0011's
`BR_ALLOWED` table; its root row and the studio's edge to BetterRobot's root are record
[0021](0021-root-imports-every-subpackage.md)'s, and the `viz` row of `ALLOWED` is record
[0019](0019-viz-imports-core-only.md)'s.

## Example
```python
BR_ALLOWED = {
    unit: {"better_robot.exceptions"} | targets
    for unit, targets in {
        "io": set(),
        "mass": {"better_robot.lie", "better_robot.spatial"},
        "core": {"better_robot", "better_robot.kinematics", "better_robot.lie"},
        "families": {"better_robot", "better_robot.io", "better_robot.kinematics", "better_robot.lie"},
        "terms": {"better_robot", "better_robot.dynamics", "better_robot.kinematics", "better_robot.optim"},
        "viz": {"better_robot.lie", "better_robot.viewer"},
        "studio": {"better_robot", "better_robot.lie", "better_robot.viewer"},  # better_robot: record 0021
        ROOT: set(),
    }.items()
}
```

## Consequences
A BetterHuman error is a BetterRobot error wherever it is raised, and a caller catches one tree. `viz` converts
rotations with BetterRobot's functions instead of a copy, and still reads no file and imports no family. A new import
of a BetterRobot submodule needs no row of its own; a new BetterRobot package does.

## Alternatives
- A BetterHuman error module re-exporting BetterRobot's classes: a second import path for each class (principle 1.6).
- For the `lie` edge (weighed in work package H5): a numpy rotation-to-quaternion copy in `viz`, the duplicate
  principle 1.3 names as its anti-example, or a public bind-pose property on `Identity`, a public name beyond record
  [0010](0010-human-viz-surface.md).
- Keep `better_robot.viewer.tools` as its own entry: the walker never reports it, so the entry could not fail.

## Compatibility
None: the table describes today's imports; no public name changes.

## Enforcement
`tests/contract/test_layer_dependencies.py` (the table above, on every import); `tests/docs/test_codemap.py` (the
package map of `docs/concepts/architecture.md` equals the table).

## Open questions
None.
