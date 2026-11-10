# 0021 The root imports every subpackage, and the studio takes `Viewer` from BetterRobot's root

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Principle 1.6 makes a package's `__all__` its public surface: the root's `__all__` lists its subpackages and the
first-level names, and each public name has one import path, the root for a first-level name. The owner's migration
plan lists `core`, `families`, `io`, `mass`, `terms`, `viz` and `studio` in `better_human.__all__`, but the
root listed `families` and `io` only, and 0011's table gave the root no edge to `studio`. The studio imports `Viewer`
from `better_robot.viewer`, although BetterRobot's root lists it as a first-level name (BetterRobot 0019), and 0011's
`BR_ALLOWED` row for the studio has no edge to BetterRobot's root.

## Decision
We will let the root import every subpackage, `studio` included, and list each in `better_human.__all__`; the studio
package's `__init__` imports no viser until `HumanStudio` or `launch` is read (0009), so `import better_human` stays
free of the `studio` extra. We will let the studio import BetterRobot's root, so that it takes `Viewer` from there.
This amends the root row of 0011's `ALLOWED` table and the `studio` row of its `BR_ALLOWED` table.

## Example
```python
ALLOWED = {..., ROOT: {"core", "families", "io", "mass", "studio", "terms", "viz"}}
BR_ALLOWED = {..., "studio": {"better_robot", "better_robot.lie", "better_robot.viewer"}}
```

## Consequences
`better_human.studio.HumanStudio` is reachable from the root's `__all__` tree, and the API snapshot and the changelog
test walk that one tree. The root and the studio import each other: the studio's modules import `better_human`, and
the root imports the studio package, whose `__init__` imports nothing of the root.

## Alternatives
Leave `studio` out of the root's `__all__`: the application would be the one public subpackage the root does not
list, so every tool that walks the public surface would add it by hand. Keep `from better_robot.viewer import
Viewer`: a second import path for a first-level name, which principle 1.6 rules out.

## Compatibility
None: names are added to the root, nothing is removed or renamed.

## Enforcement
`tests/contract/test_layer_dependencies.py` (the two rows above); `tests/contract/test_br_imports.py` (every
BetterRobot import is a path of BetterRobot's `__all__` tree, a first-level name from its root, with one named
exception: `better_robot.viewer.tools`);
`tests/contract/test_api_snapshot.py` (the root's `__all__` tree). On acceptance, 0011's Status line names this record
as superseding those two rows.

## Open questions
The studio imports `PoseGizmo`, `CameraFrustum`, `nearest_on_ray`, `ray_hit`, `vertices_in_rect` and
`vertices_in_sphere` from `better_robot.viewer.tools`, a module outside BetterRobot's `__all__` tree, and
`examples/03_ik_reach.py` imports `PoseGizmo` from it. Rules H-LAYER-2
and H-STUDIO-1 and their two tests name it as their one exception. BetterRobot's owner decides whether that module
becomes public; until then the exception stays written.
