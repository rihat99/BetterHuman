# 0011 BetterHuman packages: `io, mass, core, families, terms, viz, studio`, with two edge tables

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`core` holds the SMPL-only `RegressedRig` and `drop_joints` and so imports `better_robot.io`. Mass math lives in
`core/mass.py` while its assets live in `families/assets`. `io/convert_chumpy.py` patches `inspect.getargspec`, numpy
aliases and `sys.modules["chumpy"]` process-wide. Functions shadow modules (`families.mhr`, `families.soma`). The
layer test misses `from pkg import subpackage` edges, and `test_br_imports.py` lets `better_robot.dynamics.state` and
`better_robot.spatial.mesh` through.

## Decision
We will lay BetterHuman out as `io`, `mass`, `core`, `families`, `terms`, `viz` and `studio` under the two edge tables
below, move `RegressedRig` and `drop_joints` to `families` and `convert_chumpy` to `tools/convert/smpl/`, let no module
share a name with an exported function, and walk every import with one walker that resolves submodule imports.

## Example
```python
ALLOWED = {"io": set(), "mass": set(), "core": {"mass"}, "families": {"core", "io", "mass"}, "terms": {"core"},
           "viz": {"core", "io"}, "studio": {ROOT, "core", "families", "io", "mass", "viz"},
           ROOT: {"core", "families", "io", "mass", "terms", "viz"}}
BR_ALLOWED = {"io": set(), "mass": {"better_robot.lie", "better_robot.spatial"},
              "core": {"better_robot", "better_robot.kinematics", "better_robot.lie"},
              "families": {"better_robot", "better_robot.io", "better_robot.kinematics", "better_robot.lie"},
              "terms": {"better_robot", "better_robot.dynamics", "better_robot.kinematics", "better_robot.optim"},
              "viz": {"better_robot.viewer"}, "studio": {"better_robot.lie", "better_robot.viewer", "better_robot.viewer.tools"}}
```

## Consequences
`core` no longer imports `better_robot.io`; mass and its data sit together; the chumpy patching leaves the library. One
walker (from `better_robot.testing`, BetterRobot 0035) serves both repositories. Cost: seven packages for a small library.

## Alternatives
Keep `mass` inside `core`: a smaller graph, but `core` keeps two jobs and its assets live elsewhere. Pinocchio keeps the
model, the geometry model and the visualizers in separate modules with one-way dependencies, as here.

## Compatibility
Breaks `core.RegressedRig`, `core.drop_joints`, `core.mass.*` and `io.convert_chumpy` import paths; apps import few of
them (0001 to 0010 list the rest). Adding an edge later needs a record (11.6).

## Enforcement
`tests/contract/test_layer_dependencies.py` rewritten to walk every import and check `ALLOWED`; `tests/contract/test_br_imports.py`
replaced by a check of `BR_ALLOWED` plus the principle 1.6 consumer import test; `tests/contract/test_public_surface.py`
(no module name equals an exported function name).

## Open questions
None.

## Note (2026-11-06)
`viz -> io` is a permanent allowed edge, not a ratchet entry: the studio persists frame sets through
`better_human.io` (`read_frame_set`, `write_frame_set`; record 0013). The `ALLOWED` table above gained it, as did
`tests/contract/test_layer_dependencies.py` (commit `6f0b329`).
