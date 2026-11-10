# 0010 BetterHuman's `viz` is one `layers` module with state-driven functions

**Status:** Accepted (owner); amendment proposed in 0020. **Date:** 2026-11-01

## Context
`viz/__init__.py` imports the function `human_layers` over its own module of the same name. `skinned_layer(identity,
state)` passes the identity twice and is wrong for SOMA (0003). `forces_layer` takes joint or frame ids. Examples
repeat `theme=replace(LIGHT, up="+y")`, add layers one call at a time (13 loops) and reach into raw viser.

## Decision
We will name the module `viz/layers.py`, keep `HumanView`, `human_layers`, `mesh_layer`, `skeleton_layer` and
`frames_layer`, change `skinned_layer` to `skinned_layer(state, *, name, color, opacity)`, make `forces_layer` take
frame names, put one human theme constant in BetterHuman (BetterRobot stays human-agnostic), and add layers through
BetterRobot's iterable `Viewer.add` (BetterRobot 0022).

## Example
```python
from better_human import viz
view = viz.HumanView(state)
viewer.add(view, group="human")                  # was: for layer in view: viewer.add(layer, group="human")
viewer.add(viz.skinned_layer(state, name="skin", color=(0.8, 0.8, 0.9), opacity=0.7))
viewer.add(viz.forces_layer(state, frame_names=["left_foot", "right_foot"], forces=wrenches))
```

## Consequences
`better_human.viz.human_layers` is the function, not a shadowed module. Examples lose the repeated theme and loop
lines. The folded alternative (layers inside the studio) is not taken because BetterVideoReconstruction uses them.

## Alternatives
Fold the layers into the studio: the apps that use them would bring them back. Keep the names and fix `skinned_layer`
only: the shadowing and the id arguments stay.

## Compatibility
Breaks `skinned_layer(identity, state)`, `forces_layer` with ids and the module path `viz.human_layers`. Migration:
BetterVideoReconstruction and BetterRetarget call sites (one line each).

## Enforcement
New `tests/contract/test_public_surface.py` (no module under `better_human` shares a name with an exported function,
0011); `tests/viz/test_layers.py::test_skinned_layer_matches_state_vertices` (0003); `tests/contract/test_keyword_only.py`.

## Open questions
None.
