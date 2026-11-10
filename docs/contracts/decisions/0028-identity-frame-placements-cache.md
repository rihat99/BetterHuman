# 0028 An identity caches the placements of each contact-frame set it is asked for

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10 (recorded; decision taken earlier)

## Context
`HumanState.contact_frame_poses(frames)` gives the poses a model built with `contact_frames=frames` would report,
without building that model. Each call needs the frames' placements on their parent joints, computed from the rest
mesh, and the check that the family's rig takes contact frames at all, which builds a model. Both depend on the
identity and the frame set alone, yet the studio calls `contact_frame_poses` on every pose update of one identity.
Principle 11.6 lists a cache among the six kinds of change that start as a record; until now BetterHuman had one
cache, `HumanState._memo`, named in the design note of `src/better_human/core/human_model.py`, and no test that
names its caches.

## Decision
We will keep a second cache, `Identity._frame_placements`: a dictionary on the identity, outside its constructor
(`init=False`, so `dataclasses.replace` starts it empty), keyed by the frame set (names, vertex ids, parent joint
ids) and the grad mode, holding the model joint id of each frame's parent and the frames' placements
`(I..., N, 7)`. Only `HumanState.contact_frame_poses` reads or writes it. These two, `HumanState._memo` and
`Identity._frame_placements`, are BetterHuman's only caches; a new one needs its own record.

## Example
```python
state = identity.forward(q)
state.contact_frame_poses({"heel": (3458, None)})    # computes and checks the placements, keeps them
identity.forward(q2).contact_frame_poses({"heel": (3458, None)})    # reuses them: one compose per call
```

## Consequences
A pose update in the studio composes the cached placements with the joint poses and builds no model. The cache
grows by one entry per distinct frame set and grad mode an identity is asked for, and lives as long as the identity.
A value made under `no_grad` never serves a differentiable call.

## Alternatives
Compute the placements on every call: the studio's pose update would build a BetterRobot model each time. Ask the
caller to build a model with `contact_frames=frames` once: a frame set being edited in the studio changes on every
click, and each change would rebuild the body. A `functools` cache keyed by the identity: it would outlive the
identity and hold its tensors for the life of the process.

## Compatibility
None: no public name or layout changes.

## Enforcement
`tests/contract/test_caches.py`: the declared caches are the only fields outside a constructor, no module keeps a
store, no `functools` cache sits outside its reasoned list, only the owning function reaches each cache, a replace
starts them empty and their keys carry the grad mode.

## Open questions
1. `HumanState.vertices` and `skin_affines` return the memoised tensor itself, so an in-place edit by a caller
   changes later reads; cloning them adds an operator to the counter rows. Clone them, or document the aliasing?
