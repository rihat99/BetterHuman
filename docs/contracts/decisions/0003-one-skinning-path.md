# 0003 Every rig skins through inverse-bind affines

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`core/pipeline.py::skin_affines` branches between rest joints and `inverse_bind_poses`, plus MHR's `joint_scales` and
`meta["bind_joint_positions"]`. Six functions with no caller duplicate it (`lbs`, `linear_corrective`,
`mlp_corrective`, `dense_skinning_weights`, `skinning_transforms`, `skinning_transforms_from_inverse`).
`viz/human_layers.py::skinned_layer` builds identity-rotation binds and is off by 0.292 m for SOMA at the neutral pose
(SMPL: 3e-8 m).

## Decision
We will skin every rig with `skin_affines(joint_transforms, inverse_bind_affines)`: SMPL uses `[I | -rest_joints]`,
SOMA its authored binds, MHR folds its scales into the inverse binds at bake; `skinned_layer(state, *, name, color,
opacity)` derives binds from `inverse_bind_affines` and raises for non-rigid binds; the six dead functions go.

## Example
```python
# MHR fold at bake time, per joint: scale s, bind rotation R_b, bind translation t_b, bind position c
inverse_bind = concat(s * R_b, t_b - (s - 1) * R_b @ c)         # (J, 3, 4), pose independent
affines = skin_affines(state.joint_transforms, identity.inverse_bind_affines)
layer = viz.skinned_layer(state, name="soma", color=(0.8, 0.8, 0.9), opacity=0.7)   # correct for SOMA
```

## Consequences
One path to test; SOMA's viewer error disappears. SMPL results are bitwise equal to today's (negation and commuted
addition are exact in IEEE). Float32 MHR results move at the ulp level. The MHR fold lands in a commit of its own, with
the MHR gates measured before and after (float32 tolerance 5e-6 m); a gate's tolerance never moves (11.7).

## Alternatives
Keep per-rig branches and fix only `skinned_layer` for SOMA: the dead duplicates and the branch stay. momentum skins
with `jointState.transform · inverseBindPose`, precomputed once from the rest pose, for every character; SOMA-X
carries a per-identity bind pose. This is the same single path.

## Compatibility
Breaks `Identity.joint_scales`, `Identity.inverse_bind_poses`, `meta["bind_joint_positions"]`, the six functions and
`skinned_layer(identity, state)`. Apps: BetterRetarget's use of `lbs_affines` and `skin_affines` moves to
`core.linear_blend_skinning` and `core.skin_affines`.

## Enforcement
The parity gates in `tests/parity/` (SMPL 1e-6 m, SMPL-X, MHR LOD 0-6 in float32 and float64, SOMA);
`tests/viz/test_layers.py::test_skinned_layer_matches_state_vertices` (the client-side LBS of `skinned_layer` equals
`HumanState.vertices` at `q_neutral` for every rig; the 0.292 m probe becomes the test); `tests/core/test_rig_properties.py`
(0002).

## Open questions
None.
