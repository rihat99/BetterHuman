# 0026 `Rig` has four hooks with defaults, and its property holds in two forms

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0002](0002-rig-abstract-base.md) makes `Rig` an abstract base with four abstract members and "three hooks with
defaults"; its Example shows `stream_joint_basis`, `placements_from_rest` and a class flag `moves_joints_with_streams`.
Since then the flag went (commit `44382be`: a rig whose streams move joints says so by returning a basis from
`stream_joint_basis`), and `without_joints` was added for `families.drop_joints` (commit `21f104d`),
next to the hooks `joint_transforms`, `stream_joint_basis` and `placements_from_rest`. 0002's Enforcement names a
property, "forward kinematics at `q_neutral` reproduces `rest_vertices` through `skin_affines`", which is false by
design for SOMA, which binds in its authored A-pose while `q_neutral` is the T-pose, and for MHR with nonzero
proportions, whose joint scales scale the skin affines.

## Decision
We will give the `Rig` ABC four abstract members (`build_model`, `bake`, `joint_ids`, `corrective_features`) and four
hooks with defaults (`joint_transforms`, `stream_joint_basis`, `placements_from_rest`, `without_joints`), and state its
property in two forms, checked for every rig:

- joint form: at `q_neutral`, `skin_affines` carry every baked rest joint (`RigBake.rest_joint_translations`) to its
  forward-kinematics position, for every identity;
- mesh form: at the bind configuration, where every skin affine is the identity, forward kinematics reproduces
  `rest_vertices`; that configuration is `q_neutral` for the SMPL family and for MHR at zero proportions, and a `q`
  read from the inverse binds for SOMA. MHR with nonzero proportions has no such configuration and is listed as not
  held, with its measured gap.

This amends 0002's hook list and its enforcer's property.

## Example
```python
class Rig(ABC):
    @abstractmethod
    def build_model(self, spec, *, frames) -> Model: ...
    @abstractmethod
    def bake(self, spec, model, *, shaped_vertices, identity) -> RigBake: ...
    @abstractmethod
    def joint_ids(self, spec, model) -> Tensor: ...
    @abstractmethod
    def corrective_features(self, spec, model, *, q, identity) -> Tensor: ...
    def joint_transforms(self, spec, model, *, kinematics) -> Tensor: ...         # default: BetterRobot's
    def stream_joint_basis(self, spec, name) -> Tensor | None: return None
    def placements_from_rest(self, spec, model, *, rest_joint_translations) -> Tensor | None: return None
    def without_joints(self, spec, dropped) -> FamilySpec | None: return None    # None: drop_joints raises
```

## Consequences
A rig overrides only what its family does differently; `RegressedRig` alone implements `without_joints`. The property
test covers SOMA and MHR instead of excluding them, and a rig that breaks either form fails it.

## Alternatives
- Keep the class flag next to `stream_joint_basis`: two declarations of one fact that can disagree.
- A `drop=` option on the SMPL-family loaders instead of the `without_joints` hook (parked): one hook fewer,
  but `families.drop_joints` would no longer serve every family through one function.
- Keep the one-form property and exclude SOMA and MHR: the two families whose binds differ most would go unchecked.

## Compatibility
None for users: `Rig` subclasses live in-tree with their families (0002) and changed in the same commits.

## Enforcement
`tests/core/test_rig_properties.py` (both forms over every rig, its `NOT_HELD` table for the case that does not
hold); `tests/contract/test_abstract_or_final.py` (`Rig` abstract, every rig `@final`).

## Open questions
1. If a loader option replaces `without_joints`, this record lists three hooks and says why the fourth went.
