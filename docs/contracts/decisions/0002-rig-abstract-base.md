# 0002 `Rig` is an abstract base with declared hooks

**Status:** Accepted (owner); amendment proposed in 0026. **Date:** 2026-11-01

## Context
`core/rig.py::Rig` is a Protocol; `MhrRig` and `SomaRig` override four hooks with `return None`. Code is copied
between the rigs: `gravity(up_axis)` (3 copies), `build_model(..., preserve_joint_order=True)` (3), and a five-line
bake tail in `MhrRig.bake` and `SomaRig.bake`. `RigBake` has 8 fields, 4 of them optional. `RegressedRig` and
`drop_joints` serve the SMPL family only but live in `core`, which pulls `better_robot.io` into `core`.

## Decision
We will make `Rig` an `ABC` with four abstract members (`build_model`, `bake`, `joint_ids`, `corrective_features`)
and three hooks with defaults on the base, final frozen `RegressedRig`, `MHRRig` and `SOMARig` in `families`, a
`RigBake` without optional fields, pelvis centring inside `RegressedRig.bake`, and the common bake tail as core code.

## Example
```python
class Rig(ABC):
    moves_joints_with_streams: ClassVar[bool] = False
    @abstractmethod
    def build_model(self, spec: FamilySpec, *, frames: Sequence[FrameSpec]) -> Model: ...
    @abstractmethod
    def bake(self, spec, model, v_shaped, identity) -> RigBake: ...   # every RigBake field is required
    def stream_joint_basis(self, spec, name) -> Tensor | None: return None
    def placements_from_rest(self, spec, model, rest_joints) -> Tensor | None: return None

@dataclass(frozen=True)
class RigBake: ...   # joint_placements, rest_vertices, rest_joint_translations, inverse_bind_affines, root_offset
```

## Consequences
MHR and SOMA drop four trivial overrides each. Defaults live in one place and are called directly, with no `getattr`
probe. `core` no longer imports `better_robot.io`. Cost: one abstract class that every future family subclasses.

## Alternatives
Keep the Protocol: no place to declare defaults, the four `return None` methods stay. Rigs as data only: SOMA's
skeleton fit and twist helpers are code. momentum's `Character` bundles skeleton, parameter transform and skin weights
as data and has no overridable rig behaviour; the families here differ in code, so a base class fits better.

## Compatibility
Breaks `Rig` as a Protocol, `RegressedRig` in `core`, `MhrRig` and `SomaRig` (renamed `MHRRig`, `SOMARig`),
`native_joint_ids` (now `joint_ids`) and `RigBake` fields. Apps use `RegressedRig.joint_regressor`, which stays.

## Enforcement
New `tests/contract/test_abstract_or_final.py` (every class under `families` and the `core` rig module is abstract or
`@final`); new `tests/core/test_rig_properties.py`, a property checker over all rigs: forward kinematics at
`q_neutral` reproduces `rest_vertices` through `skin_affines` (0003).

## Open questions
None.
