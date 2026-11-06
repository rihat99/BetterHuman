# 0001 BetterHuman's `Body` becomes a frozen `HumanModel` built by one factory

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Three classes are called `Body`: BetterRobot's rigid-body record, BetterHuman's family on a device and
BetterVideoReconstruction's wrapper; in BetterRobot's glossary a body is a rigid body (3.9). `core/body.py::Body.__init__`
bakes before `self.topology` is set, rebuilds the BetterRobot model when contact frames are given, stays mutable and
forwards 8 properties (`Identity` forwards 4). "Identity" names the class, a blend group, MHR's group and
`identity_coeffs`. `Identity` and `HumanState` print 45,070 and 120,529 characters.

## Decision
We will rename `Body` to `HumanModel`, a frozen dataclass built by `HumanModel.from_spec(spec, *, density=,
contact_frames=, stream_joints=)` that computes every input first and constructs once, with `human.model` the
zero-identity BetterRobot model, no forwarding properties, the class `Identity` kept, and the spec fields `identity`
and `streams` renamed `identity_bases` and `stream_bases`.

## Example
```python
human = bh.load("smplx", dtype=torch.float64)       # HumanModel, frozen
identity = human.bake(betas=betas)                  # Identity
state = identity.forward(q, streams={"expression": expression})   # HumanState
human.spec.faces, identity.model.nq                 # no forwarding properties
```

## Consequences
A half-built object cannot be observed. "Body" keeps one meaning across the stack. `HumanModel` pairs with
`HumanState`. The method pipeline `human.bake(...)` then `identity.forward(q, ...)` stays; BetterRobot's
`forward_kinematics` is a different operation from BetterHuman's `forward` (kinematics plus skinning), which the
glossary says (BetterRobot 0027). `eq=False` on every type and one-line reprs follow from BetterRobot 0010 and principle 4.9.

## Alternatives
`BodyModel`: the field's term of art, but "body" keeps two meanings. `Human`: short, but `human` already names a
`HumanState` in `terms.py`. A mutable class with read-only properties hides nothing a frozen dataclass does not. smplx
names each family as an `nn.Module` and mutates it with `.to`; momentum's `Character` is the closest immutable
precedent.

## Compatibility
Breaks `Body` and every spelling of it: every BetterHuman doc, example and app call site; BetterVideoReconstruction
renames its wrapper in its own migration. Not reversible after the apps migrate.

## Enforcement
New `tests/contract/test_frozen.py` (`HumanModel`, `Identity` and `HumanState` are frozen with `eq=False`, and each
class's field table lists its fields, principle 1.4); new `tests/contract/test_repr.py` (one line);
`tests/contract/test_keyword_only.py`; `tests/docs/test_snippets.py` runs every page on the new names.

## Open questions
None.
