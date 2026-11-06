# 0005 BetterHuman chooses dtype and device at load; BetterRobot keeps `Model.to`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
BetterHuman has `FamilySpec.to`, `Corrective.to`, three `Rig.to` and `Body(dtype=, device=)`, routed by a hidden option
set, plus a tested float32-spec-in-float64-model MHR path (`tests/families/test_mhr.py`, "upcasts ... 3e-7 m").
BetterHuman itself calls BetterRobot's `Model.to` at three build sites (`core/rig.py`, `families/soma_rig.py`,
`families/mhr_rig.py`). `bh.load("smplx")` takes 60 s on CPU today, so "reload to move" is costly until the reader fix
of 0013 lands.

## Decision
We will choose dtype and device only at load in BetterHuman and remove every `to` there, while BetterRobot keeps
`Model.to(device=, dtype=)` beside `load(dtype=, device=)`.

## Example
```python
human = bh.load("smplx", dtype=torch.float64, device="cuda")   # chosen once
# no human.to(...): to move or recast, load again (seconds after the reader fix of 0013)
model = human.model.to(device="cpu")                           # BetterRobot's Model.to stays: one tree map over a flat pytree
```

## Consequences
About 70 lines and one numerical mode (a float32 spec in a float64 body) go. A flat pytree `Model` moves with one map,
so BetterRobot keeps the method its rigs call. The asymmetry (BetterHuman reloads, BetterRobot moves) is written in
the glossary. Benchmarks and tests pass `device` to the loader.

## Alternatives
Move both top objects: `Model.to` plus a new `HumanModel.to`, which keeps every inner `to`. No `to` anywhere: BetterRobot's
`Model.to` goes, the rigs pass `device` to `build_model`, and every BetterRobot, BetterVideoReconstruction and
BetterRetarget caller changes. smplx models are `nn.Module`s moved with `.to`; momentum casts with explicit scalar
templates. A frozen model that is rebuilt once is the simpler contract.

## Compatibility
Breaks `FamilySpec.to`, `Corrective.to`, `Rig.to`, `Body(dtype=, device=)` and the float32-spec path. Callers that
moved a loaded model reload or use `human.model.to(...)`. Partly reversible: adding `HumanModel.to` later is compatible.

## Enforcement
New `tests/contract/test_dtype_device.py`: no public type has a `to` attribute, no `def to` appears in
`src/better_human`, and a dtype mismatch between spec and request raises `DtypeMismatchError` (BetterRobot 0031); the MHR parity
gates of 0003 are re-measured after this change in a commit of its own.

## Open questions
None.
