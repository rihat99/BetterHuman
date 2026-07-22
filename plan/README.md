# Plan — API revision, package layout, honest benchmarks

The previous cycle (simplify → matrix-blend LBS → SMPL-family benchmarks)
landed in `46b1bd0` and was verified claim-by-claim by an independent
review: all findings applied or skipped-with-reason, parity within every
pin, 226/226 tests green with CUDA, zero transcription errors in the docs
tables. Historical orders/results: `git show 46b1bd0:plan/for_agents/`.

The benchmarks settled the performance question. Our deformation stack is
fast (batch-256 CUDA transforms+LBS ≈ 5 ms, correctives ≈ 0.5 ms); the gap
to upstream `smplx` is **BetterRobot Torch FK, which is kernel-launch-bound
at ~78 launches per joint** (~73% from the hand-expanded quaternion product
in `so3._quat_mul` alone). Upstream issues ~1.2 launches per joint. The
full measured attribution — forward, backward (which mirrors the forward
graph, no separate pathology), and the Warp-lane gap — lives in
[`BR_wishlist.md`](../BR_wishlist.md) at the repo root. BetterHuman does
not work around it.

Two orders remain, in order:

| Order | File | What | Depends on |
|---|---|---|---|
| 01 | [`for_agents/01_architecture.md`](for_agents/01_architecture.md) | one-call `body(q, **shape)` API; `bodies/` family subpackages; MHR split into a readable package; composition decision documented | — |
| 02 | [`for_agents/02_benchmarks_fair.md`](for_agents/02_benchmarks_fair.md) | reframe benchmarks by workload so every ratio is like-for-like; add gradient benchmarks; docs sweep | 01 |

Each order ends with the full test gate green and writes an honest
`for_agents/0N_results.md`. Subagents run on Opus; CUDA available (RTX
6000 Ada; GPU 7 was idle at planning time).

## Targets

- Order 01 is behavior-preserving: parity maxima byte-identical
  before/after; public API (`bh.SMPL/SMPLX/MHR/load`, `*Classic`)
  unchanged; no module in `bodies/` over ~300 lines; no new abstractions.
- Order 02: every printed ratio compares identical computations
  (workload A: full forward) or an explicitly framed identical workload
  (workload B: streaming poses); gradient tables for SMPL/SMPL-X vs
  upstream and MHR absolute; measured FK-inherited gaps are documented,
  not re-investigated.

## Standing decisions (owner-confirmed)

- Composition, not inheritance: `ParametricBody` composes a
  `better_robot.Model`; `BodyStructure`/`BodyValues` do not subclass
  `ModelStructure`/`ModelValues` (pytree exact-type registration, colliding
  field vocabularies, BR's own Model is itself composed; order 01 records
  the full rationale in docs).
- No BetterRobot changes from this repo; gaps go to `BR_wishlist.md` with
  measurements, never worked around.
- No Warp/CUDA kernels in BetterHuman (measured unnecessary; deformation
  is ~5 ms — the bottleneck is FK, which is BetterRobot's).
- Parity pins never bumped silently — exceed ⇒ stop and report.
- Alpha: breaking changes allowed, no back-compat shims. Everything on
  `dev`.

## Beyond this plan (roadmap, not scheduled)

MANO / SMPL-H / FLAME; STAR / SUPR; Anny; MHR follow-ups (authored closed
segmentation, landmarks, SMPL↔MHR conversion, LOD remap tools); keypoint
heads; human residuals on the BR optimizer; human→robot retargeting;
differentiable measurements.
