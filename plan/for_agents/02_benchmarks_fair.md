# Order 02 — workload-framed benchmarks + gradient benchmarks

The owner's critique of the current tables: a `× original` ratio printed
next to our *pose-only* row against upstream's *shape+pose* forward compares
two different computations. Fix the framing so every ratio compares like
with like, and add the missing backward-pass measurements. Depends on
order 01 (uses the one-call API and the new import paths).

## Ground rules (binding)

- **No BetterRobot modifications**; never modify `models/` or `references/`.
- Honest numbers: synchronized medians of >= 30 with 5 warmups, report
  device/torch/driver and shared-host caveats, no selective reruns.
- Keep the pre-timing output-agreement gate (max abs vertex error <= 1e-6 m
  vs upstream before any timing) — a benchmark of mismatched computations
  is worthless.
- Write `plan/for_agents/02_results.md` with the full tables.

## 1. Reframe `benchmarks/benchmark_smpl.py` by workload

Restructure the output into two workload tables, each internally fair:

**Workload A — full forward (new shape + new pose every call).** This is
training-style usage and the apples-to-apples comparison. Rows: upstream
`smplx` full forward vs BetterHuman `body(q, betas=...)` (the order-01
one-call path). The `× upstream` ratio lives here and only compares these
two identical computations.

**Workload B — streaming poses (shape fixed, poses vary per call).** This
is animation/tracking-style usage. Rows: BetterHuman `with_shape` once
outside timing + `vertices(q)` per call, vs upstream's per-call cost for
the same workload — which is its full forward, because upstream exposes no
shape-amortized API. State that fact in one table note; the ratio is then a
workload-level comparison (what each library costs per frame in this
usage), not a FLOP-matched one, and the note says so.

Keep: the stage-split table (FK / corrective / transforms+LBS), the
agreement table, CPU + CUDA + `--use-warp` runs, batches 1/16/256, f32.
Drop nothing else from the protocol. Apply the same two-workload framing to
`benchmarks/benchmark_mhr.py`'s table headers (its reference TorchScript
baseline is also a full shape+pose forward), without re-measuring anything
beyond what the reframe requires.

## 2. Gradient benchmarks

Extend `benchmarks/benchmark_smpl.py` with a `--grad` mode (same grid,
`torch.inference_mode` must not wrap these sections):

- **Workload A backward:** loss = `vertices.sum()`; ours: gradients wrt
  `(q, betas...)` through `body(q, betas=...)`; upstream: gradients wrt
  `(betas, body_pose, global_orient, ...)` through its full forward.
- **Workload B backward:** shape baked outside timing; ours: gradient wrt
  `q` through `vertices(q)`; upstream: gradient wrt pose inputs through its
  full forward (same note as above).
- Report per row: forward-only ms, forward+backward ms, and the
  backward/forward ratio. Known and already attributed (see
  `BR_wishlist.md` §2): backward mirrors the launch-bound BetterRobot FK
  graph, so expect ours ≈ 9–11× upstream absolute with no separate
  backward pathology — measure and document, do not re-investigate or work
  around. Only a *new* anomaly (e.g. a stage other than FK dominating
  backward) warrants a profile.
- MHR: add the same `--grad` mode to `benchmarks/benchmark_mhr.py` for the
  pose-only lane (wrt `q`) and shape+pose (wrt `q` + shape groups), no
  upstream counterpart — record absolute numbers and the ratio.
- Gradcheck correctness is already covered by tests; this order measures
  speed only.

## 3. Documentation

- Replace the SMPL/SMPL-X and MHR performance tables in
  `docs/architecture.md` with the two-workload framing and add the gradient
  tables (medians + backward/forward ratios). Replace, don't append.
- Update the observational baselines in `docs/contracts.md` to the same
  framing; keep enforced-vs-observational wording intact.
- The FK bottleneck story and the per-component attribution live in
  `BR_wishlist.md` (repo root, owner-maintained) — docs may link to it but
  the wishlist itself is not part of this order.

## Acceptance

- Both benchmark scripts run reproducibly with documented commands; every
  printed ratio compares either identical computations (workload A) or an
  explicitly-framed identical workload (workload B).
- Gradient tables present for SMPL, SMPL-X (vs upstream) and MHR
  (absolute); any pathological backward ratio reported with attribution.
- Docs updated with real medians; full suite green; `ruff check .` clean.
