# Order 05 — performance, visuals, example, docs

> **Implementation log (2026-07-21):** Added the reproducible benchmark,
> MHR's two 19-part segmentation renders, a headless-tested LOD1 example, and
> code-true docs; CUDA suite: 228 passed. Warp pose-only meets `≤2×` at batch
> 16 (1.332028×/1.761994×) but misses at 256; fair shape+pose remains on Torch
> FK and misses. See `05_results.md` for the complete CPU/CUDA tables.

The owner's explicit ask: "check that we are not super slow compared to them."
Also: segmentation renders, an example, and docs made true to the code.
Depends on orders 01–04.

## Ground rules (binding)

- **No BetterRobot modifications.** If profiling shows a BR bottleneck
  (e.g. the per-joint Python FK loop over ~175 sub-nodes), measure it,
  report it in the results file as a BR finding, and do **not** work around
  it by duplicating BR functionality in BetterHuman.
- Never modify `models/MHR/assets/` or `references/`.
- Honest numbers: report medians as measured, never rounded to "fast enough".
- Write `plan/for_agents/05_results.md` with the full benchmark table.

## 1. Benchmark vs the reference TorchScript

Yardstick: `models/MHR/assets/mhr_model.pt` (LOD1) — loads with plain torch in
our venv, runs the reference's full forward (blendshapes recomputed every
call, correctives, LBS).

Protocol:

- LOD1; CPU and CUDA; batch 1 / 16 / 256; correctives on and off; float32.
- Warm-up iterations, then median over ≥ 30 iterations;
  `torch.cuda.synchronize()` around timings; report device + torch version.
- Ours, two rows per configuration — this is the fairness note, state it in
  the table:
  1. **pose-only** (`vertices(q)` after a prior `with_shape` — our design's
     fast path; shape baked once);
  2. **shape+pose** (`with_shape` + `vertices(q)` per call — the apples-to-
     apples comparison, since the reference recomputes shape every forward).
- Report the internal split (FK / corrective MLP / LBS) via a simple timing
  breakdown, and try BR's warp FK lane where applicable (`use_warp=True`).

**Target: ours ≤ 2× the TorchScript time at CUDA batch ≥ 16.** If missed, fix
obvious wins only (the corrective matmul and LBS should dominate; no
speculative caching layers). If the FK loop dominates instead, that is a
reported BR finding. All numbers go into the results file and
`docs/architecture.md`.

## 2. Segmentation renders

MHR argmax segmentation (surface + split/exploded views) rendered via the
existing `docs/render_segmentation.py` pattern into `docs/segmentation/`,
referenced from `docs/segmentation.md` alongside the SMPL/SMPL-X images.
Note in the page that MHR has no authored closed-3D asset (argmax only;
roadmap).

## 3. Example

`examples/` gains a small MHR demo mirroring the existing examples: load →
`with_shape` (identity + proportions incl. a hand-scale tweak) → pose a few
named q entries → viz via the skinned-mesh render mode. Verify the viz mode
works with MHR meshes (18 439 verts at LOD1). Runnable with
`uv run python examples/<name>.py` given `$BETTERHUMAN_MODELS_DIR`.

## 4. Docs

- `docs/architecture.md`: MHR section — authored-skeleton path (no joint
  regressor), typed revolute tree + single-source mimic + spine stacking,
  channel-type split (rotations → q, translations/scale → proportions),
  exp2 hand scale in skinning, LODs, which seams got promoted in order 02 and
  what remains a documented seam; the benchmark summary.
- `docs/contracts.md`: already amended in order 04 (f32-oracle parity + MHR
  baselines) — verify it matches the final numbers; add the perf baseline row.
- `README.md`: model table gains MHR (LODs, flags, weights layout
  `models/MHR/…`, Apache-2.0 note with the verify-the-release-archive caveat).
- Every doc page or docstring falsified by orders 01–05 is fixed here at the
  latest (each order should have fixed its own).

## Acceptance

- Benchmark table with real measured numbers in results + architecture doc;
  target met or the miss explained with the profile split.
- Images exist and are referenced; example runs; docs true to code
  (spot-check: every MHR claim in architecture/contracts corresponds to
  shipped behavior).
- Full suite green.
