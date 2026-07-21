# Order 05 results — performance, visuals, example, and docs

Completed on branch `dev` on 2026-07-21, except for the unavailable CUDA/Warp
measurements described below. BetterRobot, `models/MHR/assets/`, and
`references/` were not modified.

## Delivered

- `benchmarks/benchmark_mhr.py`: a reproducible LOD1 float32 benchmark against
  the released TorchScript model, with the requested batch/corrective grid,
  fair pose-only vs shape+pose paths, synchronized CUDA timing support, and
  isolated FK/corrective/LBS diagnostics.
- `docs/segmentation/mhr_argmax_surface.png` and
  `mhr_argmax_split.png`, generated through the existing deterministic
  renderer. The intact mesh and exploded centroid-capped solids show all 19
  nonempty MHR reference groups.
- `examples/04_mhr_shape_pose.py`: load LOD1, bake identity and proportions
  (including accumulated hand scale), set three named revolutes, and render via
  `SkinnedMeshMode`. Its documented `uv run` headless path was executed.
- MHR support, layout/licensing caveats, rig/deformation architecture,
  segmentation, performance, parity, and singular-dynamics facts were added to
  the README and documentation. Stale SMPL-only claims and false converter
  instructions were corrected; the segmentation concept now links to a
  separate rendering guide.

## Benchmark protocol

- LOD1, float32, `compute_mass=False`; deterministic identical input rows.
- Intel Xeon Platinum 8570 (2 sockets, 56 cores/socket, SMT2), PyTorch
  `2.13.0+cu126`, BetterRobot `0.2.0`, 112 intra-op and 112 inter-op threads.
- Five warmups, then median of 30 calls using `perf_counter_ns`.
- **Pose-only:** one scalar shape baked before timing, then repeated batched
  `q`. **Shape+pose:** batched `with_shape` plus `vertices(q)` per call.
  TorchScript always recomputes native shape and pose, so shape+pose is the
  apples-to-apples row.
- The host was shared and busy during the run. Exact medians are reported as
  measured; isolated stage medians are diagnostic and are not additive.

## End-to-end CPU results

| Batch | Correctives | TorchScript shape+pose ms | BetterHuman pose-only ms | Pose/ref | BetterHuman shape+pose ms | Shape/ref |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | on | 20.169990 | 208.214920 | 10.323006× | 318.263894 | 15.779080× |
| 1 | off | 12.667384 | 226.565365 | 17.885726× | 312.925717 | 24.703263× |
| 16 | on | 48.545396 | 119.060006 | 2.452550× | 132.579338 | 2.731038× |
| 16 | off | 43.209223 | 115.589255 | 2.675106× | 117.570330 | 2.720955× |
| 256 | on | 988.033544 | 1482.259346 | 1.500212× | 1623.051953 | 1.642709× |
| 256 | off | 876.298065 | 1467.830282 | 1.675035× | 1557.705576 | 1.777598× |

The CPU batch-256 analogue meets the `≤2×` target for both BetterHuman paths.
Batch 16 misses it, and batch 1 is substantially slower. The stated acceptance
target is specifically CUDA batch ≥16, which could not be measured here.

## BetterHuman pose-only split

| Batch | Correctives | FK ms | Corrective ms | Transforms + LBS ms |
|---:|:---:|---:|---:|---:|
| 1 | on | 42.583825 | 43.082168 | 6.511565 |
| 1 | off | 42.060111 | 6.002389 | 6.326809 |
| 16 | on | 47.058248 | 5.568119 | 22.028147 |
| 16 | off | 26.154574 | 0.042477 | 24.580467 |
| 256 | on | 58.057344 | 16.201941 | 1427.753313 |
| 256 | off | 33.725507 | 0.042990 | 1346.318814 |

The BetterRobot Python/topological FK traversal is material at batch 1. At
batch 256, inverse-bind transform construction plus scaled sparse LBS dominates;
the corrective MLP is not the bottleneck. These findings do not justify a
BetterHuman copy of FK or speculative caching, so no performance workaround was
added.

## Visual and example verification

| Artifact | Size | SHA-256 |
|---|---:|---|
| MHR grouped surface | 675×975 | `480ea5af548b4ea37bbc3bb52522dfdbac3f6e289f133693bcdf2ac03af43049` |
| MHR grouped split | 975×975 | `807d4d079dd37b81526f6c27fbc226385db031340dd3f9461951e21e85a24a39` |

The images were visually inspected: all surface triangles are labeled and the
exploded view contains 19 coherent finite solids. Regenerating all ten images
left the eight existing SMPL/SMPL-X PNGs byte-identical.

The exact example smoke command succeeded:

```text
uv run python examples/04_mhr_shape_pose.py \
  --model-path models/MHR/converted/mhr_lod1.npz --headless --frames 1
Rendered 1 MHR LOD1 frames in 0.096 s with 18439 vertices
```

## Verification

- Full suite after Order 05: **223 passed, 5 skipped, 20 warnings** in
  `101.38 s`.
- `.venv/bin/ruff check .` and `git diff --check` pass.
- The benchmark, example, and changed runtime Python files pass
  `ruff format --check`.
- There is no Sphinx `conf.py` or equivalent docs build in this repository,
  so no documentation build command exists. Markdown image/link targets were
  checked directly.
- BetterRobot and the MHR reference checkout are clean.

## Deviations and findings

1. **CUDA target unjudged.** PyTorch reports no CUDA device and `warp` is not
   installed, so no CUDA batch row or true Warp row can be reported. A CPU
   result was not relabeled as CUDA success. The harness supports synchronized
   CUDA timing for a future run.
2. **Warp scope is pose-only today.** BetterRobot's fused lane accepts MHR's
   joint kinds, but a batched shaped body has batched joint placements and an
   unbatched frame-placement table, which the current Warp bridge rejects.
   `--use-warp` therefore applies only to the scalar-shape/batched-pose rows;
   the fair batched shape+pose rows remain on Torch FK. No benchmark-only model
   rewrite was added.
3. **CPU small/medium batches miss 2×.** The profile identifies BetterRobot FK
   at small batch and scaled sparse LBS at large batch. Per the plan, the BR
   traversal was reported rather than duplicated inside BetterHuman.
4. **Renderer formatting debt remains isolated.** `docs/render_segmentation.py`
   already required formatting in three unchanged render-call blocks. Its new
   lines pass lint; those unrelated blocks were not reformatted solely to make
   the diff larger.
## Primary commands

```bash
.venv/bin/python benchmarks/benchmark_mhr.py --device cpu
.venv/bin/python docs/render_segmentation.py --models-dir models
uv run python examples/04_mhr_shape_pose.py \
  --model-path models/MHR/converted/mhr_lod1.npz --headless --frames 1
.venv/bin/pytest tests/ -q
.venv/bin/ruff check .
git diff --check
```
