# Order 05 results — performance, visuals, example, and docs

Completed on branch `dev` on 2026-07-21, including the CUDA and optional Warp
follow-up. BetterRobot, `models/MHR/assets/`, and `references/` were not
modified.

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
- CUDA: NVIDIA RTX 6000 Ada Generation GPU 7 (48 GiB, compute capability 8.9,
  driver 560.35.03), selected at 19 MiB allocated and 0% utilization. PyTorch
  used CUDA 12.6; the fused run used optional `warp-lang==1.15.0`. Every timed
  CUDA call was explicitly synchronized.
- **Pose-only:** one scalar shape baked before timing, then repeated batched
  `q`. **Shape+pose:** batched `with_shape` plus `vertices(q)` per call.
  TorchScript always recomputes native shape and pose, so shape+pose is the
  apples-to-apples row.
- The CPU host was shared and busy during its run. Exact medians are reported
  as measured; isolated stage medians are diagnostic and are not additive.

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
Batch 16 misses it, and batch 1 is substantially slower.

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

## End-to-end native Torch CUDA results

| Batch | Correctives | TorchScript shape+pose ms | BetterHuman pose-only ms | Pose/ref | BetterHuman shape+pose ms | Shape/ref |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | on | 3.577612 | 71.391603 | 19.955099× | 143.908253 | 40.224667× |
| 1 | off | 2.306796 | 71.322613 | 30.918474× | 143.802566 | 62.338658× |
| 16 | on | 4.011412 | 74.835912 | 18.655753× | 149.518514 | 37.273288× |
| 16 | off | 2.426612 | 73.981340 | 30.487503× | 149.872687 | 61.762114× |
| 256 | on | 12.174137 | 159.218377 | 13.078412× | 288.315912 | 23.682657× |
| 256 | off | 9.818413 | 170.745595 | 17.390346× | 343.478717 | 34.983120× |

The native BetterRobot Torch FK traversal launches work joint by joint. It
costs 70.318152/70.748671 ms at batch 1, 72.660819/73.150610 ms at batch 16,
and 129.676297/125.557301 ms at batch 256 (correctives on/off). At batch 256,
the remaining correctives and transforms + LBS stages are
1.942327/0.008762 ms and 30.229231/27.746977 ms respectively. The fair
shape+pose path also rebuilds shaped model values and therefore remains far
outside the target on this lane.

## Warp pose-only CUDA results

The optional BetterRobot Warp lane applies only after one scalar shape is
baked. Reference medians below come from the same Warp-enabled invocation as
each BetterHuman median, so each ratio is within-run.

| Batch | Correctives | TorchScript shape+pose ms | BetterHuman Warp pose-only ms | Pose/ref | Warp FK ms | Corrective ms | Transforms + LBS ms |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | on | 2.261189 | 2.234005 | 0.987978× | 0.875298 | 1.017433 | 0.556030 |
| 1 | off | 1.175796 | 1.426103 | 1.212883× | 0.880708 | 0.017089 | 0.579417 |
| 16 | on | 2.590233 | 3.450263 | 1.332028× | 0.920807 | 1.054835 | 1.503042 |
| 16 | off | 1.300680 | 2.291790 | 1.761994× | 0.917623 | 0.016657 | 1.366592 |
| 256 | on | 12.741430 | 33.835041 | 2.655514× | 1.552150 | 1.992780 | 30.067871 |
| 256 | off | 10.096905 | 29.416434 | 2.913411× | 1.555098 | 0.008694 | 27.737855 |

Warp removes the FK bottleneck and meets `≤2×` at batch 16. Batch 256 still
misses because transforms plus scaled sparse LBS alone exceed twice the
reference. The overall CUDA batch ≥16 target is not met; only the two Warp
pose-only batch-16 cells pass. The fair batched shape+pose path cannot use
Warp: BetterRobot currently rejects the batched joint-placement/unbatched
frame-placement combination produced by a batched shape bake. Its native
ratios are 37.273288×/61.762114× at batch 16 and 23.682657×/34.983120× at
batch 256.

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

- Original CPU-only full suite: **223 passed, 5 skipped, 20 warnings** in
  `101.38 s`.
- Focused CUDA gate: **5 passed** in `7.77 s`.
- Full suite with CUDA visible: **228 passed, 20 warnings, 0 skips** in
  `125.83 s`.
- `.venv/bin/ruff check .` and `git diff --check` pass.
- The benchmark, example, and changed runtime Python files pass
  `ruff format --check`.
- There is no Sphinx `conf.py` or equivalent docs build in this repository,
  so no documentation build command exists. Markdown image/link targets were
  checked directly.
- BetterRobot and the MHR reference checkout are clean.

## Deviations and findings

1. **CUDA target is not met overall.** Only the two Warp pose-only batch-16
   cells are within `2×`; that path reaches 2.655514×/2.913411× at batch 256,
   and fair shape+pose misses at both required batches. The split attributes
   the former miss to scaled LBS and the latter to the unfused BetterRobot
   Torch FK path.
2. **Warp scope is pose-only today.** BetterRobot's fused lane accepts MHR's
   joint kinds, but a batched shaped body has batched joint placements and an
   unbatched frame-placement table, which the current Warp bridge rejects.
   `--use-warp` therefore applies only to the scalar-shape/batched-pose rows;
   the fair batched shape+pose rows remain on Torch FK. No benchmark-only model
   rewrite was added.
3. **The benchmark needed explicit CUDA cleanup.** The initial all-batch run
   retained the reference model and final batch tensors while emptying the
   allocator cache. Live references are now cleared before cache cleanup;
   batch 256 then completed without changing the measured workload.
4. **CPU small/medium batches miss 2×.** The profile identifies BetterRobot FK
   at small batch and scaled sparse LBS at large batch. Per the plan, the BR
   traversal was reported rather than duplicated inside BetterHuman.
5. **Renderer formatting debt remains isolated.** `docs/render_segmentation.py`
   already required formatting in three unchanged render-call blocks. Its new
   lines pass lint; those unrelated blocks were not reformatted solely to make
   the diff larger.
## Primary commands

```bash
.venv/bin/python benchmarks/benchmark_mhr.py --device cpu
CUDA_VISIBLE_DEVICES=7 .venv/bin/python benchmarks/benchmark_mhr.py --device cuda
uv pip install --python .venv/bin/python warp-lang==1.15.0
CUDA_VISIBLE_DEVICES=7 .venv/bin/python benchmarks/benchmark_mhr.py --device cuda --use-warp
.venv/bin/python docs/render_segmentation.py --models-dir models
uv run python examples/04_mhr_shape_pose.py \
  --model-path models/MHR/converted/mhr_lod1.npz --headless --frames 1
.venv/bin/pytest tests/ -q
CUDA_VISIBLE_DEVICES=7 .venv/bin/pytest tests/ -q
.venv/bin/ruff check .
git diff --check
```
