# Order 03 results — SMPL/SMPL-X benchmarks and final docs

Status: implementation and measurements complete on `dev`, 2026-07-21;
final repository gates are recorded below.

## Implementation

- Added `benchmarks/benchmark_smpl.py`, comparing BetterHuman with the
  vendored upstream `smplx` 0.1.28 package for SMPL and SMPL-X.
- The harness uses the same neutral NPZ fixtures as the parity suites. SMPL
  uses 10 betas; SMPL-X uses 16 betas, 10 expression coefficients, full
  hands and face, natural hand means, no hand PCA, and no face contour.
- Every batch uses one repeated shape and varying pose rows. This makes the
  scalar shape bake valid for the pose-only row while still exercising batch
  broadcasting. The fair row times `with_shape` plus `vertices(q)`.
- Every output is checked for shape, float32 dtype, device, finiteness, and
  maximum vertex error at the existing `1e-6 m` parity pin before timing.
- Timing uses five warmups, 30 samples, `perf_counter_ns`, and explicit CUDA
  synchronization. The pose-only stage split mirrors production FK,
  corrective, and transforms-plus-matrix-LBS calls.

The upstream SMPL-X call sets `return_shaped=False` so it does not compute an
optional duplicate betas-only output that BetterHuman does not return. Its
static landmark and extra-joint work remains part of the requested upstream
full forward. Each reference instance is constructed with the matching batch
size so landmark work uses canonical batch-sized storage.

## Protocol

Commands:

```bash
.venv/bin/python benchmarks/benchmark_smpl.py --device cpu
CUDA_VISIBLE_DEVICES=7 .venv/bin/python benchmarks/benchmark_smpl.py --device cuda
CUDA_VISIBLE_DEVICES=7 .venv/bin/python benchmarks/benchmark_smpl.py --device cuda --use-warp
```

All runs used PyTorch 2.13.0+cu126, BetterRobot 0.2.0, float32, batches
1/16/256, five warmups, and 30 timed calls. CPU used 112 intra/inter-op
threads on the shared dual-socket Intel Xeon Platinum 8570 host. CUDA used
physical GPU 7, an otherwise idle NVIDIA RTX 6000 Ada Generation with
driver 560.35.03. The Warp run used Warp 1.15.0.

## Pre-timing output agreement

Every value passed the `1e-6 m` pin. The worst value in all three invocations
was `5.9604644775390625e-7 m`.

### CPU

| Model | Batch | Pose-only max abs m | Shape+pose max abs m |
|---|---:|---:|---:|
| SMPL | 1 | 3.5762786865234375e-7 | 3.5762786865234375e-7 |
| SMPL | 16 | 4.76837158203125e-7 | 5.9604644775390625e-7 |
| SMPL | 256 | 5.9604644775390625e-7 | 5.9604644775390625e-7 |
| SMPL-X | 1 | 3.5762786865234375e-7 | 3.5762786865234375e-7 |
| SMPL-X | 16 | 4.76837158203125e-7 | 4.76837158203125e-7 |
| SMPL-X | 256 | 5.9604644775390625e-7 | 5.9604644775390625e-7 |

### Native CUDA

| Model | Batch | Pose-only max abs m | Shape+pose max abs m |
|---|---:|---:|---:|
| SMPL | 1 | 3.5762786865234375e-7 | 3.5762786865234375e-7 |
| SMPL | 16 | 4.1723251342773438e-7 | 4.1723251342773438e-7 |
| SMPL | 256 | 4.76837158203125e-7 | 4.76837158203125e-7 |
| SMPL-X | 1 | 2.4959444999694824e-7 | 2.4959444999694824e-7 |
| SMPL-X | 16 | 4.76837158203125e-7 | 4.76837158203125e-7 |
| SMPL-X | 256 | 5.3644180297851562e-7 | 5.9604644775390625e-7 |

### Warp CUDA

| Model | Batch | Pose-only max abs m | Shape+pose max abs m |
|---|---:|---:|---:|
| SMPL | 1 | 3.5762786865234375e-7 | 3.5762786865234375e-7 |
| SMPL | 16 | 3.5762786865234375e-7 | 4.1723251342773438e-7 |
| SMPL | 256 | 4.76837158203125e-7 | 4.76837158203125e-7 |
| SMPL-X | 1 | 2.5331974029541016e-7 | 2.4959444999694824e-7 |
| SMPL-X | 16 | 4.76837158203125e-7 | 4.76837158203125e-7 |
| SMPL-X | 256 | 4.76837158203125e-7 | 5.9604644775390625e-7 |

## CPU benchmark

| Model | Batch | Path | Median ms | × original |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream `smplx` shape+pose | 5.982045 | 1.000000 |
| SMPL | 1 | BetterHuman pose-only | 6.294569 | 1.052244 |
| SMPL | 1 | BetterHuman shape+pose | 21.256035 | 3.553306 |
| SMPL | 16 | upstream `smplx` shape+pose | 2.421118 | 1.000000 |
| SMPL | 16 | BetterHuman pose-only | 11.292286 | 4.664079 |
| SMPL | 16 | BetterHuman shape+pose | 11.746825 | 4.851819 |
| SMPL | 256 | upstream `smplx` shape+pose | 23.291602 | 1.000000 |
| SMPL | 256 | BetterHuman pose-only | 28.108368 | 1.206803 |
| SMPL | 256 | BetterHuman shape+pose | 31.323141 | 1.344826 |
| SMPL-X | 1 | upstream `smplx` shape+pose | 2.072404 | 1.000000 |
| SMPL-X | 1 | BetterHuman pose-only | 13.443112 | 6.486724 |
| SMPL-X | 1 | BetterHuman shape+pose | 9.304661 | 4.489791 |
| SMPL-X | 16 | upstream `smplx` shape+pose | 2.315695 | 1.000000 |
| SMPL-X | 16 | BetterHuman pose-only | 10.618555 | 4.585471 |
| SMPL-X | 16 | BetterHuman shape+pose | 10.957764 | 4.731954 |
| SMPL-X | 256 | upstream `smplx` shape+pose | 40.569139 | 1.000000 |
| SMPL-X | 256 | BetterHuman pose-only | 44.174468 | 1.088869 |
| SMPL-X | 256 | BetterHuman shape+pose | 54.745655 | 1.349441 |

| Model | Batch | Stage | Median ms |
|---|---:|---|---:|
| SMPL | 1 | FK | 5.195847 |
| SMPL | 1 | corrective | 0.280781 |
| SMPL | 1 | transforms + LBS | 0.732684 |
| SMPL | 16 | FK | 5.463037 |
| SMPL | 16 | corrective | 0.316877 |
| SMPL | 16 | transforms + LBS | 2.147187 |
| SMPL | 256 | FK | 3.570817 |
| SMPL | 256 | corrective | 0.851205 |
| SMPL | 256 | transforms + LBS | 19.529671 |
| SMPL-X | 1 | FK | 6.033785 |
| SMPL-X | 1 | corrective | 0.170599 |
| SMPL-X | 1 | transforms + LBS | 0.461355 |
| SMPL-X | 16 | FK | 6.312891 |
| SMPL-X | 16 | corrective | 0.299218 |
| SMPL-X | 16 | transforms + LBS | 1.920193 |
| SMPL-X | 256 | FK | 10.016085 |
| SMPL-X | 256 | corrective | 1.959046 |
| SMPL-X | 256 | transforms + LBS | 31.454771 |

Small-batch CPU values were noisy under shared-host scheduling. They are
reported as measured without selective reruns or filtering.

## Native CUDA benchmark

| Model | Batch | Path | Median ms | × original |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream `smplx` shape+pose | 1.664994 | 1.000000 |
| SMPL | 1 | BetterHuman pose-only | 17.377573 | 10.437015 |
| SMPL | 1 | BetterHuman shape+pose | 17.905309 | 10.753975 |
| SMPL | 16 | upstream `smplx` shape+pose | 1.707768 | 1.000000 |
| SMPL | 16 | BetterHuman pose-only | 18.073975 | 10.583390 |
| SMPL | 16 | BetterHuman shape+pose | 18.894050 | 11.063593 |
| SMPL | 256 | upstream `smplx` shape+pose | 3.010079 | 1.000000 |
| SMPL | 256 | BetterHuman pose-only | 22.172007 | 7.365921 |
| SMPL | 256 | BetterHuman shape+pose | 23.031094 | 7.651324 |
| SMPL-X | 1 | upstream `smplx` shape+pose | 3.183152 | 1.000000 |
| SMPL-X | 1 | BetterHuman pose-only | 41.392021 | 13.003472 |
| SMPL-X | 1 | BetterHuman shape+pose | 44.985560 | 14.132397 |
| SMPL-X | 16 | upstream `smplx` shape+pose | 3.414323 | 1.000000 |
| SMPL-X | 16 | BetterHuman pose-only | 45.384021 | 13.292246 |
| SMPL-X | 16 | BetterHuman shape+pose | 46.294748 | 13.558983 |
| SMPL-X | 256 | upstream `smplx` shape+pose | 5.671885 | 1.000000 |
| SMPL-X | 256 | BetterHuman pose-only | 30.813797 | 5.432726 |
| SMPL-X | 256 | BetterHuman shape+pose | 26.117499 | 4.604730 |

| Model | Batch | Stage | Median ms |
|---|---:|---|---:|
| SMPL | 1 | FK | 16.286674 |
| SMPL | 1 | corrective | 0.382829 |
| SMPL | 1 | transforms + LBS | 0.953956 |
| SMPL | 16 | FK | 16.791555 |
| SMPL | 16 | corrective | 0.392900 |
| SMPL | 16 | transforms + LBS | 1.035826 |
| SMPL | 256 | FK | 19.648005 |
| SMPL | 256 | corrective | 0.494661 |
| SMPL | 256 | transforms + LBS | 2.698581 |
| SMPL-X | 1 | FK | 42.404108 |
| SMPL-X | 1 | corrective | 0.443139 |
| SMPL-X | 1 | transforms + LBS | 1.075104 |
| SMPL-X | 16 | FK | 43.724548 |
| SMPL-X | 16 | corrective | 0.464864 |
| SMPL-X | 16 | transforms + LBS | 1.188697 |
| SMPL-X | 256 | FK | 22.255063 |
| SMPL-X | 256 | corrective | 0.423588 |
| SMPL-X | 256 | transforms + LBS | 3.056215 |

Every native CUDA result at batch 16 and 256 is more than 2× upstream. The
stage split identifies BetterRobot's Torch FK as the pose-only bottleneck;
correctives and matrix-blend LBS are much smaller.

## Warp CUDA benchmark

| Model | Batch | Path | Median ms | × original |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream `smplx` shape+pose | 1.855876 | 1.000000 |
| SMPL | 1 | BetterHuman pose-only, Warp requested | 2.668455 | 1.437841 |
| SMPL | 1 | BetterHuman shape+pose, Torch FK | 20.780798 | 11.197299 |
| SMPL | 16 | upstream `smplx` shape+pose | 1.237468 | 1.000000 |
| SMPL | 16 | BetterHuman pose-only, Warp requested | 1.810249 | 1.462866 |
| SMPL | 16 | BetterHuman shape+pose, Torch FK | 14.790574 | 11.952288 |
| SMPL | 256 | upstream `smplx` shape+pose | 3.149287 | 1.000000 |
| SMPL | 256 | BetterHuman pose-only, Warp requested | 4.346257 | 1.380076 |
| SMPL | 256 | BetterHuman shape+pose, Torch FK | 23.326054 | 7.406772 |
| SMPL-X | 1 | upstream `smplx` shape+pose | 3.152521 | 1.000000 |
| SMPL-X | 1 | BetterHuman pose-only, Warp requested | 2.488716 | 0.789437 |
| SMPL-X | 1 | BetterHuman shape+pose, Torch FK | 43.644657 | 13.844365 |
| SMPL-X | 16 | upstream `smplx` shape+pose | 3.401570 | 1.000000 |
| SMPL-X | 16 | BetterHuman pose-only, Warp requested | 2.721118 | 0.799960 |
| SMPL-X | 16 | BetterHuman shape+pose, Torch FK | 44.791814 | 13.167982 |
| SMPL-X | 256 | upstream `smplx` shape+pose | 5.668683 | 1.000000 |
| SMPL-X | 256 | BetterHuman pose-only, Warp requested | 5.212826 | 0.919583 |
| SMPL-X | 256 | BetterHuman shape+pose, Torch FK | 48.421574 | 8.541944 |

| Model | Batch | Stage | Median ms |
|---|---:|---|---:|
| SMPL | 1 | Warp FK | 0.438639 |
| SMPL | 1 | corrective | 0.316354 |
| SMPL | 1 | transforms + LBS | 0.779355 |
| SMPL | 16 | Warp FK | 0.391695 |
| SMPL | 16 | corrective | 0.298544 |
| SMPL | 16 | transforms + LBS | 0.784786 |
| SMPL | 256 | Warp FK | 0.767779 |
| SMPL | 256 | corrective | 0.486483 |
| SMPL | 256 | transforms + LBS | 2.597154 |
| SMPL-X | 1 | Warp FK | 0.678249 |
| SMPL-X | 1 | corrective | 0.431269 |
| SMPL-X | 1 | transforms + LBS | 1.052793 |
| SMPL-X | 16 | Warp FK | 0.697566 |
| SMPL-X | 16 | corrective | 0.451804 |
| SMPL-X | 16 | transforms + LBS | 1.161303 |
| SMPL-X | 256 | Warp FK | 0.631048 |
| SMPL-X | 256 | corrective | 0.457872 |
| SMPL-X | 256 | transforms + LBS | 3.086571 |

Warp pose-only is within 1.47× of upstream throughout and is faster for all
SMPL-X batches in this run. The fair shape+pose path retains Torch FK because
its shape-batched joint placements are not Warp-compatible; at CUDA batch
16/256 it remains 7.41–13.17× upstream.

## Documentation sweep

- Architecture now distinguishes padded `(V, K)` archive/segmentation data
  from the spawn-built dense `(V, J)` runtime matrix and documents the
  matrix-blend formulation and all SMPL-family benchmark summaries.
- Contracts now record observational SMPL/SMPL-X performance baselines and
  distinguish enforced MHR TorchScript skeleton tolerances from measured
  bit-exact P/Q/S agreement.
- The MHR architecture retains the LOD0/corrective payload sizes, neutral LOD1
  mass, and structural PSD-not-SPD CRBA rationale.
- Loader contract wording now matches selected-schema validation, and the
  stale results pointer in the README remains directed at the docs.

## Findings and deviations

- No implementation deviation from Order 03 was needed. The upstream SMPL
  loader receives the exact converted NPZ through its supported `data_struct`
  hook because direct construction treats every path as a legacy pickle;
  this matches the existing parity-oracle procedure.
- No performance target was imposed. As required, every CUDA batch-16/256
  result above 2× is reported with its stage split. Warp resolves the
  pose-only FK bottleneck, while fair shape+pose remains on Torch FK.
- The shared CPU host produced non-monotonic small-batch medians. Raw medians
  are retained; no selective rerun or filtering was used.

## Validation

- `benchmarks/benchmark_smpl.py` CPU and CUDA smoke runs: passed.
- Full CPU, native-CUDA, and Warp-CUDA benchmark grids: passed.
- Full CUDA suite: 226 passed, 0 skipped, 19 warnings in 100.82 seconds; all
  five CUDA cases executed.
- `.venv/bin/ruff check .`, benchmark format check, `git diff --check`, and
  Python byte-compilation: passed.
- Three independent final audits found no remaining benchmark-protocol,
  numeric-transcription, documentation-claim, or scope errors.
