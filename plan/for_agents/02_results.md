# Order 02 results — workload-framed and gradient benchmarks

**Status:** complete on 2026-07-22.

## Outcome

- Workload A now compares equivalent full shape-plus-pose forwards: upstream/reference
  against the BetterHuman one-call path.
- Workload B measures the per-frame streaming-pose workload: BetterHuman shape baking is
  outside timing, while the upstream/reference path must repeat its full forward because
  it has no shape-amortized API. These ratios are workload-level, not FLOP-matched.
- SMPL and SMPL-X now report forward-only, forward-plus-backward, training-step
  multiplier, and forward-plus-backward relative to upstream for both workloads.
- MHR reports absolute gradient measurements for full shape-plus-pose and streaming-pose
  lanes. Pre-timing direct reference agreement remains a hard gate.
- CPU, native CUDA/Torch, and CUDA/Warp forward protocols were measured. Gradients were
  measured on CPU and native CUDA/Torch.

## Deviations and methodology corrections

These differences from the written plan are intentional and material:

1. **MHR uses its established float32 parity contract of 5e-6 m, not the
   plan's universal 1e-6 m gate.** The official CPU direct comparison
   reached 2.7418136596679688e-6 m, so a 1e-6 m gate is
   incompatible with the existing MHR reference contract. Native CUDA reached
   1.5497207641601562e-6 m; the Warp run reached
   1.5497207641601562e-6 m for full forward and
   1.430511474609375e-6 m for streaming. All passed 5e-6 m.
   SMPL and SMPL-X remained within the requested 1e-6 m gate; their worst
   official error was 5.9604644775390625e-7 m.
2. **The ratio is reported as directly measured `F+B/F`.** The plan calls this
   “backward/forward,” but the requested numerator and the historical values are a
   complete forward-plus-backward training step. Labeling it `F+B/F` avoids presenting
   that measurement as backward-only time.
3. **Gradient timing was changed from separate blocks to paired samples.** The first,
   pre-fix SMPL CPU gradient run timed all forwards separately from all
   forward-plus-backward steps. Shared-host load produced the impossible upstream SMPL
   batch-1 Workload A result F=28.453864 ms and F+B=6.601115 ms
   (0.231994×). That entire run was rejected as methodologically invalid.
   Forward and forward-plus-backward are now sampled as a pair in every iteration, with
   gradient clearing outside the timed region. After that correction, each of the six
   official protocols below was run exactly once; no official row was selectively rerun.
4. **Official Warp protocols are forward-only.** Passing `--use-warp --grad` remains
   supported, but that hybrid is explicitly labeled “Warp forward; Torch-recompute VJP”:
   its VJP recomputes with Torch autograd. It was not included among the official gradient
   protocols below.

## Protocol

All timings use float32, 5 warmups, 30 explicitly synchronized samples, and batch sizes
1, 16, and 256. Tables report medians. Gradient forward and forward-plus-backward samples
are paired per iteration. Stage medians are independently timed diagnostic measurements;
they are **non-additive** and should not be summed to reconstruct the end-to-end median.

The CPU host was a shared dual Intel Xeon Platinum 8570 system with 112 intra-op and 112
inter-op Torch threads. CUDA measurements isolated physical GPU 7 through
`CUDA_VISIBLE_DEVICES=7`; it appeared as logical `cuda:0`, an NVIDIA RTX 6000 Ada
Generation with 48 GiB. Software was Torch 2.13.0+cu126, CUDA runtime 12.6, NVIDIA driver
560.35.03, BetterRobot 0.2.0, upstream `smplx` 0.1.28, and Warp 1.15.0. Shared-host
scheduling and concurrent load can affect these observational medians, especially CPU
and small-batch rows; the raw values are preserved without smoothing.

Official commands:

```bash
env UV_CACHE_DIR=/tmp/bh-uv-cache uv run python benchmarks/benchmark_smpl.py --device cpu --grad
env CUDA_VISIBLE_DEVICES=7 UV_CACHE_DIR=/tmp/bh-uv-cache uv run python benchmarks/benchmark_smpl.py --device cuda --grad
env CUDA_VISIBLE_DEVICES=7 UV_CACHE_DIR=/tmp/bh-uv-cache uv run python benchmarks/benchmark_smpl.py --device cuda --use-warp
env UV_CACHE_DIR=/tmp/bh-uv-cache uv run python benchmarks/benchmark_mhr.py --device cpu --grad
env CUDA_VISIBLE_DEVICES=7 UV_CACHE_DIR=/tmp/bh-uv-cache uv run python benchmarks/benchmark_mhr.py --device cuda --grad
env CUDA_VISIBLE_DEVICES=7 UV_CACHE_DIR=/tmp/bh-uv-cache uv run python benchmarks/benchmark_mhr.py --device cuda --use-warp
```

## Findings

- Native CUDA SMPL-family full-forward training steps were
  10.571821×–13.200138× upstream; streaming training steps were
  8.476885×–10.026916× upstream. The `F+B/F` multipliers broadly
  track the corresponding upstream multipliers, consistent with the already-attributed
  launch-bound FK cost rather than a new backward-specific pathology.
- Warp reduced the CUDA streaming lane to 1.182787×–1.409348×
  upstream for SMPL, 0.786707×–0.950417× for SMPL-X, and
  0.428478×–0.649820× the MHR TorchScript reference. Workload A still
  uses Torch FK because shape baking and the one-call full-forward path are outside the
  Warp streaming specialization.
- MHR native-CUDA absolute `F+B/F` multipliers ranged from 1.641031× to
  3.761233×. There is no upstream MHR gradient counterpart, so these remain
  absolute observational baselines.
- CPU results, particularly batch 1, show substantial shared-host variation. They are
  reported as measured and are not used to override the synchronized CUDA conclusions.

## Verification

```text
CUDA full suite: 230 passed, 19 warnings in 158.23s
Skipped: 0
CUDA availability-gated cases exercised: 6
```

The CUDA suite command was:

```bash
env CUDA_VISIBLE_DEVICES=7 UV_CACHE_DIR=/tmp/bh-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -ra tests
```

Targeted Ruff checks for both benchmark scripts, Python byte-compilation, and
`git diff --check` also passed.

## Complete official output tables

The following metadata and tables are transcribed from the six post-fix official outputs.
Numeric precision is preserved exactly.

### SMPL family — CPU / native Torch gradients

<details>
<summary>Run metadata</summary>

```text
torch=2.13.0+cu126
better_robot=0.2.0
reference_version=0.1.28
reference_package=/data3/rikhat.akizhanov/better/BetterHuman/references/parametric_body/smplx/smplx
device=cpu
cpu_threads=112
cpu_interop_threads=112
warmup=5; iterations=30; batches=(1, 16, 256)
use_warp=False
grad=True
timing_caveat=shared-host scheduling and concurrent load can affect medians
smpl_path=/data3/rikhat.akizhanov/better/BetterHuman/models/smpl/SMPL_NEUTRAL.npz
smplx_path=/data3/rikhat.akizhanov/better/BetterHuman/models/smplx/SMPLX_NEUTRAL.npz
```

</details>

#### Pre-timing output agreement

| Model | Batch | Workload A max abs m | Workload B max abs m |
|---|---:|---:|---:|
| SMPL | 1 | 3.5762786865234375e-07 | 3.5762786865234375e-07 |
| SMPL | 16 | 5.9604644775390625e-07 | 4.76837158203125e-07 |
| SMPL | 256 | 5.9604644775390625e-07 | 5.9604644775390625e-07 |
| SMPL-X | 1 | 3.5762786865234375e-07 | 3.5762786865234375e-07 |
| SMPL-X | 16 | 4.76837158203125e-07 | 4.76837158203125e-07 |
| SMPL-X | 256 | 5.9604644775390625e-07 | 5.9604644775390625e-07 |

#### Workload A — full forward (new shape + new pose every call)

The × upstream ratio compares equivalent full shape-plus-pose computations; BetterHuman uses its one-call API.

| Model | Batch | Implementation | Median ms | × upstream |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream smplx full forward | 5.747977 | 1.000000 |
| SMPL | 1 | BetterHuman one-call full forward | 12.415018 | 2.159893 |
| SMPL | 16 | upstream smplx full forward | 1.770128 | 1.000000 |
| SMPL | 16 | BetterHuman one-call full forward | 8.117990 | 4.586105 |
| SMPL | 256 | upstream smplx full forward | 24.879647 | 1.000000 |
| SMPL | 256 | BetterHuman one-call full forward | 32.571483 | 1.309162 |
| SMPL-X | 1 | upstream smplx full forward | 1.753289 | 1.000000 |
| SMPL-X | 1 | BetterHuman one-call full forward | 13.675982 | 7.800185 |
| SMPL-X | 16 | upstream smplx full forward | 3.085485 | 1.000000 |
| SMPL-X | 16 | BetterHuman one-call full forward | 10.912850 | 3.536834 |
| SMPL-X | 256 | upstream smplx full forward | 41.908221 | 1.000000 |
| SMPL-X | 256 | BetterHuman one-call full forward | 59.441630 | 1.418376 |

#### Workload B — streaming poses (shape fixed, poses vary per call)

BetterHuman bakes shape once outside timing. Upstream has no shape-amortized API, so its per-frame workload remains a full forward. This is a workload-level ratio, not a FLOP-matched ratio.

| Model | Batch | Implementation | Median ms | × upstream |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream smplx full forward | 5.747977 | 1.000000 |
| SMPL | 1 | BetterHuman shape-baked vertices | 6.349140 | 1.104587 |
| SMPL | 16 | upstream smplx full forward | 1.770128 | 1.000000 |
| SMPL | 16 | BetterHuman shape-baked vertices | 7.358080 | 4.156808 |
| SMPL | 256 | upstream smplx full forward | 24.879647 | 1.000000 |
| SMPL | 256 | BetterHuman shape-baked vertices | 25.725491 | 1.033997 |
| SMPL-X | 1 | upstream smplx full forward | 1.753289 | 1.000000 |
| SMPL-X | 1 | BetterHuman shape-baked vertices | 13.128313 | 7.487818 |
| SMPL-X | 16 | upstream smplx full forward | 3.085485 | 1.000000 |
| SMPL-X | 16 | BetterHuman shape-baked vertices | 11.918368 | 3.862721 |
| SMPL-X | 256 | upstream smplx full forward | 41.908221 | 1.000000 |
| SMPL-X | 256 | BetterHuman shape-baked vertices | 55.133575 | 1.315579 |

#### BetterHuman Workload B stage breakdown

| Model | Batch | Stage | Median ms |
|---|---:|---|---:|
| SMPL | 1 | FK | 5.329762 |
| SMPL | 1 | corrective | 0.177994 |
| SMPL | 1 | transforms + LBS | 0.510617 |
| SMPL | 16 | FK | 5.570056 |
| SMPL | 16 | corrective | 0.246628 |
| SMPL | 16 | transforms + LBS | 1.514430 |
| SMPL | 256 | FK | 7.015952 |
| SMPL | 256 | corrective | 0.978832 |
| SMPL | 256 | transforms + LBS | 16.607948 |
| SMPL-X | 1 | FK | 11.782476 |
| SMPL-X | 1 | corrective | 0.214475 |
| SMPL-X | 1 | transforms + LBS | 0.580283 |
| SMPL-X | 16 | FK | 6.355187 |
| SMPL-X | 16 | corrective | 0.370907 |
| SMPL-X | 16 | transforms + LBS | 1.546320 |
| SMPL-X | 256 | FK | 11.567204 |
| SMPL-X | 256 | corrective | 1.961538 |
| SMPL-X | 256 | transforms + LBS | 36.035786 |

#### Gradients: Workload A — full forward

Loss is vertices.sum(). BetterHuman differentiates q and every shape group; upstream differentiates every supplied pose and shape tensor.
F+B / F is the training-step multiplier, not a backward-only median.

| Model | Batch | Implementation | Forward ms | Forward + backward ms | F+B / F | F+B × upstream |
|---|---:|---|---:|---:|---:|---:|
| SMPL | 1 | upstream smplx full forward | 28.071696 | 64.525559 | 2.298598 | 1.000000 |
| SMPL | 1 | BetterHuman one-call full forward | 18.242398 | 34.155343 | 1.872306 | 0.529330 |
| SMPL | 16 | upstream smplx full forward | 2.731242 | 8.996616 | 3.293965 | 1.000000 |
| SMPL | 16 | BetterHuman one-call full forward | 16.611343 | 45.854965 | 2.760461 | 5.096913 |
| SMPL | 256 | upstream smplx full forward | 20.872322 | 51.779978 | 2.480796 | 1.000000 |
| SMPL | 256 | BetterHuman one-call full forward | 33.466383 | 106.526944 | 3.183103 | 2.057300 |
| SMPL-X | 1 | upstream smplx full forward | 2.117856 | 5.734379 | 2.707633 | 1.000000 |
| SMPL-X | 1 | BetterHuman one-call full forward | 15.242779 | 45.947761 | 3.014395 | 8.012684 |
| SMPL-X | 16 | upstream smplx full forward | 3.220175 | 10.879412 | 3.378515 | 1.000000 |
| SMPL-X | 16 | BetterHuman one-call full forward | 19.508656 | 54.005480 | 2.768283 | 4.964007 |
| SMPL-X | 256 | upstream smplx full forward | 41.348561 | 112.348765 | 2.717114 | 1.000000 |
| SMPL-X | 256 | BetterHuman one-call full forward | 73.301568 | 221.165732 | 3.017203 | 1.968564 |

#### Gradients: Workload B — streaming poses

Loss is vertices.sum(). BetterHuman differentiates q after baking shape outside timing. Upstream differentiates pose inputs but must still run its full forward.
F+B / F is the training-step multiplier, not a backward-only median.

| Model | Batch | Implementation | Forward ms | Forward + backward ms | F+B / F | F+B × upstream |
|---|---:|---|---:|---:|---:|---:|
| SMPL | 1 | upstream smplx full forward | 4.242632 | 6.534811 | 1.540273 | 1.000000 |
| SMPL | 1 | BetterHuman shape-baked vertices | 11.472444 | 36.634596 | 3.193269 | 5.606068 |
| SMPL | 16 | upstream smplx full forward | 2.788969 | 8.672000 | 3.109393 | 1.000000 |
| SMPL | 16 | BetterHuman shape-baked vertices | 8.107002 | 25.599815 | 3.157741 | 2.952008 |
| SMPL | 256 | upstream smplx full forward | 20.357383 | 49.028289 | 2.408379 | 1.000000 |
| SMPL | 256 | BetterHuman shape-baked vertices | 14.992769 | 66.829707 | 4.457463 | 1.363085 |
| SMPL-X | 1 | upstream smplx full forward | 2.623462 | 6.742190 | 2.569959 | 1.000000 |
| SMPL-X | 1 | BetterHuman shape-baked vertices | 12.706206 | 40.014697 | 3.149225 | 5.934970 |
| SMPL-X | 16 | upstream smplx full forward | 3.843130 | 12.407327 | 3.228443 | 1.000000 |
| SMPL-X | 16 | BetterHuman shape-baked vertices | 15.421431 | 46.431064 | 3.010814 | 3.742229 |
| SMPL-X | 256 | upstream smplx full forward | 51.169806 | 132.857832 | 2.596411 | 1.000000 |
| SMPL-X | 256 | BetterHuman shape-baked vertices | 59.711080 | 213.446547 | 3.574656 | 1.606579 |

### SMPL family — CUDA / native Torch gradients

<details>
<summary>Run metadata</summary>

```text
torch=2.13.0+cu126
better_robot=0.2.0
reference_version=0.1.28
reference_package=/data3/rikhat.akizhanov/better/BetterHuman/references/parametric_body/smplx/smplx
device=cuda (NVIDIA RTX 6000 Ada Generation)
cuda_runtime=12.6
cuda_driver=560.35.03
cpu_threads=112
cpu_interop_threads=112
warmup=5; iterations=30; batches=(1, 16, 256)
use_warp=False
grad=True
timing_caveat=shared-host scheduling and concurrent load can affect medians
smpl_path=/data3/rikhat.akizhanov/better/BetterHuman/models/smpl/SMPL_NEUTRAL.npz
smplx_path=/data3/rikhat.akizhanov/better/BetterHuman/models/smplx/SMPLX_NEUTRAL.npz
```

</details>

#### Pre-timing output agreement

| Model | Batch | Workload A max abs m | Workload B max abs m |
|---|---:|---:|---:|
| SMPL | 1 | 3.5762786865234375e-07 | 3.5762786865234375e-07 |
| SMPL | 16 | 4.1723251342773438e-07 | 4.1723251342773438e-07 |
| SMPL | 256 | 4.76837158203125e-07 | 4.76837158203125e-07 |
| SMPL-X | 1 | 2.4959444999694824e-07 | 2.4959444999694824e-07 |
| SMPL-X | 16 | 4.76837158203125e-07 | 4.76837158203125e-07 |
| SMPL-X | 256 | 5.9604644775390625e-07 | 5.3644180297851562e-07 |

#### Workload A — full forward (new shape + new pose every call)

The × upstream ratio compares equivalent full shape-plus-pose computations; BetterHuman uses its one-call API.

| Model | Batch | Implementation | Median ms | × upstream |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream smplx full forward | 1.999315 | 1.000000 |
| SMPL | 1 | BetterHuman one-call full forward | 21.644656 | 10.826039 |
| SMPL | 16 | upstream smplx full forward | 2.123617 | 1.000000 |
| SMPL | 16 | BetterHuman one-call full forward | 23.946178 | 11.276126 |
| SMPL | 256 | upstream smplx full forward | 3.261280 | 1.000000 |
| SMPL | 256 | BetterHuman one-call full forward | 25.821618 | 7.917632 |
| SMPL-X | 1 | upstream smplx full forward | 1.991998 | 1.000000 |
| SMPL-X | 1 | BetterHuman one-call full forward | 23.727799 | 11.911555 |
| SMPL-X | 16 | upstream smplx full forward | 1.909512 | 1.000000 |
| SMPL-X | 16 | BetterHuman one-call full forward | 24.667794 | 12.918376 |
| SMPL-X | 256 | upstream smplx full forward | 4.279208 | 1.000000 |
| SMPL-X | 256 | BetterHuman one-call full forward | 28.300942 | 6.613593 |

#### Workload B — streaming poses (shape fixed, poses vary per call)

BetterHuman bakes shape once outside timing. Upstream has no shape-amortized API, so its per-frame workload remains a full forward. This is a workload-level ratio, not a FLOP-matched ratio.

| Model | Batch | Implementation | Median ms | × upstream |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream smplx full forward | 1.999315 | 1.000000 |
| SMPL | 1 | BetterHuman shape-baked vertices | 21.990452 | 10.998996 |
| SMPL | 16 | upstream smplx full forward | 2.123617 | 1.000000 |
| SMPL | 16 | BetterHuman shape-baked vertices | 23.083747 | 10.870011 |
| SMPL | 256 | upstream smplx full forward | 3.261280 | 1.000000 |
| SMPL | 256 | BetterHuman shape-baked vertices | 24.895283 | 7.633592 |
| SMPL-X | 1 | upstream smplx full forward | 1.991998 | 1.000000 |
| SMPL-X | 1 | BetterHuman shape-baked vertices | 23.325769 | 11.709732 |
| SMPL-X | 16 | upstream smplx full forward | 1.909512 | 1.000000 |
| SMPL-X | 16 | BetterHuman shape-baked vertices | 24.214457 | 12.680966 |
| SMPL-X | 256 | upstream smplx full forward | 4.279208 | 1.000000 |
| SMPL-X | 256 | BetterHuman shape-baked vertices | 27.656212 | 6.462927 |

#### BetterHuman Workload B stage breakdown

| Model | Batch | Stage | Median ms |
|---|---:|---|---:|
| SMPL | 1 | FK | 17.241571 |
| SMPL | 1 | corrective | 0.451912 |
| SMPL | 1 | transforms + LBS | 1.111481 |
| SMPL | 16 | FK | 21.177231 |
| SMPL | 16 | corrective | 0.476484 |
| SMPL | 16 | transforms + LBS | 1.235575 |
| SMPL | 256 | FK | 21.399667 |
| SMPL | 256 | corrective | 0.529627 |
| SMPL | 256 | transforms + LBS | 2.773611 |
| SMPL-X | 1 | FK | 22.450940 |
| SMPL-X | 1 | corrective | 0.248315 |
| SMPL-X | 1 | transforms + LBS | 0.572035 |
| SMPL-X | 16 | FK | 23.193650 |
| SMPL-X | 16 | corrective | 0.266155 |
| SMPL-X | 16 | transforms + LBS | 0.671591 |
| SMPL-X | 256 | FK | 26.454930 |
| SMPL-X | 256 | corrective | 0.449537 |
| SMPL-X | 256 | transforms + LBS | 3.097895 |

#### Gradients: Workload A — full forward

Loss is vertices.sum(). BetterHuman differentiates q and every shape group; upstream differentiates every supplied pose and shape tensor.
F+B / F is the training-step multiplier, not a backward-only median.

| Model | Batch | Implementation | Forward ms | Forward + backward ms | F+B / F | F+B × upstream |
|---|---:|---|---:|---:|---:|---:|
| SMPL | 1 | upstream smplx full forward | 2.016026 | 7.362839 | 3.652154 | 1.000000 |
| SMPL | 1 | BetterHuman one-call full forward | 21.694166 | 80.218205 | 3.697686 | 10.895010 |
| SMPL | 16 | upstream smplx full forward | 1.701471 | 6.466421 | 3.800489 | 1.000000 |
| SMPL | 16 | BetterHuman one-call full forward | 22.375858 | 85.357652 | 3.814721 | 13.200138 |
| SMPL | 256 | upstream smplx full forward | 3.025510 | 6.961112 | 2.300807 | 1.000000 |
| SMPL | 256 | BetterHuman one-call full forward | 22.398428 | 86.299966 | 3.852947 | 12.397440 |
| SMPL-X | 1 | upstream smplx full forward | 7.191468 | 17.394033 | 2.418704 | 1.000000 |
| SMPL-X | 1 | BetterHuman one-call full forward | 65.165043 | 199.382611 | 3.059656 | 11.462702 |
| SMPL-X | 16 | upstream smplx full forward | 7.867332 | 19.639098 | 2.496284 | 1.000000 |
| SMPL-X | 16 | BetterHuman one-call full forward | 68.585478 | 210.025333 | 3.062242 | 10.694245 |
| SMPL-X | 256 | upstream smplx full forward | 9.423600 | 20.218570 | 2.145525 | 1.000000 |
| SMPL-X | 256 | BetterHuman one-call full forward | 70.283626 | 213.747098 | 3.041208 | 10.571821 |

#### Gradients: Workload B — streaming poses

Loss is vertices.sum(). BetterHuman differentiates q after baking shape outside timing. Upstream differentiates pose inputs but must still run its full forward.
F+B / F is the training-step multiplier, not a backward-only median.

| Model | Batch | Implementation | Forward ms | Forward + backward ms | F+B / F | F+B × upstream |
|---|---:|---|---:|---:|---:|---:|
| SMPL | 1 | upstream smplx full forward | 2.278359 | 7.392388 | 3.244610 | 1.000000 |
| SMPL | 1 | BetterHuman shape-baked vertices | 21.974447 | 70.566171 | 3.211283 | 9.545788 |
| SMPL | 16 | upstream smplx full forward | 2.131151 | 6.941037 | 3.256942 | 1.000000 |
| SMPL | 16 | BetterHuman shape-baked vertices | 20.352127 | 69.597196 | 3.419652 | 10.026916 |
| SMPL | 256 | upstream smplx full forward | 3.747643 | 8.268234 | 2.206249 | 1.000000 |
| SMPL | 256 | BetterHuman shape-baked vertices | 20.909514 | 70.088864 | 3.352008 | 8.476885 |
| SMPL-X | 1 | upstream smplx full forward | 7.271823 | 18.171266 | 2.498860 | 1.000000 |
| SMPL-X | 1 | BetterHuman shape-baked vertices | 64.120355 | 177.104108 | 2.762057 | 9.746382 |
| SMPL-X | 16 | upstream smplx full forward | 7.834374 | 18.430926 | 2.352571 | 1.000000 |
| SMPL-X | 16 | BetterHuman shape-baked vertices | 66.612173 | 180.349366 | 2.707454 | 9.785150 |
| SMPL-X | 256 | upstream smplx full forward | 9.238328 | 18.957483 | 2.052047 | 1.000000 |
| SMPL-X | 256 | BetterHuman shape-baked vertices | 68.362986 | 185.992442 | 2.720660 | 9.811030 |

### SMPL family — CUDA / Warp forward

<details>
<summary>Run metadata</summary>

```text
torch=2.13.0+cu126
better_robot=0.2.0
reference_version=0.1.28
reference_package=/data3/rikhat.akizhanov/better/BetterHuman/references/parametric_body/smplx/smplx
device=cuda (NVIDIA RTX 6000 Ada Generation)
cuda_runtime=12.6
cuda_driver=560.35.03
cpu_threads=112
cpu_interop_threads=112
warmup=5; iterations=30; batches=(1, 16, 256)
use_warp=True
grad=False
timing_caveat=shared-host scheduling and concurrent load can affect medians
smpl_path=/data3/rikhat.akizhanov/better/BetterHuman/models/smpl/SMPL_NEUTRAL.npz
smplx_path=/data3/rikhat.akizhanov/better/BetterHuman/models/smplx/SMPLX_NEUTRAL.npz
Warp 1.15.0 initialized:
   CUDA Toolkit 12.9, Driver 12.6
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /tmp/betterrobot-warp-cache/1.15.0
Module better_robot.kinematics._warp_kernels 763e84c load on device 'cuda:0' took 0.89 ms  (cached)
```

</details>

#### Pre-timing output agreement

| Model | Batch | Workload A max abs m | Workload B max abs m |
|---|---:|---:|---:|
| SMPL | 1 | 3.5762786865234375e-07 | 3.5762786865234375e-07 |
| SMPL | 16 | 4.1723251342773438e-07 | 3.5762786865234375e-07 |
| SMPL | 256 | 4.76837158203125e-07 | 4.76837158203125e-07 |
| SMPL-X | 1 | 2.4959444999694824e-07 | 2.5331974029541016e-07 |
| SMPL-X | 16 | 4.76837158203125e-07 | 4.76837158203125e-07 |
| SMPL-X | 256 | 5.9604644775390625e-07 | 4.76837158203125e-07 |

#### Workload A — full forward (new shape + new pose every call)

The × upstream ratio compares equivalent full shape-plus-pose computations; BetterHuman uses its one-call API.

| Model | Batch | Implementation | Median ms | × upstream |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream smplx full forward | 2.004711 | 1.000000 |
| SMPL | 1 | BetterHuman one-call full forward (Torch FK) | 18.611841 | 9.284052 |
| SMPL | 16 | upstream smplx full forward | 1.736296 | 1.000000 |
| SMPL | 16 | BetterHuman one-call full forward (Torch FK) | 19.081667 | 10.989867 |
| SMPL | 256 | upstream smplx full forward | 2.976521 | 1.000000 |
| SMPL | 256 | BetterHuman one-call full forward (Torch FK) | 20.883783 | 7.016172 |
| SMPL-X | 1 | upstream smplx full forward | 3.031563 | 1.000000 |
| SMPL-X | 1 | BetterHuman one-call full forward (Torch FK) | 40.573462 | 13.383678 |
| SMPL-X | 16 | upstream smplx full forward | 3.127746 | 1.000000 |
| SMPL-X | 16 | BetterHuman one-call full forward (Torch FK) | 26.067906 | 8.334406 |
| SMPL-X | 256 | upstream smplx full forward | 5.826313 | 1.000000 |
| SMPL-X | 256 | BetterHuman one-call full forward (Torch FK) | 55.559674 | 9.535991 |

#### Workload B — streaming poses (shape fixed, poses vary per call)

BetterHuman bakes shape once outside timing. Upstream has no shape-amortized API, so its per-frame workload remains a full forward. This is a workload-level ratio, not a FLOP-matched ratio.

| Model | Batch | Implementation | Median ms | × upstream |
|---|---:|---|---:|---:|
| SMPL | 1 | upstream smplx full forward | 2.004711 | 1.000000 |
| SMPL | 1 | BetterHuman shape-baked vertices (Warp requested) | 2.371145 | 1.182787 |
| SMPL | 16 | upstream smplx full forward | 1.736296 | 1.000000 |
| SMPL | 16 | BetterHuman shape-baked vertices (Warp requested) | 2.447046 | 1.409348 |
| SMPL | 256 | upstream smplx full forward | 2.976521 | 1.000000 |
| SMPL | 256 | BetterHuman shape-baked vertices (Warp requested) | 3.984193 | 1.338540 |
| SMPL-X | 1 | upstream smplx full forward | 3.031563 | 1.000000 |
| SMPL-X | 1 | BetterHuman shape-baked vertices (Warp requested) | 2.544034 | 0.839182 |
| SMPL-X | 16 | upstream smplx full forward | 3.127746 | 1.000000 |
| SMPL-X | 16 | BetterHuman shape-baked vertices (Warp requested) | 2.460619 | 0.786707 |
| SMPL-X | 256 | upstream smplx full forward | 5.826313 | 1.000000 |
| SMPL-X | 256 | BetterHuman shape-baked vertices (Warp requested) | 5.537430 | 0.950417 |

#### BetterHuman Workload B stage breakdown

| Model | Batch | Stage | Median ms |
|---|---:|---|---:|
| SMPL | 1 | FK (Warp requested) | 0.661066 |
| SMPL | 1 | corrective | 0.384915 |
| SMPL | 1 | transforms + LBS | 0.978911 |
| SMPL | 16 | FK (Warp requested) | 0.687351 |
| SMPL | 16 | corrective | 0.391481 |
| SMPL | 16 | transforms + LBS | 1.055481 |
| SMPL | 256 | FK (Warp requested) | 0.749190 |
| SMPL | 256 | corrective | 0.442141 |
| SMPL | 256 | transforms + LBS | 2.546454 |
| SMPL-X | 1 | FK (Warp requested) | 0.655344 |
| SMPL-X | 1 | corrective | 0.456231 |
| SMPL-X | 1 | transforms + LBS | 1.110605 |
| SMPL-X | 16 | FK (Warp requested) | 0.675551 |
| SMPL-X | 16 | corrective | 0.412083 |
| SMPL-X | 16 | transforms + LBS | 1.060273 |
| SMPL-X | 256 | FK (Warp requested) | 0.962954 |
| SMPL-X | 256 | corrective | 0.705014 |
| SMPL-X | 256 | transforms + LBS | 3.697341 |

### MHR — CPU / native Torch gradients

<details>
<summary>Run metadata</summary>

```text
torch=2.13.0+cu126
better_robot=0.2.0
device=cpu
cpu_threads=112
cpu_interop_threads=112
warmup=5; iterations=30; batches=(1, 16, 256)
use_warp=False
grad=True
timing_caveat=shared-host scheduling and concurrent load may affect observational medians
model_path=/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/converted/mhr_lod1.npz
reference_path=/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/assets/mhr_model.pt
```

</details>

#### Pre-timing output agreement

Gate: max absolute vertex error ≤ 5.0e-06 m.

| Batch | Correctives | Streaming poses max abs m | Full forward max abs m |
|---:|:---:|---:|---:|
| 1 | on | 2.7418136596679688e-06 | 2.7418136596679688e-06 |
| 16 | on | 2.6822090148925781e-06 | 2.6822090148925781e-06 |
| 256 | on | 2.6822090148925781e-06 | 2.6822090148925781e-06 |
| 1 | off | 2.6822090148925781e-06 | 2.6822090148925781e-06 |
| 16 | off | 2.6226043701171875e-06 | 2.6226043701171875e-06 |
| 256 | off | 2.6226043701171875e-06 | 2.6226043701171875e-06 |

#### Workload A — full forward (new shape + new pose every call)

Both rows compute full shape and pose; each ratio compares identical work.

| Batch | Correctives | Path | Median ms | × reference |
|---:|:---:|---|---:|---:|
| 1 | on | TorchScript full shape+pose | 23.336780 | 1.000000 |
| 1 | on | BetterHuman full shape+pose | 165.512548 | 7.092347 |
| 1 | off | TorchScript full shape+pose | 10.762407 | 1.000000 |
| 1 | off | BetterHuman full shape+pose | 154.999911 | 14.401974 |
| 16 | on | TorchScript full shape+pose | 30.333932 | 1.000000 |
| 16 | on | BetterHuman full shape+pose | 69.517568 | 2.291743 |
| 16 | off | TorchScript full shape+pose | 26.158406 | 1.000000 |
| 16 | off | BetterHuman full shape+pose | 62.134325 | 2.375310 |
| 256 | on | TorchScript full shape+pose | 911.843281 | 1.000000 |
| 256 | on | BetterHuman full shape+pose | 206.352065 | 0.226302 |
| 256 | off | TorchScript full shape+pose | 787.022646 | 1.000000 |
| 256 | off | BetterHuman full shape+pose | 249.103795 | 0.316514 |

#### Workload B — streaming poses (shape fixed, poses vary per call)

BetterHuman bakes shape once outside timing. TorchScript has no shape-amortized API, so its per-frame cost remains a full shape+pose forward; ratios compare the same user workload, not matched FLOPs.

| Batch | Correctives | Per-frame path | Median ms | × reference |
|---:|:---:|---|---:|---:|
| 1 | on | TorchScript full shape+pose | 23.336780 | 1.000000 |
| 1 | on | BetterHuman streaming poses | 60.168659 | 2.578276 |
| 1 | off | TorchScript full shape+pose | 10.762407 | 1.000000 |
| 1 | off | BetterHuman streaming poses | 82.997890 | 7.711833 |
| 16 | on | TorchScript full shape+pose | 30.333932 | 1.000000 |
| 16 | on | BetterHuman streaming poses | 37.420636 | 1.233623 |
| 16 | off | TorchScript full shape+pose | 26.158406 | 1.000000 |
| 16 | off | BetterHuman streaming poses | 51.659231 | 1.974862 |
| 256 | on | TorchScript full shape+pose | 911.843281 | 1.000000 |
| 256 | on | BetterHuman streaming poses | 136.900657 | 0.150136 |
| 256 | off | TorchScript full shape+pose | 787.022646 | 1.000000 |
| 256 | off | BetterHuman streaming poses | 148.225082 | 0.188336 |

#### BetterHuman streaming-pose breakdown

| Batch | Correctives | Path | Median ms |
|---:|:---:|---|---:|
| 1 | on | FK | 42.424694 |
| 1 | on | corrective | 33.417001 |
| 1 | on | transforms + LBS | 0.589352 |
| 16 | on | FK | 28.464218 |
| 16 | on | corrective | 4.575020 |
| 16 | on | transforms + LBS | 2.382300 |
| 256 | on | FK | 59.337605 |
| 256 | on | corrective | 24.533667 |
| 256 | on | transforms + LBS | 81.591606 |
| 1 | off | FK | 42.473900 |
| 1 | off | corrective | 0.024682 |
| 1 | off | transforms + LBS | 0.772644 |
| 16 | off | FK | 27.868542 |
| 16 | off | corrective | 0.039297 |
| 16 | off | transforms + LBS | 2.648135 |
| 256 | off | FK | 57.114863 |
| 256 | off | corrective | 0.039530 |
| 256 | off | transforms + LBS | 84.504691 |

#### BetterHuman absolute gradient performance

Workload A differentiates q and all shape groups; workload B differentiates q with shape baked outside timing. Both use Torch FK and Torch autograd, with no upstream row.
F+B / F is the training-step multiplier, not a backward-only median.

| Batch | Correctives | Workload | Forward-only ms | Forward + backward ms | F+B / F |
|---:|:---:|---|---:|---:|---:|
| 1 | on | A — full shape+pose (q + shape groups) | 212.369610 | 446.482271 | 2.102383 |
| 1 | on | B — streaming poses (q) | 130.829949 | 336.132665 | 2.569233 |
| 16 | on | A — full shape+pose (q + shape groups) | 124.484499 | 298.137320 | 2.394975 |
| 16 | on | B — streaming poses (q) | 62.761629 | 169.143155 | 2.695009 |
| 256 | on | A — full shape+pose (q + shape groups) | 255.153583 | 656.606512 | 2.573378 |
| 256 | on | B — streaming poses (q) | 158.601529 | 490.848457 | 3.094853 |
| 1 | off | A — full shape+pose (q + shape groups) | 189.611799 | 435.343932 | 2.295975 |
| 1 | off | B — streaming poses (q) | 101.297063 | 261.008742 | 2.576666 |
| 16 | off | A — full shape+pose (q + shape groups) | 174.503453 | 393.588206 | 2.255475 |
| 16 | off | B — streaming poses (q) | 51.644019 | 147.767241 | 2.861265 |
| 256 | off | A — full shape+pose (q + shape groups) | 329.645655 | 722.676883 | 2.192284 |
| 256 | off | B — streaming poses (q) | 184.607942 | 510.086399 | 2.763079 |

### MHR — CUDA / native Torch gradients

<details>
<summary>Run metadata</summary>

```text
torch=2.13.0+cu126
better_robot=0.2.0
device=cuda (NVIDIA RTX 6000 Ada Generation)
cuda_runtime=12.6
cuda_driver=560.35.03
cpu_threads=112
cpu_interop_threads=112
warmup=5; iterations=30; batches=(1, 16, 256)
use_warp=False
grad=True
timing_caveat=shared-host scheduling and concurrent load may affect observational medians
model_path=/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/converted/mhr_lod1.npz
reference_path=/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/assets/mhr_model.pt
```

</details>

#### Pre-timing output agreement

Gate: max absolute vertex error ≤ 5.0e-06 m.

| Batch | Correctives | Streaming poses max abs m | Full forward max abs m |
|---:|:---:|---:|---:|
| 1 | on | 1.5497207641601562e-06 | 1.5497207641601562e-06 |
| 16 | on | 1.5497207641601562e-06 | 1.5497207641601562e-06 |
| 256 | on | 1.5497207641601562e-06 | 1.5497207641601562e-06 |
| 1 | off | 1.4901161193847656e-06 | 1.4901161193847656e-06 |
| 16 | off | 1.430511474609375e-06 | 1.430511474609375e-06 |
| 256 | off | 1.4901161193847656e-06 | 1.4901161193847656e-06 |

#### Workload A — full forward (new shape + new pose every call)

Both rows compute full shape and pose; each ratio compares identical work.

| Batch | Correctives | Path | Median ms | × reference |
|---:|:---:|---|---:|---:|
| 1 | on | TorchScript full shape+pose | 2.252701 | 1.000000 |
| 1 | on | BetterHuman full shape+pose | 150.867692 | 66.971912 |
| 1 | off | TorchScript full shape+pose | 1.160139 | 1.000000 |
| 1 | off | BetterHuman full shape+pose | 150.326115 | 129.575952 |
| 16 | on | TorchScript full shape+pose | 2.574029 | 1.000000 |
| 16 | on | BetterHuman full shape+pose | 155.786447 | 60.522402 |
| 16 | off | TorchScript full shape+pose | 1.305433 | 1.000000 |
| 16 | off | BetterHuman full shape+pose | 313.584503 | 240.214843 |
| 256 | on | TorchScript full shape+pose | 11.318191 | 1.000000 |
| 256 | on | BetterHuman full shape+pose | 169.116997 | 14.942053 |
| 256 | off | TorchScript full shape+pose | 9.160384 | 1.000000 |
| 256 | off | BetterHuman full shape+pose | 161.538835 | 17.634505 |

#### Workload B — streaming poses (shape fixed, poses vary per call)

BetterHuman bakes shape once outside timing. TorchScript has no shape-amortized API, so its per-frame cost remains a full shape+pose forward; ratios compare the same user workload, not matched FLOPs.

| Batch | Correctives | Per-frame path | Median ms | × reference |
|---:|:---:|---|---:|---:|
| 1 | on | TorchScript full shape+pose | 2.252701 | 1.000000 |
| 1 | on | BetterHuman streaming poses | 76.708944 | 34.051986 |
| 1 | off | TorchScript full shape+pose | 1.160139 | 1.000000 |
| 1 | off | BetterHuman streaming poses | 74.335215 | 64.074404 |
| 16 | on | TorchScript full shape+pose | 2.574029 | 1.000000 |
| 16 | on | BetterHuman streaming poses | 77.473613 | 30.098184 |
| 16 | off | TorchScript full shape+pose | 1.305433 | 1.000000 |
| 16 | off | BetterHuman streaming poses | 135.909907 | 104.110939 |
| 256 | on | TorchScript full shape+pose | 11.318191 | 1.000000 |
| 256 | on | BetterHuman streaming poses | 84.796631 | 7.492066 |
| 256 | off | TorchScript full shape+pose | 9.160384 | 1.000000 |
| 256 | off | BetterHuman streaming poses | 164.078545 | 17.911754 |

#### BetterHuman streaming-pose breakdown

| Batch | Correctives | Path | Median ms |
|---:|:---:|---|---:|
| 1 | on | FK | 73.918066 |
| 1 | on | corrective | 0.921601 |
| 1 | on | transforms + LBS | 0.485233 |
| 16 | on | FK | 76.015176 |
| 16 | on | corrective | 0.953332 |
| 16 | on | transforms + LBS | 0.671344 |
| 256 | on | FK | 80.182771 |
| 256 | on | corrective | 1.629461 |
| 256 | on | transforms + LBS | 5.207095 |
| 1 | off | FK | 75.614631 |
| 1 | off | corrective | 0.009262 |
| 1 | off | transforms + LBS | 0.497956 |
| 16 | off | FK | 153.696803 |
| 16 | off | corrective | 0.018259 |
| 16 | off | transforms + LBS | 1.148986 |
| 256 | off | FK | 77.646215 |
| 256 | off | corrective | 0.009222 |
| 256 | off | transforms + LBS | 5.125349 |

#### BetterHuman absolute gradient performance

Workload A differentiates q and all shape groups; workload B differentiates q with shape baked outside timing. Both use Torch FK and Torch autograd, with no upstream row.
F+B / F is the training-step multiplier, not a backward-only median.

| Batch | Correctives | Workload | Forward-only ms | Forward + backward ms | F+B / F |
|---:|:---:|---|---:|---:|---:|
| 1 | on | A — full shape+pose (q + shape groups) | 207.695002 | 581.920783 | 2.801804 |
| 1 | on | B — streaming poses (q) | 102.178812 | 384.318314 | 3.761233 |
| 16 | on | A — full shape+pose (q + shape groups) | 310.976191 | 671.553823 | 2.159502 |
| 16 | on | B — streaming poses (q) | 208.497832 | 545.357878 | 2.615653 |
| 256 | on | A — full shape+pose (q + shape groups) | 376.295924 | 617.513200 | 1.641031 |
| 256 | on | B — streaming poses (q) | 181.165978 | 363.791832 | 2.008058 |
| 1 | off | A — full shape+pose (q + shape groups) | 349.414554 | 584.243813 | 1.672065 |
| 1 | off | B — streaming poses (q) | 172.510312 | 363.918961 | 2.109549 |
| 16 | off | A — full shape+pose (q + shape groups) | 354.854490 | 738.743698 | 2.081821 |
| 16 | off | B — streaming poses (q) | 105.754499 | 393.191090 | 3.717961 |
| 256 | off | A — full shape+pose (q + shape groups) | 218.495971 | 603.712264 | 2.763036 |
| 256 | off | B — streaming poses (q) | 107.715235 | 398.367353 | 3.698338 |

### MHR — CUDA / Warp forward

<details>
<summary>Run metadata</summary>

```text
torch=2.13.0+cu126
better_robot=0.2.0
device=cuda (NVIDIA RTX 6000 Ada Generation)
cuda_runtime=12.6
cuda_driver=560.35.03
cpu_threads=112
cpu_interop_threads=112
warmup=5; iterations=30; batches=(1, 16, 256)
use_warp=True
grad=False
timing_caveat=shared-host scheduling and concurrent load may affect observational medians
model_path=/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/converted/mhr_lod1.npz
reference_path=/data3/rikhat.akizhanov/better/BetterHuman/models/MHR/assets/mhr_model.pt
Warp 1.15.0 initialized:
   CUDA Toolkit 12.9, Driver 12.6
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /tmp/betterrobot-warp-cache/1.15.0
Module better_robot.kinematics._warp_kernels 763e84c load on device 'cuda:0' took 0.89 ms  (cached)
```

</details>

#### Pre-timing output agreement

Gate: max absolute vertex error ≤ 5.0e-06 m.

| Batch | Correctives | Streaming poses max abs m | Full forward max abs m |
|---:|:---:|---:|---:|
| 1 | on | 1.430511474609375e-06 | 1.5497207641601562e-06 |
| 16 | on | 1.3709068298339844e-06 | 1.5497207641601562e-06 |
| 256 | on | 1.3709068298339844e-06 | 1.5497207641601562e-06 |
| 1 | off | 1.3709068298339844e-06 | 1.4901161193847656e-06 |
| 16 | off | 1.3113021850585938e-06 | 1.430511474609375e-06 |
| 256 | off | 1.3709068298339844e-06 | 1.4901161193847656e-06 |

#### Workload A — full forward (new shape + new pose every call)

Both rows compute full shape and pose; each ratio compares identical work.

| Batch | Correctives | Path | Median ms | × reference |
|---:|:---:|---|---:|---:|
| 1 | on | TorchScript full shape+pose | 3.647041 | 1.000000 |
| 1 | on | BetterHuman full shape+pose (Torch FK) | 193.928811 | 53.174296 |
| 1 | off | TorchScript full shape+pose | 2.357448 | 1.000000 |
| 1 | off | BetterHuman full shape+pose (Torch FK) | 151.432239 | 64.235651 |
| 16 | on | TorchScript full shape+pose | 4.051382 | 1.000000 |
| 16 | on | BetterHuman full shape+pose (Torch FK) | 156.430489 | 38.611632 |
| 16 | off | TorchScript full shape+pose | 2.483768 | 1.000000 |
| 16 | off | BetterHuman full shape+pose (Torch FK) | 155.622264 | 62.655717 |
| 256 | on | TorchScript full shape+pose | 12.988791 | 1.000000 |
| 256 | on | BetterHuman full shape+pose (Torch FK) | 164.781857 | 12.686466 |
| 256 | off | TorchScript full shape+pose | 10.341867 | 1.000000 |
| 256 | off | BetterHuman full shape+pose (Torch FK) | 163.516999 | 15.811168 |

#### Workload B — streaming poses (shape fixed, poses vary per call)

BetterHuman bakes shape once outside timing. TorchScript has no shape-amortized API, so its per-frame cost remains a full shape+pose forward; ratios compare the same user workload, not matched FLOPs.

| Batch | Correctives | Per-frame path | Median ms | × reference |
|---:|:---:|---|---:|---:|
| 1 | on | TorchScript full shape+pose | 3.647041 | 1.000000 |
| 1 | on | BetterHuman streaming poses (Warp requested) | 2.369919 | 0.649820 |
| 1 | off | TorchScript full shape+pose | 2.357448 | 1.000000 |
| 1 | off | BetterHuman streaming poses (Warp requested) | 1.010115 | 0.428478 |
| 16 | on | TorchScript full shape+pose | 4.051382 | 1.000000 |
| 16 | on | BetterHuman streaming poses (Warp requested) | 2.230097 | 0.550453 |
| 16 | off | TorchScript full shape+pose | 2.483768 | 1.000000 |
| 16 | off | BetterHuman streaming poses (Warp requested) | 1.210952 | 0.487546 |
| 256 | on | TorchScript full shape+pose | 12.988791 | 1.000000 |
| 256 | on | BetterHuman streaming poses (Warp requested) | 8.292859 | 0.638463 |
| 256 | off | TorchScript full shape+pose | 10.341867 | 1.000000 |
| 256 | off | BetterHuman streaming poses (Warp requested) | 6.657831 | 0.643775 |

#### BetterHuman streaming-pose breakdown

| Batch | Correctives | Path | Median ms |
|---:|:---:|---|---:|
| 1 | on | FK (Warp requested) | 0.604677 |
| 1 | on | corrective | 0.922588 |
| 1 | on | transforms + LBS | 0.488313 |
| 16 | on | FK (Warp requested) | 0.619121 |
| 16 | on | corrective | 0.953587 |
| 16 | on | transforms + LBS | 0.671118 |
| 256 | on | FK (Warp requested) | 1.551654 |
| 256 | on | corrective | 1.638420 |
| 256 | on | transforms + LBS | 5.136963 |
| 1 | off | FK (Warp requested) | 0.600692 |
| 1 | off | corrective | 0.009238 |
| 1 | off | transforms + LBS | 0.488648 |
| 16 | off | FK (Warp requested) | 0.625941 |
| 16 | off | corrective | 0.009068 |
| 16 | off | transforms + LBS | 0.680054 |
| 256 | off | FK (Warp requested) | 1.545492 |
| 256 | off | corrective | 0.009011 |
| 256 | off | transforms + LBS | 5.137304 |
