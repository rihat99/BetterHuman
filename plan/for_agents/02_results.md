# Order 02 results — matrix-blend LBS

Status: complete on `dev` at working-tree state, 2026-07-21.

## Implementation

- `BodyValues` now stores one spawn-built dense `(V, J)` skinning matrix while
  retaining the padded sparse weights and indices for archive fidelity and
  segmentation.
- Both model families construct the matrix once with duplicate-safe
  `scatter_add_`.
- `lbs` has one formulation: convert each joint transform to a 3×4 affine,
  fold scale and bind pivots into that affine, blend joint affines with one
  dense contraction, then apply one affine per vertex.
- Production deformation, MHR neutral mass baking, and benchmark stage
  instrumentation all call that same implementation. The old per-influence
  gather path is gone.
- The SMPL exact-reduction test now pins the requested invariant: `s = 1`
  scaled matrix-blend output is tensor-equal to the unscaled production path.

## Parity before and after

No tolerance changed. All post-rewrite maxima remain within the existing
pins.

| Model | dtype | V before | V after | J before | J after |
|---|---|---:|---:|---:|---:|
| SMPL | float32 | 7.0780515670776367e-07 | 7.0594251155853271e-07 | 5.289912223815918e-07 | 5.289912223815918e-07 |
| SMPL | float64 | 1.0412031070661243e-07 | 1.0412031081763473e-07 | 5.7804501407687781e-08 | 5.7804501407687781e-08 |
| SMPL-X | float32 | 6.5565109252929688e-07 | 6.2584877014160156e-07 | 5.0663948059082031e-07 | 5.0663948059082031e-07 |
| SMPL-X | float64 | 5.5905974738479358e-08 | 5.5905974738479358e-08 | 8.6539269084084935e-09 | 8.6539269084084935e-09 |

| LOD | dtype | V before | V after | P before | P after | Q before | Q after | S before | S after |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | float32 | 2.0265579223632812e-06 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 2.9802322387695312e-07 | 2.9802322387695312e-07 | 0 | 0 |
| 1 | float32 | 2.9206275939941406e-06 | 2.8014183044433594e-06 | 7.5995922088623047e-07 | 7.5995922088623047e-07 | 6.5565109252929688e-07 | 6.5565109252929688e-07 | 0 | 0 |
| 2 | float32 | 1.9669532775878906e-06 | 1.9669532775878906e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 2.9802322387695312e-07 | 2.9802322387695312e-07 | 0 | 0 |
| 3 | float32 | 2.2649765014648438e-06 | 2.2649765014648438e-06 | 7.152557373046875e-07 | 7.152557373046875e-07 | 4.76837158203125e-07 | 4.76837158203125e-07 | 0 | 0 |
| 4 | float32 | 2.0265579223632812e-06 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 0 | 0 |
| 5 | float32 | 1.7881393432617188e-06 | 1.9073486328125e-06 | 4.76837158203125e-07 | 4.76837158203125e-07 | 2.9802322387695312e-07 | 2.9802322387695312e-07 | 0 | 0 |
| 6 | float32 | 1.7881393432617188e-06 | 1.7881393432617188e-06 | 2.6822090148925781e-07 | 2.6822090148925781e-07 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 0 | 0 |
| 0 | float64 | 6.7963208660870578e-07 | 6.7963208649768347e-07 | 1.8331952966743614e-07 | 1.8331952966743614e-07 | 1.3139750559831853e-07 | 1.3139750559831853e-07 | 2.4801335873902985e-08 | 2.4801335873902985e-08 |
| 1 | float64 | 8.4886980189491368e-07 | 8.4886980189491368e-07 | 2.215700635188611e-07 | 2.215700635188611e-07 | 1.9291339442562006e-07 | 1.9291339442562006e-07 | 4.6629366146078155e-08 | 4.6629366146078155e-08 |
| 2 | float64 | 5.5811346433287667e-07 | 5.5811346455492128e-07 | 1.3154689582073331e-07 | 1.3154689582073331e-07 | 1.0220597668197939e-07 | 1.0220597668197939e-07 | 5.7916624163212305e-08 | 5.7916624163212305e-08 |
| 3 | float64 | 5.6179104368503374e-07 | 5.6179104346298914e-07 | 9.9507051043445927e-08 | 9.9507051043445927e-08 | 1.0819653151195041e-07 | 1.0819653151195041e-07 | 2.9032722226318697e-08 | 2.9032722226318697e-08 |
| 4 | float64 | 5.5301331047807878e-07 | 5.5301331081114569e-07 | 2.0618963714191807e-07 | 2.0618963714191807e-07 | 1.3338477658497716e-07 | 1.3338477658497716e-07 | 2.5566294747925156e-08 | 2.5566294747925156e-08 |
| 5 | float64 | 4.4630069462314736e-07 | 4.4630069462314736e-07 | 1.2267317051062321e-07 | 1.2267317051062321e-07 | 1.2220955582231685e-07 | 1.2220955582231685e-07 | 2.2953858813679062e-08 | 2.2953858813679062e-08 |
| 6 | float64 | 5.9353756443059069e-07 | 5.9353756420854609e-07 | 1.2023582485198858e-07 | 1.2023582485198858e-07 | 1.8299445425640215e-07 | 1.8299445425640215e-07 | 2.9279848323682245e-08 | 2.9279848323682245e-08 |

The secondary TorchScript check compares two fixed external artifacts and
does not execute BetterHuman, so its before and after values are necessarily
the same. Re-running the four-case grid measured V
`1.3113021850585938e-06` m and bit-exact P/Q/S (`0` for each). The packaged
parity suite passed 103 tests before and after.

## Test and static gates

| Gate | Before | After |
|---|---:|---:|
| Collected/full CUDA tests | 226 | 226 |
| Packaged parity tests | 103 passed | 103 passed |
| Full CUDA run | 226 passed, 0 skipped, 19 warnings | 226 passed, 0 skipped, 19 warnings |

Focused changed-area tests passed 57 tests with three expected CUDA skips in
the sandbox. `ruff check .`, changed-file formatting, and `git diff --check`
passed. No test was deleted. The permitted baseline-formulation swap renamed
`test_generalized_skinning_reduces_exactly_on_smpl_fixture` to
`test_unit_scales_reduce_exactly_to_unscaled_smpl_skinning`.

## Benchmark protocol

All tables use LOD1 float32, batches 1/16/256, correctives on/off, five
warmups, 30 explicitly synchronized timed calls, PyTorch 2.13.0+cu126, and
BetterRobot 0.2.0. CPU used 112 intra/inter-op threads. CUDA used physical GPU
7, an otherwise idle NVIDIA RTX 6000 Ada Generation with driver 560.35.03;
the Warp run used Warp 1.15.0. Ratios are BetterHuman/reference from the same
invocation and corrective mode.

## CPU benchmark

| Batch | Correctives | TorchScript shape+pose ms | BetterHuman pose-only ms / ×ref | BetterHuman shape+pose ms / ×ref |
|---:|:---:|---:|---:|---:|
| 1 | on | 19.505918 | 104.505661 / 5.357639× | 175.038120 / 8.973590× |
| 1 | off | 10.832997 | 83.003433 / 7.662093× | 156.004285 / 14.400843× |
| 16 | on | 43.595248 | 32.781757 / 0.751957× | 61.089930 / 1.401298× |
| 16 | off | 38.767081 | 51.125402 / 1.318784× | 101.672627 / 2.622654× |
| 256 | on | 767.727395 | 178.638341 / 0.232685× | 270.257807 / 0.352023× |
| 256 | off | 860.573079 | 141.988578 / 0.164993× | 247.239801 / 0.287297× |

| Batch | Correctives | FK ms | Corrective ms | Transforms + LBS ms |
|---:|:---:|---:|---:|---:|
| 1 | on | 42.712647 | 36.831736 | 0.593044 |
| 1 | off | 42.148122 | 6.001499 | 23.995179 |
| 16 | on | 25.418313 | 3.209571 | 2.274361 |
| 16 | off | 47.740547 | 0.065788 | 4.028227 |
| 256 | on | 56.038709 | 24.450985 | 85.244686 |
| 256 | off | 55.463856 | 0.048177 | 84.558726 |

Shared-host load produced visibly noisy small-batch CPU stage medians. The
batch-256 matrix-blend stage nevertheless improved by 16.748883×/15.921702×
against the previously recorded implementation (correctives on/off).

## Native CUDA benchmark

| Batch | Correctives | TorchScript shape+pose ms | BetterHuman pose-only ms / ×ref | BetterHuman shape+pose ms / ×ref |
|---:|:---:|---:|---:|---:|
| 1 | on | 3.549979 | 95.193920 / 26.815347× | 190.865683 / 53.765299× |
| 1 | off | 2.275135 | 70.019388 / 30.775927× | 140.907083 / 61.933504× |
| 16 | on | 3.951922 | 73.523560 / 18.604507× | 146.560227 / 37.085810× |
| 16 | off | 2.384344 | 72.506513 / 30.409418× | 145.687805 / 61.101840× |
| 256 | on | 12.171629 | 79.546592 / 6.535411× | 155.047731 / 12.738454× |
| 256 | off | 9.428837 | 78.125182 / 8.285771× | 153.348507 / 16.263777× |

| Batch | Correctives | FK ms | Corrective ms | Transforms + LBS ms |
|---:|:---:|---:|---:|---:|
| 1 | on | 92.420127 | 0.944153 | 0.556769 |
| 1 | off | 69.554530 | 0.008936 | 0.460583 |
| 16 | on | 71.726995 | 0.944954 | 0.642901 |
| 16 | off | 71.457337 | 0.008841 | 0.645576 |
| 256 | on | 73.015869 | 1.617321 | 5.103035 |
| 256 | off | 72.660426 | 0.008654 | 5.102114 |

The batch-256 deformation stage is now approximately 5.10 ms, while native
Torch FK remains approximately 73 ms and dominates this lane.

## Warp CUDA benchmark

| Batch | Correctives | TorchScript shape+pose ms | BetterHuman pose-only ms / ×ref | BetterHuman shape+pose Torch-FK ms / ×ref |
|---:|:---:|---:|---:|---:|
| 1 | on | 2.191199 | 2.565200 / 1.170683× | 282.477651 / 128.914649× |
| 1 | off | 1.120560 | 1.062819 / 0.948471× | 175.837544 / 156.919347× |
| 16 | on | 2.516612 | 2.263239 / 0.899320× | 184.140042 / 73.169818× |
| 16 | off | 1.236523 | 1.306156 / 1.056314× | 184.246376 / 149.003598× |
| 256 | on | 11.200383 | 8.333692 / 0.744054× | 192.926257 / 17.224970× |
| 256 | off | 9.071364 | 6.679759 / 0.736357× | 187.913769 / 20.715051× |

| Batch | Correctives | Warp FK ms | Corrective ms | Transforms + LBS ms |
|---:|:---:|---:|---:|---:|
| 1 | on | 0.632403 | 0.945547 | 0.578162 |
| 1 | off | 0.615940 | 0.010177 | 0.565254 |
| 16 | on | 0.658878 | 0.977999 | 0.771719 |
| 16 | off | 0.642872 | 0.009840 | 0.749222 |
| 256 | on | 1.688651 | 1.925943 | 5.245006 |
| 256 | off | 1.596782 | 0.009864 | 5.182707 |

The Warp pose-only path meets the `≤2×` target in both corrective modes at
batch 16 and 256. The batch-256 deformation stage improved by
5.732667×/5.352001× against the prior per-influence formulation. The fair
shape+pose rows remain on BetterRobot Torch FK and still miss the target; no
kernel or caching workaround was added.

## Findings and deviations

- No implementation deviation was required. The expected summation-order
  rounding moved vertex maxima only; every existing pin held.
- The CPU host was busy enough to distort small-batch stage medians. Raw values
  are retained rather than filtered or rerun selectively.
- The secondary TorchScript grid currently measures a worst vertex error of
  `1.3113021850585938e-06` m, not the stale `1.220703125e-06` value in the
  contracts. Its P/Q/S agreement is bit-exact. Order 03 must correct that
  documentation while preserving the enforced tolerance wording.

Order 03 was not started until this implementation, report, and all Order 02
gates were complete.
