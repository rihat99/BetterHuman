# Order 01 results — overengineering removal

Status: complete on `dev` at working-tree state, 2026-07-21.

## Finding accounting

| Finding | Status | Result |
|---|---|---|
| F1 | Applied with deviation | Removed the world/root scalar checks, five converter-constant string checks, and four corrective re-checks. Retained `up_axis`, `forward_axis`, and `coordinate_axis`: the converter reads them from FBX rather than writing literals, while runtime assumes `+y/+z/+x`. |
| F2 | Applied | Stored `_TreeBinding` once as `self._tree`; all former exploded binding attributes now resolve through it, including the benchmark reach-in. |
| F3 | Applied | Removed the provisional neutral FK and used zero rest-joint/template placeholders until the existing shape bake overwrites them. |
| F4 | Applied | The NPZ loader reads and validates only the selected schema. Excluded payloads may be absent or invalid and are never indexed; the guarded test proves lazy exclusion. |
| F5 | Applied | Removed the unused `pose_feature_6d_from_matrix` API, export, direct test, and its gradcheck half. |
| F6 | Skipped | Optional archive-format change would require regenerating and revalidating all seven multi-GB converted archives for about 15.8 MB total savings. |
| F7 | Applied | Removed the dead `_TreeBinding.corrective_euler_indices` member; the local index tensor remains only to construct the accumulating one-hot transform. |
| F8 | Applied | Removed the production-unused 7D overload from `skinning_transforms`. Non-identity bind coverage now exercises `skinning_transforms_from_inverse`; one overload-only test was deleted. |
| F9 | Applied | Removed the unused optional MLP bias argument and retained no-bias output and gradient coverage. |
| F10 | Applied | Re-expressed partitions, host weights, permutations, limits, round trips, and mimic expansion through archive data and public behavior. Private reaches remain only for corrective binding presence and authored-row alignment, with comments explaining the missing public surface. |
| F11 | Skipped | Pinned `pymomentum` exposes no equivalent load-time FBX metadata, and its FBX SDK integration is unavailable in the parity environment. |

## Numerical parity

The same script was run before and after the refactor. Every printed maximum
below is exactly equal before and after.

| Model | dtype | V before | V after | J before | J after |
|---|---|---:|---:|---:|---:|
| SMPL | float32 | 7.0780515670776367e-07 | 7.0780515670776367e-07 | 5.289912223815918e-07 | 5.289912223815918e-07 |
| SMPL | float64 | 1.0412031070661243e-07 | 1.0412031070661243e-07 | 5.7804501407687781e-08 | 5.7804501407687781e-08 |
| SMPL-X | float32 | 6.5565109252929688e-07 | 6.5565109252929688e-07 | 5.0663948059082031e-07 | 5.0663948059082031e-07 |
| SMPL-X | float64 | 5.5905974738479358e-08 | 5.5905974738479358e-08 | 8.6539269084084935e-09 | 8.6539269084084935e-09 |

| LOD | dtype | V before | V after | P before | P after | Q before | Q after | S before | S after |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | float32 | 2.0265579223632812e-06 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 2.9802322387695312e-07 | 2.9802322387695312e-07 | 0 | 0 |
| 1 | float32 | 2.9206275939941406e-06 | 2.9206275939941406e-06 | 7.5995922088623047e-07 | 7.5995922088623047e-07 | 6.5565109252929688e-07 | 6.5565109252929688e-07 | 0 | 0 |
| 2 | float32 | 1.9669532775878906e-06 | 1.9669532775878906e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 2.9802322387695312e-07 | 2.9802322387695312e-07 | 0 | 0 |
| 3 | float32 | 2.2649765014648438e-06 | 2.2649765014648438e-06 | 7.152557373046875e-07 | 7.152557373046875e-07 | 4.76837158203125e-07 | 4.76837158203125e-07 | 0 | 0 |
| 4 | float32 | 2.0265579223632812e-06 | 2.0265579223632812e-06 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 0 | 0 |
| 5 | float32 | 1.7881393432617188e-06 | 1.7881393432617188e-06 | 4.76837158203125e-07 | 4.76837158203125e-07 | 2.9802322387695312e-07 | 2.9802322387695312e-07 | 0 | 0 |
| 6 | float32 | 1.7881393432617188e-06 | 1.7881393432617188e-06 | 2.6822090148925781e-07 | 2.6822090148925781e-07 | 3.5762786865234375e-07 | 3.5762786865234375e-07 | 0 | 0 |
| 0 | float64 | 6.7963208660870578e-07 | 6.7963208660870578e-07 | 1.8331952966743614e-07 | 1.8331952966743614e-07 | 1.3139750559831853e-07 | 1.3139750559831853e-07 | 2.4801335873902985e-08 | 2.4801335873902985e-08 |
| 1 | float64 | 8.4886980189491368e-07 | 8.4886980189491368e-07 | 2.215700635188611e-07 | 2.215700635188611e-07 | 1.9291339442562006e-07 | 1.9291339442562006e-07 | 4.6629366146078155e-08 | 4.6629366146078155e-08 |
| 2 | float64 | 5.5811346433287667e-07 | 5.5811346433287667e-07 | 1.3154689582073331e-07 | 1.3154689582073331e-07 | 1.0220597668197939e-07 | 1.0220597668197939e-07 | 5.7916624163212305e-08 | 5.7916624163212305e-08 |
| 3 | float64 | 5.6179104368503374e-07 | 5.6179104368503374e-07 | 9.9507051043445927e-08 | 9.9507051043445927e-08 | 1.0819653151195041e-07 | 1.0819653151195041e-07 | 2.9032722226318697e-08 | 2.9032722226318697e-08 |
| 4 | float64 | 5.5301331047807878e-07 | 5.5301331047807878e-07 | 2.0618963714191807e-07 | 2.0618963714191807e-07 | 1.3338477658497716e-07 | 1.3338477658497716e-07 | 2.5566294747925156e-08 | 2.5566294747925156e-08 |
| 5 | float64 | 4.4630069462314736e-07 | 4.4630069462314736e-07 | 1.2267317051062321e-07 | 1.2267317051062321e-07 | 1.2220955582231685e-07 | 1.2220955582231685e-07 | 2.2953858813679062e-08 | 2.2953858813679062e-08 |
| 6 | float64 | 5.9353756443059069e-07 | 5.9353756443059069e-07 | 1.2023582485198858e-07 | 1.2023582485198858e-07 | 1.8299445425640215e-07 | 1.8299445425640215e-07 | 2.9279848323682245e-08 | 2.9279848323682245e-08 |

The packaged parity suite was also unchanged: 103 passed before and 103 passed
after.

## Tests and static gates

| Gate | Result |
|---|---|
| Focused changed-area tests | 54 passed, 2 CUDA-gated skips in the sandbox |
| Full suite, CUDA visible | 226 passed, 0 skipped, 19 warnings in 91.16 s |
| Ruff | `ruff check .` passed |
| Formatting | Changed Python files passed `ruff format --check`; the repository-wide check still reports six untouched pre-existing files |
| Patch hygiene | `git diff --check` passed |

The prior full-suite count was 228. The exact two deleted tests are:

- `tests/core/test_pipeline.py::test_matrix_6d_pose_feature_matches_euler_entry_and_is_zero_at_rest` (F5)
- `tests/core/test_pipeline.py::test_full_bind_poses_and_unit_scales_reduce_exactly_to_legacy_skinning` (F8)

The Euler half of the former combined 6D-feature gradcheck remains under the singular name
`test_float32_gradcheck_6d_pose_feature`; the full-bind gradcheck remains on
the production inverse-bind path as
`test_float32_gradcheck_inverse_bind_transforms_and_scaled_lbs`.

## LOC accounting

| File | Before | After | Delta |
|---|---:|---:|---:|
| `src/better_human/bodies/mhr.py` | 791 | 756 | -35 |
| `src/better_human/io/load.py` | 145 | 116 | -29 |
| `src/better_human/core/pipeline.py` | 214 | 194 | -20 |
| `tests/bodies/test_mhr.py` | 473 | 517 | +44 |
| `tests/io/test_load.py` | 174 | 176 | +2 |
| `tests/core/test_pipeline.py` | 356 | 293 | -63 |
| `tests/bodies/test_smpl.py` | 170 | 174 | +4 |
| `benchmarks/benchmark_mhr.py` | 326 | 326 | 0 |
| **Total** | **2649** | **2552** | **-97** |

## CUDA performance run

The existing pinned protocol ran on an otherwise idle NVIDIA RTX 6000 Ada:
PyTorch 2.13.0+cu126, BetterRobot 0.2.0, Warp 1.15.0, five warmups, 30 timed
iterations, explicit synchronization, float32, LOD1.

| Batch | Correctives | TorchScript shape+pose ms | BetterHuman pose-only Warp ms | BetterHuman shape+pose Torch-FK ms |
|---:|:---:|---:|---:|---:|
| 1 | on | 3.271207 | 2.390250 | 285.119098 |
| 1 | off | 2.114105 | 0.776293 | 140.865380 |
| 16 | on | 4.000013 | 3.591153 | 295.218605 |
| 16 | off | 2.408837 | 1.945176 | 146.289692 |
| 256 | on | 12.979529 | 33.829921 | 330.098519 |
| 256 | off | 10.283672 | 29.437234 | 178.906521 |

At batch 256, the isolated raw medians were FK 1.546289/1.547522 ms,
corrective 1.608754/0.008462 ms, and transforms plus LBS
30.065510/27.749241 ms (correctives on/off). These are observational timings;
Order 01 defines no performance pass/fail threshold.

## Deviations from the order

- F1's premise that all eight metadata strings are converter literals is
  false for the three axes. Those load-bearing checks were retained instead
  of silently accepting coordinates the runtime does not transform.
- For F8/F9, affected tests were minimally redirected to the supported
  inverse-bind/no-bias paths instead of deleting their behavioral and gradient
  coverage. This is a justified exception to the order's literal F10-only
  test-rewrite sentence and is safer than shrinking the safety net further.

F6 and F11 were explicitly optional and are recorded as skipped, not silent
deviations. Orders 02 and 03 were not started.
