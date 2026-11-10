# Benchmarks

Measured wall times of the body operations, against each family's upstream code where it runs. They are
measurements, not gates; the counters that gate are in `benchmarks/thresholds/` ([Run the
benchmarks](../guides/run-the-benchmarks.md)).

## The runner at commit 50ea3dc

`benchmarks/run.py` at the clean commit `50ea3dc`, float32, median ms: one Torch thread on an idle core of an AMD
EPYC 9654, torch 2.13, `NUMPY_MADVISE_HUGEPAGE=0`, a shared host (load average 15 to 26).

| counter row | B=1 | B=64 |
|---|---:|---:|
| `fk_vertices/smplx` | 3.10 | 127.6 |
| `bake_forward/smpl` | 1.71 | 45.0 |
| `bake_forward_dempster/smpl` | 12.2 | 254.9 |

Load rows, the family loader against the raw read of the same archive, ms (ratio), in two runs:

| row | run 1 | run 2 |
|---|---:|---:|
| `load/smpl` | 97.5 / 85.1 (1.15) | 118.2 / 127.9 (0.92) |
| `load/smplx` | 795.1 / 798.3 (1.00) | 790.1 / 753.9 (1.05) |
| `load/mhr_lod1` | 704.5 / 1018.2 (0.69) | 685.1 / 969.4 (0.71) |
| `load/soma` | 376.5 / 115.5 (3.26) | 333.6 / 77.7 (4.29) |

SOMA's loader takes three to four times its raw read, above the 1.5 times decision record
[0013](../contracts/decisions/0013-bh-package-data-and-load-speed.md) names (open question 1 of record
[0024](../contracts/decisions/0024-archive-load-speed-measured-again.md)).

## The earlier runner at commit 9ad4093

The earlier `benchmarks/run.py` at commit `9ad4093`, float32, median ms. Ratio = ours / upstream (below 1 = ours
faster).
Upstream = vendored `smplx` forward for SMPL and SMPL-X, TorchScript `mhr_model.pt` for MHR.
CPU: AMD EPYC 9654, 8 threads. CUDA: NVIDIA RTX PRO 6000 Blackwell.

| family | op | CPU B=1 | CPU B=1024 | CUDA B=1 | CUDA B=1024 |
|---|---|---:|---:|---:|---:|
| SMPL | vertices (ratio) | 1.21 (0.69) | 296 (0.95) | 1.19 (0.80) | 8.05 (0.99) |
| SMPL-X | vertices (ratio) | 3.02 (0.42) | 464 (0.74) | 1.25 (0.47) | 12.3 (0.82) |
| MHR | vertices (ratio) | 22.4 (0.59) | 1310 (0.25) | 2.50 (1.15) | 26.8 (1.02) |
| SMPL | bake with mass (B=1) | 11.2 | | 20.8 | |
| SMPL-X | bake with mass (B=1) | 23.5 | | 45.4 | |
| MHR | bake with mass (B=1) | 20.3 | | 21.8 | |
| SMPL | fk / rnea | 0.26 / 0.31 | 6.87 / 15.8 | 0.60 / 0.73 | 0.74 / 0.81 |
| SMPL-X | fk / rnea | 0.35 / 0.37 | 12.0 / 25.6 | 0.79 / 0.72 | 0.83 / 0.82 |
| MHR | fk / rnea | 0.94 / 0.55 | 41.2 / 45.2 | 1.61 / 0.77 | 2.08 / 1.55 |

Caveats:

- Inputs differ slightly. MHR "default" vertices include the expression stream; SMPL-X "default"
  passes no stream.
- SMPL bake without mass is 0.41 ms on CPU: mass integration dominates `bake`.
- Batch 8192 vertex cells (ours and upstream) were skipped where the estimate exceeded 8 GB
  (for example SMPL 10.8 GB, SMPL-X 16.5 GB, MHR TorchScript 29.0 GB). fk and rnea ran at 8192.
- Full tables: `git show 9ad4093:benchmarks/results/p4_bodies_cpu.json`, and the same for `p4_bodies_cuda.json`.
