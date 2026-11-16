# Run the benchmarks

`benchmarks/run.py` is the one place BetterHuman measures speed. It records two kinds of numbers: counters, which do
not change from run to run and gate the full tier, and wall time, which does change and is recorded with the
machine's load beside it. This guide runs it, reads its output and re-measures the thresholds.

## Run the benchmark

From the repository root:

```bash
uv run python benchmarks/run.py --counters-only                       # counter rows, checked against the thresholds
uv run python benchmarks/run.py                                       # counter rows with wall time
NUMPY_MADVISE_HUGEPAGE=0 uv run python benchmarks/run.py --licensed   # also the family sweep and the load rows
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--batches` | `1,64,1024` | sweep batch sizes `B` of `q (B, nq)` |
| `--licensed` | off | also the rows that read the licensed weights from `$BETTERHUMAN_MODELS_DIR` |
| `--repeats` | `20` | most timed calls per row; fewer once a row has taken 2 s |
| `--threads` | `1` | Torch threads for the run |
| `--rows` | all | comma-separated row ids to measure |
| `--counters-only` | off | counter rows only, no wall time; exit 1 on a counter above its threshold |
| `--timing-only` | off | wall time only, no counters |
| `--max-call-s` | `20` | skip the larger batches of a sweep row once one call takes longer |
| `--out` | `benchmarks/results/<commit>_<device>.json` | output file; `benchmarks/results/` is ignored, results are never committed |

Every number is `float32`. The runner stops when `BETTER_ROBOT_INTERNAL_CHECKS=1`, because BetterRobot's internal
checks add host syncs and operators. The counter rows read the tracked random-weight archives in
`tests/support/random_weights/`, so they need no weights; the `--licensed` rows find theirs through
`io.weight_path`, which raises with the fix when `$BETTERHUMAN_MODELS_DIR` is unset or a file is missing.

## What each row measures

Counter rows run in the neutral pose with seeded identity coefficients:

| Row | Function |
|-----|----------|
| `fk_vertices/smplx/B{1,64}` | `identity.forward(q).vertices` of one baked SMPL-X identity, without part inertias |
| `fk_streams/smplx/B{1,64}` | the same with the expression stream |
| `joint_poses/smplx/B{1,64}` | `identity.forward(q).joint_poses` |
| `contact_frame_poses/smplx/B{1,64}` | `identity.forward(q).contact_frame_poses(frames)` of two frames, the identity's placements cached |
| `to_classic/smplx/B{1,64}` | `better_human.to_classic(identity, q)` |
| `bake_forward/smpl/B{1,64}` | `human.bake(betas=betas).forward(q).vertices`, one identity per pose, `density=None` |
| `bake_forward_dempster/smpl/B{1,64}` | the same with the default `density="Dempster"`: 24 part inertias per bake |

Each counter row has four counters, the same for every run with one torch version and device: `graph_breaks` (what
`torch._dynamo.explain` finds), `recompiles` (graphs compiled when `torch.compile(fn, backend="eager")`, first called
at the row's batch, is then called at batches 1, 2, 3 and 64), `ops` (top-level `aten::` operators of one call) and
`host_syncs` (calls of the operators that make the host wait for a CUDA queue, counted on CPU too). The random
archives have each family's vertex and joint counts but a random kinematic tree, so their graph breaks, recompiles
and host syncs equal the licensed archives' and their operator counts do not.

With `--licensed`, sweep rows `<family>/<op>/B<batch>/<variant>` time SMPL, SMPL-X, MHR (level of detail 1) and
SOMA: `bake` at batch 1 with and without part inertias, then at each `--batches` size `vertices` under `no_grad`,
`vertices_backward` (the gradient of `sum(vertices)` with respect to `q`), and BetterRobot's `fk` and `rnea` on the
baked model; MHR's released TorchScript model runs beside them. Load rows `load/{smpl,smplx,mhr_lod1,soma}` give
the family loader's time beside the raw read of the same archive, and their `ratio`.

Wall time (`median_ms`) is per call, after one warm-up call, with the minor page faults per call and the load
average when the row started. The JSON's `meta` records the commit, dirty flag, command, the BetterRobot file
imported, torch and Python versions, CPU, threads, CPU affinity, the allocator and huge-page variables and the load
average at start and end. Wall time compares only on the same machine, thread count and similar load.

## The counter gate

The thresholds live in `benchmarks/thresholds/torch-<version>.json`, one file per torch version: the runner's own
`--counters-only` output from a clean commit. `tests/contract/test_compile_counters.py` runs `--counters-only` in the
full tier (it is marked `slow` and takes about a minute) and fails when a counter is above its threshold, when a
counter row has no threshold, or when the installed torch has no thresholds file. Read the file like any result:

```python
import json
from pathlib import Path

import torch

path = Path("benchmarks/thresholds") / f"torch-{torch.__version__.split('+')[0]}.json"
thresholds = json.loads(path.read_text())
counters = ("graph_breaks", "recompiles", "ops", "host_syncs")
print(thresholds["meta"]["commit"], thresholds["meta"]["command"])
print(f"{'row':31}", *(f"{name:>12}" for name in counters))
for row in thresholds["rows"]:
    print(f"{row['row']:31}", *(f"{row[name]:>12}" for name in counters))
```
```output
af974d0 benchmarks/run.py --counters-only --out benchmarks/thresholds/torch-2.13.0.json
row                             graph_breaks   recompiles          ops   host_syncs
fk_vertices/smplx/B1                       0            1          111            0
fk_vertices/smplx/B64                      0            2          111            0
fk_streams/smplx/B1                        0            1          192            0
fk_streams/smplx/B64                       0            2          192            0
joint_poses/smplx/B1                       0            1           79            0
joint_poses/smplx/B64                      0            2           79            0
contact_frame_poses/smplx/B1               0            1          124            0
contact_frame_poses/smplx/B64              0            2          124            0
to_classic/smplx/B1                        0            1           84            0
to_classic/smplx/B64                       0            2           84            0
bake_forward/smpl/B1                       0            1          186            0
bake_forward/smpl/B64                      0            2          186            0
bake_forward_dempster/smpl/B1              0            1         3430            0
bake_forward_dempster/smpl/B64             0            2         3430            0
```

When a change adds a counter, remove what added it. When the change needs it, or when a change lowers a counter,
re-measure on a clean commit and commit the thresholds file alone, with a `Why:` line:

```bash
NUMPY_MADVISE_HUGEPAGE=0 uv run python benchmarks/run.py --counters-only --out benchmarks/thresholds/torch-<version>.json
```

## Time on a host with huge pages

On a host whose transparent huge pages stall large reads (see
[Run the tests](run-the-tests.md#speed-up-loads-on-a-host-with-huge-pages)), a load can take seconds longer. Set `NUMPY_MADVISE_HUGEPAGE=0` for every run you compare;
`meta` records the setting. At batch 1024 and above the runner also re-runs each row in a child process with glibc's
allocator thresholds fixed and records that time and its page faults beside the default ones, because the default
allocator's behaviour depends on what the process freed before; a "no slower" verdict at those batches reads the
fixed-allocator number.

## Where next

- [Benchmarks](../reference/benchmarks.md): the numbers measured so far.
- [Run the tests](run-the-tests.md): the tiers that run the counter gate.
