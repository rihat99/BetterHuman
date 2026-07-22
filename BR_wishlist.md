# BetterRobot wishlist (from BetterHuman)

Measured BetterRobot bottlenecks that cap BetterHuman performance. Per the
standing rule, BetterHuman never modifies or works around BetterRobot; this
file is the hand-off. Every number below was measured on an idle RTX 6000
Ada (GPU 7), torch 2.13.0+cu126, better_robot 0.2.0, float32, synchronized
medians of 30 after 5 warmups. FK wall-clock is CPU-launch-bound and so
carries run-to-run noise; the launch counts and ratios are exact and stable.

## 1. Torch FK issues ~78 kernel launches per joint (the one big item)

`br.forward_kinematics` is ~20× slower than the upstream `smplx` package's
equivalent and the gap is pure kernel-launch volume — the profile shows no
CPU-GPU syncs, no memcpys, no quadratic term, and near-zero GPU-side time:

| Model | joints | FK b=1 | FK b=256 | launches/call (b=256) | launches/joint |
|---|--:|--:|--:|--:|--:|
| SMPL | 25 | 17.7 ms | 18.8 ms | 1 937 | 77.5 |
| SMPL-X | 56 | 50.0 ms | 52.3 ms | 4 386 | 78.3 |
| MHR LOD1 | 203 | 92.7 ms | 96.3 ms | 16 432 | 80.9 |

Upstream `smplx` `batch_rigid_transform` (one batched 4×4 matmul per joint):
24 joints → 37 launches (1.5/joint), 1.13 ms at b=256; 55 joints → 68
launches (1.2/joint), 2.64 ms. **~63× fewer launches per joint.** FK time is
flat in batch size because the CPU cannot queue tiny kernels faster than
the GPU drains them (~5.6 µs per `cudaLaunchKernel` + dispatch).

Where the launches come from (share of FK launches ≈ share of wall time):

| Share | Source | Location |
|--:|---|---|
| ~73% | `so3._quat_mul` — Hamilton product hand-expanded into ~28 scalar elementwise kernels (16 `mul` + 6 `add` + 6 `sub` + `stack`) | `better_robot/lie/so3.py:42-50` |
| ~15% | `so3.act` — 2 `linalg_cross` + 2 `mul` + 2 `add` per call | `better_robot/lie/so3.py:169-176` |
| ~5% | `se3.compose` glue (translation add + `cat`) | `better_robot/lie/se3.py:30-38` |
| ~5-7% | per-joint `joint_transform` (spherical normalize/cat, revolute axis-angle) | `joint_models/spherical.py:25-34`, `revolute.py:18-21`, `so3.py:253-262` |

The FK loop (`kinematics/forward.py:176`, `for j in structure.topo_order:`)
calls `se3.compose` twice per joint (`forward.py:191` parent-frame,
`forward.py:198` world), so items 1–3 — everything inside `se3.compose` —
account for ~93% of all launches.

**Ask:** make the quaternion algebra issue O(1) kernels per call instead of
O(30) — e.g. express `_quat_mul`/`act` as batched tensor ops (einsum/gather
formulations), fuse them (custom op / `torch.compile` on the lie layer), or
add an upstream-style lane that materializes per-joint rotation matrices in
one batched call and does one matmul per joint. Any of these cuts
~78 launches/joint toward ~1–2 and closes most of the 20× forward gap.
Cost is strictly linear in joints, so MHR (203 joints) benefits most.

## 2. Backward inherits the same tax — no separate fix needed

Backward is a 1:1 mirror of the forward tiny-op graph (SMPL-X b=256:
13 286 launches fwd+bwd vs 4 386 fwd-only; top autograd nodes are
`MulBackward0`/`AddBackward0`/`SubBackward0` from the same `so3.py` lines).
Measured forward+backward vs upstream `smplx` (loss = `vertices.sum()`,
grads wrt pose[/shape]):

| Model | b | lane | ours f+b | upstream f+b | ours bwd/fwd | upstream bwd/fwd |
|---|--:|---|--:|--:|--:|--:|
| SMPL | 256 | pose-only | 83.0 ms | 7.4 ms | 5.6 | 2.9 |
| SMPL | 256 | shape+pose | 98.3 ms | 8.2 ms | 6.3 | 3.2 |
| SMPL-X | 256 | pose-only | 206.1 ms | 22.0 ms | 3.6 | 3.3 |
| SMPL-X | 256 | shape+pose | 245.3 ms | 21.8 ms | 4.2 | 3.3 |
| MHR | 256 | pose-only | 582.9 ms | — | 5.9 | — |

There is no separate backward pathology: fixing item 1 fixes both
directions at once.

## 3. Warp FK lane rejects shape-batched models

The Warp lane solves item 1 for the pose-only path (SMPL-X pose-only drops
to ~0.9× upstream), but it rejects batched `joint_placements` (with
unbatched frame placements), so any body whose shape was baked with a
batch — the fair shape+pose path, i.e. training-style usage — falls back to
Torch FK and pays the full launch tax. Supporting batched joint placements
in the Warp lane (or fixing item 1, which makes the Torch lane fast enough)
unlocks the shape+pose lane.

## Non-items (measured clean — no action)

- No `.item()`/sync/memcpy on the FK hot path; no quadratic-in-joints term.
- `Model.with_values` on the shape-bake path rebuilds `ModelValues` only
  (metadata-only validate, no `validate_consistency` syncs): 0.7 ms at
  b=256 for SMPL/SMPL-X. Model rebuild cost is a non-issue.
- BetterHuman's `body.fk` wrapper adds ≤0.14 ms — a pass-through.

## Known BetterHuman-side consequence (not a BR ask)

MHR's shape bake runs one internal neutral FK (`bodies/mhr/deformation.py`,
`br.forward_kinematics(robot, q_neutral)`) with shape-batched placements to
obtain shaped rest joints — ~102 ms of the 108 ms `with_shape` cost at
b=256, i.e. MHR shape+pose currently pays the FK launch tax twice. This is
inherent to needing shaped rest-joint positions and is not worked around;
item 1 shrinks it ~20× automatically.
