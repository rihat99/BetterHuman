# BetterHuman

BetterHuman is a PyTorch library of parametric human body models on a
robotics core. It gives human body models one API while BetterRobot supplies
the configuration manifold, kinematics, dynamics, optimization, and viewer.
Shape is baked into an immutable body model; pose remains a batched,
differentiable robotics query.

BetterHuman is currently alpha software. APIs may change, and model weights
are not distributed with the library.

## Get started

During alpha, BetterHuman uses an editable path dependency on BetterRobot.
Clone both repositories as siblings so the layout matches `pyproject.toml`:

```text
workspace/
├── BetterRobot/
└── BetterHuman/
```

Then create the environment with [uv](https://docs.astral.sh/uv/):

```bash
cd BetterHuman
uv sync
```

Install the optional browser viewer for the demos:

```bash
uv sync --extra viewer
```

## Supply model weights

Register at the official [SMPL](https://smpl.is.tue.mpg.de/) or
[SMPL-X](https://smpl-x.is.tue.mpg.de/) site, accept the applicable license,
and download those weights there. Obtain MHR from its author release.
BetterHuman never downloads or redistributes model files.

Converted files use this layout:

```text
models/
├── smpl/
│   ├── SMPL_FEMALE.npz
│   ├── SMPL_MALE.npz
│   └── SMPL_NEUTRAL.npz
├── smplx/
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_NEUTRAL.npz
└── MHR/
    ├── assets/                     # original author release
    └── converted/
        ├── mhr_lod0.npz
        ├── mhr_lod1.npz           # default
        └── ... mhr_lod6.npz
```

Point BetterHuman at the containing directory:

```bash
export BETTERHUMAN_MODELS_DIR="$PWD/models"
```

An explicit `model_path=` always takes precedence over the environment
variable.

Legacy Chumpy-backed SMPL pickles can be converted without installing Chumpy:

```bash
mkdir -p models/smpl
uv run python -m better_human.io.convert_chumpy \
  /path/to/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \
  models/smpl/SMPL_NEUTRAL.npz \
  --model smpl
```

SMPL-X is supplied as NPZ and does not use this legacy converter.

The upstream MHR repository states Apache-2.0, but the downloaded asset
directory used here contains no separate `LICENSE` or `NOTICE`. Verify the
terms accompanying the exact release archive before redistributing its assets.
Convert the author files with `tools/convert/mhr/convert_mhr.py`; the runtime
does not import `pymomentum`.

## Create and pose a body

```python
import torch

import better_human as bh

body = bh.SMPL(gender="neutral", num_betas=10, compute_mass=False)
body = body.with_shape(betas=torch.tensor([0.5, 0.0, -0.2, 0.0, 0.0,
                                           0.0, 0.0, 0.0, 0.0, 0.0]))

q = body.robot.q_neutral
velocity = torch.zeros(body.robot.nv)
velocity[body.robot.idx_vs[body.robot.joint_id("left_elbow")]] = 0.5
q = body.robot.integrate(q, velocity)

vertices = body.vertices(q)  # (6890, 3), with autograd support
```

Set `compute_mass=True` (the default) when dynamics need shape-dependent body
mass and inertia.

## Supported models

| Model | Spawn controls | Surface |
|---|---|---:|
| SMPL | `gender`, `num_betas` | 6,890 vertices |
| SMPL-X | `gender`, `num_betas`, `num_expression`, `use_hands`, `use_face`, `flat_hand_mean` | 10,475 vertices |
| MHR | `lod=0..6` (default 1), `use_expression`, `use_correctives` | 73,639 / 18,439 / 10,661 / 4,899 / 2,461 / 971 / 595 vertices |

`use_hands=False` and `use_face=False` remove those SMPL-X articulation
subtrees while preserving its full surface topology. All models also accept
`dtype`, `device`, `density`, and `compute_mass`; MHR supports float32/float64
and scalar or 19-part density.

## Run the examples

```bash
uv run python examples/01_smpl_shape_pose.py
uv run python examples/02_smplx_partial.py
uv run python examples/03_ik_reach.py
uv run python examples/04_mhr_shape_pose.py
```

The first demo exposes shape and joint sliders, the second compares full and
body-only SMPL-X, the third drives an SMPL wrist target through BetterRobot
IK, and the fourth shows MHR identity, proportions, hand scale, and named
revolute pose controls. Every example has a deterministic no-server smoke path:

```bash
uv run python examples/01_smpl_shape_pose.py --headless --frames 1
```

Start with [the vision](docs/vision.md), then see
[architecture](docs/architecture.md), [tensor and convention contracts](docs/contracts.md),
and [design principles](docs/design_principles.md). Measured parity and
performance baselines live in the contracts and architecture docs; the
current implementation orders live in [`plan/`](plan/).
