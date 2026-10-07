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

An explicit `path` argument (`bh.load("smpl", path)`) always takes precedence over the environment
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

## Quick start

```python
import torch
import better_human as bh
from better_human.viz import HumanView

body = bh.load("smpl", gender="neutral")         # one BetterRobot model per family
identity = body.bake(betas=torch.zeros(1, 10))   # bake shape once (mass, joints, contact frames)
q = identity.q_neutral.expand(4, -1).clone()     # (B, nq): free-flyer root + spherical joints
q[:, 1] = 1.0                                    # root 1 m up (y-up, metres)
state = identity.forward(q)                      # HumanState for 4 poses
print(state.vertices().shape, state.joints.shape)  # (4, 6890, 3) (4, 24, 3)

view = HumanView(identity.forward(identity.q_neutral))   # viewer layers, updated in place
# from better_robot.viewer import Viewer
# viewer = Viewer(); [viewer.add(layer, group="human") for layer in view]
# view.update(state_for_next_frame); viewer.show()
```

Per-frame groups such as SMPL-X expression go to `forward`:
`identity.forward(q, stream={"expression": torch.zeros(B, 10)})`.
See `docs/concepts/architecture.md` for the design and `docs/contracts/bh-addendum.md` for shapes, units and gates.

## Supported families

| family | loader | identity groups | stream groups |
|---|---|---|---|
| SMPL | `bh.load("smpl")` | `betas` | |
| SMPL-H | `bh.load("smplh")` | `betas` | |
| SMPL-X | `bh.load("smplx", hands=, face=)` | `betas` | `expression` |
| MANO | `bh.load("mano")` | `betas` | |
| FLAME | `bh.load("flame")` (no asset tested here) | `betas` | `expression` |
| MHR | `bh.load("mhr", lod=1)` | `identity`, `proportions` | `expression` |

Options `mass`, `density`, `contact_frames` and `stream_joints` go to `Body`; the rest go to the loader.

## Run the examples

```bash
uv run python examples/01_smpl_shape_pose.py    # SMPL shape sliders and pose
uv run python examples/02_smplx_partial.py      # full SMPL-X next to body-only SMPL-X
uv run python examples/03_ik_reach.py           # drag a target, the left wrist follows (LM solve)
uv run python examples/04_mhr_shape_pose.py     # MHR LOD1 identity sliders and arm swing
```

Add `--check` to build the scene and exit (used in CI).

## Benchmarks

```bash
uv run python benchmarks/run.py --device cpu --threads 8 --batches 1 64 1024 8192 \
  --out benchmarks/results/p4_bodies_cpu.json
```

Measured numbers are in `docs/contracts/bh-addendum.md`.
