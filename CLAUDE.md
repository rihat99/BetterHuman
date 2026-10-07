# BetterHuman — Agent Guide

Parametric human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR) as BetterRobot models.
Design: `docs/concepts/architecture.md` (site: `uv run mkdocs serve`). Shapes, units, gates, baselines: `docs/contracts/bh-addendum.md`.

## Commands

```bash
export BETTERHUMAN_MODELS_DIR=$PWD/models               # licensed weights, never committed
CUDA_VISIBLE_DEVICES="" uv run pytest tests -q          # full suite
uv run ruff check src tests examples benchmarks         # lint (line length 120)
uv run python examples/01_smpl_shape_pose.py --check    # each example 01-04 supports --check
uv run python benchmarks/run.py --device cpu --threads 8
```

## Layering rule

`better_robot` public API -> `core` -> `families` -> `terms` / `viz`. Lower layers never import
higher ones. Import BetterRobot through its public surface only (`better_robot.Model`,
`ModelBuilder`, `io.build_model`, `kinematics`, `dynamics`, `lie`, `spatial`, `viewer`).
`tests/contract/test_br_imports.py` enforces this. Never edit BetterRobot from this repo.

## Where things live

| path | what |
|---|---|
| `src/better_human/__init__.py` | `load`, `Body`, `FamilySpec`, `Identity`, `HumanState` |
| `core/spec.py`, `core/rig.py` | `FamilySpec` data, `Rig` protocol, `RegressedRig`, `drop_joints` |
| `core/body.py` | `Body` -> `bake` -> `Identity` -> `forward` -> `HumanState` |
| `core/pipeline.py`, `core/mass.py` | blend shapes and skinning; segmentation and inertias |
| `families/smpl_family.py`, `families/tables.py` | SMPL-family loaders; file patterns and joint names |
| `families/classic.py`, `families/mhr.py` | classic converters; MHR loader and converters |
| `families/mhr_rig.py` | `AuthoredRig` for MHR |
| `terms.py`, `viz/human_layers.py` | optimisation terms; viewer layers and `HumanView` |
| `tests/parity/` | upstream goldens and gates |

## Numerical contracts

- Metres, y-up, BR pose order in `q`; native joint order in `HumanState.joints`.
- `q (B..., nq)` broadcasts with identity dims `I...`; never hard-code `nq`.
- Family behaviour goes in the `Rig` or the loader, never in `Body` branches.
- Native parameters enter and leave only through `*_from_classic` / `*_to_classic`.

## Gates that must not be weakened

- SMPL-X parity 1e-6 m (float32 and float64, expression streamed); current worst 5.1e-7 m.
- MHR parity on all 7 LODs and vs TorchScript (tolerances in `tests/parity/`).
- `stream_joints=False` error stays in (1e-5, 2e-3) m.
Never skip, xfail or loosen a test to get green; report the number instead.

## Housekeeping

- Never `rm`. Move files with `trash <paths>` (goes to `<repo>/trash/`).
- No new dependencies without asking. Parity tests skip only when a licensed asset is absent.
