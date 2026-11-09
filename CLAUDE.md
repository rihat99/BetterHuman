# BetterHuman — Agent Guide

Parametric human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR, SOMA) as BetterRobot models.
Design: `docs/concepts/architecture.md` (site: `uv run mkdocs serve`). Shapes, units, gates, baselines: `docs/contracts/bh-addendum.md`.

## Commands

```bash
export BETTERHUMAN_MODELS_DIR=$PWD/models                       # licensed weights, never committed
uv run pytest tests/contract -q -m "not slow and not licensed"  # fast tier, 1.0 s
uv run pytest tests -q -m "not cuda and not licensed"           # full tier, no weights needed, 10 to 15 s
uv run pytest tests -q -m "licensed and not cuda"               # release tier on CPU, 2.5 min; fails without weights
uv run pytest tests -q -m cuda                                  # release tier on a free GPU, 10 s
uv run ruff check src tests examples benchmarks                 # lint (line length 120)
uv run python examples/01_smpl_shape_pose.py --check            # each example 01-04 supports --check
uv run python benchmarks/run.py --device cpu --threads 8 --out ./temp/bench_cpu.json  # temp/ is ignored
```

Tier times are wall times measured on 2026-11-06 (384-core host, load average 8 to 15, default huge pages). The fast
tier took 0.97 to 1.00 s in five runs (median 0.98 s); the SOMA cases that read or write its 40 MB archives are in
the full tier, which took 10.4 and 14.8 s in two runs and 4.6 s with `NUMPY_MADVISE_HUGEPAGE=0`, because the host's
transparent huge pages stall those loads. The CPU release tier took 15 to 17 min while they stalled and 2.5 min with
`NUMPY_MADVISE_HUGEPAGE=0`. `docs/contributing/testing.md` has the tiers in full and the huge-page story.

## Layering rule

Each package imports only the packages of its row in decision record 0011's two edge tables, `ALLOWED` (among
BetterHuman's packages) and `BR_ALLOWED` (into BetterRobot), which `tests/contract/test_layer_dependencies.py` holds
and checks on every import; an edge that breaks them today sits in `tests/contract/ratchets/edges.json` until its fix
lands:

```text
io, mass  -> nothing
core      -> mass
families  -> core io mass
terms     -> core
viz       -> core io
studio    -> better_human core families io mass viz
the root  -> core families io mass terms viz
```

BetterRobot is imported through the public modules of the importer's `BR_ALLOWED` row only. Never edit BetterRobot
from this repo.

## Where things live

| path | what |
|---|---|
| `src/better_human/__init__.py` | `load`, `HumanModel`, `FamilySpec`, `Identity`, `HumanState` |
| `src/better_human/core/spec.py`, `src/better_human/core/rig.py` | `FamilySpec` data; the abstract `Rig` and `RigBake` |
| `src/better_human/core/human_model.py` | `HumanModel.from_spec` -> `bake` -> `Identity` -> `forward` -> `HumanState` |
| `src/better_human/core/skinning.py`, `src/better_human/core/mass.py` | blend shapes and skinning; segmentation and inertias |
| `src/better_human/families/smpl_family.py`, `src/better_human/families/tables.py` | SMPL-family loaders; file patterns and joint names |
| `src/better_human/families/regressed_rig.py` | `RegressedRig` (SMPL family), `drop_joints` |
| `src/better_human/families/classic.py`, `src/better_human/families/mhr.py` | classic converters; MHR loader and converters |
| `src/better_human/families/mhr_rig.py` | `MhrRig` |
| `src/better_human/families/soma.py`, `src/better_human/families/soma_rig.py` | SOMA loader and converters; `SomaRig` |
| `src/better_human/io/load.py`, `src/better_human/io/schema.py` | weight paths (`models_dir`, `weight_path`); internal archive reader and schemas |
| `src/better_human/io/formats.py`, `src/better_human/io/frame_sets.py` | file formats, `check_format`, the migrations (spec: `docs/reference/formats/`); frame sets |
| `src/better_human/terms.py`, `src/better_human/viz/human_layers.py` | optimisation terms; viewer layers and `HumanView` |
| `tests/parity/` | parity gates; their goldens live in `$BETTERHUMAN_MODELS_DIR/goldens/` (release tier) |

## Numerical contracts

- Metres, y-up, BR pose order in `q`; native joint order in `HumanState.joint_translations`.
- `q (B..., nq)` broadcasts with identity dims `I...`; never hard-code `nq`.
- Family behaviour goes in the `Rig` or the loader, never in `HumanModel` branches.
- Native parameters enter and leave only through `*_from_classic` / `*_to_classic`.

## Gates that must not be weakened

- SMPL-X parity 1e-6 m (float32 and float64, expression streamed); current worst 5.1e-7 m.
- MHR parity on all 7 LODs and vs TorchScript (tolerances in `tests/parity/`).
- SOMA parity on 14 goldens and their identity batch, all 110 skin joints, float32 and float64 (tolerances in
  `tests/parity/`).
- `stream_joints=False` error stays in (1e-5, 2e-3) m.
Never skip, xfail or loosen a test to get green; report the number instead.

## Housekeeping

- Never `rm`. Move files with `trash <paths>` (goes to `<repo>/trash/`).
- No new dependencies without asking. Tests that read licensed weights are marked `licensed` and fail, never
  skip, when the weights are missing.
