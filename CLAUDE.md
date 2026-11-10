# BetterHuman — Agent Guide

Parametric human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR, SOMA) as BetterRobot models.
Design, shapes, units: `docs/concepts/architecture.md` (site: `uv run mkdocs serve`). Rules: `docs/contracts/rules.md`,
each with its enforcer in `docs/contracts/index.md`; gates: `docs/reference/tests.md`; benchmarks:
`docs/guides/run-the-benchmarks.md` and `docs/reference/benchmarks.md`.

## Commands

```bash
export BETTERHUMAN_MODELS_DIR=$PWD/models                       # licensed weights, never committed
uv run pytest tests/contract -q -m "not slow and not licensed"  # fast tier, about 6 s
uv run pytest tests -q -m "not cuda and not licensed"           # full tier, no weights needed, about 2 min
uv run pytest tests -q -m "licensed and not cuda"               # release tier on CPU, about 3 min, needs the weights
uv run pytest tests -q -m cuda                                  # release tier on a free GPU, 10 s
uv run ruff check . && uv run ruff format --check .             # lint and format, as the pre-commit hook runs them
uv run python tools/api_snapshot.py --check                     # the public API against its snapshot
node ../BetterRobot/tools/docs_screenshots.mjs temp/shots --repo .  # every docs page as light/dark PNGs (BETTER_ROBOT_CHROME)
uv run python examples/01_smpl_shape_pose.py --check            # each example 01-04 supports --check
uv run python benchmarks/run.py --counters-only                 # the counter gate (also in the full tier), about 50 s
```

Tier times are pytest's, measured on 2026-11-10 on a shared 384-thread host (load average 20 to 80) with
`NUMPY_MADVISE_HUGEPAGE=0`; `uv run` adds about 3 s. The full tier's counter gate,
`tests/contract/test_compile_counters.py`, takes about 50 s of it. With the host's default transparent huge pages the
large archive loads stall: the full tier took 201 s before the counter gate joined it, and the CPU release tier 15
to 17 min. `docs/guides/run-the-tests.md` has the tiers in full and the huge-page story.

## Layering rule

Each package imports only the packages of its row in decision record 0011's two edge tables, as the proposed records
0019, 0021 and 0022 amend them: `ALLOWED` (among BetterHuman's packages) and `BR_ALLOWED` (into BetterRobot), which
`tests/contract/test_layer_dependencies.py` holds and checks on every import; an edge that breaks them today sits in
`tests/contract/ratchets/edges.json` until its fix lands:

```text
io, mass  -> nothing
core      -> mass
families  -> core io mass
terms     -> core
viz       -> core
studio    -> better_human core families io mass viz
the root  -> core families io mass studio terms viz
```

BetterRobot is imported through the public modules of the importer's `BR_ALLOWED` row only, a first-level name
(`Viewer`, `forward_kinematics`) from its root (`tests/contract/test_br_imports.py`). Never edit BetterRobot from
this repo.

## Where things live

| path | what |
|---|---|
| `src/better_human/__init__.py` | the subpackages; `load`, `HumanModel`, `Identity`, `HumanState`, `FamilySpec`, `from_classic`, `to_classic`; `__version__` from the installed metadata |
| `src/better_human/core/spec.py`, `src/better_human/core/rig.py` | `FamilySpec` data; the abstract `Rig` and `RigBake` |
| `src/better_human/core/human_model.py` | `HumanModel.from_spec` -> `bake` -> `Identity` -> `forward` -> `HumanState` |
| `src/better_human/core/skinning.py` | blend shapes and skinning |
| `src/better_human/mass/` | part topology, densities and inertias; its data are versioned archives read by `src/better_human/io/mass_data.py` |
| `src/better_human/families/smpl_family.py`, `src/better_human/families/_tables.py` | SMPL-family loaders; file patterns and joint names |
| `src/better_human/families/regressed_rig.py` | `RegressedRig` (SMPL family), `drop_joints` |
| `src/better_human/core/classic.py`, `src/better_human/families/classic.py` | `from_classic` / `to_classic` and the abstract `ClassicFormat`; every `*Classic` and the three formats |
| `src/better_human/families/mhr_family.py` | MHR loader |
| `src/better_human/families/mhr_rig.py` | `MHRRig` |
| `src/better_human/families/soma_family.py`, `src/better_human/families/soma_rig.py` | SOMA loader; `SOMARig` |
| `src/better_human/io/load.py`, `src/better_human/io/schema.py` | weight paths (`models_dir`, `weight_path`); internal archive reader and schemas |
| `src/better_human/io/formats.py`, `src/better_human/io/frame_sets.py` | file formats, `check_format`, the migrations (spec: `docs/reference/formats/`); frame sets |
| `src/better_human/terms/`, `src/better_human/viz/layers.py` | optimisation terms (state, fitting, physics); viewer layers and `HumanView` |
| `tests/parity/` | parity gates; the licensed goldens live in `$BETTERHUMAN_MODELS_DIR/goldens/` (release tier), the random-weight goldens in `tests/parity/smplx/random/` (full tier) |
| `tests/docs/test_snippets.py` | runs every Python fence of `docs/` and the README on the random-weight archives of `tests/support/random_weights/` and compares its output |
| `docs/contracts/decisions/` | decision records; `tests/contract/test_decision_records.py` checks their template, and that an accepted record keeps its text |
| `tools/api_snapshot.py` | `--check` the public API against `tests/contract/public_api.json`; `--write` after an intended change; `--diff tests/contract/public_api_before_0.2.0.json` lists what changed since 0.1 |
| `tools/convert/`, `tools/parity/`, `tools/render_segmentation.py` | weight converters; oracle dumps, each folder with its environment and README; the images of `docs/concepts/segmentation.md` |

## Numerical contracts

- Metres, y-up, BR pose order in `q`; native joint order in `HumanState.joint_translations`.
- `q (B..., nq)` broadcasts with identity dims `I...`; never hard-code `nq`.
- Family behaviour goes in the `Rig` or the loader, never in `HumanModel` branches.
- Native parameters enter and leave only through `bh.from_classic` / `bh.to_classic`.
- Every shape, layout and format another repository reads is on `docs/contracts/coupling-standard.md`, which adds
  BetterHuman's to BetterRobot's page and only grows.

## Gates that must not be weakened

- SMPL-X parity 1e-6 m (float32 and float64, expression streamed); current worst 5.1e-7 m.
- MHR parity on all 7 LODs and vs TorchScript (tolerances in `tests/parity/`).
- SOMA parity on 14 goldens and their identity batch, all 110 skin joints, float32 and float64 (tolerances in
  `tests/parity/`).
- SMPL, SMPL+H and SMPL-X parity on the random-weight goldens, 1e-6 m in float32 and float64, full tier
  (`tests/parity/smplx/test_random_weights.py`; worst 3.6e-7 m).
- `stream_joints=False` error stays in (1e-5, 2e-3) m.
- The counters (graph breaks, recompiles, operators, host syncs) of the benchmark runner's counter rows never exceed
  `benchmarks/thresholds/torch-2.13.0.json` (`tests/contract/test_compile_counters.py`, full tier).
Never skip, xfail or loosen a test to get green; report the number instead.

## Housekeeping

- Never `rm`. Move files with `trash <paths>` (goes to `<repo>/trash/`).
- A new word in a public name enters `docs/reference/glossary.md` in the same change, unless BetterRobot's glossary
  has it (`tests/contract/test_vocabulary.py::test_glossary_words`).
- No new dependencies without asking. Tests that read licensed weights are marked `licensed` and fail, never
  skip, when the weights are missing.
