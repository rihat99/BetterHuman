# BetterHuman

Parametric human bodies (SMPL, SMPL-H, SMPL-X, MANO, FLAME, MHR, SOMA) as batched, differentiable BetterRobot
models. This file holds what the code cannot tell. `AGENTS.md` sends every other agent here.

## Commands

From the repository root. Wall times measured on a shared 384-thread AMD EPYC host at load average 16 to 113, with
`NUMPY_MADVISE_HUGEPAGE=0` (below).

```bash
uv sync --extra studio                                           # the environment and the dev group, under a second
uv run pytest tests/contract -q -m "not slow and not cuda and not licensed"   # fast tier, about 13 s
uv run pytest tests -q -m "not cuda and not licensed"            # full tier, about 2 min
bash tools/release_tier.sh --gpu <index>                         # release tier, weights and a free GPU, about 7.5 min
uv run pytest tests/docs -q -m "not licensed"                    # every docs snippet against its output, about 25 s
uv run mkdocs build --strict                                     # the docs site into site/, about 3 s
uv run ruff check . && uv run ruff format --check .              # lint and format, under a second
uv run python tools/api_snapshot.py --check                      # the public API against its snapshot, about 2.5 s
uv run python benchmarks/run.py --counters-only                  # the counter gate, about 50 s
node ../BetterRobot/tools/docs_screenshots.mjs temp/shots --repo .   # light and dark PNG of every page, about 5 min
```

While editing a package, run its folder beside the fast tier (`uv run pytest tests/core -q -m "not cuda and not
licensed"`, about 6 s). `.pre-commit-config.yaml` runs ruff and the fast tier before a commit and the full tier
before a push, once installed (`uv run pre-commit install`). `docs/guides/run-the-tests.md` explains the tiers;
`docs/reference/tests.md` says what each group of tests guards.

## What the tiers need

- Nothing skips: `tests/conftest.py` loads BetterRobot's plugin `better_robot.testing.no_skips`, which turns every
  skip and xfail into a failure (`tests/contract/test_no_skips.py`). A test that reads licensed weights carries the
  `licensed` marker and one that needs a GPU `cuda`; the fast and full tiers deselect both.
- The fast and full tiers need no weights: their docs snippets, examples and parity read the random-weight
  archives of `tests/support/random_weights/`.
- The weights live in `$BETTERHUMAN_MODELS_DIR`, one folder per family (`docs/guides/load-a-family.md` names each
  folder and file) and the licensed goldens in `goldens/<family>/`. Nothing guesses a default: a run that selects a
  `licensed` test without the variable stops at collection with the fix. A `models` folder at the root is ignored.
- On a host whose transparent huge pages are in `madvise` mode, NumPy's huge-page advice stalls each read of a large
  archive for seconds; `NUMPY_MADVISE_HUGEPAGE=0` turns it off for a run (`docs/guides/run-the-tests.md`). Neither
  the library nor the tests set it.
- `benchmarks/run.py` refuses `BETTER_ROBOT_INTERNAL_CHECKS=1`: BetterRobot's checks add host syncs and operators.
- The screenshot command runs BetterRobot's script from a BetterRobot clone beside this one, needs
  `BETTER_ROBOT_CHROME` (a headless Chromium) and builds the site with a plain `uv run`.

## Gates that must not be weakened

Never skip, xfail, loosen a tolerance or delete an assertion to get green; report the number instead. Gate data (a
tolerance, a counter threshold, a golden) changes only in a commit of its own that says why.

- SMPL and SMPL-X against the `smplx` package, 1e-6 m in float32 and float64, expression streamed:
  `tests/parity/smplx/test_smpl_parity.py` and `tests/parity/smplx/test_smplx_parity.py`, release tier.
- SMPL, SMPL-H and SMPL-X on the random-weight archives, the same 1e-6 m: `tests/parity/smplx/test_random_weights.py`,
  full tier.
- SMPL-X with `stream_joints=False`: the joint error stays in (1e-5, 2e-3) m.
- MHR at levels of detail 0 to 6 against `pymomentum` and against its TorchScript model; SOMA mid on 14 goldens and
  their identity batch, all 110 skin joints, float32 and float64. Their tolerances are `PARITY_ATOLS` and the
  constants beside it in each test (`docs/reference/tests.md` lists them).
- Counters: graph breaks, recompiles, operators and host syncs of the counter rows never exceed
  `benchmarks/thresholds/torch-2.13.0.json` (rule C-PERF-3, `tests/contract/test_compile_counters.py`, full tier).
- Ratchets in `tests/contract/ratchets/` only shrink; the test that reads each one fails on a stale entry, so an
  entry leaves in the commit that fixes it. Open today: dead_citations 5, positional_parameters 12 (the public
  records' positional fields and `terms.baked_identity`'s `*groups`, both parked with the owner); the other four are
  empty.
- The public API equals `tests/contract/public_api.json` (`tests/contract/test_api_snapshot.py`); an intended change
  is recorded with `uv run python tools/api_snapshot.py --write` and an entry in `docs/reference/changelog.md`
  (`tests/contract/test_changelog.py`).

## Absences, each with its enforcer

- No package imports outside its row of `ALLOWED` (among BetterHuman's packages) and `BR_ALLOWED` (into
  BetterRobot) in `tests/contract/test_layer_dependencies.py`, nor by a computed name; `docs/concepts/architecture.md`
  prints both tables, and `tests/docs/test_codemap.py` checks that copy.
- No BetterRobot import outside its `__all__` tree but the one exception rules H-LAYER-2 and H-STUDIO-1 of
  `docs/contracts/rules.md` name (`tests/contract/test_br_imports.py`, `tests/contract/test_studio_boundary.py`).
- No class in `src` that is neither abstract nor `@final`, and none that derives from a `@final` class: a partial
  model is a loader option (`tests/contract/test_abstract_or_final.py`); `load` dispatches to every family loader
  (`tests/contract/test_keyword_only.py::test_load_shows_every_option_and_dispatches_to_every_family_loader`).
- No `to` method: dtype and device are chosen at load, and a tensor in another dtype raises
  (`tests/contract/test_dtype_device.py`); no mutable public dataclass, and none that holds tensors compares by
  value (`tests/contract/test_frozen.py`).
- No cache beyond `HumanState._memo` and `Identity._frame_placements` (record 0028), each reached by one function of
  its class, and no `functools` cache outside the reasoned `FUNCTOOLS_CACHES` (`tests/contract/test_caches.py`).
- No public function, constructor (dataclass fields included), method, class method or static method with more
  than two positional parameters, a `*args` counting as three, beyond the counted sites of
  `tests/contract/ratchets/positional_parameters.json` (`tests/contract/test_keyword_only.py`); no public function
  takes a joint index, and no module shares a name with an exported function (`tests/contract/test_public_surface.py`).
- No public call changes a tensor its caller holds (`tests/contract/test_no_inplace.py`).
- No builtin `ValueError`, `TypeError`, `RuntimeError`, `KeyError`, `IndexError` or `AssertionError` raised in `src`
  outside `PLAIN_RAISES`: every error derives from BetterRobot's `BetterRobotError`
  (`tests/contract/test_error_tree.py`).
- No weight lookup but `better_human.io.models_dir` and `better_human.io.weight_path`, in the package, tests,
  examples, tools, benchmarks or docs scripts (`tests/contract/test_models_dir.py`).
- No licensed weights or goldens computed from them in git: git holds no array file, `.npz`, `.npy`, `.pt` or any
  other format, beyond the random-weight archives and goldens, the format fixtures and the shipped data
  (`tests/contract/test_golden_provenance.py`).
- No file format without its name and version, no pickle load, and no module but `src/better_human/io/formats.py`
  and `src/better_human/io/schema.py` naming `format_version` (`tests/contract/test_formats.py`).
- No word of a public name outside BetterHuman's or BetterRobot's glossary (`docs/reference/glossary.md`,
  `tests/contract/test_vocabulary.py`).
- No skip or xfail spelled in a test module, deselected ones included (`tests/contract/test_no_skips.py`).
- No dead path citation, no line-number citation and no plan or review code in tracked text
  (`tests/contract/test_citations.py`).
- No plan code and no history on a docs page outside the changelog, the migration pages and the records
  (`tests/docs/test_docs_text.py`).
- No tracked symlink, file over 1 MiB, or directory without a tracked file (`tests/contract/test_repository_layout.py`).
- No agent file over its length and no capitals for emphasis (`tests/contract/test_agent_files.py`).

## Conventions the code does not announce

- Units are metres and radians, the world is y-up, and `q` has BetterRobot's layout (poses
  `[tx, ty, tz, qx, qy, qz, qw]`, quaternion scalar last).
- `HumanState.joint_translations` and `joint_poses` are in the family's native joint order; `HumanState.kinematics`
  is in the model's joint order. Every layout another repository reads is in `docs/contracts/coupling-standard.md`,
  which only grows.
- `q (B..., nq)` broadcasts with the identity's batch axes `I...`; read `nq` from the model, never hard-code it.
- A family's own behaviour goes in its `Rig` or its loader, never in a `HumanModel` branch (rule H-API-1).
- Native parameters enter and leave only through `bh.from_classic` and `bh.to_classic` (record 0004).
- BetterRobot changes in its own repository, never from this one.

## Working rules

These come from the principles repository (its README's ten laws and chapters 01-12; chapter 10 governs this
file); no test checks them, a reviewer does.

- Write the least code that solves the problem; a change's net line count per package is a design signal
  (principle 4.1).
- Add an abstraction only when two in-tree callers need it.
- No forwarding layer: a module whose public functions each forward to one other module merges into it (1.2).
- One obvious place per concept (Law 2); prefer a few deep modules to many tiny ones, and a family of modules
  becomes a folder at three (1.2, 3.8).
- Delete only after a search shows no caller; back every claim in a report with its command and output (4.1, 10.5).
- Update the docs and the nearest agent file in the change that changes a contract. A change of a contract or a
  layout gets a decision record (`docs/contracts/index.md` says when); an accepted record keeps its text, Status line
  aside (`tests/contract/test_decision_records.py`).
- A new family meets the inclusion bar of `docs/contracts/scope.md`; before reopening a settled idea, read its
  "Commonly rejected".

## Where things live

- Rules and their enforcers: `docs/contracts/index.md`; scope, non-goals and rejected ideas:
  `docs/contracts/scope.md`; the public plan: `docs/reference/roadmap.md`.
- Procedures: `.claude/skills/add-body-family/SKILL.md`, `.claude/skills/regenerate-goldens/SKILL.md` and
  `.claude/skills/release/SKILL.md`.
- `tools/convert/` turns upstream releases into the archives the loaders read; `tools/parity/` holds one folder per
  oracle, each with its pinned environment, dump scripts and a README; `tools/render_segmentation.py` draws the
  images of `docs/concepts/segmentation.md`.
- `agent_notes/` (one folder per task: plan, state, notes) and `trash/` are ignored whole; nothing tracked cites
  them.
