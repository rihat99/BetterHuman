# Test suite

What each group of tests guards. How to run them: [Run the tests](../guides/run-the-tests.md).

## Markers

| Marker | Meaning | Deselected by |
|--------|---------|---------------|
| `licensed` | reads licensed weights or goldens from `$BETTERHUMAN_MODELS_DIR` | the fast and full tiers |
| `cuda` | needs a CUDA device | the fast and full tiers |
| `slow` | takes seconds | the fast tier |

The release tier, `bash tools/release_tier.sh --gpu <index>`, runs the `cuda` tests on that GPU and the `licensed`
tests last ([Run the tests](../guides/run-the-tests.md#run-a-tier)).

pytest runs strictly (`[tool.pytest.ini_options]` in `pyproject.toml`): an unknown marker or config key is an error, a
passing `xfail` fails, an empty parameter set fails at collection, and every warning is an error unless a named filter
with its reason lets it through. `tests/conftest.py` loads BetterRobot's plugin `better_robot.testing.no_skips`,
which fails every skipped or xfailed outcome, however it was produced.

## Parity

Each family's forward pass is compared with its authors' code, the oracle, on goldens the oracle wrote. The
tolerances are part of the contract: a parity failure means the library changed its answer, and the fix is in the
code, never in the tolerance. The goldens computed from licensed weights live in
`$BETTERHUMAN_MODELS_DIR/goldens/<family>/` (decision record [0015](../contracts/decisions/0015-licensed-goldens-out-of-git.md)),
so those tests are `licensed`; the random-weight goldens are tracked (record
[0016](../contracts/decisions/0016-random-weight-goldens.md)) and run in the full tier.

| Gate | Test | Tolerance |
|------|------|-----------|
| SMPL against the `smplx` package | `tests/parity/smplx/test_smpl_parity.py` | `FLOAT64_ATOL`, `FLOAT32_ATOL` there |
| SMPL-X against the `smplx` package, float32 and float64, expression streamed | `tests/parity/smplx/test_smplx_parity.py` | `FLOAT64_ATOL`, `FLOAT32_ATOL` there |
| SMPL-X with `stream_joints=False` | `tests/parity/smplx/test_smplx_parity.py` | the error stays inside a band, asserted there: expression moves joints upstream |
| SMPL, SMPL-H and SMPL-X on random weights | `tests/parity/smplx/test_random_weights.py` | `tests/parity/smplx/test_random_weights.py::ATOL`, the licensed gate |
| MHR, levels of detail 0 to 6, against `pymomentum` | `tests/parity/mhr/test_mhr_parity.py` | `tests/parity/mhr/test_mhr_parity.py::PARITY_ATOLS` (vertices, positions, quaternions, scales) |
| MHR against its TorchScript model | `tests/parity/mhr/test_mhr_torchscript.py` | `VERTEX_ATOL_METERS`, `POSITION_ATOL_METERS`, `QUATERNION_ATOL`, `SCALE_ATOL` there |
| SOMA mid against SOMA-X v0.3.4, one by one and as one identity batch | `tests/parity/soma/test_soma_parity.py` | `tests/parity/soma/test_soma_parity.py::PARITY_ATOLS`, with `PARITY_ATOLS_BY_CASE` for the 5-sigma identity |
| SOMA bake against the goldens' rest shape and binds | `tests/parity/soma/test_soma_parity.py` | `tests/parity/soma/test_soma_parity.py::BAKE_ATOLS`, with `BAKE_ATOLS_BY_CASE` |

Every golden stamps what produced it (record [0014](../contracts/decisions/0014-golden-provenance.md)); the READMEs of
`tools/parity/smplx/`, `tools/parity/mhr/` and `tools/parity/soma/` give the commands that regenerate them.

## Contract tests

`tests/contract/` guards properties that no unit test sees. The [contracts index](../contracts/index.md) names the
rule each one enforces.

| Test | Guards |
|------|--------|
| `test_layer_dependencies.py` | every import between BetterHuman's units, and into BetterRobot's, is in the edge tables |
| `test_br_imports.py` | no module imports a private BetterRobot module |
| `test_studio_boundary.py` | the studio reads only public BetterHuman and BetterRobot names and writes no environment variable |
| `test_public_surface.py` | no module shares a name with an exported function; no public function takes a joint index |
| `test_api_snapshot.py` | the public API equals its snapshot, `tests/contract/public_api.json` |
| `test_changelog.py` | the changelog groups changes by type and names every added public name, and every removed public name and every breaking signature change is on the migration page |
| `test_vocabulary.py` | one name per thing, acronyms in capitals, no vague module name, every word of every public name in a glossary |
| `test_naming.py` | the field tables of `HumanModel`, `Identity` and `HumanState`, field by field |
| `test_keyword_only.py` | public callables take at most two positional parameters |
| `test_frozen.py` | public dataclasses are frozen; every one that holds tensors compares and hashes by identity; mapping fields are read-only |
| `test_caches.py` | `HumanState._memo` and `Identity._frame_placements` are the only caches (record [0028](../contracts/decisions/0028-identity-frame-placements-cache.md)): each reached by one function, empty after `dataclasses.replace`, keyed by the grad mode; no module-level store and no unreasoned `functools` cache |
| `test_no_inplace.py` | no public call (from `from_spec` and `bake` through every `HumanState` output, the classic conversions and every term) changes a tensor its caller holds: version counters and values |
| `test_memo_writes.py` | `HumanState._memo` and `Identity._frame_placements` appear in tracked Python only as `self.<memo>` inside their own class |
| `test_repr.py` | every public class, built from the random-weight archive or large tensors, prints one short line that names it and no tensor; `HumanModel`, `Identity` and `HumanState` name the family |
| `test_abstract_or_final.py` | every class in `src` is abstract or `@final` |
| `test_dtype_device.py` | dtype and device are chosen at load; no `to`; a tensor in another dtype raises |
| `test_error_tree.py` | every BetterHuman error derives from BetterRobot's; `src` raises builtin errors only where listed |
| `test_coupling_standard.py` | every line of the [coupling standard](../contracts/coupling-standard.md) against the library, and BetterRobot's lines BetterHuman relies on |
| `test_compile_counters.py` | the CPU counters (graph breaks, recompiles, operators, host syncs) of the benchmark runner's counter rows stay within `benchmarks/thresholds/torch-<version>.json`; `slow` |
| `test_formats.py` | every file format names itself and its version, reads its past versions and has a spec page whose `json` examples its reader reads |
| `test_models_dir.py` | tests, examples and tools find weights only through `better_human.io` |
| `test_golden_provenance.py` | every golden stamps its provenance; git holds no `.npz` but the random-weight archives and goldens |
| `test_license.py` | the licence file, `pyproject.toml` and the README agree on Apache-2.0 |
| `test_citations.py` | every path and `path::symbol` cited in tracked text resolves; no line number is cited |
| `test_repository_layout.py` | the tracked top-level entries are the standard set; each `src` package has its `tests/` folder and every other `tests/` folder is a named suite; no tracked symlink and no tracked file over 1 MiB (record [0015](../contracts/decisions/0015-licensed-goldens-out-of-git.md)); no directory of the working tree without a tracked file, ignored ones aside |
| `test_agent_files.py` | the root `CLAUDE.md` is at most 200 lines and a nested one 60; no capitals for emphasis in an agent file or skill; `AGENTS.md` is one line naming `CLAUDE.md`, and no other tool's instruction file exists at any depth; the shared `.claude/` files are tracked and the rest ignored |
| `test_decision_records.py` | every decision record follows the template, has an allowed status and an index row, and an accepted record keeps its accepted text |
| `test_no_skips.py` | no test module spells a skip or an xfail, the deselected ones included |
| `test_pytest_config.py` | pytest runs strictly, with exactly the tier markers; each warning filter after `error` names its warning under a comment with the reason |
| `test_python_version.py` | the Python floor, the lint target and the pinned version agree |

A ratchet is a JSON file under `tests/contract/ratchets/` that lists today's violations of a rule, each with the fix
that removes it; it only shrinks, and the fix removes its entry.

## Docs tests

`tests/docs/` keeps the pages and the README true:

| Test | Guards |
|------|--------|
| `test_snippets.py` | every `python` fence runs and every `output` fence equals what it prints (`better_robot.testing.snippets`); a page outside `LICENSED` opens no weights file outside `tests/support/` and the package data; every referenced image exists |
| `test_readme_first_screen.py` | the README's first `python` fence has its `output` fence docked under it, closing by line 40: the first screen runs an example and shows its output (principle 7.2) |
| `test_api_reference.py` | every path of `tests/contract/public_api.json` has an entry on a page of the API reference |
| `test_codemap.py` | the package map of the [architecture page](../concepts/architecture.md#the-package-map) equals the edge tables `ALLOWED` and `BR_ALLOWED` |
| `test_contract_matrix.py` | the [enforcement matrix](../contracts/index.md#enforcement-matrix) equals the rules' `Enforcement:` paragraphs; every rule has an enforcer that exists; every contract test is in the index; a strength word stands only in a rule heading |
| `test_docs_text.py` | a page outside the changelog, the migration pages and the decision records names no plan code and tells no history: it says what holds now (principle 7.9); the README's family table, the weights layout of [Load a family](../guides/load-a-family.md) and this page's test tables equal what they copy |
