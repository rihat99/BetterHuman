# Contracts

A *contract* is a rule the code follows so the library grows without refactoring. BetterHuman follows
BetterRobot's design contracts, which live in the BetterRobot repository (start at
`BetterRobot/docs/contracts/00-index.md`); their rule IDs (`C-*`) are cited here as written there, since a
cross-repository link is not possible. BetterHuman adds its own rules (`H-*`), each with an ID, a strength, a
rationale, an example and a test that enforces it. When a BetterHuman rule and a BetterRobot rule disagree,
BetterRobot's numerical contracts win.

| Page | Covers |
|------|--------|
| [Scope](scope.md) | what BetterHuman does, its non-goals, the bar for a new family or feature, commonly rejected ideas |
| [Public surface](public-surface.md) | what is public, what users may do, horizons, compatible changes |
| [Rules](rules.md) | BetterHuman's own rules, `H-*` |
| [Coupling standard](coupling-standard.md) | what BetterHuman fixes on top of BetterRobot's coupling standard: the shapes, layouts and file formats other repositories read |
| [Decisions](decisions/index.md) | the decision records; the ones that bind both libraries live in BetterRobot and are cited as `BetterRobot NNNN` |

## Enforcement matrix

| Rule | Enforcer |
|------|----------|
| H-LAYER-1 | `tests/contract/test_layer_dependencies.py` |
| H-LAYER-2 | `tests/contract/test_br_imports.py` |
| H-API-1 | `tests/contract/test_keyword_only.py::test_load_shows_every_option_and_dispatches_to_every_family_loader`, `tests/contract/test_abstract_or_final.py` |
| H-API-3 | `tests/contract/test_dtype_device.py` |
| H-ASSET-1 | `tests/contract/test_golden_provenance.py::test_git_holds_no_npz_but_the_allowed_ones` |
| H-ASSET-2 | `tests/io/test_load.py`, `tests/contract/test_models_dir.py` |
| H-FORMAT-1 | `tests/contract/test_formats.py` |
| H-PARITY-1 | `tests/parity/smplx/test_smpl_parity.py`, `tests/parity/smplx/test_smplx_parity.py`, `tests/parity/smplx/test_random_weights.py`, `tests/parity/mhr/test_mhr_parity.py`, `tests/parity/mhr/test_mhr_torchscript.py`, `tests/parity/soma/test_soma_parity.py` |
| H-PARITY-2 | `tests/contract/test_golden_provenance.py` |
| H-STUDIO-1 | `tests/contract/test_studio_boundary.py` |
| H-STUDIO-2 | `tests/studio/test_persistence.py::test_studio_paths_layout` |
| H-LICENSE-1 | `tests/contract/test_license.py` |

This table repeats the `Enforcement:` paragraphs of [Rules](rules.md), and `tests/docs/test_contract_matrix.py`
fails when the two differ, when a rule has no enforcer or cites one that does not exist, and when a strength word
appears outside a rule's heading on a contracts page.

## BetterRobot rules and records that hold here

| BetterRobot rule or record | BetterHuman enforcer |
|------|----------|
| C-API-2 arguments after the second are keyword-only | `tests/contract/test_keyword_only.py` |
| C-STATE-1 frozen records (record [0001](decisions/0001-human-model.md)) | `tests/contract/test_frozen.py`, `tests/contract/test_repr.py` |
| C-EXT-2 every class abstract or `@final` (records [0002](decisions/0002-rig-abstract-base.md), [0008](decisions/0008-typed-family-spec.md)) | `tests/contract/test_abstract_or_final.py` |
| C-EXT-6 pre-1.0 versioning, the public API snapshot (BetterRobot 0024) | `tests/contract/test_api_snapshot.py`, `tests/contract/test_changelog.py` |
| C-ERR-2 errors from the `BetterRobotError` tree (BetterRobot 0031) | `tests/contract/test_error_tree.py` |
| C-PERF-3 the counters of the hot entry points do not grow (BetterRobot 0011): graph breaks, recompiles, top-level operators and host-syncing operators of the counter rows of `benchmarks/run.py` stay within `benchmarks/thresholds/torch-<version>.json`; marked `slow` | `tests/contract/test_compile_counters.py` |
| C-NAME-2 descriptive names; one vocabulary (BetterRobot 0023, 0027, 0030) | `tests/contract/test_naming.py`, `tests/contract/test_vocabulary.py` |
| C-API-4 joints named, not indexed (BetterRobot 0029); no module named like a function (records [0010](decisions/0010-human-viz-surface.md), [0011](decisions/0011-human-package-layout.md)) | `tests/contract/test_public_surface.py` |
| BetterRobot 0026, the coupling standard | `tests/contract/test_coupling_standard.py` |
| BetterRobot 0033, the test tiers: nothing skips | `tests/contract/test_no_skips.py`, `tests/contract/test_pytest_config.py` |
| BetterRobot 0034, Python 3.12 | `tests/contract/test_python_version.py` |
| C-DOC-2 every docs snippet runs and its output is compared | `tests/docs/test_snippets.py` |
| C-DOC-3 the enforcement matrix equals the rules | `tests/docs/test_contract_matrix.py` |
| principle 7.7, the codemap is the edge table | `tests/docs/test_codemap.py` |
| principle 7.8, citations resolve | `tests/contract/test_citations.py` |

Every test module under `tests/contract/` is in one of the two tables (`tests/docs/test_contract_matrix.py` checks
it).

## Shared code and its one exception

Code that both libraries need lives once, in `better_robot.testing` (BetterRobot 0035): the docs runner, the fake
viser server, the import walker and the citation resolver. The written exception is the three static theme files,
`docs/stylesheets/extra.css`, `docs/stylesheets/guidebook.css` and `docs/javascripts/mathjax.js`: they hold no
behaviour, so each repository keeps a byte-identical copy instead of reading package data at build time.

## When to write a decision record

Write a record under `decisions/` when a change does any of the following:

- adds, removes or re-orders a unit in the edge tables, or an edge into BetterRobot;
- changes a public data layout (the pose `q`, a state's shapes, a file format's fields);
- adds a family, a rig hook, or a way to extend BetterHuman;
- adds an optional dependency, or replaces one the public API exposes;
- rejects a contract rule for a whole package.

Only the owner accepts a record: one an agent or the integrator writes is `Proposed (integrator, awaiting owner)`
until then. An accepted record changes only in its Status line; a new record supersedes it or amends part of it.
