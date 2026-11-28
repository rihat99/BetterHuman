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

## Enforcement matrix

| Rule | Enforcer |
|------|----------|
| H-LAYER-1 | `tests/contract/test_layer_dependencies.py` |
| H-LAYER-2 | `tests/contract/test_br_imports.py` |
| H-API-1 | `tests/contract/test_keyword_only.py::test_load_shows_every_option_and_dispatches_to_every_family_loader`, `tests/contract/test_abstract_or_final.py` |
| H-API-3 | `tests/contract/test_dtype_device.py` |
| H-ASSET-1 | `tests/contract/test_golden_provenance.py::test_git_holds_no_array_file_but_the_allowed_ones` |
| H-ASSET-2 | `tests/io/test_load.py`, `tests/contract/test_models_dir.py` |
| H-FORMAT-1 | `tests/contract/test_formats.py` |
| H-PARITY-1 | `tests/parity/smplx/test_smplx_parity.py`, `tests/parity/mhr/test_mhr_parity.py`, `tests/parity/mhr/test_mhr_torchscript.py`, `tests/parity/soma/test_soma_parity.py` |
| H-PARITY-2 | `tests/contract/test_golden_provenance.py` |
| H-STUDIO-1 | `tests/contract/test_studio_boundary.py` |
| H-STUDIO-2 | `tests/studio/test_persistence.py::test_studio_paths_layout`, `tests/studio/test_app.py::test_the_studio_runs_on_the_stand_in_server`, `tests/studio/test_app.py::test_a_save_name_that_is_not_one_file_name_is_refused_and_writes_nothing` |
| H-LICENSE-1 | `tests/contract/test_license.py` |

This table repeats the `Enforcement:` paragraphs of [Rules](rules.md), and `tests/docs/test_contract_matrix.py`
fails when the two differ, when a rule has no enforcer or cites one that does not exist, and when a strength word
appears outside a rule's heading on a contracts page.

## BetterRobot rules that hold here

| BetterRobot rule | BetterHuman enforcer |
|------|----------|
| C-API-2 arguments after the second are keyword-only: every function, constructor (dataclass fields included), method, class method and static method of the `__all__` tree, a `*args` counting as three; today's sites in the counted ratchet `tests/contract/ratchets/positional_parameters.json` | `tests/contract/test_keyword_only.py` |
| C-STATE-1 frozen records | `tests/contract/test_frozen.py`, `tests/contract/test_repr.py` |
| C-STATE-3 never mutate an input tensor: `from_spec`, `bake`, `forward`, every `HumanState` output, the classic conversions and every term | `tests/contract/test_no_inplace.py` |
| C-EXT-2 every class abstract or `@final` | `tests/contract/test_abstract_or_final.py` |
| C-EXT-6 pre-1.0 versioning, the public API snapshot | `tests/contract/test_api_snapshot.py` |
| C-ERR-2 errors from the `BetterRobotError` tree | `tests/contract/test_error_tree.py` |
| C-PERF-3 the counters of the hot entry points do not grow: graph breaks, recompiles, top-level operators and host-syncing operators of the counter rows of `benchmarks/run.py` stay within `benchmarks/thresholds/torch-<version>.json`; marked `slow` | `tests/contract/test_compile_counters.py` |
| C-NAME-2 descriptive names: the fields of `HumanModel`, `Identity` and `HumanState`, field by field | `tests/contract/test_api_snapshot.py` (the constructor fields, in order), `tests/contract/test_caches.py` (the late cache fields) |
| principles 3.4, 3.5, 3.8 and 3.9, one vocabulary: one name per thing, acronyms in capitals, no vague module name, every word of every public name in BetterHuman's or BetterRobot's glossary | `tests/contract/test_vocabulary.py` |
| C-API-4 joints named, not indexed; no module named like a function | `tests/contract/test_public_surface.py` |
| the coupling standard | `tests/contract/test_coupling_standard.py` |
| the test tiers: nothing skips | `tests/contract/test_no_skips.py`, `tests/contract/test_pytest_config.py` |
| Python 3.12 | `tests/contract/test_python_version.py` |
| C-DOC-2 every docs snippet runs and its output is compared | `tests/docs/test_snippets.py` |
| C-DOC-3 the enforcement matrix equals the rules | `tests/docs/test_contract_matrix.py` |
| principle 7.7, the codemap is the edge table | `tests/docs/test_codemap.py` |
| principle 7.8, citations resolve; tracked text gives reasons, not plan or review codes | `tests/contract/test_citations.py` |
| principles 9.1, 9.2, 9.3 and 9.6, one repository layout: the standard top-level entries, a `tests/` folder per package, no tracked symlink or file over 1 MiB, no ghost directory | `tests/contract/test_repository_layout.py` |
| principles 10.1, 10.2 and 10.3, the agent files: one `CLAUDE.md` of at most 200 lines, a nested one of at most 60, a one-line `AGENTS.md`, no capitals for emphasis, the shared `.claude/` files tracked | `tests/contract/test_agent_files.py` |
| principles 1.5, 5.6 and 11.6, caches: `HumanState._memo` and `Identity._frame_placements` only, each owned by one function of its class and keyed by the grad mode | `tests/contract/test_caches.py` |

Every test module under `tests/contract/` is in one of the two tables (`tests/docs/test_contract_matrix.py` checks
it).

## Shared code and its one exception

Code that both libraries need lives once, in `better_robot.testing`: the docs runner, the fake
viser server, the import walker and the citation resolver. The written exception is the two static theme files,
`docs/stylesheets/extra.css` and `docs/stylesheets/guidebook.css`: they hold no behaviour, so each repository keeps a
byte-identical copy instead of reading package data at build time.

## Changes the owner decides

Ask the owner before a change of one of the six kinds of principle 11.6:

- a change to the public surface that is not an addition: a public name removed or renamed, a signature or a public
  data layout changed (the pose `q`, a state's shapes);
- an added dependency, optional or not;
- an added file format or format version;
- a change to the coupling standard (`docs/contracts/coupling-standard.md`);
- an added edge in the edge tables: a unit added, removed or re-ordered, or an edge into BetterRobot;
- a second backend, a custom kernel, a compile wrapper or a cache.

Ask also before a new family, a rig hook or another way to extend BetterHuman, and before rejecting a contract rule
for a whole package. The commit message of such a change says what changed and why.
