# Contracts

BetterHuman follows BetterRobot's design contracts. They live in the BetterRobot repository under
`BetterRobot/docs/contracts/` (start at `BetterRobot/docs/contracts/00-index.md`); rule IDs (`C-*`) are cited
here as written there. A cross-repository link is not possible, so cite the page path and the rule ID.

BetterHuman adds its own rules (`H-*`), its numeric conventions, weight-file layout, parity gates, baselines
and design principles in [`bh-addendum.md`](bh-addendum.md).

Decisions that shaped these rules are recorded in [`decisions/`](decisions/index.md); records that bind both
libraries live in BetterRobot and are cited as `BetterRobot NNNN`.

## BetterRobot contract families that apply unchanged

| BR page | rule IDs | how it applies in BetterHuman |
|---|---|---|
| `BetterRobot/docs/contracts/02-tensors.md` | `C-TENS-*` | trailing event axes, arbitrary leading batch axes; preserve dtype and device; no silent cast, move or squeeze; no host sync in hot paths |
| `BetterRobot/docs/contracts/03-state-mutability.md` | `C-STATE-*` | public ops never mutate inputs; `HumanState` is created per query and never mutated (it only memoises its outputs) |
| `BetterRobot/docs/contracts/07-dependencies-layering.md` | `C-DEP-*` | imports point down the package DAG; optional dependencies stay lazy; BH uses only BR's public surface |
| BR numerical contracts (pose `[t, qxyz, qw]`, twist `[linear, angular]`, `nq != nv`, `kinematics.integrate`) | `C-*` | `q` is a BR configuration; move on the manifold with `kinematics.integrate(identity.model, q, v)` |
| BR naming, style, testing and packaging pages | `C-*` | same conventions; BH-specific deviations are listed as `H-*` rules |

When a BetterHuman rule and a BetterRobot rule disagree, BetterRobot's numerical contracts win.

## Enforcement in BetterHuman

| test | rules |
|---|---|
| `tests/contract/test_br_imports.py` | `C-DEP-*`: only BetterRobot's public surface is imported |
| `tests/contract/test_layer_dependencies.py` | `H-LAYER-1`, record 0011: the unit and BetterRobot edge tables; ratchet `tests/contract/ratchets/edges.json` |
| `tests/contract/test_citations.py` | principle 7.8: cited paths and `path::symbol`s resolve, no line numbers; ratchet `tests/contract/ratchets/dead_citations.json` |
| `tests/contract/test_keyword_only.py` | `H-API-2` / `C-API-2`: keyword-only arguments; ratchet `tests/contract/ratchets/keyword_only.json` |
| `tests/parity/` | `H-PARITY-1`: parity gates |
| `tests/contract/test_models_dir.py` | `H-ASSET-2`, record 0017: weights paths only through `better_human.io` |
| `tests/contract/test_formats.py` | records 0012, 0013: every file format round-trips, reads its past versions through migration (fixtures `tests/support/formats/`), refuses a later version or another format, and has a page under `docs/reference/formats/`; only `src/better_human/io/formats.py` and `src/better_human/io/schema.py` name `format_version` (8.8); no `allow_pickle=True` in `src` but its listed exception; the fixtures match `tests/support/formats/SHA256SUMS` (written by `make_fixtures.py` there) and each spec page's field tables equal the reader's schema |
| `tests/conftest.py` | BetterRobot 0033: every skipped or xfailed outcome fails |
| `tests/contract/test_no_skips.py` | BetterRobot 0033: no skip or xfail spelling in any test module, deselected ones included |
| `tests/contract/test_pytest_config.py` | BetterRobot 0033: strict pytest, the tier markers |
| `tests/contract/test_python_version.py` | BetterRobot 0034: one Python version |
