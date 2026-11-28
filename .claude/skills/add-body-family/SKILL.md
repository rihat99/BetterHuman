---
name: add-body-family
description: Add a parametric body family to BetterHuman - weight format and converter, rig, loader, classic format, parity against its authors' code, tests and docs rows. Use when a task adds a body model the library does not load yet.
---

# Add a body family

A family lands when it meets the inclusion bar of `docs/contracts/scope.md`: parity against its authors' released
code run in a pinned environment, the one `FamilySpec`, `Rig` and skinning path, and its own parameters through a
classic format with an exact round trip. SOMA is the latest complete example: `git grep -il soma` lists every file a
family touches.

1. Weights. A family whose upstream files the loaders cannot read gets a converter,
   `tools/convert/<family>/convert_<family>.py` (with a README when it needs its own environment, as
   `tools/convert/soma/README.md`) and a line in `docs/install.md`. It writes a versioned archive: a format name and
   version in `src/better_human/io/formats.py`, a `ModelSchema` in `src/better_human/io/schema.py`, a spec page
   under `docs/reference/formats/`, a reader in `READERS` of
   `tests/contract/test_formats.py`, and fixtures written by `tests/support/formats/make_fixtures.py` and listed in
   `tests/support/formats/SHA256SUMS`. A family in the `smplx` layout reads its upstream file, with its file pattern
   and joint names in `src/better_human/families/_tables.py`.
2. Rig. A joint regressor over the mesh reuses `RegressedRig` (`src/better_human/families/regressed_rig.py`);
   anything else subclasses `Rig` (`src/better_human/core/rig.py`, four abstract members) in
   `src/better_human/families/<family>_rig.py`, as `MHRRig` and `SOMARig`. Family behaviour stays in the rig or the
   loader; `src/better_human/core/human_model.py` gets no branch.
3. Loader. `src/better_human/families/<family>_family.py`: `<family>(path=None, *, dtype, device, ...)` finds its
   file through `better_human.io.weight_path` (`tests/contract/test_models_dir.py`), maps every part to a density
   category (`_DENSITY_CATEGORIES`, `tests/families/test_density_categories.py`) and returns a `FamilySpec`
   (`src/better_human/core/spec.py`). Export it from `src/better_human/families/__init__.py`, add it to `_LOADERS`
   and the `family` literal of `load` in `src/better_human/__init__.py`
   (`tests/contract/test_keyword_only.py::test_load_shows_every_option_and_dispatches_to_every_family_loader`) and
   to `LOADERS` of `tests/families/test_loader_arguments.py`.
4. Classic format. `<Family>Classic` and `<Family>Format`, a `ClassicFormat`, in
   `src/better_human/families/classic.py`; its round trip in float32 and float64 in
   `tests/families/test_classic_formats.py`.
5. Parity. The oracle's pinned environment, `dump_<family>.py` and a README in `tools/parity/<family>/`; goldens in
   `$BETTERHUMAN_MODELS_DIR/goldens/<family>/` (the regenerate-goldens skill), read through
   `tests/support/goldens.py` by `tests/parity/<family>/test_<family>_parity.py`, marked `licensed`; the oracle's pin
   in `PINS` and the goldens in `LICENSED` of `tests/contract/test_golden_provenance.py`. An oracle that loads random
   arrays in the family's layout also gets a tracked random-weight archive and goldens, as the SMPL
   family's from `tools/parity/smplx/make_random_archive.py`, so the full tier runs the parity.
6. Tests. `tests/families/test_<family>.py` (sizes, root frame, batching, gradients) and the family's case in
   `tests/core/test_rig_properties.py`, `tests/io/test_archive_reader.py` and `tests/viz/test_layers.py`; a test
   that reads the weights is marked `licensed`. The family's acronym in `ACRONYMS` of
   `tests/contract/test_vocabulary.py`, every new word in `docs/reference/glossary.md`.
7. Docs, in the same change: the parity test in the `Enforcement:` paragraph of H-PARITY-1 in
   `docs/contracts/rules.md` and in its row of `docs/contracts/index.md` (`tests/docs/test_contract_matrix.py`); the
   family in the weights layout of `docs/install.md`, the README's family table, `docs/index.md` and the rig list of
   `docs/concepts/architecture.md`; every new layout another repository reads in
   `docs/contracts/coupling-standard.md` and `tests/contract/coupling_standard_lines.txt`; a rig of its own gets a
   load row in `LOADS` of `benchmarks/run.py`; then
   `uv run python -m better_robot.testing.api_snapshot better_human tests/contract/public_api.json --write`, and the
   commit message names the new public names.
8. Check: the fast tier, the full tier, then with the weights `uv run pytest tests/parity/<family> -q -m licensed`.
