---
name: regenerate-goldens
description: Regenerate BetterHuman's parity goldens - the licensed goldens in the models directory, and the tracked random-weight archives and their goldens. Use when an oracle, a weight release or an archive layout changes; a BetterHuman code change never regenerates them.
---

# Regenerate parity goldens

Goldens are gate data. Regenerate them only when the oracle, the weights or an archive layout changes, never to make
a change pass. Dump into a new directory, compare every array with the current files at a tolerance far below the
gate (a re-dump on another CPU or BLAS path differs in the last bits), and replace the files only after review.

Two kinds, and who can regenerate each:

- Licensed goldens, computed from the licensed weights: `$BETTERHUMAN_MODELS_DIR/goldens/` holds `smpl_neutral`,
  `smplx_neutral`, `mhr_lod0` to `mhr_lod6` and `soma_mid`. Only a machine with the weights and the
  oracle's environment can dump them. They stay in the models directory, and no array of a golden or of the weights
  is copied into a tracked file; git holds no array file, of any format, beyond the allowed ones
  (`tests/contract/test_golden_provenance.py::test_git_holds_no_array_file_but_the_allowed_ones`).
- Random-weight archives and goldens, tracked: `tests/support/random_weights/{smpl,smplh,smplx}_random.npz` and
  `tests/parity/smplx/random/{smpl,smplh,smplx}/`. Anyone can regenerate them: every array comes from a
  fixed seed, and the archive is the same bytes on every run. Regenerate the archives only when their layout
  changes, then the goldens from them, in a commit of their own that says why.

The commands live once, in the README of each oracle's folder:

1. SMPL, SMPL+H and SMPL-X, licensed and random-weight: `tools/parity/smplx/README.md` (the `smplx` oracle locked
   in `tools/parity/smplx/uv.lock`).
2. MHR: `tools/parity/mhr/README.md` (conda environment `tools/parity/mhr/environment.yml`; the converter runs
   before the dump).
3. SOMA: `tools/parity/soma/README.md` (an environment outside the repository from
   `tools/parity/soma/requirements.txt`; the dump runs with CUDA hidden).

Checks:

- Every golden stamps its `provenance`. `tests/contract/test_golden_provenance.py` holds the oracle
  pins, generators and dtypes, checks the random-weight goldens in every tier, rebuilds each archive from its seed
  (`slow`), and checks the licensed goldens in the release tier.
- Random-weight parity: `uv run pytest tests/parity/smplx/test_random_weights.py -q`.
- Licensed parity, with the weights: `uv run pytest tests/parity tests/contract/test_golden_provenance.py -q -m licensed`.
- The format fixtures of `tests/support/formats/` are not goldens: they are frozen, and
  `tests/support/formats/make_fixtures.py` rewrites them only to add a format or a version.
