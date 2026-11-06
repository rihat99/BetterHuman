# 0015 Licensed-derived goldens live in the models directory, not in git

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
The 104 goldens in `tests/parity/goldens` are outputs of licensed SMPL, SMPL-X, MHR and SOMA weights. Whether outputs
derived from licensed weights may go to a public remote is unresolved until each licence is read. Removing files from
history later takes a history rewrite. Whether the shipped segmentation files may be redistributed is also unverified.
The fast and full tiers must run without licensed weights (BetterRobot 0033).

## Decision
We will move the licensed-derived goldens out of git now into `$BETTERHUMAN_MODELS_DIR/goldens`, reached through
`io.models_dir()` and used only in the release tier, keep only random-weight goldens (0016) in git, and leave history
unchanged (a rewrite is the owner's call before any public push).

## Example
```text
$BETTERHUMAN_MODELS_DIR/goldens/{smpl_neutral,smplx_neutral,mhr_lod0..mhr_lod6,soma_mid}/   # 104 stamped files (0014)
tests/parity/smplx/random/{smpl,smplh,smplx}/          # random-weight goldens only (0016); fast and full tiers
```
```python
golden_dir = io.models_dir() / "goldens"      # raises with the fix when the variable is unset (0017)
```

## Consequences
A clone of the repository contains no output of a licensed asset from now on. Until 0016 covers a family, that
family's parity gate runs only in the release tier on a machine that has the models directory: MHR and SOMA until
their oracle work lands. Old commits still hold the files, so a public push still waits for the licence reading and
the owner's decision on history. The goldens do not travel with a clone.

## Alternatives
Keep the files in git while the remote stays private, with no public push until each licence is read: the files then
sit in every clone and in all later history. Keep only random-weight goldens once 0016 lands, moving the licensed files
out then: it leaves the licensed files in git for as long as 0016 takes. The smplx project ships code
only and asks users to download the weights themselves.

## Compatibility
Breaks nothing in the public surface. A test that read `tests/parity/goldens` for a licensed family now reads the
models directory. Reversible by moving the files back, except that history keeps them either way.

## Enforcement
`tests/contract/test_golden_provenance.py` (0014): every tracked golden's provenance names a random-weight archive;
`tests/contract/test_repository_layout.py::test_no_symlinks_and_size` (the 1 MiB file-size rule, 9.3); the release-tier
command fails when the models directory or a golden is missing (BetterRobot 0033).

## Open questions
None.
