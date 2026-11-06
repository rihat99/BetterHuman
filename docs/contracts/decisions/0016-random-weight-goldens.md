# 0016 Random-weight goldens run the SMPL-family parity machinery without licensed weights

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
Random-weight goldens let the fast and full tiers exercise the parity machinery without licensed weights. Today 270 of
382 BetterHuman tests need licensed weights; after tiering the fast and full tiers would run about 110 tests and no
parity. The `smplx` oracle loads any archive of the right layout. pymomentum and py-soma-x read their own asset formats,
so MHR and SOMA need converter work. BetterHuman's README example and docs snippets skip without weights.

## Decision
We will generate, for SMPL, SMPL-H and SMPL-X now, a random archive per family in the upstream layout
(`tests/support/random_weights/`) and `smplx` oracle goldens from each (`tests/parity/smplx/random/`), track them in
git, run them in the fast and full tiers, and add MANO, FLAME, MHR and SOMA random-weight goldens later.

## Example
```bash
uv run pytest -q tests/parity        # full tier: SMPL-family parity on tracked random-weight goldens, no licence needed
```
```python
human = bh.load("smplx", path=RANDOM_ARCHIVE)   # the README's first example runs on the random archive, output compared
```

## Consequences
The classic round trips, mass and skinning machinery are gated on every full-tier run. The README's first example runs
on the random archive with its output compared. Random weights test the code path, not the quality of the licensed
model, so the release tier keeps the licensed parity (0015). MHR and SOMA have no fast or full parity until their
converter work.

## Alternatives
All families now: MHR and SOMA need converter work in the oracle environments first. None: the fast and full tiers run
no parity at all. The `smplx` oracle loads any archive of the right layout, which is what makes a random archive
possible; pymomentum and py-soma-x do not.

## Compatibility
Adds tracked files (small, within the 1 MiB rule) and tools. Nothing public changes. Reversible: yes.

## Enforcement
`tests/parity/smplx/test_random_weights.py` runs in the full tier against the random-weight goldens with the gates'
tolerances (1e-6 m for SMPL, SMPL-H and SMPL-X); `tests/contract/test_golden_provenance.py` (0014) covers the tracked
files; the README example runs in `tests/docs/`.

## Open questions
None.
