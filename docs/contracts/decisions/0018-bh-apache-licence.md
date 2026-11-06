# 0018 BetterHuman is licensed Apache-2.0

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
BetterRobot is Apache-2.0 (`LICENSE`, `pyproject.toml`). BetterHuman is MIT (`LICENSE`, dated 2025-09-10). BetterHuman
depends on BetterRobot and ships data derived from licensed body models (segmentations, density tables). BetterHuman has one
author today, so a licence change needs no one else's consent. The MHR and SOMA-X code are
Apache-2.0; the SMPL family's weights carry a non-commercial research licence that no code licence changes.

## Decision
We will relicense BetterHuman's code under Apache-2.0, so that one licence with a patent grant covers the stack.

## Example
```toml
# BetterHuman pyproject.toml
license = "Apache-2.0"
```
```text
LICENSE          Apache License, Version 2.0 (same text as BetterRobot)
README.md        "Code: Apache-2.0. Body-model weights and data derived from them keep their own licences."
```

## Consequences
One licence across both libraries and one rule for contributions. The patent grant protects users of the combined
stack. The README must say that the weights and the data derived from them keep their own terms. Redistribution terms
of the shipped segmentation files are checked before any public push (0015).

## Alternatives
Keep both licences and say in BetterHuman's README that MIT code may depend on Apache-2.0 code: legal, but two
licences to explain and no patent grant on one side. Relicense BetterRobot to MIT: gives up the patent grant that
the lower library has. Copies of BetterHuman already received under MIT stay under MIT.

## Compatibility
Prospective only: new releases are Apache-2.0. Reversible only for future releases, never for copies already given.
No code change in the apps.

## Enforcement
New `tests/contract/test_license.py` in BetterHuman: the `LICENSE` text, `pyproject.toml` `license` and the README
line agree; the release checklist (8.10) re-reads the line.

## Open questions
None.
