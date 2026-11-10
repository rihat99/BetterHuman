# 0025 A stamped golden's `asset_sha256` is the hash of the assets at stamp time

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0014](0014-golden-provenance.md) stamps a `provenance` field into every existing golden, with an
`asset_sha256` from the local asset hashes. The 104 goldens stamped on 2026-11-06 took the hash of the asset files on
disk that day, not a hash taken when each golden was dumped: the SMPL, SMPL-X and MHR assets of that machine were
restored on 2026-10-10, while their goldens date from 2026-07-21. Commit `9d2e3fc` wrote this into 0014 as a note,
which an accepted record's frozen text does not allow; this record carries it.

## Decision
We will read `asset_sha256` as the hash of the assets at the time the golden was stamped, and say so where the goldens
are regenerated (the READMEs of `tools/parity/smplx/`, `tools/parity/mhr/` and `tools/parity/soma/`). A golden its
dump script writes hashes the files that dump read. This amends what 0014's provenance field promises.

## Example
```python
json.loads(str(np.load(path)["provenance"]))["asset_sha256"]
# SMPL, SMPL-X, MHR goldens of 2026-07-21: the assets on disk on 2026-11-06, restored on 2026-10-10
# SOMA goldens: the assets their dump read (dumped again on 2026-11-06, bit for bit equal to the earlier files)
```

## Consequences
That the SMPL, SMPL-X and MHR goldens were computed from the hashed files is supported by their parity errors, which
did not change and sit far inside the gates, but it is not proven bit for bit. Regenerating those goldens with their
dump scripts would make the hash exact.

## Alternatives
Mark the hash of the restored assets unknown: true, but it throws away the one link the goldens have to a file.

## Compatibility
None: no golden changes.

## Enforcement
`tests/contract/test_golden_provenance.py` (every golden carries the field; a tracked golden names a random-weight
archive).

## Open questions
1. Regenerate the SMPL, SMPL-X and MHR licensed goldens with their dump scripts, so their hashes are exact (a release
   tier run on a machine with the weights)?
