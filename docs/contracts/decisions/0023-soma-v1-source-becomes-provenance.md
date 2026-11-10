# 0023 A version-1 SOMA archive's `source` becomes its `provenance`

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0012](0012-bh-converted-archives.md) migrates a version-1 archive to version 2 in memory by adding
`provenance="unknown (v1)"`. A version-1 SOMA archive is not without provenance: its converter (`tools/convert/soma/`)
wrote a JSON field `source` naming the package, the asset tag and commit, and the assets' sha256. The migration that
keeps it landed on 2026-11-06 (commit `0a25bb2`) together with an amendment section appended to 0012, which ruling R-3
does not allow: an accepted record changes only in its Status line, so the amendment is this record.

## Decision
We will turn a version-1 SOMA archive's `source` into its `provenance` and drop `source`, and make the reader require
`source` of version-1 SOMA archives only (`src/better_human/io/formats.py::PAST_FIELDS`). A version-1 MHR archive keeps
`provenance="unknown (v1)"`. This amends 0012's migration.

## Example
```python
PAST_FIELDS = {("better_human.soma", 1): ("source",)}   # the fields only that version carries, which its migration reads

def _soma_v1_to_v2(fields):
    kept = {name: value for name, value in fields.items() if name != "source"}
    return {**kept, "format": np.str_(SOMA), "format_version": np.int64(2), "provenance": np.str_(fields["source"])}
```

## Consequences
A SOMA archive converted before format version 2 still says which assets it came from. A version-1 SOMA archive
without `source` is refused, naming the field, instead of migrating to an archive that claims no provenance.

## Alternatives
`"unknown (v1)"` for SOMA as for MHR: one migration shape for both formats, at the price of discarding the hashes the
file carries.

## Compatibility
None for files: every version-1 SOMA archive the converter wrote carries `source`.

## Enforcement
`tests/contract/test_formats.py::test_past_version_reads_through_migration_as_the_latest` (a version-1 SOMA fixture
migrates to its own provenance) and
`tests/contract/test_formats.py::test_a_past_version_without_the_field_its_migration_reads_is_refused`.

## Open questions
None.
