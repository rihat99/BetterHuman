# 0012 BetterHuman archives carry `format`, `format_version` 2 and `provenance`

**Status:** Accepted (owner); amendment proposed in 0023. **Date:** 2026-11-01

## Context
`mhr_lod{N}.npz` has `format_version=1` and no `format`, and the loader compares the whole metadata dict, including
axis strings nothing reads (`families/mhr.py::METADATA`), so no migration is possible. `MHR_SCHEMA` requires 66 fields,
36 never read, plus 4 unread corrective fields. The SOMA converter writes a `source` provenance JSON that `SOMA_SCHEMA`
ignores. Re-converting needs the pymomentum and py-soma-x environments.

## Decision
We will define format v2 as `format="better_human.mhr"` or `"better_human.soma"`, a checked `format_version=2`, a
required `provenance` (converter, upstream versions, source sha256) and a schema of the fields the reader uses, with
unknown fields ignored, one pure `migrate_v1_to_v2` per format that adds `format` and `provenance="unknown (v1)"` in
memory so files on disk stay valid, and one `io.check_format` raising one `FormatError`.

## Example
```python
FORMAT, FORMAT_VERSION = "better_human.mhr", 2

def migrate_v1_to_v2(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """v2 adds `format` and `provenance`; v1 files keep their arrays and gain provenance "unknown (v1)"."""
    return {**fields, "format": np.str_(FORMAT), "provenance": np.str_("unknown (v1)"), "format_version": np.int64(2)}

fields = upgrade(read_npz(path))      # core code reads only v2
```

## Consequences
Old archives load without re-conversion; the reader no longer fails on an irrelevant axis string; a later version is
refused with both versions named. The cost is one migration function per format and a fixture per past version.

## Alternatives
Format v2 with mandatory re-conversion: needs both oracle environments working. Keep v1 and the equality check: no
room for a migration. SQLite's file format keeps every old file readable for the life of the project, the standard
8.7 sets.

## Compatibility
Files on disk remain valid. Readers older than this release refuse v2 archives written by the new converters; nothing
else changes. Reversible: no (a promise to read v1 and v2 forever).

## Enforcement
Per 8.7 and 8.8, in `tests/contract/test_formats.py`: a round-trip test, one v1 fixture and one v2 fixture in
`tests/support/formats/`, a test that a later version is refused, a test that `upgrade(v1)` equals the v2 fixture, and
a check that no `format_version` comparison exists outside `io.check_format` and the migration module.

## Open questions
None.
