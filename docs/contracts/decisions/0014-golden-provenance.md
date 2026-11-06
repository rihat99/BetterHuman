# 0014 Every golden file stamps its provenance

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
104 golden files (23.6 MiB) back the parity gates. SMPL (13) and SMPL-X (29) carry no provenance; MHR (48) carry the
oracle, dtype and seed but no version, generator or asset hash; SOMA (14) carry only `format_version` and `seed`. The
SOMA converter and the dump script default to an agent scratchpad under `/tmp`.

## Decision
We will stamp one `provenance` field `{oracle: "<package>==<version>", generator: <the dump script's repository path>,
asset_sha256, dtype, seed}` into the existing files with the arrays untouched, from the tool lockfiles and local asset
hashes (SOMA's asset hash marked unknown if the assets are gone), in a commit of its own, and make the SOMA tools take
the asset directory as a required argument.

## Example
```python
golden = np.load(path)
json.loads(str(golden["provenance"]))
# {"oracle": "smplx==<version>", "generator": "tools/parity/smplx/dump_smplx.py", "asset_sha256": "<64 hex>", "dtype": "float64", "seed": <int>}
```

## Consequences
Every golden says what produced it and can be regenerated. The stamping commit changes only a metadata field (11.7).
The contract test runs wherever the goldens live: over `$BETTERHUMAN_MODELS_DIR/goldens` for the licensed files (0015,
release tier) and over the tracked random-weight goldens (0016) in the fast tier.

## Alternatives
Regenerate every golden in the oracle environments with dump scripts that write provenance: stronger, but it needs the
pymomentum, py-soma-x and smplx environments working at once, and a regenerated number could hide a regression. The
principles require a generator and a version for every golden (6.4).

## Compatibility
Adds a field; readers ignore it. Reversible: yes.

## Enforcement
New `tests/contract/test_golden_provenance.py`: loads every golden, checks the keys and that the `generator` path
exists in the repository; it fails when a golden has no provenance, and when a tracked golden's provenance does not
name a random-weight archive (0015).

## Open questions
None.
