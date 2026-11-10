# 0024 Archives load with `np.load`, once per load; 0013's slow member was the host's huge pages

**Status:** Proposed (integrator, awaiting owner). **Date:** 2026-11-10

## Context
Record [0013](0013-bh-package-data-and-load-speed.md) found `bh.load("smplx")` taking 60 s on CPU and blamed numpy's
reading of compressed npz members (`posedirs` alone 5.7 s against 0.27 s through `zipfile.read` plus
`np.load(BytesIO)`), so it decided that the npz reader reads each member's bytes once. Measured again on 2026-11-06
(commit `a6b2bce` appended the findings to 0013 as a note, which ruling R-3 does not allow; this record carries them):

- the 5.7 s was the host's transparent huge pages: with them set to `madvise` and memory fragmented, the first touch
  of each large array waits for the kernel to compact memory (`docs/guides/run-the-tests.md`);
- `bh.load("smplx")` was slow because `families.smplx` decompressed the archive twice, once for its hand arrays and
  once for the rest; commit `a90dc45` decompresses it once, and the load then takes 1.00 times the raw read;
- the `np.load(BytesIO)` reader was slower than `np.load` on every archive (SMPL-X 0.79 s against 0.60 s per load,
  medians of three, huge pages off), so it was not landed.

## Decision
We will read an archive with `np.load`, once per load, and keep no second reader. A slow load on a host with huge
pages is fixed in the environment (`NUMPY_MADVISE_HUGEPAGE=0`), not in the reader. This amends 0013's reader
decision and corrects its context.

## Example
```bash
NUMPY_MADVISE_HUGEPAGE=0 uv run pytest tests -q -m "not cuda and not licensed"   # the full tier, without the stalls
```

## Consequences
One reader, numpy's. The test guide tells a user on such a host to set `NUMPY_MADVISE_HUGEPAGE=0`; without it,
`bh.load` there is as slow as numpy's first touch of each array.

## Alternatives
The `zipfile.read` plus `np.load(BytesIO)` reader 0013 proposed: slower on every archive once the huge-page stalls are
taken out, and a second code path for the same files.

## Compatibility
None: no file format and no public name changes.

## Enforcement
No test fails when a load slows: the ratio of a load to the raw read of its members is a benchmark measurement, for
the runner `benchmarks/run.py` (open question 1).

## Open questions
1. Record 0013's Enforcement reads like a gate (a load at most 1.5 times the raw read), but the benchmark runner only
   reports the ratio, and SOMA measured 3.75 to 5.67 in work package X3: is 1.5 a gate the runner enforces, and on
   which families?
