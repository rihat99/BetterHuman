# 0013 BetterHuman package data is versioned and loads without pickle

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`families/assets/smpl_3d_segmentation.npy` is a pickled dict loaded with `allow_pickle=True`; `body_densities.json` is
keyed by index strings; the frame-set JSON has no version and silently accepts a legacy bare-int form; studio poses
store `q` without joint names. `bh.load("smplx")` takes 60 s on CPU because numpy streams compressed npz members
20x slower than `zipfile.read` plus `np.load(BytesIO)` (the `posedirs` member alone: 5.7 s against 0.27 s). The shipped
frame sets are consumed by two apps through the checkout path.

## Decision
We will store the segmentation as `.npz` (faces plus per-part offsets) with `format` and `format_version`, key the
density tables by category name with a version, store frame sets as `{"format": "better_human.frame_set",
"format_version": 1, "frames": {...}}` with a migration for the bare-int form, ship `better_contacts.json` as `io`
package data read by `io.shipped_frame_set(name)` (`contacts.json` becomes a test fixture and `studio_data/` leaves the
root), version the studio formats and store joint names in poses, and make the npz reader read each member's bytes once.

## Example
```python
frames = bh.io.shipped_frame_set("smplx_contacts")       # was Path(bh.__file__).parents[2] / "studio_data" / ...
frames = bh.io.read_frame_set(path)                      # raises FormatError on a later version, listing both
# {"format": "better_human.frame_set", "format_version": 1, "frames": {"left_wrist": {"vertex": 5432, "joint": "left_wrist"}}}
```

## Consequences
`bh.load("smplx")` takes seconds, which makes load-at-new-dtype (0005) and the examples practical. No shipped file
executes pickle. Load time joins the benchmark (5.7). A stored studio pose survives a joint-order change.

## Alternatives
Store converted weights uncompressed: every user re-converts and upstream SMPL-X, which is not ours, gains nothing. The
reader fix needs no file format change. SQLite versions its file header and refuses later versions (8.7).

## Compatibility
Breaks the old frame-set path reads in BetterVideoReconstruction (`bvr/body.py`) and BetterRetarget
(`betterretarget/human.py`); each switches to `io.shipped_frame_set`. Existing studio files migrate by version.

## Enforcement
8.7 and 8.8 checks per format in `tests/contract/test_formats.py` (round trip, a fixture per past version, a later
version refused), which also fails on `allow_pickle=True` anywhere in `src`; the load rows of `benchmarks/run.py`,
where loading an archive takes at most 1.5 times the raw read of its members.

## Open questions
None.
