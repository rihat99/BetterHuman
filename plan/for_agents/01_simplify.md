# Order 01 — remove the reviewed overengineering

> **Implementation log (2026-07-21):** Complete. F1–F5 and F7–F10 were
> applied; optional F6/F11 were skipped. The three FBX-derived axis guards
> were retained as a documented F1 deviation. All parity maxima are unchanged;
> the full CUDA gate is 226 passed, 0 skipped. See `01_results.md`.

A pure refactor driven by a completed adversarial review of commits
`1814394`/`8c32ec7`. The findings below were verified against the code with
file:line evidence — do not re-litigate them, but do re-verify each line
number before editing (the review predates you). The review's overall
verdict: the architecture is sound; the fat is localized. Your job is to
remove exactly that fat and nothing else.

## Ground rules (binding)

- Read `~/.claude/handoffs/betterhuman_mhr_context.md`,
  `docs/design_principles.md`, and `docs/architecture.md` first.
- **No BetterRobot modifications.** Never modify `models/MHR/assets/` or
  `references/`.
- **Pure refactor**: run the full parity suite (SMPL, SMPL-X, MHR) before
  and after and diff the *measured maxima* — they must be byte-identical.
  No pinned tolerance changes. No numerics changes of any kind.
- The safety net only grows in coverage; tests may be *rewritten* only as
  finding F10 prescribes (private-attr assertions re-expressed through
  public surface, keeping the behavioral intent).
- Surgical: touch only what a finding names. No drive-by "improvements".
- Write `plan/for_agents/01_results.md`: per-finding applied/skipped (with
  reason), LOC delta per file, before/after parity maxima, exact test counts.

## Execution order

F10 first (it unblocks F2), then F1 → F3 → F4 → F5 → F2 → F7.
F6, F8, F9, F11 are optional — apply only if genuinely straightforward.

## Findings (verified; severity, location, action)

**F10 (do first) — tests pin private structure.**
`tests/bodies/test_mhr.py:77-95, 106-107, 138-139, 155-157, 285-297` assert
on `body._pose_parameter_indices`, `_proportion_parameter_indices`,
`_pose_host_weights`, `_corrective_full_q_indices`,
`_corrective_first_layer_weight`, `_q_to_classic_permutation`. Re-express
the pose/proportion partition and host-weight assertions through public
behavior (`from_classic`/`to_classic`, `robot.q_permutation`,
`expand_configuration`); keep a private reach-in only where no public
surface exists, and say so in a comment. The parity tests touch no private
attrs — leave them untouched.

**F1 (high) — validation of impossible states.**
`src/better_human/bodies/mhr.py:108-125` re-checks 8 metadata strings the
converter writes as literal constants and the library never branches on;
`mhr.py:626-638` re-checks 4 corrective invariants the converter asserts.
Keep the minimal honest set: `format_version`, `lod`, the
parent-before-child order check (`mhr.py:124-125` — load-bearing because
BetterRobot silently *repairs* non-topological order), the partition-count
check in `_parameter_partitions`, and the `(132,131,179,178)` structural
assert. Delete the 8-string block and the 4 corrective-expectation checks
(~28 lines).

**F3 (med) — dead neutral FK + throwaway provisional values.**
`mhr.py:596-600` runs a full neutral FK whose outputs (`rest_joints`,
`v_shaped`) are immediately overwritten by `_bake_mhr_shape` at `mhr.py:693`
(verified: the bake never reads them). Replace with cheap placeholders and
delete the first FK (~6 lines + one FK traversal per spawn).

**F4 (med) — `io/load.py` validates fields it then discards.**
`src/better_human/io/load.py:121-145` (`_read_array_header_views`) parses
NPY headers of *excluded* fields solely to schema-validate data that is
thrown away. Validate the selected sub-schema only; delete the header-view
machinery (~28 lines). The lazy-exclusion behavior itself (not materializing
2.6 GB LOD0 correctives) must be preserved — keep/adapt the guarded test in
`tests/io/test_load.py:113-133` so it still proves no excluded payload is
read.

**F5 (med) — speculative API with zero in-tree callers.**
`src/better_human/core/pipeline.py:76-88` `pose_feature_6d_from_matrix` is
used only by its own tests. Delete it, its `__all__` entry, and its tests
(~14 src + ~20 test lines). Alpha: no deprecation shim.

**F2 (high, after F10) — `_TreeBinding` exploded into separate attrs.**
`mhr.py:658-695` unpacks the `_TreeBinding` dataclass (`mhr.py:57-69`) into
~10 individual `object.__setattr__` calls. Store the binding once
(`self._tree`) and reference its fields at use sites
(`_bake_mhr_shape`, `_mhr_pose_corrective`, `from_classic`, `to_classic`).
Do not invent new grouping dataclasses for the remaining loose attrs —
only reuse the one that already exists.

**F7 (low) — dead attribute.**
`_corrective_euler_indices` (`mhr.py:66`, `mhr.py:684`) is never read after
construction (only the derived one-hot `_corrective_euler_transform` is
used). Drop the field and the setattr; keep the local tensor just long
enough to build the one-hot.

**Optional (apply only if straightforward, skip freely with a note):**

- **F6**: the archive stores both `corrective_first_layer_indices` and the
  redundant `corrective_first_layer_mask` (~2.25 MB × 7 LODs), triple-checked
  across converter / schema / `dense_weight_from_sparse`
  (`pipeline.py:112-113`). Dropping the mask touches the converter, the
  schema, and requires regenerating all 7 converted npz in the conda
  `betterhuman-mhr-parity` env — only do it if you run and re-verify the
  full parity suite against regenerated archives.
- **F8**: the `(...,7)` bind branch of `skinning_transforms`
  (`pipeline.py:145-150`) has no production caller (SMPL passes 3-D, MHR
  uses `skinning_transforms_from_inverse`). May be deleted with its tests.
- **F9**: the unused optional `first_layer_bias` of `mlp_corrective`
  (MHR ships no biases).
- **F11**: the ~95-line hand-rolled binary-FBX `GlobalSettings` parser in
  `tools/convert/mhr/convert_mhr.py:66-163` extracts constants the config
  already hardcodes. Only remove if pymomentum exposes the same metadata —
  verify in the pinned env first; otherwise leave.

## Leave alone (reviewed and justified — do not churn)

`_build_mhr_tree` (intrinsic complexity, ordered by the original plan);
the scaled-LBS machinery and `_accumulated_scales`; the
`_shape_bake_policy`/`_pose_corrective_policy` seams; the `classic.py`
helper extraction; the 2π-roll negative-quaternion handling in `to_classic`
(correctness-critical, tested); the one-hot `corrective_euler_transform`
matmul (it is an *accumulating* scatter for the stacked spine rows, kept as
matmul for clean autograd); `dump_mhr.py`'s heavy self-checks (they feed
the parity net); the converter's parse guards.

## Acceptance

- Parity maxima byte-identical before/after (paste both tables).
- Full suite green (expect 228 passed, 0 skipped with CUDA visible;
  count will drop only by tests deliberately deleted under F5/F8 — list
  every deleted test by name).
- `ruff check .` clean; LOC report per file; every finding accounted for
  as applied or skipped-with-reason.
