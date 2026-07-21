# Order 04 — SMPL-X and the generic partial-model reduction

> **Agent log — 2026-07-20:** Completed on `dev`; the 29-case oracle grid
> peaked at `5.590597473847936e-08` (float64) and
> `6.556510925292969e-07` (float32). All six partial-model/hand-mean
> combinations passed at `5e-7`; no plan deviations. See
> [`04_results.md`](04_results.md).

Read first: `docs/architecture.md` (§Partial-model spawning, §Classic
conversion), `docs/contracts.md`, `references/design/smplx.md` (§3.2–3.3
hands/expression, §4 SMPL-X fields, §7 the traps),
`references/design/BEST_PRACTICES.md` §6, order 03's results file. Ground
rules in `plan/README.md` apply.

## Context

SMPL-X (10475 verts, 55 LBS joints: 1 global/pelvis + 21 body + jaw +
2 eyes + 30 fingers, 300 shape + 100 expression columns, hand PCA) is the model that motivates
partial spawning: most users want the body without face DOFs or without
articulated fingers. The old repo solved this with `SMPLXmini`/`SMPLXmid`
class forks — explicitly banned. This order delivers SMPL-X plus the one
generic reduction mechanism, as constructor parameters.

## Deliverables

1. **`core/reduction.py`** — generic, model-agnostic reduction applied at
   load time, before the BR tree is built:
   - drop **terminal subtrees** of joints: reassign each dropped joint's
     skinning weights to its surviving ancestor, renormalize. v1 supports
     terminal-subtree removal only — every v1 spawn flag drops leaves or
     leaf chains (fingers below the wrists, jaw, eyes), so a
     reparent-mid-chain branch would have zero callers (design principle
     2); it arrives with the first model that needs it. Reject non-terminal
     drop sets with a clear error;
   - slice corrective basis columns belonging to dropped joints (their
     pose features are identically rest ⇒ exact zero contribution — assert
     this property in a test, don't just claim it). Mind the indexing:
     `posedirs` columns are indexed by the (J−1) **non-root** joints, so
     dropped joint `j` owns columns `[(j−1)·9, j·9)`; this is well-defined
     only because the root is never droppable — assert that too;
   - slice shape/expression basis columns (`num_betas`,
     `num_expression`);
   - emit reduced `BodyStructure`/`BodyValues` + joint tables for the BR
     builder. Pure functions; no SMPL-X specifics inside.
2. **`bodies/smplx.py`** — `SMPLX(model_path=None, gender="neutral",
   num_betas=16, num_expression=10, use_hands=True, use_face=True,
   flat_hand_mean=False, dtype=..., device=...)`:
   - `use_face=False` drops jaw+eyes joints and all expression columns;
     `use_hands=False` drops the 30 finger joints; both are just calls
     into `core/reduction.py` with model-specific joint sets;
   - `SMPLXClassic` dataclass mirroring the native parameterization
     (`global_orient`, `body_pose`, `jaw_pose`, `leye_pose`, `reye_pose`,
     `left_hand_pose`, `right_hand_pose`, `expression`, `transl`) with
     hand poses acceptable as full axis-angle or PCA coefficients
     (`num_pca_comps` on the dataclass/converter, PCA expansion + the
     baked `hands_mean` handled in `from_classic`/`to_classic`, honoring
     `flat_hand_mean`);
   - the known traps are encoded, not rediscovered: SMPL-X native pose
     ordering is jaw+eyes **before** hands; expression =
     `shapedirs[:, :, 300:]`; `left_hand_pose=0` ≠ flat hand unless
     `flat_hand_mean` (see `references/design/smplx.md`).
   - Note: `expression` is a shape group — it flows through `with_shape`,
     not through pose. `SMPLXClassic.expression` exists only for classic
     interop and converter tests.
3. **Parity** (extend the order-03 harness):
   - full SMPL-X vs oracle goldens (float64 ≤ 1e-6): grid must exercise
     betas, expression, jaw/eyes, hands both as PCA and full axis-angle,
     `flat_hand_mean` both ways, root wart case. SMPL-X goldens
     (~0.25 MB/case × ~25-30 cases) are **regenerate-on-demand**, not
     committed — document the regeneration command and skip the parity
     tests (with explicit reason) when goldens are absent; the tree gets
     the same `preserve_joint_order=True` + alignment-test treatment as
     order 03;
   - partial-spawn correctness as a *property*, since the oracle can't
     spawn partials: a reduced body at any pose of the surviving DOFs must
     match the full body evaluated with dropped DOFs at rest, to a
     float32-epsilon-level tolerance measured and pinned (weight
     pre-summation changes float summation order near dropped joints —
     bit-identity is not achievable and must not be claimed). The
     mathematical identity behind the property: an at-rest joint's
     skinning transform equals its surviving ancestor's, transitively down
     the dropped chain.
4. **Tests**: reduction unit tests (terminal-subtree validation errors,
   weight renormalization sums to 1, corrective-column-slice exactness); SMPL-X round-trips including
   PCA hands; gradcheck `expression → vertices` (float32, vertex subset);
   registry `load("smplx", ...)`; batched + CUDA smoke as in order 03.

## Constraints

- No new classes beyond `SMPLX` + its classic dataclass. If `smpl.py` and
  `smplx.py` start sharing family-specific code, it moves to one shared
  family module — never copy-paste (the smplx-package disease).
- Keypoint/landmark output heads stay out (plan decision 6).

## Acceptance

- Full SMPL-X parity table green (float64 ≤ 1e-6; float32 recorded).
- Partial-spawn property tests green for `use_hands=False`,
  `use_face=False`, and both together; `num_betas`/`num_expression`
  slicing verified against the oracle at reduced counts.
- Full gate green; ruff clean.

## Results file

`plan/for_agents/04_results.md`: parity tables (full + partial property),
reduction findings, exact joint sets dropped per flag, deviations.
