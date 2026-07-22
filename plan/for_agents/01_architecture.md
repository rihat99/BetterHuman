# Order 01 — end-to-end API, package layout, MHR readability

> **Implementation log (2026-07-22): complete.** Exact parity maxima are
> unchanged; CUDA gate: 230 passed, 0 skipped. Added `mhr/deformation.py` to
> reconcile the prescribed split with the ~300-line limit. Two BetterRobot
> rationale claims were corrected against source; details are in `01_results.md`.

The owner's asks: a single end-to-end call taking shape and pose; family
subfolders under `bodies/` with an unchanged public API; and an MHR module
that reads elegantly. This is an API + layout revision, not a rewrite: the
math, the numerics, and the composition architecture stay exactly as they
are.

## Ground rules (binding)

- Read `~/.claude/handoffs/betterhuman_mhr_context.md`,
  `docs/architecture.md`, and `docs/design_principles.md` first.
- **No BetterRobot modifications.** Never modify `models/` weights/assets or
  `references/`.
- **No numerics changes of any kind.** Run the full parity suite before and
  after; every measured maximum must be byte-identical. No pin changes.
- Alpha: breaking import-path changes are fine, but the *public* names
  (`bh.SMPL`, `bh.SMPLX`, `bh.MHR`, `bh.load`, the `*Classic` types) must
  keep working unchanged.
- **No new abstractions.** This order adds one dunder method and moves
  code between files. If you find yourself writing a new class, stop.
- Write `plan/for_agents/01_results.md`: what moved where, LOC per file
  before/after, before/after parity maxima, exact test counts.

## 1. End-to-end call

Add exactly one method to `ParametricBody` (`core/body.py`):

```python
def __call__(self, q: torch.Tensor, **shape_groups: torch.Tensor) -> torch.Tensor:
    """Vertices from pose and optional shape in one call."""
    body = self.with_shape(**shape_groups) if shape_groups else self
    return body.vertices(q)
```

- `body(q)` == `body.vertices(q)`; `body(q, betas=b)` ==
  `body.with_shape(betas=b).vertices(q)`. Nothing else. The amortized
  two-step (`with_shape` once, `vertices` per frame) remains the documented
  fast path for fixed-shape streaming.
- Tests: end-to-end equals the two-step result exactly (SMPL and MHR, CPU;
  one CUDA case); unknown-group and bad-shape errors still raise through
  the call.
- Do not add caching, do not return tuples/structs, do not add a `forward`
  alias. One method, one return value.
- Docs: short "one-call API" paragraph in `docs/architecture.md`'s API
  section showing both usage patterns and when to prefer which.

This is ergonomics, not performance: FK dominates runtime (see
`BR_wishlist.md`), and no BetterHuman-side API change moves that needle.

## 2. `bodies/` family subpackages

Reorganize `src/better_human/bodies/` into family subpackages. Pure file
moves plus import fixes — function bodies are copied verbatim:

```text
bodies/
  __init__.py          re-exports: SMPL, SMPLClassic, SMPLX, SMPLXClassic,
                       MHR, MHRClassic, load  (public API unchanged)
  registry.py          stays
  config_body.py       stays (shared config base)
  assets/              stays
  smpl_family/
    __init__.py        re-exports SMPL, SMPLClassic, SMPLX, SMPLXClassic
    smpl.py            was bodies/smpl.py
    smplx.py           was bodies/smplx.py
    shared.py          was bodies/smpl_family.py
    config_smpl.py     was bodies/config_smpl.py
    config_smplx.py    was bodies/config_smplx.py
  mhr/
    __init__.py        re-exports MHR, MHRClassic
    ...                see §3 for the module split
```

- `git mv` so history follows the files.
- Update every internal import (src, tests, benchmarks, tools) to the new
  paths. Grep for stragglers; `examples/` too if it imports these modules.
- Do not merge, split, or "improve" any smpl-family function while moving
  it. The only content change allowed in `shared.py` etc. is the import
  block and the module docstring.
- Future families (MANO/FLAME, STAR/SUPR, Anny) each get their own
  subpackage later; do not create empty placeholders now.

## 3. MHR decomposition for readability

`bodies/mhr.py` is 762 lines with a 180-line tree builder and a 168-line
`__init__`. The complexity is intrinsic (verified twice by review — do not
simplify the logic), but one file forces readers to hold five concerns at
once. Split it into the `bodies/mhr/` package along its existing seams:

```text
bodies/mhr/
  __init__.py     re-exports MHR, MHRClassic
  config.py       was bodies/config_mhr.py
  archive.py      _tensor, _strings, _string_scalar, _validate_metadata,
                  _parameter_partitions, _parameter_bases, _segmentation
                  (reading + validating the converted NPZ)
  tree.py         _TreeBinding, _origin, _build_mhr_tree,
                  _accumulated_scales  (archive -> BetterRobot tree)
  body.py         class MHR (__init__, from_classic, to_classic),
                  _bake_mhr_shape, _mhr_pose_corrective
  classic.py      MHRClassic + any pure conversion helpers that
                  from_classic/to_classic already delegate to core/classic.py
                  (if nothing beyond the dataclass moves here, put MHRClassic
                  in body.py and skip this file — do not create a
                  one-dataclass module for symmetry's sake)
```

Rules for the split:

- Function bodies move verbatim. The private-name underscores may drop where
  a function becomes a module-internal helper imported by `body.py`
  (`archive.tensor(...)` reads better than `archive._tensor(...)`), but
  signatures and logic are untouched.
- Each module gets a real docstring stating its single concern and the one
  or two domain facts a reader needs (e.g. `tree.py`: why the spine rows
  stack into one revolute + mimics; the static-FBX-binds-are-authoritative
  rule; `body.py`: the bake-overwrites-placeholders flow). Domain knowledge
  belongs in module docstrings, not inline comment noise.
- `MHR.__init__` may be reorganized into a few clearly named private
  methods on MHR (e.g. `_bind_archive`, `_bind_corrective`) ONLY if each is
  called exactly once from `__init__` and the split makes the read
  top-to-bottom shorter. No new dataclasses, no builder classes.
- Target: no module over ~300 lines; the parity/round-trip/gradcheck test
  files keep passing without tolerance changes; `tests/bodies/test_mhr.py`
  import updates only.

## 4. Record the composition decision

Add a short "Design decision: composition, not inheritance" subsection to
`docs/architecture.md` (near the ParametricBody description) capturing,
in the owner's words later verifiable against BetterRobot source:

- `ParametricBody` composes a `better_robot.Model` instead of
  `BodyStructure`/`BodyValues` subclassing `ModelStructure`/`ModelValues`.
- Why: (1) BR itself composes — `Model` is `ModelStructure` + `ModelValues`
  behind a property facade, so composition is the host library's idiom;
  (2) `ModelValues` is registered in torch's pytree registry by exact
  class, and BR reconstructs values via `cls(**tensor_fields)` — subclass
  instances would flatten wrong or be silently rebuilt as the parent,
  dropping body fields; (3) `ModelValues.validate` pins exact unbatched
  event shapes (`(njoints, 7)` ...) while `BodyValues` carries batched
  shape-baked products `(..., V, 3)`; (4) the vocabularies collide —
  `joint_names`/`parents` mean the 127-native skin joints in
  `BodyStructure` and the 203-joint BR tree in `ModelStructure`, and
  inheritance would force one meaning to win; (5) BR is alpha — inheriting
  would couple BetterHuman to BR's exact ~70-field constructor list.
- One body-has-a-robot also keeps the roadmap open (several meshes/LODs
  sharing one skeleton).

## Acceptance

- Full suite green with CUDA (expect 226 passed + the new `__call__`
  tests; 0 skipped); parity maxima byte-identical before/after (paste both
  tables).
- `from better_human import SMPL, SMPLX, MHR, load` and all `*Classic`
  imports work exactly as before; `ruff check .` clean.
- No module in `bodies/` over ~300 lines except `smpl_family/smplx.py` if
  it already is; report final LOC per file.
- `benchmarks/benchmark_mhr.py` and `benchmarks/benchmark_smpl.py` still
  run (CPU smoke) after the import updates.
