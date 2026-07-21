# Order 01 results

Completed on `dev` from base commit `ca37bd5` on 2026-07-20.

## Delivered

- Replaced the setuptools project with a Hatchling-built uv project at
  version `0.1.0.dev0`.
- Added the editable sibling `../BetterRobot` dependency and matched its
  Python requirement, PyTorch CUDA 12.6 index, build layout, and Ruff rules.
- Reduced the package to `src/better_human/__init__.py`, containing only
  `__version__`.
- Added import, editable-dependency, single FK, and batch-4 FK coverage in
  `tests/test_smoke.py`.
- Replaced the conda-era README and `.gitignore`; generated and tracked
  `uv.lock`.

## Deletion evidence

The baseline contained 39 tracked legacy files: 6 scripts, 32 package files,
and 1 test. The old initializer was replaced at the same path; the other 38
tracked files were deleted.

- Scripts: `scripts/convert.py`, `scripts/mano_viser_demo.py`,
  `scripts/smpl_viser_demo.py`, `scripts/smplh_viser_demo.py`,
  `scripts/smplx_viser_demo.py`, `scripts/star_viser_demo.py`.
- Core: `src/better_human/core/__init__.py`,
  `src/better_human/core/humanoid.py`.
- SMPL package: `src/better_human/smpl/__init__.py`,
  `src/better_human/smpl/base.py`, `src/better_human/smpl/dynamics.py`,
  `src/better_human/smpl/kinematics.py`, `src/better_human/smpl/mass.py`,
  `src/better_human/smpl/sequence.py`,
  `src/better_human/smpl/visualize.py`.
- SMPL configs: `src/better_human/smpl/config/__init__.py`,
  `body_densities.json`, `mano.json`, `mano_vert_segmentation.json`,
  `smpl.json`, `smpl_3d_segmentation.npy`, `smpl_vert_segmentation.json`,
  `smplh.json`, `smplx.json`, `smplx_mini.json`,
  `smplx_mini_3d_segmentation.npy`, `smplx_vert_segmentation.json` (all
  paths under `src/better_human/smpl/config/`).
- SMPL models: `flame.py`, `mano.py`, `smpl.py`, `smplh.py`, `smplx.py`,
  `smplx_mini.py`, `star.py` (all under
  `src/better_human/smpl/model/`).
- Other package files: `src/better_human/urdf/__init__.py`,
  `src/better_human/utils/__init__.py`,
  `src/better_human/utils/lie.py`.
- Test: `tests/test_load.py`.

The exact ignored/untracked deletion targets were the entire `temp/` tree
(101 files, including its nested STAR checkout), the entire generated
`src/better_human.egg-info/` tree (`PKG-INFO`, `SOURCES.txt`,
`dependency_links.txt`, `requires.txt`, `top_level.txt`), all legacy
`__pycache__/` trees, and the pre-existing
`src/better_human/smpl/config/smplx_mid.json` and
`src/better_human/smpl/model/smplx_mid.py` files.

## Resolved environment

| Component | Resolved version |
|---|---:|
| Python on this host | 3.13.11 |
| BetterHuman | 0.1.0.dev0 (editable) |
| BetterRobot | 0.2.0 (editable `../BetterRobot`) |
| NumPy | 2.5.1 |
| PyTorch | 2.13.0+cu126 |
| pytest | 9.1.1 |
| Ruff | 0.15.22 |
| uv | 0.11.7 |

The universal lock also resolves NumPy 2.2.6 for Python below 3.11 and 2.4.6
for Python 3.11. Hatchling is an isolated build requirement; uv does not
record an exact Hatchling version in this lockfile.

## Verification

- `uv sync`: passed; 76 packages resolved.
- `uv lock --check`: passed.
- `uv run pytest tests/ -q`: 2 passed in 2.28 seconds on the final run.
- `uv run ruff check src/ tests/`: all checks passed.
- FK measurements: `nq == 99`, `nv == 75`; joint poses `(25, 7)` and
  `(4, 25, 7)`.
- `git diff HEAD --check`: clean.
- `models/` (74 files) and `references/` (8,370 files) had the same combined
  path/size/mtime manifest before and after the work
  (`1e12ec801fada88a15763f372770d3e065e4d1015a680a3f0e50917cb52e2258`).
- BetterRobot was not modified.

## Deviations and findings

- **Plan deviations:** none.
- The working tree initially contained an `SMPLXMid` import plus two
  untracked `smplx_mid` files. Order 01 explicitly classified these as
  old-tree leftovers, so they were removed rather than preserved.
- The initial `.gitignore` already had an unstaged `references` entry; the
  required concise replacement incorporates it as `references/`.
- The workspace-level `../CLAUDE.md` still describes BetterHuman as
  reference-only. The newer project plan clearly supersedes it; the external
  file was left unchanged as out of scope.
