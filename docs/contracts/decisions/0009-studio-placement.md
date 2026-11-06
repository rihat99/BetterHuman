# 0009 The studio is `better_human.studio`, an optional application subpackage

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
`viz/studio/` (1,846 lines) is an application inside the library. Its data root is the checkout
(`persistence.py::DEFAULT_ROOT`, broken in a wheel); it writes `os.environ["BETTERHUMAN_MODELS_DIR"]`; it imports
`core.mass._majority_face_parts`, reaches `Layer._visible`, `Layer._handles` and viser's `_tab_handles`; it draws from
numpy's global RNG and draws one authored frame in two orientations. All its tests skip without SMPL weights.

## Decision
We will move the studio to `better_human.studio`, an optional `studio` extra under the application profile, with a
required data root (the command-line flag `--data-root`, no environment variable), weights through `io.models_dir()` (0017), public
BetterHuman and BetterRobot names only (BetterRobot 0022), its own versioned file formats (0013), and the vertex-segmentation JSON
files moved with it.

## Example
```bash
uv sync --extra studio
uv run python -m better_human.studio --data-root ~/better/studio_data   # no default root inside the checkout
```
```python
from better_human.studio import HumanStudio, launch    # no longer exported from better_human.viz
```

## Consequences
The library has no application-shaped code in `viz`. The studio can move out of the repository later at the cost of a
directory. It must work against the public viewer API, so its smoke test runs on the stand-in server (BetterRobot 0035), and on
the random-weight archive (0016) in the full tier.

## Alternatives
Keep `viz.studio` and fix its internals: the application stays inside the library's public surface. Move it to a repo
`apps/` folder or its own repository: the right end state if it grows, but the viewer API work (BetterRobot 0022) is needed either
way. Pinocchio's Python visualizers wrap external viewers (gepetto-viewer, meshcat) behind optional imports and ship no application.

## Compatibility
Breaks `better_human.viz.studio.*`. Nothing else imports it; users launch it by a different module path and pass the
data root explicitly.

## Enforcement
The BetterHuman edge tables (0011) in `tests/contract/test_layer_dependencies.py`: `studio` may import `ROOT`, `core`,
`families`, `io`, `mass`, `viz` and, from BetterRobot, only `lie`, `viewer`, `viewer.tools`; ruff `SLF001`; new
`tests/contract/test_studio_boundary.py` (no `_`-prefixed attribute of a BetterRobot or BetterHuman object in `studio`,
no write to `os.environ` in `src`); the smoke test `tests/studio/test_app.py` on the stand-in server.

## Open questions
None.
