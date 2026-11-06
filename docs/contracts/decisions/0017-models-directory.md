# 0017 One way to find the weights: `io.models_dir()`

**Status:** Accepted (owner). **Date:** 2026-11-01

## Context
One weights directory has four lookup rules: the library's `$BETTERHUMAN_MODELS_DIR`, BetterHuman tests'
`Path(__file__).parents[2] / "models"` (22 files), BetterVideoReconstruction's `Path(bh.__file__).parents[2]`
fallback, and a hard-coded absolute path in BetterRetarget. In a worktree with no `models/` directory 204 of 275
BetterHuman tests skip and the run is green, so no parity gate can be checked there.

## Decision
We will provide `io.models_dir() -> Path`, which reads `$BETTERHUMAN_MODELS_DIR` and raises with the fix when it is
unset, and `io.weight_path(folder, filename, *, path=None)`, which replaces `resolve_model_path`, and make tests,
examples, BetterVideoReconstruction and BetterRetarget use them.

## Example
```python
from better_human import io
io.models_dir()                                    # Path("/data/models"); raises AssetNotFoundError("set BETTERHUMAN_MODELS_DIR") if unset
io.weight_path("smplx", "SMPLX_NEUTRAL.npz")       # models_dir() / "smplx" / "SMPLX_NEUTRAL.npz", or `path` when given
```

## Consequences
A missing weights directory is an error with the fix, never a green skip. The goldens (0015) and the weights share one
root. Every machine sets one environment variable. Nothing guesses a location.

## Alternatives
As above plus a default such as `~/.cache/better_human/models` when the variable is unset: a guess about where the
user keeps licensed files, and a silent skip again. smplx requires the model folder as an explicit argument
(`model_path`) and never searches.

## Compatibility
Breaks `resolve_model_path` (renamed `weight_path`) and every hand-built weights path in tests and apps. Reversible: a
default can be added later.

## Enforcement
New `tests/contract/test_models_dir.py`: no `Path(__file__).parents[...] / "models"` and no `better_human.__file__`
in `tests`, `examples` or `tools`; the `licensed` collection guard in `tests/conftest.py` fails, not skips, when the
models directory is unset (BetterRobot 0033); the same grep runs in BetterVideoReconstruction and BetterRetarget.

## Open questions
None.
