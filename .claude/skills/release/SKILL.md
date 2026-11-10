---
name: release
description: Prepare a BetterHuman release - public-API diff, changelog section, version, release tier. Use when a task prepares or checks a release; the tag, the push and the BetterRobot source stay with the owner.
---

# Prepare a BetterHuman release

Each step names the check that holds it; run them on a clean checkout of the commit to release.

1. Diff the public API against the last release's snapshot:
   `uv run python tools/api_snapshot.py --diff <old public_api.json>` (the 0.1 line's is
   `tests/contract/public_api_before_0.2.0.json`). Every removed path has its row in
   `docs/reference/migration-0.2.0.md` and its replacement in the changelog (`tests/contract/test_changelog.py`).
2. In `docs/reference/changelog.md`, the `## Unreleased` heading becomes `## vX.Y.Z (YYYY-MM-DD)` under a new
   `## Unreleased` heading (`tests/contract/test_changelog.py`).
3. Set `version` in `pyproject.toml`, `0.MINOR.PATCH` before 1.0, where a minor release may break (rule C-EXT-6),
   then `uv lock`: the lockfile records the version, and the hooks' `uv run --locked` fails without it.
4. Run the full tier, then the release tier with the weights in `$BETTERHUMAN_MODELS_DIR` and
   `NUMPY_MADVISE_HUGEPAGE=0` on a host with transparent huge pages (`docs/guides/run-the-tests.md`):
   `uv run pytest tests -q -m "licensed and not cuda"`, then on a GPU nobody else uses
   `CUDA_VISIBLE_DEVICES=<index> uv run pytest tests -q -m cuda`. Every licensed test passes; a missing weight
   file fails, never skips.
5. Each change is a commit of its own. The tag, the push and the BetterRobot source in `[tool.uv.sources]` of
   `pyproject.toml` are the owner's steps.
