---
name: release
description: Prepare a BetterHuman release - BetterRobot pin, public-API diff, version, release tier. Use when a task prepares or checks a release; the tag and the push stay with the owner.
---

# Prepare a BetterHuman release

A release is one commit and one tag. The commit pins BetterRobot and raises the version, the release tier passes on
it, and the tag names it. Run the steps in order on a clean checkout of the commit to release; each step names the
check that holds it. The tag, the push and the BetterRobot revision are the owner's steps. No version has a tag yet:
the first release is 0.1.0, the version `pyproject.toml` already holds, and its tag will be `v0.1.0`.

## Pin BetterRobot by revision

Between releases `[tool.uv.sources]` in `pyproject.toml` pins a commit of BetterRobot's `dev` branch, or reads the
checkout beside this one while both libraries change together:
`better-robot = { path = "../BetterRobot", editable = true }`. A release names one tagged BetterRobot commit, never a
branch or a path:
`better-robot = { git = "https://github.com/rihat99/BetterRobot", rev = "<sha>" }`. Once the owner has pushed the
BetterRobot tag, `git ls-remote` prints its commit on the line ending `^{}`:

```bash
git ls-remote https://github.com/rihat99/BetterRobot "refs/tags/v0.1.0^{}"
```

Set `rev` to it, record it in the lockfile, check that the import comes from it, and run the full tier, in a commit
of its own before the release commit:

```bash
uv lock
uv run python -c "import better_robot; print(better_robot.__file__, better_robot.__version__)"   # a file under .venv
uv run pytest tests -q -m "not cuda and not licensed"
```

When the new revision lowers a counter of the benchmark rows, re-measure the thresholds in a commit of their own
(`benchmarks/run.py --counters-only`; its docstring says how).

## Review the public-API diff

BetterRobot's snapshot command with `--diff` prints the public paths added (`+`), removed (`-`) and changed (`~`)
since an older snapshot. The older snapshot is the last release tag's `tests/contract/public_api.json`; 0.1.0, the
first release, has none to compare with:

```bash
# for the release after 0.1.0
git show v0.1.0:tests/contract/public_api.json > /tmp/public_api_0.1.0.json
uv run python -m better_robot.testing.api_snapshot better_human /tmp/public_api_0.1.0.json --diff
```

Every `+` path has a docstring, and every `-` or `~` path came with a commit message that names its replacement; a
reviewer checks the messages. Before 1.0 a minor release may change or remove a public name and keeps no deprecated
alias (`docs/contracts/public-surface.md`).

## Raise the version

One commit, `Release: X.Y.Z`, holds two changes; 0.1.0 needs neither, since both files already hold it:

1. `version` in `pyproject.toml` becomes `0.MINOR.PATCH`; `better_human.__version__` reads it from the installed
   package.
2. `uv lock` writes the version into `uv.lock`. The release tier starts with `uv lock --check` and the hooks run
   through `uv run --locked`; both fail on a lockfile that holds the old version.

```bash
uv lock
uv run python -c "import better_human; print(better_human.__version__)"   # now X.Y.Z
```

## Run the release tier

On the release commit, with the licensed weights in `$BETTERHUMAN_MODELS_DIR` and the `nvidia-smi` index of a GPU
nobody else uses:

```bash
NUMPY_MADVISE_HUGEPAGE=0 bash tools/release_tier.sh --gpu <index>   # ends with "== release tier passed"
```

It stops at its first failure; the script lists its steps. Its `licensed` tests read every family's weights and the
parity goldens: a missing file fails its test, never skips, so without the MANO weights and the original SMPL pickle
four licensed tests fail and the tier does not pass. It tests the Python, torch and BetterRobot of `uv.lock` only;
no tier tests the lower bounds in `pyproject.toml`. Its full tier fails on any warning a test raises, deprecations
included (`filterwarnings` starts with `error`, `tests/contract/test_pytest_config.py`).

## Tag the release (the owner)

The owner tags the release commit `vX.Y.Z` and pushes the tag:

```bash
git tag -a v0.1.0 -m "BetterHuman 0.1.0"
git push origin v0.1.0
```

## Once per clone

A clone does not install the hooks of `.pre-commit-config.yaml`. Install both stages once; the worktrees of a clone
share its hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```
