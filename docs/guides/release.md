# Release a version

A release is one commit and one tag. The commit raises the version and dates the changelog section, the release
tier passes on it, and the tag names it. BetterHuman depends on one BetterRobot commit, so the release fixes that
commit too. Run the steps in order on a clean checkout of the commit to release. Version 0.2.0 is the first release
with a tag: the 0.1 line has none.

## Move the BetterRobot pin

BetterHuman depends on one BetterRobot commit, named under `[tool.uv.sources]` in its `pyproject.toml`:
`better-robot = { git = "https://github.com/rihat99/BetterRobot", rev = "<sha>" }`. `uv sync` installs that commit
into `.venv`; no BetterRobot checkout is needed. A pin names a tagged BetterRobot release, never a branch. To move
it, set `rev` to the commit of the release's tag, which `git ls-remote` prints on the line ending `^{}`:

```bash
git ls-remote https://github.com/rihat99/BetterRobot "refs/tags/v0.3.0^{}"
```

Then record the revision in the lockfile, check that the import comes from it, and run the full tier on it, in a
commit of its own before the release commit:

```bash
uv lock
uv run python -c "import better_robot; print(better_robot.__file__, better_robot.__version__)"
uv run pytest tests -q -m "not cuda and not licensed"
```

The printed file lies under `.venv`. When the new revision lowers a counter of the benchmark rows, re-measure the
thresholds in a commit of their own ([Run the benchmarks](run-the-benchmarks.md)).

## Review the public-API diff

`tools/api_snapshot.py --diff` prints the public paths added (`+`), removed (`-`) and changed (`~`) since an older
snapshot. The older snapshot is the last release's `tests/contract/public_api.json`. For 0.2.0 it is
`tests/contract/public_api_before_0.2.0.json`, the public tree of the 0.1 line, which has no tag:

```bash
# for 0.2.0
uv run python tools/api_snapshot.py --diff tests/contract/public_api_before_0.2.0.json
# for the release after it
git show v0.2.0:tests/contract/public_api.json > /tmp/public_api_0.2.0.json
uv run python tools/api_snapshot.py --diff /tmp/public_api_0.2.0.json
```

Every `-` path is a Removed entry of the changelog that names its replacement, and a row of the migration page;
every `+` path has a docstring and an Added entry. `tests/contract/test_changelog.py` checks the Removed entries
and the 0.2.0 migration page; a reviewer checks the Added entries and the docstrings. Before 1.0 a minor release
may change or remove a public name, and keeps no deprecated alias ([Public surface](../contracts/public-surface.md)).

## Raise the version and close the changelog section

One commit, `Release: X.Y.Z`, holds three changes:

1. `version` in `pyproject.toml` becomes `0.MINOR.PATCH`; `better_human.__version__` reads it from the installed
   package.
2. In `docs/reference/changelog.md`, the `## Unreleased` heading becomes `## vX.Y.Z (YYYY-MM-DD)` with the day of
   the release, and a new, empty `## Unreleased` heading opens the file (`tests/contract/test_changelog.py`).
3. `uv lock` writes the version into `uv.lock`. The release tier starts with `uv lock --check` and the hooks run
   through `uv run --locked`; both fail on a lockfile that holds the old version.

```bash
uv lock
uv run python -c "import better_human; print(better_human.__version__)"   # now X.Y.Z
```

## Run the release tier

On the release commit, with the licensed weights in `$BETTERHUMAN_MODELS_DIR` and the `nvidia-smi` index of a GPU
nobody else uses:

```bash
bash tools/release_tier.sh --gpu <index>   # ends with "== release tier passed"
```

It stops at its first failure; [Run the tests](run-the-tests.md) lists its steps, how to give it the weights, and
the `NUMPY_MADVISE_HUGEPAGE=0` setting a host with transparent huge pages needs. Its `licensed` tests read every
family's weights and the parity goldens ([Load a family](load-a-family.md) names the files): a missing file fails
its test, never skips, so without the MANO weights and the original SMPL pickle four licensed tests fail and the
tier does not pass. It tests the Python, torch and BetterRobot of `uv.lock` only; no tier tests the lower
bounds in `pyproject.toml`. Its full tier fails on any warning a test raises, deprecations included
(`filterwarnings` starts with `error`, `tests/contract/test_pytest_config.py`).

## Tag the release

The owner tags the release commit with the changelog heading's name and pushes the tag:

```bash
git tag -a v0.2.0 -m "BetterHuman 0.2.0"
git push origin v0.2.0
```

## Once per clone

A clone does not install the hooks of `.pre-commit-config.yaml`. Install both stages once; the worktrees of a clone
share its hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Where next

- [Changelog](../reference/changelog.md) and [Migrating to 0.2.0](../reference/migration-0.2.0.md): what each
  release changed, and the replacement of every renamed or removed name.
- [Run the tests](run-the-tests.md): the three tiers and what the release tier needs.
- [Public surface](../contracts/public-surface.md): what a release promises.
