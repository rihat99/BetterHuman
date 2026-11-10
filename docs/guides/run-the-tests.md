# Run the tests

The test suite is the contract: parity tests pin the numbers against each family's upstream code, contract tests pin
the architecture, and docs tests keep every page runnable. This guide shows how to run it; [Test
suite](../reference/tests.md) lists what each group guards.

## Run a tier

`uv sync --extra studio` installs everything the tests need (the `dev` group comes by default). Each tier is one
command (BetterRobot decision record 0033); `.pre-commit-config.yaml` runs the fast tier before a commit and the full
tier before a push, and `uv run pre-commit install` installs the hooks.

| Tier | Command | Needs |
|------|---------|-------|
| fast | `uv run pytest tests/contract -q -m "not slow and not licensed"` | nothing |
| full | `uv run pytest tests -q -m "not cuda and not licensed"` | nothing: the docs, examples and random-weight parity cases read the archives in `tests/support/random_weights/` |
| release | `uv run pytest tests -q -m "licensed and not cuda"`, then `uv run pytest tests -q -m cuda` with a free GPU visible | the licensed weights, then a GPU |

While you edit one package, run the fast tier plus that package's folder, for example `uv run pytest tests/core -q`.

## Give the release tier the weights

The release tier reads the licensed weights and the parity goldens computed from them, both under
`$BETTERHUMAN_MODELS_DIR` (the goldens in `goldens/<family>/`, decision record
[0015](../contracts/decisions/0015-licensed-goldens-out-of-git.md)):

```bash
export BETTERHUMAN_MODELS_DIR=/path/to/models
uv run pytest tests -q -m "licensed and not cuda"
```

Nothing skips. A run that selects a `licensed` test while the variable is unset, or names no folder, stops at
collection with the fix; a file missing inside the folder fails the test that reads it. `tools/parity/smplx/`,
`tools/parity/mhr/` and `tools/parity/soma/` each have a README with the commands that regenerate their goldens.

## Speed up loads on a host with huge pages

On a host whose transparent huge pages are set to `madvise` and whose memory is fragmented, NumPy's advice to back
large arrays with huge pages makes the first touch of each array wait for the kernel to compact memory: reading the
SMPL-X archive took 6 to 12 s that way, against 0.8 s with the advice off, on the same machine in the same hour.
Turn it off for a run:

```bash
NUMPY_MADVISE_HUGEPAGE=0 uv run pytest tests -q -m "not cuda and not licensed"
```

The tier commands stay as written, and neither the library nor the tests change the setting. A timing you compare
records the setting and the load average with its numbers.

## Check the docs pages

`tests/docs/test_snippets.py` runs every `python` fence of each page and compares every `output` fence with what
the fence prints; a page runs on the random-weight archives with `$BETTERHUMAN_MODELS_DIR` unset, unless the test
lists it as needing the licensed weights:

```bash
uv run pytest tests/docs -q -k first-human                                        # one page
BR_DOCS_CAPTURE=1 BR_DOCS_CAPTURE_DIR=/tmp/out uv run pytest tests/docs -q -k first-human   # write real outputs
uv run pytest tests/docs -q -m licensed                                           # the pages that need weights
uv run mkdocs build --strict                                                      # broken links fail the build
```

The capture run writes `<page>.outputs.json` with each fence's output; copy it into the `output` fence.

## Never weaken a test

Do not skip, `xfail`, loosen a tolerance or delete an assertion to land a change. If a parity or contract test
fails, the change is wrong, or the contract changes on purpose: then the test, the docs and `CLAUDE.md` change
together, with a decision record when the change is one the [contracts index](../contracts/index.md) names.

## Where next

- [Test suite](../reference/tests.md): the markers, the parity tolerances and every contract test.
- [Contracts](../contracts/index.md): the rules the contract tests enforce.
