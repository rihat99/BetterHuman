# Run the tests

The test suite is the contract: parity tests pin the numbers against each family's upstream code, contract tests pin
the architecture, and docs tests keep every page runnable. This guide shows how to run it; [Test
suite](../reference/tests.md) lists what each group guards.

## Run a tier

`uv sync --extra studio` installs everything the tests need (the `dev` group comes by default). Each tier is one
command:

| Tier | Command | Time | Run by |
|------|---------|------|--------|
| fast | `uv run pytest tests/contract -q -m "not slow and not cuda and not licensed"` | about 7 s | the pre-commit hook, after every edit |
| full | `uv run pytest tests -q -m "not cuda and not licensed"` | about 105 s | the pre-push hook, the release tier |
| release | `bash tools/release_tier.sh --gpu <index>` | about 7.5 min | by hand before a tag, with the weights and a free GPU |

The fast and full tiers need nothing: the docs, examples and random-weight parity cases read the archives in
`tests/support/random_weights/`. Measured at commit `ae30f6e` on a shared AMD EPYC 9654 host (384 threads), load
average 16 to 113, with `NUMPY_MADVISE_HUGEPAGE=0` (below): fast 6.8 s and full 103 to 106 s as pytest reports them
(`uv run` adds about 2.5 s), of which the counter gate `tests/contract/test_compile_counters.py` takes about 50 s
([Run the benchmarks](run-the-benchmarks.md)); the release tier's steps 461 s in all, of which its licensed step
took 184 s. Without the setting the full tier took 214 s.

The release tier stops at its first failure. It checks the lockfile (`uv lock --check`) and that torch can use GPU
`<index>`, the `nvidia-smi` index of a GPU nobody else is using (the run stops there when torch cannot). Then it runs
ruff check and ruff format --check, the public-API snapshot check (`tools/api_snapshot.py --check`), the strict docs
build, the `cuda` tests and the CUDA counters (`benchmarks/run.py --device cuda --counters-only`) on that GPU, the
full tier, the full tier again with BetterRobot's internal checks on (`BETTER_ROBOT_INTERNAL_CHECKS=1`; the tests
leave them off, as users run), and last the `licensed` tests (`-m "licensed and not cuda"`). The `cuda` and
`licensed` steps read the weights under `$BETTERHUMAN_MODELS_DIR`.

While you edit one package, run the fast tier plus that package's folder, for example `uv run pytest tests/core -q`.
The hooks live in `.pre-commit-config.yaml`, BetterRobot's file but for the two marker expressions, and run through
`uv run --locked`, so a lockfile that does not match `pyproject.toml` fails them: ruff check, ruff format --check
and the fast tier before each commit, the full tier before each push. `uv run pre-commit install` installs both
stages; `uv run pre-commit run --all-files --hook-stage pre-push` runs one stage without installing it.

## Give the release tier the weights

The release tier reads the licensed weights and the parity goldens computed from them, both under
`$BETTERHUMAN_MODELS_DIR` (the goldens in `goldens/<family>/`):

```bash
export BETTERHUMAN_MODELS_DIR=/path/to/models
bash tools/release_tier.sh --gpu <index>
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
together, and the owner decides when the change is one the [contracts index](../contracts/index.md) names.

## Where next

- [Test suite](../reference/tests.md): the markers, the parity tolerances and every contract test.
- [Contracts](../contracts/index.md): the rules the contract tests enforce.
- [Run the benchmarks](run-the-benchmarks.md): the counter gate the full tier runs, and wall time.
