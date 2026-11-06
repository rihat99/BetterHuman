# Testing

BetterHuman has three test tiers, each one command run on this machine (decision record
`BetterRobot/docs/contracts/decisions/0033-test-tiers.md`). No test skips: a test that cannot run everywhere
carries a marker that the tier commands select on, and a selected test whose weights or device is missing fails.

| tier | command | runs | fails when |
|---|---|---|---|
| fast | `uv run pytest tests/contract -q -m "not slow and not licensed"` | the contract tests that need no weights | a test fails |
| full | `uv run pytest tests -q -m "not cuda and not licensed"` | every test that needs neither weights nor a GPU | a test fails |
| release | `uv run pytest tests -q -m "licensed and not cuda"`, then `uv run pytest tests -q -m cuda` with a free GPU visible | the tests that read licensed weights on the CPU, then the GPU tests | a test fails, or the weights or the device is missing |

`.pre-commit-config.yaml` defines the hooks: lint, the format check and the fast tier before a commit, the full tier
before a push.

## Markers

| marker | meaning | deselected by |
|---|---|---|
| `licensed` | reads licensed weights through `better_human.io.weight_path` | fast, full |
| `cuda` | needs a CUDA device | fast, full |
| `slow` | takes seconds | fast |

pytest runs strictly (`[tool.pytest.ini_options]` in `pyproject.toml`): an unknown marker or config key is an error,
a passing `xfail` fails, an empty parameter set fails at collection, and every warning is an error unless a named
filter with its reason lets it through. `tests/contract/test_pytest_config.py` and `tests/contract/test_no_skips.py`
hold these rules.

## Weights

`$BETTERHUMAN_MODELS_DIR` names the directory that holds the licensed weights; `better_human.io.models_dir()` is its
one reader and `better_human.io.weight_path(folder, filename)` names a file under it. When a run selects a
`licensed` test and the variable is unset or names no directory, collection stops with a usage error that names the
variable and the fix (`tests/conftest.py`); a file missing inside the directory fails the test that reads it.

```bash
export BETTERHUMAN_MODELS_DIR=/path/to/models
uv run pytest tests -q -m "licensed and not cuda"
```

The parity goldens are computed from licensed weights, so they live in the same directory, in
`$BETTERHUMAN_MODELS_DIR/goldens/<family>/` (decision record 0015), never in git. Tests reach them through
`tests/support/goldens.py::golden_dir`, called inside fixtures only, so collecting the fast and full tiers never needs
the directory, and `tests/contract/test_golden_provenance.py` checks that every golden stamps what produced it
(decision record 0014). `tools/parity/smplx/README.md`, `tools/parity/mhr/README.md` and
`tools/parity/soma/README.md` give the commands that regenerate them.

## Timing on a busy host

Load times of the weights depend on the host's memory state through transparent huge pages. NumPy advises the kernel
to back every array of 4 MiB or more with huge pages, and on a host whose `/sys/kernel/mm/transparent_hugepage`
settings are `madvise` and whose memory is fragmented, the first touch of such an array waits for the kernel to
compact memory. Reading the SMPL-X archive took 6 to 12 s that way, against 0.8 s with `NUMPY_MADVISE_HUGEPAGE=0`, in
the same hour on the same machine. A load-time gate is therefore a ratio measured in one process (the loader
against a raw read of the same archive, both in the same memory state), and a timing comparison records the
`NUMPY_MADVISE_HUGEPAGE` setting and the load average with its numbers. The library never changes this setting.

## Stale bytecode and the assertion hook

pytest caches each test module after rewriting its asserts, keyed by the interpreter, the pytest version and the
source file's modification time and size, but not by `enable_assertion_pass_hook`, so a cache written by an ordinary
run silently disables that hook in a later run. The parity margin recorder, which relies on the hook, therefore sets
its own `sys.pycache_prefix` to a fresh directory, and its command needs no flag for it.
