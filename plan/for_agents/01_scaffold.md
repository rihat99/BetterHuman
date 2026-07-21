# Order 01 — Clean slate, uv project, BetterRobot dependency, test skeleton

> **Agent log — 2026-07-20:** Completed on `dev`; `uv sync`, 2 smoke tests,
> and Ruff pass. The plan-named local `smplx_mid` leftovers were removed;
> `models/` and `references/` stayed unchanged. No plan deviations; see
> [`01_results.md`](01_results.md).

Read first: `docs/design_principles.md`, `docs/architecture.md`,
`docs/contracts.md`, `plan/README.md` (ground rules apply, especially:
never touch `models/` or `references/`).

## Context

The old library is dead: hand-rolled FK, `SMPLXmini`-style class forks,
conda-era tooling. Nothing in it is reused. This order clears the tree and
erects the uv-based project that every later order builds in.

## Deliverables

1. **Delete the old code** (git rm, evidence in results file):
   `src/better_human/` contents, `src/better_human.egg-info/`, `scripts/`,
   `tests/`, `temp/`. Untracked leftovers (e.g. `__pycache__`, uncommitted
   `smplx_mid` files from the old working tree) are removed too. `models/`,
   `references/`, `docs/`, `plan/`, `LICENSE`, `.vscode/` stay.
2. **New `pyproject.toml`** managed by uv:
   - project `better-human`, package `src/better_human/`, version `0.1.0.dev0`,
     `requires-python` matching BetterRobot's.
   - dependencies: `better-robot` (via `[tool.uv.sources] better-robot =
     { path = "../BetterRobot", editable = true }`), `numpy`, `torch`
     (torch index config copied from BetterRobot's pyproject so both
     projects resolve the same wheel family).
   - dev group: `pytest`, `ruff`. Copy BetterRobot's ruff configuration
     verbatim as the starting point.
   - Match BetterRobot's build backend and layout conventions; when in
     doubt, mirror `../BetterRobot/pyproject.toml`.
3. **Package skeleton**: `src/better_human/__init__.py` containing only
   `__version__`. No placeholder modules, no empty subpackages — packages
   appear when their first real code lands (design principle 3).
4. **Test skeleton**: `tests/test_smoke.py` proving the foundation wiring:
   - `import better_human` works;
   - `better_robot` imports from the editable path;
   - build the 24-joint human tree via
     `better_robot.io.builders.smpl_like.make_smpl_like_model`, run
     `forward_kinematics` at `q_neutral`, assert `nq == 99`, `nv == 75`,
     and world joint pose shape `(25, 7)`;
   - one batched FK call `(B=4)` asserting output batch shape.
5. **`.gitignore`**: prune stale conda-era entries, ensure `uv.lock` is
   tracked, `models/`, `temp/`, `references`, egg-info/caches ignored.
6. **README**: replace the conda instructions with the uv equivalents
   (`uv sync`, `uv run pytest`); keep it short — the full quickstart is
   order 06's job.
7. Commit `uv.lock`.

## Acceptance

- `uv sync` succeeds from a clean checkout on this host.
- `uv run pytest tests/ -q` green, including the BR smoke FK test.
- `git ls-files src scripts tests temp` shows only the new files (old tree
  gone from the index); `models/` and `references/` untouched (`git status`
  clean of them).
- `uv run ruff check src/ tests/` clean.

## Results file

Write `plan/for_agents/01_results.md`: exact deletion list, dependency
versions resolved, test count, deviations as findings.
