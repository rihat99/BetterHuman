"""The repository's tree: its top-level entries, a tests folder per src package, no tracked symlink or large file, no
ghost directory (principles 9.1, 9.2, 9.3, 9.6; decision record 0015).

Each kind of file has one home, so a second home fails here. A tracked symlink over an ignored directory replaces
that directory on checkout (``models`` holds the licensed weights), and a large file weighs on every clone forever:
a file that truly cannot shrink goes in ``LARGE_ON_PURPOSE`` with its reason. A directory with no tracked file in it
(only ``__pycache__`` or stale output) looks like a package to ``ls`` and ``grep``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = "src/better_human"
LIMIT = 1024 * 1024
LARGE_ON_PURPOSE: dict[str, str] = {}
# Principle 9.1's set without `.github/`: this major has no CI (BetterRobot record 0033). `agent_notes/` is ignored
# whole.
TOP_LEVEL = {
    ".gitignore",
    ".pre-commit-config.yaml",
    ".python-version",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "benchmarks",
    "docs",
    "examples",
    "mkdocs.yml",
    "pyproject.toml",
    "src",
    "tests",
    "tools",
    "uv.lock",
}
# The folders of tests/ that are not a src package: one per cross-cutting suite.
SUITES = {
    "contract": "rules as tests",
    "parity": "comparisons against an oracle, one folder per oracle",
    "docs": "every snippet of README.md and docs/, with its output",
    "examples": "every example, headless",
    "benchmarks": "the smoke test of benchmarks/run.py",
    "support": "fixtures, random-weight archives and helpers that only these tests import, never collected",
}
GHOST_ROOTS = ("src", "tests", "tools")


def _tracked() -> list[tuple[str, str]]:
    """``(mode, path)`` of every tracked file."""
    listing = subprocess.run(["git", "ls-files", "-s"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [(line.split()[0], line.split("\t", 1)[1]) for line in listing.splitlines()]


def _tracked_paths() -> list[str]:
    return [path for _, path in _tracked()]


def test_top_level_entries() -> None:
    """The tracked top-level entries are exactly ``TOP_LEVEL``."""
    entries = {path.split("/", 1)[0] for path in _tracked_paths()}
    extra, missing = sorted(entries - TOP_LEVEL), sorted(TOP_LEVEL - entries)
    assert not extra, f"move these into one of the standard homes (principle 9.1): {extra}"
    assert not missing, f"these standard top-level entries are missing (principle 9.1): {missing}"


def test_tests_mirror_src() -> None:
    """Every package under ``src/better_human/`` has a ``tests/`` folder of the same name, and every ``tests/`` folder
    is a package's or a suite's."""
    paths = _tracked_paths()
    packages = {path.split("/")[2] for path in paths if Path(path).match(f"{PACKAGE}/*/__init__.py")}
    folders = {path.split("/")[1] for path in paths if path.startswith("tests/") and path.count("/") >= 2}
    untested, unknown = sorted(packages - folders), sorted(folders - packages - SUITES.keys())
    assert not untested, f"give each of these src packages its tests/ folder (principle 9.2): {untested}"
    assert not unknown, (
        "these tests/ folders are neither a src package nor a suite of SUITES; move their tests into the folder of "
        f"the package they test or of their suite (principle 9.2): {unknown}"
    )


def test_no_symlinks_and_size() -> None:
    """Every tracked path is a regular file of at most 1 MiB, or on the reasoned list."""
    tracked = _tracked()
    symlinks = [path for mode, path in tracked if mode == "120000"]
    large = [
        f"{path}: {(REPO / path).stat().st_size} B"
        for mode, path in tracked
        if mode != "120000" and path not in LARGE_ON_PURPOSE and (REPO / path).stat().st_size > LIMIT
    ]
    assert not symlinks, f"replace these tracked symlinks by files or an ignore line (principle 9.3): {symlinks}"
    assert not large, "shrink these under 1 MiB (or list one in LARGE_ON_PURPOSE with its reason):\n  " + "\n  ".join(
        large
    )


def test_no_ghost_directories() -> None:
    """Every directory under ``src/``, ``tests/`` and ``tools/`` on disk, ``__pycache__`` aside, holds a tracked file."""
    holding = {str(Path(path).parent) for path in _tracked_paths()}
    holding |= {str(parent) for folder in list(holding) for parent in Path(folder).parents}
    ghosts = []
    for root in GHOST_ROOTS:
        for folder, subfolders, _ in os.walk(REPO / root):
            subfolders[:] = [name for name in subfolders if name != "__pycache__"]
            relative = Path(folder).relative_to(REPO).as_posix()
            if relative not in holding:
                ghosts.append(relative)
    assert not ghosts, f"move these directories, which hold no tracked file, to the trash (principle 9.6): {ghosts}"
