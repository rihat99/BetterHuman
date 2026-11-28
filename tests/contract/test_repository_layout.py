"""The repository's tree: its top-level entries, a tests folder per src package, no tracked symlink or large file, no
ghost directory (principles 9.1, 9.2, 9.3, 9.6).

The top-level entries are fixed and every tests folder is a src package's or a suite's, so a second home for the
tests or a new top-level folder fails here; a second image folder under ``docs/`` does not. A tracked symlink over an
ignored directory replaces that directory on checkout (``models`` holds the licensed weights). BetterRobot's
``better_robot.testing.repository.layout_problems`` holds the rules; this module holds BetterHuman's data.
"""

from __future__ import annotations

from pathlib import Path

from better_robot.testing import repository

REPO = Path(__file__).resolve().parents[2]
LARGE_ON_PURPOSE: dict[str, str] = {}
# Principle 9.1's set without `.github/`: this major has no CI. `agent_notes/` is ignored whole.
TOP_LEVEL = {
    ".claude",
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


def test_the_tree_has_the_standard_layout() -> None:
    """The top-level entries are ``TOP_LEVEL``; each src package has its tests folder and each tests folder is a
    package's or a suite's; no tracked symlink, no file over 1 MiB outside ``LARGE_ON_PURPOSE``, no directory without
    a tracked file."""
    problems = repository.layout_problems(
        REPO, package="better_human", top_level=TOP_LEVEL, suites=SUITES, large_on_purpose=LARGE_ON_PURPOSE
    )
    assert not problems, "\n".join(problems)
