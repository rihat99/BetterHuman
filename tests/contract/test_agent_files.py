"""The agent files stay short, plain and shared (principles 10.1, 10.2, 10.3).

``CLAUDE.md`` at the root is the agent file, at most 200 lines; a nested ``CLAUDE.md`` holds the traps of its folder
in at most 60. ``AGENTS.md`` is one line that sends every other agent to ``CLAUDE.md``, and no other tool's
instruction file exists. A rule that must hold is a check, so no agent file or skill uses a word in capitals for
emphasis. The shared agent configuration (``.claude/settings.json``, ``.claude/rules/``, ``.claude/skills/``) is
tracked, so every contributor's agent reads the same; the rest of ``.claude/`` is per machine and ignored.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_LIMIT = 200
NESTED_LIMIT = 60
EMPHASIS = re.compile(r"\b(?:ALWAYS|NEVER|IMPORTANT|CRITICAL|MUST|SHOULD|REQUIRED|NOTE|WARNING|DO|NOT|NO|ONLY)\b")
# Paths in parts: none of them may exist, and a whole path in tracked text is a citation that must resolve.
OTHER_INSTRUCTION_FILES = (Path(".cursorrules"), Path(".cursor", "rules"), Path(".github", "copilot-instructions.md"))
SHARED = ("settings.json", "rules/example", "skills/example")  # under .claude/
PER_MACHINE = ("settings.local.json", "autoresearch")  # under .claude/


def _git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()


def _agent_files() -> list[str]:
    """Every ``CLAUDE.md`` and every file of ``.claude/``, tracked or new, ignored ones aside."""
    listed = set(_git("ls-files", "--cached", "--others", "--exclude-standard"))
    return sorted(
        path
        for path in listed
        if (path.endswith("CLAUDE.md") or path.startswith(".claude/")) and (REPO / path).is_file()
    )


def test_agent_files_are_short() -> None:
    """The root ``CLAUDE.md`` has at most 200 lines and a nested one at most 60."""
    long = []
    for path in _agent_files():
        if not path.endswith("CLAUDE.md"):
            continue
        limit = ROOT_LIMIT if path == "CLAUDE.md" else NESTED_LIMIT
        count = len((REPO / path).read_text().splitlines())
        if count > limit:
            long.append(f"{path}: {count} lines, limit {limit}")
    assert not long, (
        "move procedures into .claude/skills/ and design notes into module docstrings, and drop what the code shows "
        "(principle 10.2):\n  " + "\n  ".join(long)
    )


def test_no_capitals_for_emphasis() -> None:
    """No word in capitals such as ``NEVER`` or ``IMPORTANT`` in an agent file or skill."""
    found = [
        f"{path}: {match[0]}"
        for path in _agent_files()
        if path.endswith(".md")
        for match in EMPHASIS.finditer((REPO / path).read_text())
    ]
    assert not found, (
        "write the word in lower case; a rule that must hold becomes a contract test (principles 6.1, 10.2):\n  "
        + "\n  ".join(found)
    )


def test_one_agent_file() -> None:
    """``AGENTS.md`` is one line that names ``CLAUDE.md`` only, and no other tool's instruction file exists."""
    lines = (REPO / "AGENTS.md").read_text().splitlines()
    named = set(re.findall(r"[\w./-]+\.md\b", "\n".join(lines)))
    others = [str(path) for path in OTHER_INSTRUCTION_FILES if (REPO / path).exists()]
    assert len(lines) == 1 and named == {"CLAUDE.md"}, f"AGENTS.md is one line that names CLAUDE.md only: {lines}"
    assert not others, f"fold these into CLAUDE.md and remove them (principle 10.1): {others}"


def test_shared_agent_configuration_is_tracked() -> None:
    """``.gitignore`` admits exactly the shared parts of ``.claude/``, and each of their files on disk is tracked."""
    # check-ignore exits 1 when it ignores none of the paths; only 128 is an error.
    probe = subprocess.run(
        ["git", "check-ignore", "--no-index", *SHARED, *PER_MACHINE],
        cwd=REPO / ".claude",
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode in (0, 1), probe.stderr
    ignored = set(probe.stdout.splitlines())
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", ".claude")
    assert ignored == set(PER_MACHINE), (
        f".gitignore admits {list(SHARED)} under .claude/ and ignores the rest (principle 10.3); it ignores "
        f"{sorted(ignored)}"
    )
    assert not untracked, f"commit these shared agent files, or move them out of .claude/ (principle 10.3): {untracked}"
