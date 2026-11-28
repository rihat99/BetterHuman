"""The agent files stay short, plain and shared (principles 10.1, 10.2, 10.3).

``CLAUDE.md`` at the root is the agent file, at most 200 lines; a nested ``CLAUDE.md`` holds the traps of its folder
in at most 60. ``AGENTS.md`` at the root is one line that sends every other agent to ``CLAUDE.md``, and no other
tool's instruction file exists at any depth, a nested ``AGENTS.md`` included. A rule that must hold is a check, so no
agent file or skill uses a word in capitals for emphasis. The shared agent configuration (``.claude/settings.json``,
``.claude/rules/``, ``.claude/skills/``) is tracked; the rest of ``.claude/`` is per machine and ignored.
BetterRobot's ``better_robot.testing.repository.agent_file_problems`` holds the rules.
"""

from __future__ import annotations

from pathlib import Path

from better_robot.testing import repository

REPO = Path(__file__).resolve().parents[2]


def test_agent_files_are_short_plain_and_shared() -> None:
    problems = repository.agent_file_problems(REPO)
    assert not problems, "\n".join(problems)
