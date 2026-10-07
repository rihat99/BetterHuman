"""BetterHuman uses only BetterRobot's public surface."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).parents[2] / "src" / "better_human"
PRIVATE = re.compile(r"better_robot\.(io\.builders|io\.ir|data_model\.|_|kinematics\._|dynamics\._|optim2\._|viewer\._)")
IMPORT = re.compile(r"^\s*(from|import)\s+(better_robot[\w.]*)", re.MULTILINE)


def test_no_private_better_robot_imports() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        for match in IMPORT.finditer(path.read_text()):
            if PRIVATE.search(match.group(2)):
                offenders.append(f"{path.relative_to(SRC)}: {match.group(0).strip()}")
    assert not offenders, "\n".join(offenders)


def test_no_old_api_left() -> None:
    text = "\n".join(path.read_text() for path in SRC.rglob("*.py") if "studio" not in path.parts)
    for symbol in ("RenderContext", "ParametricBody", "better_robot.tasks", " Data,"):
        assert symbol not in text, symbol
