"""No test module spells a skip or an xfail, including the tests the running tier deselects.

The rule itself holds at run time:
BetterRobot's plugin ``better_robot.testing.no_skips`` (loaded by ``tests/conftest.py``) fails every skipped or xfailed
outcome however it was produced, but it sees only the tests a run selects, so a skip in a ``licensed`` or ``cuda``
test would surface only in the release tier. Until BetterRobot ships a shared scan, this static one reads every module
under ``tests/`` in the fast tier and finds the usual spellings: the ``pytest.skip``, ``pytest.importorskip`` and
``pytest.xfail`` calls, the ``pytest.mark.skip``, ``pytest.mark.skipif`` and ``pytest.mark.xfail`` markers, and
``from pytest import`` of the three calls. Other routes (``unittest.SkipTest``, a renamed ``pytest`` import, a skip
raised by a plugin) are left to the plugin.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).parents[1]
SKIP_CALLS = {"skip", "importorskip", "xfail"}
SKIP_MARKS = {"skip", "skipif", "xfail"}


def _offences(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "pytest":
            found += [(node.lineno, f"from {node.module} import {a.name}") for a in node.names if a.name in SKIP_CALLS]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "pytest":
            if node.attr in SKIP_CALLS:
                found.append((node.lineno, f"pytest.{node.attr}"))
        elif isinstance(node, ast.Attribute) and node.attr in SKIP_MARKS:
            inner = node.value
            if isinstance(inner, ast.Attribute) and inner.attr == "mark":
                found.append((node.lineno, f"mark.{node.attr}"))
    return found


def test_no_test_skips_or_expects_failure() -> None:
    offences = [
        f"{path.relative_to(TESTS.parent)}:{line}: {what}"
        for path in sorted(TESTS.rglob("*.py"))
        for line, what in _offences(ast.parse(path.read_text()))
    ]
    assert not offences, "skips and xfails under tests/ (mark the test licensed or cuda instead):\n" + "\n".join(
        offences
    )


def test_the_scan_sees_every_spelling() -> None:
    source = (
        "import pytest\nfrom pytest import skip, importorskip\n"
        "pytest.skip('x')\npytest.importorskip('x')\npytest.xfail('x')\n"
        "@pytest.mark.skip\ndef a(): pass\n@pytest.mark.skipif(True, reason='x')\ndef b(): pass\n"
        "@pytest.mark.xfail\ndef c(): pass\npytestmark = pytest.mark.skipif(True, reason='x')\n"
    )
    assert len(_offences(ast.parse(source))) == 9
