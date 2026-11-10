"""The enforcement matrix of the contracts index equals the rules' ``Enforcement:`` paragraphs (BetterRobot C-DOC-3).

Each ``## H-<AREA>-<n> ... (MUST|SHOULD)`` heading of ``docs/contracts/rules.md`` has a paragraph starting with
``Enforcement:`` that names its enforcers, test paths (``tests/...py``, optionally ``::test_name``); the row of
``docs/contracts/index.md``'s matrix for that rule carries exactly the same set. Every rule has an enforcer and
every cited test exists, in the matrix and in the table of BetterRobot rules alike: a plan or a reviewer's eye is not
a check. Every module of ``tests/contract/`` is cited by the index, so a contract test always names its rule. A
strength word appears on a contracts page only in a rule's heading, so no rule escapes the matrix without an ID.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTRACTS = REPO / "docs" / "contracts"
RULES = CONTRACTS / "rules.md"
INDEX = CONTRACTS / "index.md"
RULE = re.compile(r"^## (?P<id>H-[A-Z]+-\d+) .+ \((?:MUST|SHOULD)\)$", re.MULTILINE)
TEST = re.compile(r"`(?P<path>tests/[\w/]+\.py)(?:::(?P<name>\w+))?`")
STRENGTH = re.compile(r"\b(?:MUST|SHOULD)\b")


def _enforcers(text: str) -> set[str]:
    return {f"{m['path']}::{m['name']}" if m["name"] else m["path"] for m in TEST.finditer(text)}


def _page_rules() -> dict[str, set[str]]:
    """Rule id -> the enforcers its ``Enforcement:`` paragraph names."""
    text = RULES.read_text()
    heads = list(RULE.finditer(text))
    rules = {}
    for i, head in enumerate(heads):
        body = text[head.end() : heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        paragraph = next((p for p in body.split("\n\n") if p.startswith("Enforcement:")), "")
        rules[head["id"]] = _enforcers(paragraph)
    return rules


def _index_section(title: str) -> str:
    return INDEX.read_text().split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def _matrix_rows() -> dict[str, set[str]]:
    rows = re.findall(r"^\| (H-[A-Z]+-\d+) \| (.*) \|$", _index_section("Enforcement matrix"), re.MULTILINE)
    return {rule: _enforcers(cell) for rule, cell in rows}


def test_every_rule_is_in_the_matrix_with_the_same_enforcers() -> None:
    """The matrix row of each rule names the enforcers its page names, and the matrix has no other row."""
    assert _matrix_rows() == _page_rules()


def test_every_rule_has_an_enforcer_that_exists() -> None:
    """A rule names at least one test, and every test the index or the rules page cites exists."""
    assert all(_page_rules().values()), [rule for rule, enforcers in _page_rules().items() if not enforcers]
    missing = []
    for match in TEST.finditer(RULES.read_text() + INDEX.read_text()):
        target = REPO / match["path"]
        if not target.is_file() or (match["name"] and not re.search(rf"^def {match['name']}\(", target.read_text(), re.M)):
            missing.append(match[0])
    assert not missing, f"cite an existing test: {sorted(set(missing))}"


def test_every_contract_test_is_cited_by_the_index() -> None:
    """A module of ``tests/contract/`` is in the matrix or in the table of BetterRobot rules."""
    cited = {match["path"] for match in TEST.finditer(INDEX.read_text())}
    modules = {path.relative_to(REPO).as_posix() for path in (REPO / "tests" / "contract").glob("test_*.py")}
    assert not modules - cited, f"name the rule each of these enforces in the index: {sorted(modules - cited)}"


def test_strength_words_sit_in_rule_headings() -> None:
    """Outside a rule heading, no contracts page (records aside) says MUST or SHOULD."""
    stray = [
        f"{page.name}:{number}"
        for page in sorted(CONTRACTS.glob("*.md"))
        for number, line in enumerate(page.read_text().splitlines(), start=1)
        if STRENGTH.search(line) and not RULE.match(line)
    ]
    assert not stray, f"give these rules an ID and an enforcer, or drop the strength word: {stray}"
