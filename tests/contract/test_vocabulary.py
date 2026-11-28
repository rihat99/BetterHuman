"""Four vocabulary rules of chapter 3 over BetterHuman's ``__all__`` tree (principles 3.4, 3.5, 3.8, 3.9).

- 3.4: one name per thing. No public class or function is listed under two names, or bound by a listed module under
  a name other than its own (``core.Spec = FamilySpec``, listed or not), and no public class reuses a class name of a
  library ``src`` imports (``LIBRARIES``, BetterRobot's public tree among them). ``FIELDS_HOLDING_A_CLASS`` holds the
  permanent, reasoned exceptions.
- 3.5: an acronym is written in capitals inside CapWords (``SMPLXClassic``, ``MHRRig``), never ``SmplxClassic``.
- 3.8: no module is named ``utils``, ``common``, ``misc`` or ``helpers``.
- 3.9: every word of every public name, method, field and parameter is a term or an everyday word of
  ``docs/reference/glossary.md`` or of BetterRobot's glossary, which ``better_robot.testing.glossary.read()`` reads
  from the installed BetterRobot. A word of BetterRobot's page keeps its meaning:
  BetterHuman's page defines it only in its table of words BetterRobot uses with another meaning.

BetterRobot's ``better_robot.testing.naming_problems`` checks 3.4's two names, 3.5 and 3.8, and
``glossary.unknown_words`` 3.9; the reused class names and the glossary's own rules stay here.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from better_human import core
from better_robot.testing import glossary, naming_problems, public_objects

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "reference" / "glossary.md"
COLLISIONS = "## Words BetterRobot uses with another meaning"
# The third-party packages src imports; BetterRobot's public classes join them in _library_class_names.
LIBRARIES = ("torch", "numpy", "viser")
FIELDS_HOLDING_A_CLASS = {
    "classic_type": "a ClassicFormat field whose value is the family's classic dataclass, not a second name for it",
}
ACRONYMS = ("MHR", "SOMA", "SMPL", "SMPLH", "SMPLX", "MANO", "FLAME", "MLP")


def _library_class_names() -> dict[str, str]:
    """Class name to the first library in ``LIBRARIES``, then ``better_robot``, whose namespace lists it."""
    names: dict[str, str] = {}
    for library in LIBRARIES:
        module = importlib.import_module(library)
        for name in dir(module):
            if not name.startswith("_") and inspect.isclass(getattr(module, name, None)):
                names.setdefault(name, library)
    for value in public_objects("better_robot").values():
        if inspect.isclass(value):
            names.setdefault(value.__name__, "better_robot")
    return names


def test_naming_rules() -> None:
    """One name per public object, acronyms in capitals inside class names, no module named utils, common, misc or
    helpers."""
    problems = naming_problems("better_human", acronyms=ACRONYMS, ignore=FIELDS_HOLDING_A_CLASS)
    assert not problems, "\n".join(problems)


def test_an_unlisted_alias_in_a_subpackage_is_a_second_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "Spec", core.FamilySpec, raising=False)
    assert naming_problems("better_human", acronyms=ACRONYMS, ignore=FIELDS_HOLDING_A_CLASS) == [
        "better_human.core.Spec = FamilySpec: a second name of a public object; remove it (principle 3.4)"
    ]


def test_no_public_class_reuses_a_library_class_name() -> None:
    taken = _library_class_names()
    reused = sorted(
        f"{value.__name__} (also {taken[value.__name__]}.{value.__name__})"
        for path, value in public_objects("better_human").items()
        if inspect.isclass(value) and value.__name__ in taken and path.rsplit(".", 1)[-1] not in FIELDS_HOLDING_A_CLASS
    )
    assert not reused, "rename these public classes (principle 3.4):\n  " + "\n  ".join(reused)


def _vocabulary() -> set[str]:
    """BetterRobot's terms and everyday words and BetterHuman's: the words a public name may use."""
    (robot_terms, robot_everyday), (human_terms, human_everyday) = glossary.read(), glossary.read(PAGE)
    return robot_terms | set(robot_everyday) | human_terms | set(human_everyday)


def test_glossary_words() -> None:
    missing = glossary.unknown_words("better_human", _vocabulary())
    assert not missing, (
        "define these words in docs/reference/glossary.md, or rename (principle 3.9):\n  " + "\n  ".join(missing)
    )


def test_glossary_lists_each_everyday_word_once_and_only_while_a_name_uses_it() -> None:
    terms, everyday = glossary.read(PAGE)
    used = glossary.used_entries("better_human", _vocabulary())
    assert everyday == sorted(set(everyday)), "keep the everyday words sorted, each once"
    assert sorted(set(everyday) & terms) == [], "a word is a term or an everyday word, not both"
    assert sorted(set(everyday) - used) == [], "no public name uses these everyday words; remove them"


def _collision_terms(tmp_path: Path) -> set[str]:
    """The terms of the page's table of words BetterRobot uses with another meaning, read as a page of its own."""
    section = PAGE.read_text(encoding="utf-8").split(f"\n{COLLISIONS}\n", 1)[1].split("\n## ", 1)[0]
    page = tmp_path / "collisions.md"
    page.write_text(section, encoding="utf-8")
    return glossary.read(page)[0]


def test_a_betterrobot_word_keeps_its_meaning_unless_listed_with_another(tmp_path: Path) -> None:
    robot_terms, robot_everyday = glossary.read()
    robot = robot_terms | set(robot_everyday)
    terms, everyday = glossary.read(PAGE)
    assert sorted(word for word in everyday if glossary.entry(word, robot)) == [], (
        "BetterRobot's glossary lists these words; remove them from BetterHuman's everyday words"
    )
    redefined, listed = {term for term in terms if glossary.entry(term, robot)}, _collision_terms(tmp_path)
    assert sorted(redefined - listed) == [], (
        f"BetterRobot's glossary defines these words: use them in its meaning, or move the row under {COLLISIONS!r}"
    )
    assert sorted(listed - redefined) == [], (
        f"BetterRobot's glossary does not list these words; move the row out of {COLLISIONS!r}"
    )
