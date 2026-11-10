"""Four vocabulary rules of chapter 3 over BetterHuman's ``__all__`` tree (principles 3.4, 3.5, 3.8, 3.9).

- 3.4: one name per thing. No public class or function is listed under two names, or bound by a listed module under
  a name other than its own (``core.Spec = FamilySpec``, listed or not), and no public class reuses a class name of a
  library ``src`` imports (``LIBRARIES``, BetterRobot's public tree among them). ``FIELDS_HOLDING_A_CLASS`` holds the
  permanent, reasoned exceptions.
- 3.5: an acronym is written in capitals inside CapWords (``SMPLXClassic``, ``MHRRig``), never ``SmplxClassic``.
- 3.8: no module is named ``utils``, ``common``, ``misc`` or ``helpers``.
- 3.9: every word of every public name, method, field and parameter is a term or an everyday word of
  ``docs/reference/glossary.md`` or of BetterRobot's glossary, which ``better_robot.testing.glossary.read()`` reads
  from the installed BetterRobot (BetterRobot decision record 0027). A word of BetterRobot's page keeps its meaning:
  BetterHuman's page defines it only in its table of words BetterRobot uses with another meaning.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections import defaultdict
from pathlib import Path

import pytest

import better_human
from better_human import core
from better_robot.testing import glossary, public_objects

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_human"
PAGE = REPO / "docs" / "reference" / "glossary.md"
COLLISIONS = "## Words BetterRobot uses with another meaning"
# The root __all__ lists families and io only; core, mass, terms, viz and studio are public subpackages too
# (decision record 0011), so each is walked as the root would walk it (tests/contract/test_api_snapshot.py does the same).
PACKAGES = ["better_human"] + [
    f"better_human.{info.name}" for info in pkgutil.iter_modules(better_human.__path__) if info.name[0] != "_"
]
# The third-party packages src imports; BetterRobot's public classes join them in _library_class_names.
LIBRARIES = ("torch", "numpy", "viser")
FIELDS_HOLDING_A_CLASS = {
    "classic_type": "a ClassicFormat field whose value is the family's classic dataclass, not a second name for it",
}
ACRONYMS = ("MHR", "SOMA", "SMPL", "SMPLH", "SMPLX", "MANO", "FLAME", "MLP")
BANNED_MODULES = {"utils", "common", "misc", "helpers"}


def _public_tree() -> dict[str, object]:
    return {path: value for package in PACKAGES for path, value in public_objects(package).items()}


def _public_classes_and_functions() -> dict[str, object]:
    return {
        path: value
        for path, value in _public_tree().items()
        if (inspect.isclass(value) or inspect.isfunction(value))
        and path.rsplit(".", 1)[-1] not in FIELDS_HOLDING_A_CLASS
    }


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


def test_no_public_object_is_listed_under_two_names() -> None:
    names: defaultdict[int, set[str]] = defaultdict(set)
    for path, value in _public_classes_and_functions().items():
        names[id(value)].add(path.rsplit(".", 1)[-1])
    assert sorted(sorted(found) for found in names.values() if len(found) > 1) == []


def _second_names() -> list[str]:
    """``module.name = Object`` for each BetterHuman class or function a listed module binds under a public name that
    is not the object's own."""
    return sorted(
        f"{path}.{name} = {value.__name__}"
        for path, module in _public_tree().items()
        if inspect.ismodule(module)
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and value.__module__.startswith("better_human")
        and value.__name__ != name
    )


def test_no_module_binds_a_public_object_under_a_second_name() -> None:
    assert _second_names() == []


def test_an_unlisted_alias_in_a_subpackage_is_a_second_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "Spec", core.FamilySpec, raising=False)
    assert _second_names() == ["better_human.core.Spec = FamilySpec"]


def test_no_public_class_reuses_a_library_class_name() -> None:
    taken = _library_class_names()
    reused = sorted(
        f"{value.__name__} (also {taken[value.__name__]}.{value.__name__})"
        for value in _public_classes_and_functions().values()
        if inspect.isclass(value) and value.__name__ in taken
    )
    assert not reused, "rename these public classes (principle 3.4):\n  " + "\n  ".join(reused)


def test_acronyms_are_capitals_inside_class_names() -> None:
    wrong = []
    for value in _public_classes_and_functions().values():
        if not inspect.isclass(value):
            continue
        for acronym in ACRONYMS:
            # An acronym word: starts with a capital and ends at the next capital or the end of the name.
            for match in re.finditer(rf"(?=({acronym}))", value.__name__, flags=re.IGNORECASE):
                start, text = match.start(), match.group(1)
                end = start + len(text)
                at_word = text[0].isupper() and (end == len(value.__name__) or value.__name__[end].isupper())
                if at_word and text != acronym:
                    wrong.append(f"{value.__name__}: write {acronym}")
    assert wrong == []


def test_no_module_has_a_name_that_says_nothing() -> None:
    names = {path.relative_to(SRC).as_posix() for path in SRC.rglob("*") if path.suffix == ".py" or path.is_dir()}
    vague = sorted(path for path in names if Path(path).stem.removeprefix("_") in BANNED_MODULES)
    assert vague == [], "name the module after what it computes (principle 3.8)"


def _public_names() -> dict[str, str]:
    """Every public name, method, field and parameter, mapped to the first public path that shows it."""
    names: dict[str, str] = {}
    for path, value in _public_tree().items():
        names.setdefault(path.rsplit(".", 1)[-1], path)
        function = value.__func__ if isinstance(value, staticmethod | classmethod) else value
        # A class whose constructor is a builtin's (an exception, an abstract base) has no parameters of its own.
        if inspect.isfunction(function) or (inspect.isclass(function) and inspect.isfunction(function.__init__)):
            for parameter in inspect.signature(function).parameters:
                if parameter not in ("self", "cls") and not parameter.startswith("_"):
                    names.setdefault(parameter, f"{path}({parameter}=)")
    return names


def _vocabulary() -> set[str]:
    """BetterRobot's terms and everyday words and BetterHuman's: the words a public name may use."""
    (robot_terms, robot_everyday), (human_terms, human_everyday) = glossary.read(), glossary.read(PAGE)
    return robot_terms | set(robot_everyday) | human_terms | set(human_everyday)


def test_glossary_words() -> None:
    vocabulary = _vocabulary()
    missing = []
    for name, path in sorted(_public_names().items()):
        words = glossary.words(name)
        if glossary.spelling(words, vocabulary) is None:
            spans = [range(start, end) for start in range(len(words)) for end in range(start + 1, len(words) + 1)]
            known = {
                i for span in spans if glossary.entry("".join(words[span.start : span.stop]), vocabulary) for i in span
            }
            missing.append(f"{name} ({path}): " + ", ".join(word for i, word in enumerate(words) if i not in known))
    assert not missing, (
        "define these words in docs/reference/glossary.md, or rename (principle 3.9):\n  " + "\n  ".join(missing)
    )


def test_glossary_lists_each_everyday_word_once_and_only_while_a_name_uses_it() -> None:
    terms, everyday = glossary.read(PAGE)
    vocabulary = _vocabulary()
    used = {entry for name in _public_names() for entry in glossary.spelling(glossary.words(name), vocabulary) or ()}
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
