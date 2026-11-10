"""Every path of the public API snapshot has an entry on an API reference page.

``tests/contract/public_api.json`` lists every public path. A page of ``docs/reference/api/`` renders a module with a
``::: <module>`` directive, and mkdocstrings shows every member without a ``_`` prefix, documented or not
(``show_if_no_docstring: true`` in ``mkdocs.yml``). So a path has an entry when it is a directive's module, or lies
under one with no ``_`` segment; a ``_`` name the filter hides (``__version__``) is named on its module's page instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).parents[2]
API_PAGES = REPO / "docs" / "reference" / "api"
DIRECTIVE = re.compile(r"^::: (?P<module>[\w.]+)$", re.MULTILINE)


def _missing(paths: list[str], pages: dict[str, str]) -> list[str]:
    """The ``paths`` with no entry on ``pages`` (page text by the module its directive renders)."""
    missing = []
    for path in paths:
        module = max((name for name in pages if path == name or path.startswith(name + ".")), key=len, default=None)
        if module is None:
            missing.append(path)
        elif path != module and any(part.startswith("_") for part in path[len(module) + 1 :].split(".")):
            if f"`{path.rsplit('.', 1)[-1]}`" not in pages[module]:
                missing.append(path)
    return missing


def test_every_public_path_has_a_reference_entry() -> None:
    assert re.search(r"^ +show_if_no_docstring: true$", (REPO / "mkdocs.yml").read_text(), re.MULTILINE)
    pages = {
        match["module"]: page.read_text()
        for page in sorted(API_PAGES.glob("*.md"))
        for match in DIRECTIVE.finditer(page.read_text())
    }
    missing = _missing(sorted(json.loads((REPO / "tests" / "contract" / "public_api.json").read_text())), pages)
    assert not missing, "give these public paths an entry on a docs/reference/api page:\n  " + "\n  ".join(missing)


def test_a_path_off_every_page_is_missing() -> None:
    pages = {"pkg": "`__version__` is the version.", "pkg.sub": ""}
    paths = ["pkg", "pkg.f", "pkg.sub.C.m", "pkg.__version__", "pkg.sub.__all__", "other.g"]
    assert _missing(paths, pages) == ["pkg.sub.__all__", "other.g"]
