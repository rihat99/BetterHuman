"""H-LAYER-1: imports inside ``better_human`` follow the package DAG.

``core`` and ``io`` import nothing from the package; ``families`` imports ``core`` and ``io``; ``terms`` imports
``core``; ``viz`` imports anything below it and the package root. The edges below are the ones that exist; adding
one is a contract decision.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).parents[2] / "src" / "better_human"
ROOT = "__init__"
ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "io": set(),
    "families": {"core", "io"},
    "terms": {"core"},
    "viz": {ROOT, "core", "families", "io", "terms"},
    ROOT: {ROOT, "core", "families", "io", "terms"},
}


def _package_of(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ROOT if parts == ("__init__",) else parts[0]


def _imported_packages(path: Path) -> set[str]:
    module_parts = path.relative_to(SRC).parts[:-1]
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = module_parts[: len(module_parts) - (node.level - 1)]
            dotted = [*base, *(node.module.split(".") if node.module else [])]
        elif isinstance(node, ast.ImportFrom) and node.module:
            dotted = node.module.split(".")
            if dotted[0] != "better_human":
                continue
            dotted = dotted[1:]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "better_human":
                    targets.add(alias.name.split(".")[1] if "." in alias.name else ROOT)
            continue
        else:
            continue
        targets.add(dotted[0].removesuffix(".py") if dotted else ROOT)
    return targets


def test_imports_follow_package_dag() -> None:
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        package = _package_of(path)
        assert package in ALLOWED, f"new package {package!r}: add it to ALLOWED in this test"
        for target in _imported_packages(path) - {package}:
            if target not in ALLOWED[package]:
                violations.append(f"{path.relative_to(SRC)}: {package} -> {target}")
    assert not violations, "imports against the H-LAYER-1 DAG:\n" + "\n".join(violations)
