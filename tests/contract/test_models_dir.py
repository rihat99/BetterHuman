"""Tests, examples and tools find weights only through ``better_human.io``.

Decision record 0017 (one way to find the weights): ``io.models_dir()`` is the one reader of
``$BETTERHUMAN_MODELS_DIR`` and ``io.weight_path`` the one way to name a file under it. Two spellings would bring back
a second lookup rule, and the scan rejects both: a ``models`` directory joined onto a path built from ``__file__``
(directly or through a name assigned from one), and ``__file__`` of ``better_human`` or one of its modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).parents[2]
SCANNED = ("tests", "examples", "tools")


def _uses_file(node: ast.AST, derived: set[str]) -> bool:
    return any(isinstance(sub, ast.Name) and (sub.id == "__file__" or sub.id in derived) for sub in ast.walk(node))


def _offences(tree: ast.AST) -> list[tuple[int, str]]:
    aliases: set[str] = set()
    derived: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or "better_human" for a in node.names if a.name.split(".")[0] == "better_human"}
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "better_human":
            aliases |= {a.asname or a.name for a in node.names}
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign | ast.AnnAssign) and node.value]
    while True:  # names assigned from __file__, then from those names, until nothing new
        new = {
            target.id
            for node in assignments
            if _uses_file(node.value, derived)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name) and target.id not in derived
        }
        if not new:
            break
        derived |= new
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
            and node.right.value.split("/")[0] == "models"
            and _uses_file(node.left, derived)
        ):
            found.append((node.lineno, f'<path from __file__> / "{node.right.value}"'))
        elif isinstance(node, ast.Attribute) and node.attr == "__file__":
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in aliases:
                found.append((node.lineno, ast.unparse(node)))
    return found


def test_weights_are_found_through_io() -> None:
    offences = [
        f"{path.relative_to(REPO)}:{line}: {what}"
        for folder in SCANNED
        for path in sorted((REPO / folder).rglob("*.py"))
        for line, what in _offences(ast.parse(path.read_text()))
    ]
    assert not offences, "weights paths built by hand (use better_human.io.weight_path):\n" + "\n".join(offences)


def test_the_scan_sees_every_spelling() -> None:
    source = (
        "from pathlib import Path\nimport better_human as bh\nimport better_human.io\nfrom better_human import io\n"
        "a = Path(__file__).parents[2] / 'models' / 'smpl'\nROOT = Path(__file__).parent\nBASE = ROOT.parent\n"
        "b = BASE / 'models/smplx'\nc = Path(bh.__file__)\nd = better_human.io.__file__\ne = io.__file__\n"
        "fine = Path(tmp) / 'models'\n"
    )
    assert sorted(line for line, _ in _offences(ast.parse(source))) == [5, 8, 9, 10, 11]
