"""The package, tests, examples, tools, benchmarks and docs scripts find weights only through ``better_human.io``.

Decision record 0017 (one way to find the weights): ``io.models_dir()`` is the one reader of
``$BETTERHUMAN_MODELS_DIR`` and ``io.weight_path`` the one way to name a file under it. A second lookup rule comes
back through a ``models`` path joined onto a base built from ``__file__``, the working directory (``cwd()``,
``getcwd()``) or the home directory (``home()``, ``expanduser``), directly or through a name assigned from one, with
``/``, ``+``, ``joinpath``, ``os.path.join``, ``Path(base, ...)`` or an f-string; through a relative or home-relative
``Path("models/...")``, ``Path("~/models/...")`` or ``os.path.join("models", ...)``; through ``__file__`` of
``better_human`` or one of its modules; or through a read of the variable itself (``os.environ[...]``, ``.get``,
``getenv``) anywhere but ``src/better_human/io/load.py``. The scan rejects each of these.
"""

from __future__ import annotations

import ast
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).parents[2]
SCANNED = ("src", "tests", "examples", "tools", "benchmarks", "docs")
#: The one module that reads the variable: ``io.models_dir``.
READER = Path("src/better_human/io/load.py")
BASE_CALLS = {"cwd", "getcwd", "home", "expanduser"}
JOIN_CALLS = {"Path", "PurePath", "join", "expanduser"}
#: The variable's name and the constant ``better_human.io.load`` spells it with.
MODELS_ENV_SPELLINGS = {"BETTERHUMAN_MODELS_DIR", "MODELS_ENV"}


def _call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    return getattr(node.func, "attr", getattr(node.func, "id", ""))


def _is_base(node: ast.AST, derived: set[str]) -> bool:
    """Whether ``node`` uses ``__file__``, the working or home directory, or a name assigned from one of them."""
    return any(
        (isinstance(sub, ast.Name) and (sub.id == "__file__" or sub.id in derived)) or _call_name(sub) in BASE_CALLS
        for sub in ast.walk(node)
    )


def _is_models(node: ast.AST) -> bool:
    """Whether ``node`` is a string (or an f-string's leading text) whose first path part, after ``./`` or ``~/``, is
    ``models``."""
    if isinstance(node, ast.JoinedStr) and node.values:
        node = node.values[0]
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.lstrip("./~").split("/")[0] == "models"
    )


def _names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Names bound to ``better_human`` or its modules, and names assigned from a base or from such a name."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= {a.asname or "better_human" for a in node.names if a.name.split(".")[0] == "better_human"}
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "better_human":
            aliases |= {a.asname or a.name for a in node.names}
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign | ast.AnnAssign) and node.value]
    derived: set[str] = set()
    while True:  # names assigned from a base, then from those names, until nothing new
        new = {
            target.id
            for node in assignments
            if _is_base(node.value, derived)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name) and target.id not in derived
        }
        if not new:
            return aliases, derived
        derived |= new


def _joined_parts(node: ast.AST) -> list[ast.AST]:
    """The operands ``node`` joins into one path or string, in order; empty when it joins nothing."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div | ast.Add):
        return [node.left, node.right]
    if _call_name(node) == "joinpath":
        return [node.func.value, *node.args]
    if _call_name(node) in JOIN_CALLS:
        return node.args
    if isinstance(node, ast.JoinedStr):
        return node.values
    return []


def _builds_weights_path(node: ast.AST, aliases: set[str], derived: set[str]) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "__file__":
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        return isinstance(root, ast.Name) and root.id in aliases
    if _call_name(node) in JOIN_CALLS and node.args and _is_models(node.args[0]):
        return True  # relative to the working directory
    return any(_is_base(left, derived) and _is_models(right) for left, right in pairwise(_joined_parts(node)))


def _names_models_env(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and node.value in MODELS_ENV_SPELLINGS) or (
        getattr(node, "id", getattr(node, "attr", None)) in MODELS_ENV_SPELLINGS
    )


def _reads_models_env(node: ast.AST) -> bool:
    """Whether ``node`` reads the variable: ``environ[...]`` (not an assignment), ``.get(...)``, ``getenv(...)``."""
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        return _names_models_env(node.slice)
    return _call_name(node) in ("get", "getenv") and bool(node.args) and _names_models_env(node.args[0])


def _offences(tree: ast.AST, reader: bool = False) -> list[tuple[int, str]]:
    aliases, derived = _names(tree)
    return [
        (node.lineno, ast.unparse(node))
        for node in ast.walk(tree)
        if _builds_weights_path(node, aliases, derived) or (not reader and _reads_models_env(node))
    ]


def test_weights_are_found_through_io() -> None:
    offences = [
        f"{path.relative_to(REPO)}:{line}: {what}"
        for folder in SCANNED
        for path in sorted((REPO / folder).rglob("*.py"))
        for line, what in _offences(ast.parse(path.read_text()), reader=path.relative_to(REPO) == READER)
    ]
    assert not offences, "weights paths built by hand (use better_human.io.weight_path):\n" + "\n".join(offences)


def test_the_scan_sees_every_spelling() -> None:
    source = (
        "import os\nfrom pathlib import Path\nimport better_human as bh\nimport better_human.io\n"
        "from better_human import io\n"
        "a = Path(__file__).parents[2] / 'models' / 'smpl'\nROOT = Path(__file__).parent\nBASE = ROOT.parent\n"
        "b = BASE / 'models/smplx'\nc = Path(bh.__file__)\nd = better_human.io.__file__\ne = io.__file__\n"
        "f = ROOT.joinpath('models', 'smpl')\ng = os.path.join(os.path.dirname(__file__), 'models')\n"
        "h = f'{ROOT}/models/smpl'\ni = str(BASE) + '/models'\nj = Path('models/SOMA')\nk = Path.cwd() / 'models'\n"
        "m = os.path.join('models', 'smpl')\nn = Path(os.getcwd(), './models')\n"
        "fine = Path(tmp) / 'models'\nfine = Path(tmp).joinpath('models')\nfine = f'{tmp}/models'\n"
        "fine = ', '.join(['models'])\nfine = Path(__file__).parent / 'goldens.py'\n"
        "o = Path.home() / 'models' / 'smpl'\np = Path('~/models/smpl').expanduser()\n"
        "q = os.path.expanduser('~/models')\nr = Path(os.environ['BETTERHUMAN_MODELS_DIR'])\n"
        "s = os.environ.get(io.load.MODELS_ENV)\nt = os.getenv(MODELS_ENV)\n"
        "fine = os.environ['HOME']\nos.environ['BETTERHUMAN_MODELS_DIR'] = str(tmp)\n"
    )
    assert sorted({line for line, _ in _offences(ast.parse(source))}) == [6, *range(9, 21), *range(26, 32)]
