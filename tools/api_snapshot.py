"""Write, check or diff the public-API snapshot of ``better_human`` (BetterRobot decision records 0024 and 0035).

``tests/contract/public_api.json`` maps every path of the public tree to one line describing it
(``better_robot.testing.public_api`` of the root and of each public subpackage). ``--write`` rewrites it after an
intended API change; ``--check`` prints the differences and exits 1 when the package no longer matches it;
``--diff OLD.json`` prints the paths added (``+``), removed (``-``) and changed (``~``) since an older snapshot, the
draft of the next migration table (``tests/contract/public_api_before_0.2.0.json`` is the 0.1 side).

    uv run python tools/api_snapshot.py --check
    uv run python tools/api_snapshot.py --diff tests/contract/public_api_before_0.2.0.json
"""

from __future__ import annotations

import argparse
import json
import pkgutil
import sys
from pathlib import Path

import better_human
from better_robot.testing import public_api

SNAPSHOT = Path(__file__).resolve().parents[1] / "tests" / "contract" / "public_api.json"


def current_api() -> dict[str, str]:
    """``public_api`` of ``better_human`` and of each public subpackage, merged into one map."""
    # The root __all__ lists families and io only; core, mass, terms, viz and studio are public subpackages too
    # (decision record 0011), so each is walked as the root would walk it.
    subpackages = [info.name for info in pkgutil.iter_modules(better_human.__path__) if info.name[0] != "_"]
    packages = ["better_human", *(f"better_human.{name}" for name in subpackages)]
    return {path: line for package in packages for path, line in public_api(package).items()}


def differences(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Lines ``- path: old``, ``+ path: new`` and ``~ path: old -> new``, sorted by path."""
    lines = [f"- {path}: {old[path]}" for path in old.keys() - new.keys()]
    lines += [f"+ {path}: {new[path]}" for path in new.keys() - old.keys()]
    lines += [f"~ {path}: {old[path]} -> {new[path]}" for path in old.keys() & new.keys() if old[path] != new[path]]
    return sorted(lines, key=lambda line: line[2:])


def main() -> int:
    """Run the command line; the exit code is 1 only for ``--check`` with differences."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help=f"rewrite {SNAPSHOT.name} from the installed package")
    action.add_argument("--check", action="store_true", help=f"compare the package with {SNAPSHOT.name}")
    action.add_argument("--diff", type=Path, metavar="OLD.json", help="compare the package with an older snapshot")
    arguments = parser.parse_args()
    current = current_api()
    if arguments.write:
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        return 0
    lines = differences(json.loads((arguments.diff or SNAPSHOT).read_text()), current)
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    return 1 if arguments.check and lines else 0


if __name__ == "__main__":
    sys.exit(main())
