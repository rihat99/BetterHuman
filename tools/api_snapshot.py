"""Write, check or diff the public-API snapshot of ``better_human``.

``tests/contract/public_api.json`` maps every path of the ``__all__`` tree to one line describing it
(``better_robot.testing.public_api``; the root lists every subpackage). ``--write`` rewrites it after an
intended API change; ``--check`` prints the differences and exits 1 when the package no longer matches it;
``--diff OLD.json`` prints the paths added (``+``), removed (``-``) and changed (``~``) since an older snapshot,
such as the last release's.

    uv run python tools/api_snapshot.py --check
    uv run python tools/api_snapshot.py --diff /tmp/public_api_0.1.0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from better_robot.testing import public_api

SNAPSHOT = Path(__file__).resolve().parents[1] / "tests" / "contract" / "public_api.json"


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
    current = public_api("better_human")
    if arguments.write:
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        return 0
    lines = differences(json.loads((arguments.diff or SNAPSHOT).read_text()), current)
    sys.stdout.write("".join(f"{line}\n" for line in lines))
    return 1 if arguments.check and lines else 0


if __name__ == "__main__":
    sys.exit(main())
