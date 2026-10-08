"""``FamilySpec.meta``: the documented key set is the set core and the families read."""

from __future__ import annotations

import re
from pathlib import Path

import better_human
from better_human.core.spec import META_KEYS

_READ = re.compile(r"""meta(?:\[|\.get\()\s*["']([a-z0-9_]+)["']|["']([a-z0-9_]+)["']\s+in\s+(?:spec\.)?meta\b""")


def test_documented_meta_keys_match_the_keys_core_reads() -> None:
    root = Path(better_human.__file__).parent
    read = set()
    for path in [*root.joinpath("core").rglob("*.py"), *root.joinpath("families").rglob("*.py")]:
        for match in _READ.finditer(path.read_text()):
            read.add(match.group(1) or match.group(2))
    assert read == set(META_KEYS)
