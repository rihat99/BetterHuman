"""``FamilySpec.meta``: the documented key set is the set core and the families read."""

from __future__ import annotations

import re
from pathlib import Path

from better_human.core.spec import META_KEYS

SRC = Path(__file__).parents[2] / "src" / "better_human"

_READ = re.compile(r"""meta(?:\[|\.get\()\s*["']([a-z0-9_]+)["']|["']([a-z0-9_]+)["']\s+in\s+(?:spec\.)?meta\b""")


def test_documented_meta_keys_match_the_keys_core_reads() -> None:
    read = set()
    for path in [*SRC.joinpath("core").rglob("*.py"), *SRC.joinpath("families").rglob("*.py")]:
        for match in _READ.finditer(path.read_text()):
            read.add(match.group(1) or match.group(2))
    assert read == set(META_KEYS)
