"""Every repository path cited in tracked text resolves, and no line number is cited (principle 7.8).

Citations are found by ``better_robot.testing.dead_citations`` (BetterRobot decision record 0035, the shared testing
module): a ``src/``, ``tests/``, ``docs/``, ``tools/``, ``examples/`` or ``benchmarks/`` path must be tracked, a
``path::symbol`` must name a symbol the file defines, and a ``.py:<line>`` citation is always dead. The dead
citations that exist today sit in ``tests/contract/ratchets/dead_citations.json`` (``"<file>: <citation>"`` to the
reason); the ratchet only shrinks, and the fix that resolves a citation removes its entry. Permanent exceptions: the
citations of accepted decision records, whose text never changes after acceptance (principle 8.2), each
with where the cited thing is now, and this module's own repetition of them. Tracked text states a reason, never the
code of a plan, a review or its finding (``PLAN_CODE``).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from better_robot.testing import dead_citations

REPO = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().relative_to(REPO).as_posix()
PLAN_CODE = re.compile(
    r"\bwp/[A-Z]|\blane [A-Z]\b|\bR-\d+\b|\(P-\d+\)|\bP\d+-D\d+\b|\bPhase \d+ pass|\bowning WP\b|\bwork package\b"
    r"|\bruling R-|\breview finding\b|\b[A-Z]\d{1,2}(?:fix)? (?:review|finding)\b"
)
TEXT_SUFFIXES = (".md", ".py", ".toml", ".yml", ".yaml", ".json", ".sh", ".txt")
CITATION_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "dead_citations.json").read_text())
_RECORDS = "docs/contracts/decisions/"
_PACKAGE_RELATIVE = "written relative to src/better_human/, as records of that date did"
FROZEN_RECORD_CITATIONS: dict[str, str] = {
    f"{_RECORDS}0001-human-model.md: core/body.py::Body.__init__": (
        "record 0001 replaced Body by src/better_human/core/human_model.py::HumanModel"
    ),
    f"{_RECORDS}0002-rig-abstract-base.md: core/rig.py::Rig": _PACKAGE_RELATIVE,
    f"{_RECORDS}0003-one-skinning-path.md: core/pipeline.py::skin_affines": (
        "now src/better_human/core/skinning.py::skin_affines"
    ),
    f"{_RECORDS}0003-one-skinning-path.md: viz/human_layers.py::skinned_layer": (
        "now src/better_human/viz/layers.py::skinned_layer"
    ),
    f"{_RECORDS}0005-dtype-and-device-at-load.md: core/rig.py": _PACKAGE_RELATIVE,
    f"{_RECORDS}0005-dtype-and-device-at-load.md: families/mhr_rig.py": _PACKAGE_RELATIVE,
    f"{_RECORDS}0005-dtype-and-device-at-load.md: families/soma_rig.py": _PACKAGE_RELATIVE,
    f"{_RECORDS}0010-human-viz-surface.md: viz/__init__.py": _PACKAGE_RELATIVE,
    f"{_RECORDS}0010-human-viz-surface.md: viz/layers.py": _PACKAGE_RELATIVE,
    f"{_RECORDS}0011-human-package-layout.md: core/mass.py": "now the package mass under src/better_human/",
    f"{_RECORDS}0011-human-package-layout.md: io/convert_chumpy.py": "now tools/convert/smpl/convert_chumpy.py",
    f"{_RECORDS}0012-bh-converted-archives.md: families/mhr.py::METADATA": (
        "gone; the MHR archive's fields are src/better_human/io/schema.py::MHR_SCHEMA"
    ),
    f"{_RECORDS}0013-bh-package-data-and-load-speed.md: betterretarget/human.py": (
        "a module of BetterRetarget, another repository"
    ),
    f"{_RECORDS}0013-bh-package-data-and-load-speed.md: bvr/body.py": (
        "a module of BetterVideoReconstruction, another repository"
    ),
    f"{_RECORDS}0013-bh-package-data-and-load-speed.md: families/assets/smpl_3d_segmentation.npy": (
        "now src/better_human/io/assets/smpl_segmentation.npz"
    ),
}


def _dead() -> set[str]:
    """Every dead citation but this module's repetition of the frozen ones."""
    repeated = {f"{THIS}: {key.split(': ', 1)[1]}" for key in FROZEN_RECORD_CITATIONS}
    return set(dead_citations(REPO)) - repeated


def test_cited_paths_resolve() -> None:
    """A new dead citation fails: cite a tracked path or ``path::symbol``, never a line number."""
    new = sorted(_dead() - CITATION_RATCHET.keys() - FROZEN_RECORD_CITATIONS.keys())
    assert not new, "cite a tracked path or path::symbol, never a line number:\n  " + "\n  ".join(new)


def test_citation_ratchet_has_no_stale_entries() -> None:
    """A citation that resolves now leaves ``tests/contract/ratchets/dead_citations.json`` in the commit that fixes it,
    and a frozen one that resolves leaves ``FROZEN_RECORD_CITATIONS``."""
    stale = sorted(CITATION_RATCHET.keys() - _dead())
    assert not stale, "remove these entries from tests/contract/ratchets/dead_citations.json:\n  " + "\n  ".join(stale)
    stale = sorted(FROZEN_RECORD_CITATIONS.keys() - _dead())
    assert not stale, "remove these entries from FROZEN_RECORD_CITATIONS:\n  " + "\n  ".join(stale)


def test_no_plan_code_in_tracked_text() -> None:
    """Tracked text states a reason, never a plan code such as ``wp/H9``, ``H3 review L7``, ``ruling R-3`` or a
    review finding; this module and the docs-text lint are not read, since their patterns spell the codes."""
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    found = [
        f"{path}: {match[0]}"
        for path in listing.split("\0")
        if path.endswith(TEXT_SUFFIXES) and path not in (THIS, "tests/docs/test_docs_text.py")
        for match in PLAN_CODE.finditer((REPO / path).read_text())
    ]
    assert not found, "write the reason instead of the plan code:\n  " + "\n  ".join(found)
