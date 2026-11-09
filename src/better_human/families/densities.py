"""The density category of each part, from its name: how the family loaders declare ``Parts.density_categories``.

The shipped density tables are keyed by SMPL's 24 joint names (``pelvis`` .. ``right_hand``). A part named like a
category takes it; any other name (SMPL-X fingers and face joints, MHR segments) maps by keyword to a region (fingers
-> hand, eye and jaw -> head, twist joints -> their limb, toes and ball -> foot, ...), with the side its name gives
for a limb. A name the map misses has no category (``None``), and ``density="<table>"`` raises for that family at load
(decision record 0007).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

# Region keyword -> (base category name, whether the category needs a side
# prefix).  Ordered most-specific first; the first keyword found in a part name
# wins.  ``spine`` is handled separately to pick spine1/2/3 by trailing digit.
_REGION_RULES: tuple[tuple[tuple[str, ...], str, bool], ...] = (
    (("thumb", "index", "middle", "ring", "pinky", "finger", "palm", "knuckle", "hand"), "hand", True),
    (("wrist",), "wrist", True),
    (("lowarm", "forearm", "elbow"), "elbow", True),
    (("uparm", "upperarm", "shoulder"), "shoulder", True),
    (("clavicle", "collar", "scapula"), "collar", True),
    (("talocrural", "subtalar", "transversetarsal", "ankle", "heel"), "ankle", True),
    (("toe", "ball", "foot"), "foot", True),
    (("lowleg", "shin", "calf", "knee"), "knee", True),
    (("upleg", "thigh", "hip"), "hip", True),
    (("eye", "jaw", "teeth", "tongue", "skull", "nose", "head"), "head", False),
    (("neck",), "neck", False),
    (("spine", "chest", "thorax", "abdomen", "torso", "back"), "spine", False),
    (("pelvis", "sacrum", "root", "world", "hips"), "pelvis", False),
)


def density_categories(part_names: Sequence[str], tables: Mapping[str, Mapping[str, float]]) -> tuple[str | None, ...]:
    """The category of each part among the categories of ``tables`` (table -> category -> kg/m³), or ``None``."""
    known = {category for table in tables.values() for category in table}
    return tuple(_category(name, known) for name in part_names)


def _category(name: str, known: set[str]) -> str | None:
    if name in known:
        return name
    lowered = name.lower()
    for keywords, region, needs_side in _REGION_RULES:
        if not any(keyword in lowered for keyword in keywords):
            continue
        if region == "spine":
            category = _spine_category(lowered)
        elif needs_side:
            side = _side(name)
            category = None if side is None else f"{side}_{region}"
        else:
            category = region
        if category is not None and category in known:
            return category
    return None


def _side(name: str) -> str | None:
    """Return ``"left"``, ``"right"`` or ``None`` from a part name's tokens."""
    tokens = name.lower().split("_")
    if "l" in tokens or "left" in tokens or name.lower().startswith("left"):
        return "left"
    if "r" in tokens or "right" in tokens or name.lower().startswith("right"):
        return "right"
    return None


def _spine_category(name: str) -> str:
    """Map a spine-like part to spine1/2/3 by its trailing digit (default 2)."""
    match = re.search(r"(\d+)", name)
    if match is None:
        return "spine2"
    return f"spine{min(int(match.group(1)) + 1, 3)}"
