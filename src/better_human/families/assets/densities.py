"""Name-keyword density mapping from arbitrary parts to SMPL density categories.

The shipped ``body_densities.json`` tables are authored for SMPL's 24 native
joints (category index == SMPL joint index).  SMPL-X and MHR expose finer or
differently named joints, so this module maps each part name to one of the 24
SMPL categories by keyword (fingers -> hand, eye/jaw/tongue -> head, twist
procs -> their limb segment, toes/ball -> foot, ...) and resolves the category's
density.  Parts that match no rule fall back to the table mean.  Pure torch and
string logic; no viser or model dependencies.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch

#: The 24 SMPL native joint names in kinematic-tree order.  Category index ``i``
#: in ``body_densities.json`` names SMPL joint ``i``; this is the SMPL
#: part-index -> category-name mapping (identical to ``SMPL_CONFIG.joint_names``).
SMPL_CATEGORY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

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


def part_categories(
    part_names: Sequence[str],
    category_names: Sequence[str] = SMPL_CATEGORY_NAMES,
) -> list[str | None]:
    """Map each part name to a SMPL density category, or ``None`` if unmapped.

    An exact category-name match wins first (so SMPL-X's shared joints map to
    themselves); otherwise a keyword rule assigns a region, prefixing the body
    side for limb categories.

    :param part_names: names to classify (e.g. ``structure.joint_names``).
    :param category_names: the 24 category names in index order.
    :returns: one category name (or ``None``) per input part.
    """
    category_set = set(category_names)
    result: list[str | None] = []
    for name in part_names:
        result.append(_category_for(name, category_set))
    return result


def _category_for(name: str, category_set: set[str]) -> str | None:
    if name in category_set:
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
        if category is not None and category in category_set:
            return category
    return None


def density_for_parts(
    part_names: Sequence[str],
    table: Mapping[str, float],
    category_names: Sequence[str] = SMPL_CATEGORY_NAMES,
) -> torch.Tensor:
    """Resolve a per-part density tensor from a SMPL density table by name.

    :param part_names: parts to assign a density (native joint order).
    :param table: one ``body_densities.json`` table keyed by category index
        string (``"0"`` .. ``"23"``) with density values in kg/m^3.
    :param category_names: the 24 category names in index order.
    :returns: ``(len(part_names),)`` density tensor; unmapped parts take the
        table mean.
    """
    values = [float(table[str(index)]) for index in range(len(category_names))]
    mean = sum(values) / len(values)
    by_name = {name: values[index] for index, name in enumerate(category_names)}
    categories = part_categories(part_names, category_names)
    resolved = [mean if category is None else by_name[category] for category in categories]
    return torch.tensor(resolved, dtype=torch.float32)


__all__ = ["SMPL_CATEGORY_NAMES", "density_for_parts", "part_categories"]
