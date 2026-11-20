"""The mass data BetterHuman ships: closed part segmentations and named density tables, as versioned archives.

Specs: ``docs/reference/formats/segmentation.md`` and ``docs/reference/formats/density_tables.md``. The family loaders
read them here and hand the arrays to their ``Parts``; ``better_human.mass`` reads no file.
"""

from __future__ import annotations

from pathlib import Path

import torch
from better_robot.exceptions import UnknownNameError

from .formats import FormatError
from .load import load_model_npz
from .schema import DENSITY_TABLES_SCHEMA, SEGMENTATION_SCHEMA

_ASSETS = Path(__file__).parent / "assets"


def shipped_segmentation(family: str) -> tuple[int, dict[int, torch.Tensor]]:
    """The closed segmentation shipped for ``family`` (``"smpl"``): ``(V, {part: faces (F_p, 3) long})``.

    Each part's faces index ``[vertices (V, 3); joints (J, 3)]``: an index ``>= V`` is a joint that caps the part.
    Raises :class:`~better_robot.exceptions.UnknownNameError` listing the shipped families for another ``family``,
    and :class:`~better_human.io.FormatError` for part offsets that do not run from 0 to the face count without
    decreasing, or a negative face index (the joint count that bounds the indices is the family's:
    ``PartTopology.from_closed_faces`` checks it).
    """
    shipped = sorted(path.name.removesuffix("_segmentation.npz") for path in _ASSETS.glob("*_segmentation.npz"))
    if family not in shipped:
        raise UnknownNameError("shipped segmentation", family, valid=shipped)
    path = _ASSETS / f"{family}_segmentation.npz"
    fields = load_model_npz(path, SEGMENTATION_SCHEMA)
    faces, offsets = fields["faces"], fields["part_offsets"].tolist()
    fix = "Reconvert it (spec: docs/reference/formats/segmentation.md)."
    if offsets[0] != 0 or offsets[-1] != len(faces) or any(end < start for start, end in zip(offsets, offsets[1:])):
        raise FormatError(
            f"{path}: part_offsets must run from 0 to the face count {len(faces)} without decreasing. {fix}"
        )
    if bool((faces < 0).any()):
        raise FormatError(f"{path}: a face index is negative. {fix}")
    parts = {part: faces[start:end] for part, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))}
    return int(fields["num_vertices"]), parts


def shipped_density_tables() -> dict[str, dict[str, float]]:
    """The shipped density tables: table name (``"Dempster"``, ...) -> density category -> kg/m³."""
    fields = load_model_npz(_ASSETS / "density_tables.npz", DENSITY_TABLES_SCHEMA)
    categories = [str(category) for category in fields["categories"]]
    return {
        str(name): dict(zip(categories, row, strict=True))
        for name, row in zip(fields["table_names"], fields["densities"].tolist(), strict=True)
    }
