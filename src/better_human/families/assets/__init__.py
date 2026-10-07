"""Tracked SMPL-family metadata assets and their loaders.

Authored closed-3D segmentation face tables and named density tables are
derived metadata, not license-gated weights, so they ship inside the package.
The loaders resolve a :class:`~better_human.families.tables.FamilyTable`
asset filename against this directory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .densities import density_for_parts

_ASSET_DIR = Path(__file__).parent


def load_closed_segmentation(
    filename: str,
    device: torch.device | None = None,
) -> dict[int, torch.Tensor]:
    """Load an authored closed-3D segmentation as per-part face tables.

    :param filename: asset filename, e.g. ``"smpl_3d_segmentation.npy"``.
    :param device: device for the returned face tables.
    :returns: mapping from native joint index to an ``(F, 3)`` ``long`` face
        table indexing the concatenation ``[shaped_vertices; rest_joints]``.
    """
    payload = np.load(_ASSET_DIR / filename, allow_pickle=True).item()
    return {
        int(part): torch.as_tensor(np.asarray(faces), dtype=torch.long, device=device)
        for part, faces in payload["body_faces"].items()
    }


def load_density_table(
    filename: str,
    table_name: str,
    joint_names: Sequence[str],
    exemplar: torch.Tensor,
) -> torch.Tensor:
    """Resolve one named density table to a per-part tensor in native order.

    The tables are authored for SMPL's 24 joints, so a part is looked up by
    name: SMPL's own joints match their entry exactly, and any other name
    (SMPL-X fingers and face joints, MHR segments) is mapped onto the closest
    SMPL category by keyword — fingers take the hand entry, jaw and eyes the
    head entry (see :mod:`better_human.families.assets.densities`).

    :param filename: density-table asset filename, e.g. ``"body_densities.json"``.
    :param table_name: table key, e.g. ``"Dempster"``.
    :param joint_names: native joint names, one per physical part.
    :param exemplar: tensor whose dtype and device the result matches.
    :returns: ``(len(joint_names),)`` density tensor in native joint order.
    """
    tables = json.loads((_ASSET_DIR / filename).read_text())
    try:
        table = tables[table_name]
    except KeyError as error:
        raise ValueError(f"{filename} has no density table {table_name!r}; available: {sorted(tables)}") from error
    return density_for_parts(joint_names, table).to(dtype=exemplar.dtype, device=exemplar.device)


__all__ = ["load_closed_segmentation", "load_density_table"]
