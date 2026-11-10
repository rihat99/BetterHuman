"""Every SMPL-family loader reads a file of exactly its table's vertex count (FormatError naming the file).

The shipped landmarks (SMPL, SMPL-H, SMPL-X), contact frames (SMPL-X) and closed segmentation (SMPL) index the
authored meshes: a file with fewer or more vertices is not the family's file, and with more its landmarks would land
on other vertices. Synthetic files of each family's joint count, without weights.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import better_human as bh
from better_human.families import _tables
from better_human.io import FormatError, shipped_frame_set
from better_human.io.mass_data import shipped_segmentation
from tests.support.synthetic import LAYOUTS, write_family

TABLES = (_tables.SMPL, _tables.SMPLH, _tables.SMPLX, _tables.MANO, _tables.FLAME)


@pytest.mark.parametrize("family", list(LAYOUTS))
@pytest.mark.parametrize("more", [False, True], ids=["fewer", "more"])
def test_a_file_with_another_vertex_count_raises(family: str, more: bool, tmp_path: Path) -> None:
    expected = getattr(_tables, family.upper()).num_vertices
    found = expected + 1 if more else 50
    path = write_family(tmp_path, family, num_vertices=found)
    message = f"{path}: {family} weights hold {found} vertices; expected {expected}."
    with pytest.raises(FormatError, match=re.escape(message)):
        getattr(bh.families, family)(path)


@pytest.mark.parametrize("table", TABLES, ids=lambda table: table.name)
def test_the_shipped_vertex_data_fit_the_table_vertex_count(table: _tables.FamilyTable) -> None:
    if table.keypoint_vertex_ids:
        assert max(table.keypoint_vertex_ids) < table.num_vertices
    if table.closed_segmentation is not None:
        assert shipped_segmentation(table.closed_segmentation)[0] == table.num_vertices


def test_the_shipped_contact_frames_fit_smplx() -> None:
    assert max(vertex for vertex, _ in shipped_frame_set("smplx_contacts").values()) < _tables.SMPLX.num_vertices
