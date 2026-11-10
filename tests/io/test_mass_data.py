"""The shipped mass data readers check the archive's semantics, not only its fields' shapes and dtypes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from better_human.io import FormatError, mass_data


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("reversed_offsets", "part_offsets must run from 0 to the face count"),
        ("short_offsets", "part_offsets must run from 0 to the face count"),
        ("negative_face", "a face index is negative"),
    ],
)
def test_a_segmentation_with_broken_semantics_is_a_format_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str, match: str
) -> None:
    with np.load(mass_data._ASSETS / "smpl_segmentation.npz") as archive:
        fields = {name: archive[name] for name in archive.files}
    if corruption == "reversed_offsets":
        fields["part_offsets"] = fields["part_offsets"][::-1].copy()
    elif corruption == "short_offsets":
        fields["part_offsets"] = fields["part_offsets"][:-1].copy()
    else:
        fields["faces"] = fields["faces"].copy()
        fields["faces"][7, 1] = -1
    np.savez(tmp_path / "smpl_segmentation.npz", **fields)
    monkeypatch.setattr(mass_data, "_ASSETS", tmp_path)
    with pytest.raises(FormatError, match=rf"smpl_segmentation\.npz: {match}.*Reconvert"):
        mass_data.shipped_segmentation("smpl")


def test_the_shipped_segmentation_passes_its_checks() -> None:
    num_vertices, parts = mass_data.shipped_segmentation("smpl")
    assert num_vertices == 6890 and sorted(parts) == list(range(24))
