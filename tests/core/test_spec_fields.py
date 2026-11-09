"""Decision record 0008: every ``FamilySpec`` field is typed; no ``Mapping[str, Any]`` bag, and ``provenance`` is the
only ``Mapping[str, str]``.

``UNTYPED_RATCHET`` names the bag that exists today and the work package that types it; it only shrinks.
"""

from __future__ import annotations

import dataclasses

from better_human import FamilySpec

UNTYPED_RATCHET = {"meta": "the SMPL-family hand pose PCA moves to the SMPL-family classic format (H4, record 0004)"}


def _annotations() -> dict[str, str]:
    return {field.name: str(field.type).replace(" ", "") for field in dataclasses.fields(FamilySpec)}


def test_no_field_is_an_untyped_mapping() -> None:
    untyped = {name for name, annotation in _annotations().items() if "Any" in annotation}
    assert untyped == set(UNTYPED_RATCHET), f"type these fields, or drop them from UNTYPED_RATCHET: {untyped}"


def test_provenance_is_the_only_string_to_string_mapping() -> None:
    string_maps = [name for name, annotation in _annotations().items() if annotation == "Mapping[str,str]"]
    assert string_maps == ["provenance"]
