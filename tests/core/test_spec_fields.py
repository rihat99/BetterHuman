"""Decision record 0008: every ``FamilySpec`` field is typed; no ``Mapping[str, Any]`` bag, and ``provenance`` is the
only ``Mapping[str, str]``."""

from __future__ import annotations

import dataclasses

from better_human import FamilySpec


def _annotations() -> dict[str, str]:
    return {field.name: str(field.type).replace(" ", "") for field in dataclasses.fields(FamilySpec)}


def test_no_field_is_an_untyped_mapping() -> None:
    untyped = sorted(name for name, annotation in _annotations().items() if "Any" in annotation)
    assert untyped == [], f"give these fields a type (record 0008): {untyped}"


def test_provenance_is_the_only_string_to_string_mapping() -> None:
    string_maps = [name for name, annotation in _annotations().items() if annotation == "Mapping[str,str]"]
    assert string_maps == ["provenance"]
