"""Principle 2.9: public records are frozen, and a record that holds tensors compares by
identity.

Every dataclass a public ``__all__`` exports is frozen; each one with a field whose annotation names a tensor also sets
``eq=False``, because ``==`` on tensors is elementwise and a generated ``__eq__`` or ``__hash__`` over them raises or
hashes by tensor identity. Frozen goes all the way down: every mapping field is read-only, and a ``HumanModel`` cannot
be built inconsistent.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from better_robot.exceptions import ArgumentError

import better_human as bh
from tests.support.synthetic import flame_spec

PACKAGES = ["better_human", "better_human.core", "better_human.families", "better_human.io", "better_human.mass"]


def _public_dataclasses() -> list[tuple[str, type]]:
    found = {}
    for package in PACKAGES:
        module = importlib.import_module(package)
        for name in module.__all__:
            value = getattr(module, name)
            if inspect.isclass(value) and dataclasses.is_dataclass(value):
                found[f"{value.__module__}.{value.__qualname__}"] = value
    return sorted(found.items())


@pytest.mark.parametrize(("qualname", "cls"), _public_dataclasses(), ids=[name for name, _ in _public_dataclasses()])
def test_public_dataclass_is_frozen(qualname: str, cls: type) -> None:
    assert cls.__dataclass_params__.frozen, f"{qualname} must be @dataclass(frozen=True)"


def _tensor_records() -> list[tuple[str, type]]:
    return [
        (qualname, cls)
        for qualname, cls in _public_dataclasses()
        if any("Tensor" in str(item.type) for item in dataclasses.fields(cls))
    ]


def test_the_tensor_records_include_the_three_of_record_0001() -> None:
    assert {bh.HumanModel, bh.Identity, bh.HumanState} <= {cls for _, cls in _tensor_records()}


@pytest.mark.parametrize(("qualname", "cls"), _tensor_records(), ids=[name for name, _ in _tensor_records()])
def test_tensor_records_compare_by_identity(qualname: str, cls: type) -> None:
    assert not cls.__dataclass_params__.eq, f"{qualname} must set eq=False"
    assert cls.__eq__ is object.__eq__ and cls.__hash__ is object.__hash__


def _mapping_fields(record: object, path: str) -> list[tuple[str, Mapping]]:
    found = []
    for item in dataclasses.fields(record):
        value = getattr(record, item.name)
        if isinstance(value, Mapping) and not item.name.startswith("_"):
            found.append((f"{path}.{item.name}", value))
            found += [
                (f"{path}.{item.name}[{key!r}]", inner) for key, inner in value.items() if isinstance(inner, Mapping)
            ]
    return found


@pytest.mark.slow
def test_every_mapping_field_is_read_only(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    identity = human.bake(betas=torch.zeros(100, dtype=torch.float64))
    state = identity.forward(human.model.q_neutral, {"expression": torch.zeros(50, dtype=torch.float64)})
    records = {"spec": human.spec, "spec.parts": human.spec.parts, "human": human, "identity": identity, "state": state}
    mappings = [found for path, record in records.items() for found in _mapping_fields(record, path)]
    assert {"spec.identity_bases", "spec.parts.density_tables['Dempster']", "human.stream_joint_bases"} <= {
        path for path, _ in mappings
    }
    for path, mapping in mappings:
        with pytest.raises(TypeError, match="does not support item assignment"):
            mapping["changed"] = None  # type: ignore[index]
        assert "changed" not in mapping, path


@pytest.mark.slow
def test_a_human_model_with_density_but_no_topology_cannot_be_built(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    with pytest.raises(ArgumentError, match="density and part_topology are both None or both set"):
        dataclasses.replace(human, density=torch.tensor(1000.0, dtype=torch.float64))
