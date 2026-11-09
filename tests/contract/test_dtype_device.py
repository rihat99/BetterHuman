"""Decision record 0005: dtype and device are chosen once, at load; no BetterHuman object has ``to``, and a tensor in
another dtype than the model's raises instead of being cast."""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import torch
from better_robot.exceptions import DtypeMismatchError

import better_human as bh
from tests.support.synthetic import flame_spec

SRC = Path(__file__).resolve().parents[2] / "src" / "better_human"


def test_no_class_or_function_in_src_is_named_to() -> None:
    defined = [
        f"{path.relative_to(SRC.parent)}:{node.lineno}"
        for path in sorted(SRC.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "to"
    ]
    assert defined == [], f"load at the dtype and device instead (record 0005): {defined}"


@pytest.mark.parametrize("dtype", [torch.int64, torch.float16, torch.bfloat16, torch.complex64, "float32"])
def test_load_accepts_only_float32_and_float64(dtype: object) -> None:
    with pytest.raises(DtypeMismatchError, match=r"is not torch\.float32 or torch\.float64.*pass dtype=torch\.float32"):
        bh.load("smpl", dtype=dtype)


def _tensors(value: object, path: str, seen: set[int]) -> Iterator[tuple[str, torch.Tensor]]:
    """Every tensor reachable from ``value`` through dataclass fields, mappings, sequences and the attributes of
    BetterHuman and BetterRobot objects."""
    if id(value) in seen or isinstance(value, (str, bytes, int, float, bool, type(None), torch.dtype, torch.device)):
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield path, value
    elif dataclasses.is_dataclass(value):
        for item in dataclasses.fields(value):
            yield from _tensors(getattr(value, item.name), f"{path}.{item.name}", seen)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _tensors(item, f"{path}[{key!r}]", seen)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{path}[{index}]", seen)
    elif type(value).__module__.startswith(("better_human.", "better_robot.")) and hasattr(value, "__dict__"):
        yield from _tensors(vars(value), path, seen)


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_loader_builds_every_tensor_at_the_requested_dtype(tmp_path: Path, dtype: torch.dtype) -> None:
    spec = flame_spec(tmp_path, dtype=dtype)
    human = bh.HumanModel.from_spec(spec, density=None, contact_frames={"chin": (0, None)})
    identity = human.bake()
    state = identity.forward(human.model.q_neutral)
    state.vertices, state.joint_poses  # fill the memo, so the walk reaches every output
    tensors = list(_tensors(state, "state", set()))
    floating = [(path, tensor.dtype) for path, tensor in tensors if tensor.is_floating_point()]
    assert len(floating) > 30
    assert [(path, found) for path, found in floating if found != dtype] == []
    assert [path for path, tensor in tensors if tensor.device != spec.device] == []


@pytest.mark.slow
def test_identity_values_in_another_dtype_raise(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    with pytest.raises(
        DtypeMismatchError, match=r"betas is torch.float32; pass it in the model's dtype, torch.float64"
    ):
        human.bake(betas=torch.zeros(100, dtype=torch.float32))
