"""Decision record 0005: dtype and device are chosen once, at load; no BetterHuman object has ``to``, and a tensor in
another dtype than the model's raises instead of being cast."""

from __future__ import annotations

import ast
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


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_loader_builds_every_tensor_at_the_requested_dtype(tmp_path: Path, dtype: torch.dtype) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path, dtype=dtype), density=None)
    identity = human.bake()
    state = identity.forward(human.model.q_neutral)
    assert identity.rest_vertices.dtype == dtype
    assert state.vertices.dtype == dtype


@pytest.mark.slow
def test_identity_values_in_another_dtype_raise(tmp_path: Path) -> None:
    human = bh.HumanModel.from_spec(flame_spec(tmp_path), density=None)
    with pytest.raises(
        DtypeMismatchError, match=r"betas is torch.float32; pass it in the model's dtype, torch.float64"
    ):
        human.bake(betas=torch.zeros(100, dtype=torch.float32))
