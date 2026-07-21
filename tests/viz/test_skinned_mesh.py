"""Skinned-mesh render-mode tests against BetterRobot's recording backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from better_robot.viewer import RenderContext, RenderMode
from better_robot.viewer.renderers.testing import MockBackend

import better_human as bh
from better_human.viz import SkinnedMeshMode

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"


@dataclass(frozen=True)
class _FakeBody:
    robot: object
    structure: SimpleNamespace
    values: SimpleNamespace
    posed_vertices: torch.Tensor

    def vertices_from_data(self, data: object) -> torch.Tensor:
        return self.posed_vertices


def _fake_body(device: torch.device | str = "cpu") -> _FakeBody:
    rest = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    return _FakeBody(
        robot=object(),
        structure=SimpleNamespace(faces=torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)),
        values=SimpleNamespace(v_shaped=rest),
        posed_vertices=rest + 0.25,
    )


@pytest.fixture(scope="module")
def body() -> bh.SMPL:
    """Load the licensed local neutral fixture when available."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.SMPL(MODEL_PATH, compute_mass=False)


def _bent_elbow(body: bh.SMPL) -> torch.Tensor:
    tangent = body.robot.q_neutral.new_zeros(body.robot.nv)
    joint_id = body.robot.joint_id("left_elbow")
    tangent[body.robot.idx_vs[joint_id] + 1] = 0.6
    return body.robot.integrate(body.robot.q_neutral, tangent)


def test_fake_backend_contract_needs_no_licensed_weights() -> None:
    body = _fake_body()
    backend = MockBackend()
    mode = SkinnedMeshMode(cast(Any, body))
    context = RenderContext(backend=backend, namespace="/fake")

    mode.attach(context, cast(Any, body.robot), cast(Any, object()))
    assert len(backend.calls_for("add_mesh")) == 1
    torch.testing.assert_close(backend.calls_for("add_mesh")[0].args[1], body.values.v_shaped)

    mode.update(cast(Any, object()))
    assert len(backend.calls_for("add_mesh")) == 2
    torch.testing.assert_close(backend.calls_for("add_mesh")[-1].args[1], body.posed_vertices)


def test_attach_once_then_update_pushes_posed_vertices(body: bh.SMPL) -> None:
    backend = MockBackend()
    mode = SkinnedMeshMode(body)
    context = RenderContext(backend=backend, namespace="/human")

    assert isinstance(mode, RenderMode)
    mode.attach(context, body.robot, body.fk(body.robot.q_neutral))
    attach_calls = backend.calls_for("add_mesh")
    assert len(attach_calls) == 1
    rest_vertices = attach_calls[0].args[1]
    assert rest_vertices.dtype == body.values.v_shaped.dtype
    assert rest_vertices.device == body.values.v_shaped.device
    torch.testing.assert_close(rest_vertices, body.values.v_shaped)
    torch.testing.assert_close(attach_calls[0].args[2], body.structure.faces)

    data = body.fk(_bent_elbow(body))
    mode.update(data)
    update_calls = backend.calls_for("add_mesh")
    assert len(update_calls) == 2
    posed_vertices = update_calls[-1].args[1]
    torch.testing.assert_close(posed_vertices, body.vertices_from_data(data))
    assert not torch.equal(posed_vertices, rest_vertices)

    mode.set_visible(False)
    assert backend.visible["/human/surface"] is False
    mode.detach()
    assert "/human/surface" not in backend.nodes


def test_batch_index_selects_one_shape_and_pose(body: bh.SMPL) -> None:
    betas = torch.stack((torch.zeros(10), torch.linspace(-0.5, 0.5, 10)))
    shaped = body.with_shape(betas=betas)
    backend = MockBackend()
    mode = SkinnedMeshMode(shaped)
    context = RenderContext(backend=backend, namespace="/batched", batch_index=1)
    data = shaped.fk(shaped.robot.q_neutral)

    mode.attach(context, shaped.robot, data)
    torch.testing.assert_close(backend.calls_for("add_mesh")[0].args[1], shaped.values.v_shaped[1])
    mode.update(data)
    torch.testing.assert_close(
        backend.calls_for("add_mesh")[-1].args[1],
        shaped.vertices_from_data(data)[1],
    )


def test_attach_rejects_a_different_robot(body: bh.SMPL) -> None:
    other = bh.SMPL(MODEL_PATH, compute_mass=False)
    mode = SkinnedMeshMode(body)
    context = RenderContext(backend=MockBackend(), namespace="/wrong")

    with pytest.raises(ValueError, match="body.robot"):
        mode.attach(context, other.robot, other.fk(other.robot.q_neutral))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mode_keeps_cuda_vertices_at_the_backend_boundary() -> None:
    body = _fake_body(torch.device("cuda"))
    backend = MockBackend()
    mode = SkinnedMeshMode(cast(Any, body))
    context = RenderContext(backend=backend, namespace="/cuda")

    mode.attach(context, cast(Any, body.robot), cast(Any, object()))
    mode.update(cast(Any, object()))

    for call in backend.calls_for("add_mesh"):
        assert call.args[1].is_cuda
        assert call.args[1].dtype == torch.float32
        assert call.args[2].is_cuda
