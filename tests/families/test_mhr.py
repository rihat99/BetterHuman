"""Behaviour of the MHR family on the new core: sizes, mimic spine, toggles, the root branch, batching, mass.

The classic round trip runs through tests/families/test_classic_formats.py.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from better_human import HumanModel, families, from_classic, to_classic
from better_human.families import MHRClassic
from better_human.io import weight_path
from better_robot.exceptions import ArgumentError, DtypeMismatchError
from better_robot.kinematics import full_configuration

pytestmark = pytest.mark.licensed


def _random_classic(generator: torch.Generator, batch: tuple[int, ...] = ()) -> MHRClassic:
    """Moderate identity, proportions and pose values ``(*batch, 45)``, ``(*batch, 204)``."""
    identity = 0.5 * torch.randn(*batch, 45, generator=generator)
    model_parameters = 0.3 * torch.randn(*batch, 204, generator=generator)
    return MHRClassic(identity_coeffs=identity, model_parameters=model_parameters)


@pytest.fixture(scope="module")
def body() -> HumanModel:
    return HumanModel.from_spec(families.mhr(weight_path("MHR/converted", "mhr_lod1.npz"), lod=1))


def test_configuration_sizes(body: HumanModel) -> None:
    identity, q, stream = from_classic(body, MHRClassic())
    assert (body.model.nq, body.model.nv, body.model.nq_full) == (132, 131, 179)
    assert (identity.model.nq, identity.model.nv) == (132, 131)
    assert q.shape == (132,)
    assert set(stream) == {"expression"}
    assert stream["expression"].shape == (72,)


def test_mimic_expansion_matches_all_spine_twist_weights(body: HumanModel) -> None:
    with np.load(weight_path("MHR/converted", "mhr_lod1.npz"), allow_pickle=False) as arrays:
        parameter_index = arrays["model_parameter_names"].tolist().index("spine_twist0")
        rows = torch.from_numpy(arrays["parameter_transform_rows"])
        columns = torch.from_numpy(arrays["parameter_transform_columns"])
        weights = torch.from_numpy(arrays["parameter_transform_values"])
    rotation_entries = (rows // 7 >= 2) & (rows % 7 >= 3) & (rows % 7 <= 5)
    expected = weights[rotation_entries][columns[rotation_entries] == parameter_index]
    assert len(expected) > 1

    model_parameters = torch.zeros(204)
    model_parameters[parameter_index] = 1.0
    _, q, _ = from_classic(body, MHRClassic(model_parameters=model_parameters))
    model = body.model
    q_full = full_configuration(model, q)
    source_id = model.joint_id("spine_twist0")
    driven = [joint for joint, source in enumerate(model.mimic_source_ids) if int(source) == source_id]
    contributions = q_full.index_select(-1, torch.tensor([int(model.idx_qs_full[joint]) for joint in driven]))
    torch.testing.assert_close(contributions.sort().values, expected.to(q_full.dtype).sort().values)


def test_expression_toggle() -> None:
    path = weight_path("MHR/converted", "mhr_lod6.npz")
    with_expression = HumanModel.from_spec(families.mhr(path, lod=6, correctives=False))
    without_expression = HumanModel.from_spec(families.mhr(path, lod=6, expression=False, correctives=False))
    classic = _random_classic(torch.Generator().manual_seed(0))

    assert "expression" not in without_expression.spec.stream_bases
    identity, q, stream = from_classic(without_expression, classic)
    assert stream == {}
    assert to_classic(identity, q, streams=stream).expression is None
    with pytest.raises(ValueError, match="loaded without expression"):
        from_classic(without_expression, dataclasses.replace(classic, expression=torch.zeros(72)))

    reference, q_reference, zero_stream = from_classic(with_expression, classic)
    torch.testing.assert_close(zero_stream["expression"], torch.zeros(72))
    neutral_face = reference.forward(q_reference, zero_stream).vertices
    torch.testing.assert_close(identity.forward(q).vertices, neutral_face, rtol=0, atol=1e-6)
    expression = 0.5 * torch.randn(72, generator=torch.Generator().manual_seed(1))
    reference, q_reference, stream = from_classic(with_expression, dataclasses.replace(classic, expression=expression))
    assert (reference.forward(q_reference, stream).vertices - neutral_face).abs().amax() > 1e-4


def test_correctives_toggle() -> None:
    path = weight_path("MHR/converted", "mhr_lod6.npz")
    classic = _random_classic(torch.Generator().manual_seed(2))
    zero_pose = MHRClassic(identity_coeffs=classic.identity_coeffs)
    vertices = {}
    for correctives in (True, False):
        body = HumanModel.from_spec(families.mhr(path, lod=6, correctives=correctives))
        for name, value in (("posed", classic), ("rest", zero_pose)):
            identity, q, stream = from_classic(body, value)
            vertices[name, correctives] = identity.forward(q, stream).vertices
    assert (vertices["posed", True] - vertices["posed", False]).abs().amax() > 1e-4
    torch.testing.assert_close(vertices["rest", True], vertices["rest", False], rtol=0, atol=1e-6)


def test_a_negated_root_quaternion_round_trips(body: HumanModel) -> None:
    """``to_classic`` puts a negative-scalar root quaternion on the Euler branch that ``from_classic`` maps back to it."""
    classic = dataclasses.replace(
        _random_classic(torch.Generator().manual_seed(3), (4,)),
        expression=0.3 * torch.randn(4, 72, generator=torch.Generator().manual_seed(4)),
    )
    identity, q, stream = from_classic(body, classic)
    root_quaternion = torch.arange(3, 7)
    q = q.index_copy(-1, root_quaternion, -q.index_select(-1, root_quaternion))
    state = identity.forward(q, stream)

    restored = to_classic(identity, q, streams=stream)
    torch.testing.assert_close(restored.identity_coeffs, classic.identity_coeffs, rtol=0, atol=0)
    torch.testing.assert_close(restored.expression, classic.expression, rtol=0, atol=0)
    identity_back, q_back, stream_back = from_classic(body, restored)
    torch.testing.assert_close(q_back, q, rtol=0, atol=1e-6)
    restored_state = identity_back.forward(q_back, stream_back)
    torch.testing.assert_close(restored_state.vertices, state.vertices, rtol=0, atol=1e-6)
    torch.testing.assert_close(restored_state.joint_poses[..., :3], state.joint_poses[..., :3], rtol=0, atol=1e-6)


def test_identity_batch_broadcasts_against_pose_batch(body: HumanModel) -> None:
    generator = torch.Generator().manual_seed(4)
    identities = _random_classic(generator, (2, 1))
    poses = _random_classic(generator, (3,))
    identity, _, _ = from_classic(body, identities)
    _, q, _ = from_classic(body, MHRClassic(model_parameters=poses.model_parameters))
    vertices = identity.forward(q).vertices
    assert vertices.shape == (2, 3, body.spec.num_vertices, 3)
    torch.testing.assert_close(vertices, identity.forward(q, {"expression": torch.zeros(72)}).vertices)
    for i in range(2):
        single, _, _ = from_classic(
            body,
            MHRClassic(
                identity_coeffs=identities.identity_coeffs[i, 0], model_parameters=identities.model_parameters[i, 0]
            ),
        )
        for j in range(3):
            torch.testing.assert_close(vertices[i, j], single.forward(q[j]).vertices, rtol=0, atol=1e-6)


def test_mass_lives_on_part_bodies_only(body: HumanModel) -> None:
    assert body.part_body_ids.shape == (body.spec.parts.count,)
    assert body.part_body_ids.unique().numel() == body.spec.parts.count
    identity, _, _ = from_classic(body, _random_classic(torch.Generator().manual_seed(6)))
    inertias = identity.model.body_inertias
    assert torch.isfinite(inertias).all()
    masses = inertias[..., 0]
    on_parts = torch.zeros(masses.shape[-1], dtype=torch.bool)
    on_parts[body.part_body_ids] = True
    assert (masses[..., on_parts] > 0).all()
    assert (masses[..., ~on_parts] == 0).all()


def test_spec_and_classic_validation() -> None:
    path = weight_path("MHR/converted", "mhr_lod6.npz")
    with pytest.raises(DtypeMismatchError, match="torch.float32 or torch.float64"):
        families.mhr(path, lod=6, dtype=torch.float16)
    for lod in (-1, 7, 1.0, True):
        with pytest.raises(ArgumentError, match="lod must be one of"):
            families.mhr(path, lod=lod)
    with pytest.raises(ArgumentError, match="must be bool"):
        families.mhr(path, lod=6, expression=1)
    body = HumanModel.from_spec(families.mhr(path, lod=6, correctives=False, device="cpu"))
    assert body.spec.device == torch.device("cpu")
    with pytest.raises(ArgumentError, match="MHRRig cannot drop joints"):
        families.drop_joints(body.spec, [body.spec.joint_names[-1]])
    with pytest.raises(ValueError, match=r"\(\.\.\., 45\)"):
        from_classic(body, MHRClassic(identity_coeffs=torch.zeros(44)))
    with pytest.raises(DtypeMismatchError, match="model_parameters.dtype=torch.int64 is not float32 or float64"):
        from_classic(body, MHRClassic(model_parameters=torch.zeros(204, dtype=torch.int64)))
    with pytest.raises(ValueError, match="float32"):
        from_classic(body, MHRClassic(identity_coeffs=torch.zeros(45, dtype=torch.float64)))
    identity, q, stream = from_classic(body, MHRClassic())
    with pytest.raises(ValueError, match="132"):
        to_classic(identity, q[:-1], streams=stream)
