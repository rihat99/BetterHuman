"""MHR tree, conversion, deformation, and early-parity tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
import better_robot as br
from better_robot.data_model.reduced_coordinates import expand_configuration

import better_human as bh
from better_human.bodies import MHRClassic

MODELS_DIR = Path(__file__).parents[2] / "models" / "MHR" / "converted"
GOLDENS_DIR = Path(__file__).parents[1] / "parity" / "goldens"
LOD_VERTEX_COUNTS = (73639, 18439, 10661, 4899, 2461, 971, 595)


def _model_path(lod: int) -> Path:
    path = MODELS_DIR / f"mhr_lod{lod}.npz"
    if not path.is_file():
        pytest.skip(f"converted MHR LOD{lod} fixture is absent: {path}")
    return path


@pytest.fixture(scope="module")
def body() -> bh.MHR:
    """Load the smallest LOD with correctives for focused behavior tests."""
    return bh.MHR(_model_path(6), lod=6, compute_mass=False)


@pytest.fixture
def deterministic_gradcheck() -> Iterator[None]:
    """Make finite-difference reentrancy checks repeatable, then restore state."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _zero_classic(exemplar: torch.Tensor) -> MHRClassic:
    return MHRClassic(
        identity_coeffs=exemplar.new_zeros(45),
        model_parameters=exemplar.new_zeros(204),
    )


def test_registry_exports_and_data_driven_tree(body: bh.MHR) -> None:
    loaded = bh.load(
        "mhr",
        model_path=_model_path(6),
        lod=6,
        use_correctives=False,
        compute_mass=False,
    )
    assert isinstance(loaded, bh.MHR)
    assert (body.robot.nq, body.robot.nv) == (132, 131)
    assert (body.robot.nq_full, body.robot.nv_full) == (179, 178)
    assert (body.robot.njoints, body.robot.nbodies) == (203, 203)
    assert sum(model.kind.startswith("revolute") for model in body.robot.joint_models) == 172
    assert all(model.kind != "spherical" for model in body.robot.joint_models)
    assert body.structure.joint_names[:2] == ("body_world", "root")
    assert body.structure.native_pose_joint_indices.shape == (127,)
    assert body.native_joint_poses(body.fk(body.robot.q_neutral)).shape == (127, 7)


def test_parameter_partitions_and_public_q_follow_archive_names(body: bh.MHR) -> None:
    with np.load(_model_path(6), allow_pickle=False) as arrays:
        parameter_names = arrays["model_parameter_names"].tolist()
        lower_limits = torch.from_numpy(arrays["model_parameter_lower_limits"])
        upper_limits = torch.from_numpy(arrays["model_parameter_upper_limits"])
    pose_indices = body._pose_parameter_indices.tolist()
    proportion_indices = body._proportion_parameter_indices.tolist()
    assert pose_indices == [*range(6, 130), 151]
    assert proportion_indices == [*range(130, 151), *range(152, 204)]
    assert proportion_indices[14:16] == [144, 145]

    model_parameters = torch.linspace(-0.4, 0.4, 204)
    _, q = body.from_classic(MHRClassic(torch.zeros(45), model_parameters))
    for position, parameter_index in enumerate(pose_indices):
        joint_id = body.robot.joint_id(parameter_names[parameter_index])
        q_index = body.robot.idx_qs[joint_id]
        expected = model_parameters[parameter_index] * body._pose_host_weights[position]
        torch.testing.assert_close(q[q_index], expected)
        expected_limits = torch.sort(
            torch.stack((lower_limits[parameter_index], upper_limits[parameter_index]))
            * body._pose_host_weights[position]
        ).values
        torch.testing.assert_close(body.robot.lower_pos_limit[q_index], expected_limits[0])
        torch.testing.assert_close(body.robot.upper_pos_limit[q_index], expected_limits[1])


def test_all_lods_spawn_and_forward_without_corrective_payloads() -> None:
    for lod, vertex_count in enumerate(LOD_VERTEX_COUNTS):
        body = bh.MHR(
            _model_path(lod),
            lod=lod,
            use_correctives=False,
            compute_mass=False,
        )
        assert body._corrective_first_layer_weight is None
        assert body._corrective_output_weight is None
        assert body.vertices(body.robot.q_neutral).shape == (vertex_count, 3)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 2e-6), (torch.float64, 1e-12)],
)
def test_classic_round_trip_includes_shape_pose_and_scaled_hosts(
    dtype: torch.dtype,
    tolerance: float,
) -> None:
    body = bh.MHR(
        _model_path(6),
        lod=6,
        use_correctives=False,
        compute_mass=False,
        dtype=dtype,
    )
    identity = torch.linspace(-0.7, 0.7, 45, dtype=dtype)
    model_parameters = torch.linspace(-0.45, 0.45, 204, dtype=dtype)
    model_parameters[:6] = torch.tensor((0.2, -0.1, 0.3, 0.4, -0.35, 0.25), dtype=dtype)
    with np.load(_model_path(6), allow_pickle=False) as arrays:
        parameter_names = arrays["model_parameter_names"].tolist()
    spine_indices = [index for index, name in enumerate(parameter_names) if name.startswith("spine_")]
    model_parameters[spine_indices] = torch.linspace(-0.8, 0.8, len(spine_indices), dtype=dtype)
    expression = torch.linspace(0.3, -0.3, 72, dtype=dtype)
    classic = MHRClassic(identity, model_parameters, expression)

    shaped, q = body.from_classic(classic)
    assert shaped is not body
    assert torch.count_nonzero(body.values.shape_parameters["identity"]) == 0
    assert torch.count_nonzero(body.values.shape_parameters["proportions"]) == 0
    recovered = shaped.to_classic(q)
    torch.testing.assert_close(recovered.identity_coeffs, identity, rtol=0, atol=tolerance)
    torch.testing.assert_close(recovered.model_parameters, model_parameters, rtol=0, atol=tolerance)
    torch.testing.assert_close(recovered.expression, expression, rtol=0, atol=tolerance)

    reshaped, recovered_q = body.from_classic(recovered)
    torch.testing.assert_close(recovered_q, q, rtol=0, atol=tolerance)
    torch.testing.assert_close(
        reshaped.values.shape_parameters["proportions"],
        shaped.values.shape_parameters["proportions"],
        rtol=0,
        atol=tolerance,
    )

    spine_twist = parameter_names.index("spine_twist0")
    external_q = q.index_select(-1, body._q_to_classic_permutation)
    pose_position = body._pose_parameter_indices.tolist().index(spine_twist)
    host_weight = body._pose_host_weights[pose_position]
    torch.testing.assert_close(host_weight, host_weight.new_tensor(0.01), rtol=0, atol=3e-10)
    torch.testing.assert_close(
        external_q[7 + pose_position],
        model_parameters[spine_twist] * host_weight,
        rtol=0,
        atol=tolerance,
    )


def test_classic_round_trip_preserves_negative_root_quaternion_sign() -> None:
    body = bh.MHR(
        _model_path(6),
        lod=6,
        use_correctives=False,
        compute_mass=False,
        dtype=torch.float64,
    )
    shaped, q = body.from_classic(_zero_classic(body.values.v_template))
    signed_q = q.index_copy(0, torch.arange(3, 7), -q[3:7])
    recovered = shaped.to_classic(signed_q)
    _, round_trip = body.from_classic(recovered)
    torch.testing.assert_close(round_trip, signed_q, rtol=0, atol=1e-12)


def test_classic_broadcasts_and_expression_none_is_canonical_zero(body: bh.MHR) -> None:
    identity = torch.zeros(2, 1, 45)
    model_parameters = torch.zeros(1, 3, 204)
    shaped, q = body.from_classic(MHRClassic(identity, model_parameters))
    assert q.shape == (2, 3, 132)
    assert shaped.values.v_shaped.shape == (2, 3, 595, 3)
    assert shaped.values.joint_scales.shape == (2, 3, 127)
    assert shaped.vertices(q).shape == (2, 3, 595, 3)
    recovered = shaped.to_classic(q)
    assert recovered.expression.shape == (2, 3, 72)
    assert torch.count_nonzero(recovered.expression) == 0


def test_expression_can_be_disabled() -> None:
    body = bh.MHR(
        _model_path(6),
        lod=6,
        use_expression=False,
        use_correctives=False,
        compute_mass=False,
    )
    assert tuple(body.structure.parameter_groups) == ("identity", "proportions")
    classic = _zero_classic(body.values.v_template)
    shaped, q = body.from_classic(classic)
    assert shaped.to_classic(q).expression is None
    with pytest.raises(ValueError, match="use_expression=True"):
        body.from_classic(
            MHRClassic(
                classic.identity_coeffs,
                classic.model_parameters,
                torch.zeros(72),
            )
        )


def test_classic_boundary_rejects_wrong_shape_dtype_and_q_width(body: bh.MHR) -> None:
    with pytest.raises(ValueError, match=r"identity_coeffs must have shape \(\.\.\., 45\)"):
        body.from_classic(MHRClassic(torch.zeros(44), torch.zeros(204)))
    with pytest.raises(TypeError, match="model_parameters must be a floating"):
        body.from_classic(MHRClassic(torch.zeros(45), torch.zeros(204, dtype=torch.int64)))
    with pytest.raises(ValueError, match="dtype=torch.float32"):
        body.from_classic(MHRClassic(torch.zeros(45, dtype=torch.float64), torch.zeros(204)))
    shaped, q = body.from_classic(_zero_classic(body.values.v_template))
    with pytest.raises(ValueError, match=r"q must have shape \(\.\.\., 132\)"):
        shaped.to_classic(q[:-1])


def test_constructor_rejects_dtypes_unsupported_by_better_robot() -> None:
    with pytest.raises(TypeError, match="torch.float32 or torch.float64"):
        bh.MHR(
            _model_path(6),
            lod=6,
            use_correctives=False,
            compute_mass=False,
            dtype=torch.float16,
        )


def test_mimic_expansion_matches_all_spine_twist_weights(body: bh.MHR) -> None:
    with np.load(_model_path(6), allow_pickle=False) as arrays:
        parameter_names = arrays["model_parameter_names"].tolist()
        parameter_index = parameter_names.index("spine_twist0")
        rows = torch.from_numpy(arrays["parameter_transform_rows"])
        columns = torch.from_numpy(arrays["parameter_transform_columns"])
        weights = torch.from_numpy(arrays["parameter_transform_values"])
    rotation_entries = (rows // 7 >= 2) & ((rows % 7 >= 3) & (rows % 7 <= 5))
    selected = columns[rotation_entries] == parameter_index
    expected = weights[rotation_entries][selected]
    assert len(expected) > 1

    model_parameters = torch.zeros(204)
    model_parameters[parameter_index] = 1.0
    _, q = body.from_classic(MHRClassic(torch.zeros(45), model_parameters))
    q_full = expand_configuration(body.robot.structure, q)
    contributions = q_full.index_select(-1, body._corrective_full_q_indices)[selected]
    torch.testing.assert_close(contributions, expected)


def test_all_rotation_transform_entries_match_expanded_configuration(body: bh.MHR) -> None:
    with np.load(_model_path(6), allow_pickle=False) as arrays:
        rows = torch.from_numpy(arrays["parameter_transform_rows"]).long()
        columns = torch.from_numpy(arrays["parameter_transform_columns"]).long()
        weights = torch.from_numpy(arrays["parameter_transform_values"])
    native_joints = rows // 7
    channels = rows % 7
    rotation_entries = (native_joints >= 2) & (channels >= 3) & (channels <= 5)
    rotation_rows = rows[rotation_entries]
    rotation_columns = columns[rotation_entries]
    rotation_weights = weights[rotation_entries]
    assert len(rotation_rows) == 172
    counts = torch.bincount(rotation_columns, minlength=204)
    assert torch.count_nonzero(counts > 1) == 16

    generator = torch.Generator().manual_seed(20260721)
    model_parameters = torch.randn(100, 204, generator=generator) * 0.1
    _, q = body.from_classic(MHRClassic(torch.zeros(100, 45), model_parameters))
    expanded = expand_configuration(body.robot.structure, q).index_select(
        -1,
        body._corrective_full_q_indices,
    )
    expected = model_parameters.index_select(-1, rotation_columns) * rotation_weights
    torch.testing.assert_close(expanded, expected, rtol=0, atol=2e-7)

    pose_columns = body._pose_parameter_indices.cpu()
    local_columns = torch.full((204,), -1, dtype=torch.long)
    local_columns[pose_columns] = torch.arange(len(pose_columns))
    transform = torch.zeros(375, 125, dtype=torch.float64)
    transform.index_put_(
        (
            (rotation_rows // 7 - 2) * 3 + rotation_rows % 7 - 3,
            local_columns[rotation_columns],
        ),
        rotation_weights.double(),
        accumulate=True,
    )
    assert torch.linalg.matrix_rank(transform) == 119


def test_shape_and_pose_paths_retain_float32_gradients(body: bh.MHR) -> None:
    identity = (torch.randn(45) * 0.02).requires_grad_()
    model_parameters = (torch.randn(204) * 0.01).requires_grad_()
    shaped, q = body.from_classic(MHRClassic(identity, model_parameters))
    loss = shaped.vertices(q)[::97].square().sum()
    identity_gradient, parameter_gradient = torch.autograd.grad(loss, (identity, model_parameters))
    assert torch.isfinite(identity_gradient).all()
    assert torch.isfinite(parameter_gradient).all()
    assert torch.count_nonzero(identity_gradient) > 0
    assert torch.count_nonzero(parameter_gradient) > 0


def test_float32_gradcheck_shape_groups_to_vertices(
    body: bh.MHR,
    deterministic_gradcheck: None,
) -> None:
    del deterministic_gradcheck
    identity = torch.linspace(-0.01, 0.01, 45).requires_grad_()
    expression = torch.linspace(0.01, -0.01, 72).requires_grad_()
    proportions = torch.zeros(73)
    proportions[14:16] = torch.tensor((0.08, -0.06))
    proportions.requires_grad_()
    q = body.robot.q_neutral.clone()
    q[7:] = torch.linspace(-0.03, 0.03, body.robot.nq - 7)

    def vertices_from_shape(
        identity_value: torch.Tensor,
        expression_value: torch.Tensor,
        proportion_value: torch.Tensor,
    ) -> torch.Tensor:
        shaped = body.with_shape(
            identity=identity_value,
            expression=expression_value,
            proportions=proportion_value,
        )
        return shaped.vertices(q)

    assert torch.autograd.gradcheck(
        vertices_from_shape,
        (identity, expression, proportions),
        eps=1e-3,
        atol=3e-2,
        rtol=8e-2,
        fast_mode=True,
    )
    hand_scale_gradient = torch.autograd.grad(
        vertices_from_shape(identity, expression, proportions).square().sum(),
        proportions,
    )[0][14:16]
    assert torch.isfinite(hand_scale_gradient).all()
    assert torch.count_nonzero(hand_scale_gradient) == 2


def test_float32_gradcheck_q_to_vertices(
    body: bh.MHR,
    deterministic_gradcheck: None,
) -> None:
    del deterministic_gradcheck
    proportions = torch.zeros(73)
    proportions[14:16] = torch.tensor((0.08, -0.06))
    shaped = body.with_shape(
        identity=torch.linspace(-0.01, 0.01, 45),
        expression=torch.linspace(0.01, -0.01, 72),
        proportions=proportions,
    )
    q = shaped.robot.q_neutral.clone()
    q[7:] = torch.linspace(-0.03, 0.03, shaped.robot.nq - 7)
    q.requires_grad_()

    assert torch.autograd.gradcheck(
        shaped.vertices,
        (q,),
        eps=1e-3,
        atol=3e-2,
        rtol=8e-2,
        fast_mode=True,
    )


def test_shape_and_q_broadcast_independent_leading_axes(body: bh.MHR) -> None:
    shaped = body.with_shape(
        identity=torch.zeros(2, 1, 45),
        expression=torch.zeros(72),
        proportions=torch.zeros(1, 1, 73),
    )
    q = body.robot.q_neutral.expand(1, 3, body.robot.nq)
    vertices = shaped.vertices(q)
    assert vertices.shape == (2, 3, LOD_VERTEX_COUNTS[6], 3)
    assert vertices.dtype == torch.float32
    assert vertices.device.type == "cpu"


def test_lod1_combined_early_smoke_parity() -> None:
    golden_path = GOLDENS_DIR / "mhr_lod1" / "combined_00.npz"
    if not golden_path.is_file():
        pytest.skip(f"MHR golden is absent: {golden_path}")
    with np.load(golden_path, allow_pickle=False) as golden:
        classic = MHRClassic(
            identity_coeffs=torch.from_numpy(golden["identity_coeffs"]),
            model_parameters=torch.from_numpy(golden["model_parameters"]),
            expression=torch.from_numpy(golden["face_expression"]),
        )
        expected_vertices = torch.from_numpy(golden["vertices"]) * 0.01
        expected_state = torch.from_numpy(golden["skel_state"])
    body = bh.MHR(_model_path(1), lod=1, compute_mass=False)
    shaped, q = body.from_classic(classic)
    data = shaped.fk(q)
    actual_poses = shaped.native_joint_poses(data)
    actual_vertices = shaped.vertices_from_data(data)

    torch.testing.assert_close(actual_vertices, expected_vertices, rtol=0, atol=1e-3)
    torch.testing.assert_close(actual_poses[..., :3], expected_state[..., :3] * 0.01, rtol=0, atol=1e-3)
    quaternion_error = torch.minimum(
        (actual_poses[..., 3:7] - expected_state[..., 3:7]).abs().amax(dim=-1),
        (actual_poses[..., 3:7] + expected_state[..., 3:7]).abs().amax(dim=-1),
    )
    assert quaternion_error.max() < 1e-3
    torch.testing.assert_close(shaped.values.joint_scales, expected_state[..., 7], rtol=0, atol=1e-6)


def test_lod1_mass_is_positive_plausible_and_only_on_physical_bodies() -> None:
    body = bh.MHR(
        _model_path(1),
        lod=1,
        use_correctives=False,
        compute_mass=True,
    )
    masses = body.values.body_inertias[..., 0]
    physical = masses[masses > 0]
    assert len(physical) == 19
    assert physical.min() > 0
    torch.testing.assert_close(physical.sum(), physical.new_tensor(81.481842), rtol=0, atol=2e-4)
    assert torch.count_nonzero(masses == 0) == body.robot.nbodies - 19


def test_mhr_crba_is_psd_with_documented_structural_nullspace() -> None:
    body = bh.MHR(
        _model_path(1),
        lod=1,
        use_expression=False,
        use_correctives=False,
        compute_mass=True,
        dtype=torch.float64,
    )
    mass_matrix = br.crba(body.robot, body.robot.q_neutral)
    torch.testing.assert_close(mass_matrix, mass_matrix.mT, rtol=0, atol=1e-12)
    eigenvalues = torch.linalg.eigvalsh(mass_matrix)
    assert eigenvalues.min() >= -1e-12
    numerical_rank = int(torch.count_nonzero(torch.linalg.svdvals(mass_matrix) > 1e-10))
    assert numerical_rank == 50
    assert torch.count_nonzero(torch.diagonal(mass_matrix) == 0) == 72
    assert numerical_rank < 125 < body.robot.nv


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mhr_forward_and_backward_preserve_device() -> None:
    device = torch.device("cuda")
    body = bh.MHR(
        _model_path(6),
        lod=6,
        compute_mass=False,
        device=device,
    )
    identity = torch.linspace(-0.01, 0.01, 45, device=device).requires_grad_()
    model_parameters = torch.linspace(-0.02, 0.02, 204, device=device).requires_grad_()
    expression = torch.linspace(0.01, -0.01, 72, device=device).requires_grad_()
    shaped, q = body.from_classic(MHRClassic(identity, model_parameters, expression))
    vertices = shaped.vertices(q)
    gradients = torch.autograd.grad(vertices[::31].square().sum(), (identity, model_parameters, expression))
    assert vertices.is_cuda and vertices.dtype == torch.float32
    for gradient in gradients:
        assert gradient.is_cuda
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
