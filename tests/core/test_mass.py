"""Closed part meshes and shape-dependent human mass properties."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from better_robot.lie import se3, so3
from better_robot.spatial import Inertia

import better_human as bh
from better_human.bodies.assets import load_density_table
from better_human.core.mass import (
    PartMeshTopology,
    body_inertias_from_shape,
    build_part_topology,
    meshes_from_topology,
    segment_vertices,
)

MODEL_PATH = Path(__file__).parents[2] / "models" / "smpl" / "SMPL_NEUTRAL.npz"
SMPLX_MODEL_PATH = Path(__file__).parents[2] / "models" / "smplx" / "SMPLX_NEUTRAL.npz"
SEGMENTATION_PATH = Path(bh.__file__).parent / "bodies" / "assets" / "smpl_3d_segmentation.npy"
CONSERVATION_RTOL = 2e-2
AUTHORED_VOLUME_RTOL = 1e-4
COM_ATOL = 3e-3
GRADCHECK_BETAS = 10


@pytest.fixture(scope="module")
def mass_body() -> bh.SMPL:
    """Load the neutral SMPL fixture with default mass computation."""
    if not MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL fixture is absent: {MODEL_PATH}")
    return bh.SMPL(MODEL_PATH)


def _box_mesh() -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=torch.long,
    )
    return vertices, faces


def _assert_closed_edges(faces: torch.Tensor) -> None:
    edges = (
        torch.cat(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]),
            dim=0,
        )
        .sort(dim=-1)
        .values
    )
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    torch.testing.assert_close(counts, torch.full_like(counts, 2))


def test_segment_vertices_selects_strongest_padded_influence() -> None:
    weights = torch.tensor([[0.2, 0.8, 0.0], [0.6, 0.1, 0.3]], dtype=torch.float32)
    indices = torch.tensor([[0, 3, 0], [2, 1, 4]], dtype=torch.long)
    torch.testing.assert_close(segment_vertices(weights, indices), torch.tensor([3, 2]))


def test_part_meshes_caps_mixed_face_partition_as_closed_surfaces() -> None:
    vertices, faces = _box_mesh()
    part_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    num_parts = int(part_ids.max().item()) + 1
    meshes = meshes_from_topology(vertices, build_part_topology(faces, part_ids, num_parts))
    masses = []
    for part_vertices, part_faces in meshes:
        _assert_closed_edges(part_faces)
        masses.append(Inertia.from_mesh(part_vertices, part_faces).mass)
    torch.testing.assert_close(torch.stack(masses).sum(), Inertia.from_mesh(vertices, faces).mass)


def test_authored_caps_accept_rotated_se3_part_frames() -> None:
    vertices = torch.tensor(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=torch.float32,
    )
    faces = torch.tensor(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=torch.long)
    topology = PartMeshTopology(
        vertex_indices=(torch.arange(3),),
        cap_joint_indices=(torch.tensor([0]),),
        boundary_loops=((),),
        faces=(faces,),
    )
    pose = torch.cat(
        (
            torch.tensor((0.0, 0.0, 1.0)),
            so3.from_euler(torch.tensor((0.2, -0.3, 0.4))),
        )
    ).unsqueeze(0)

    actual = body_inertias_from_shape(vertices, pose, topology, torch.tensor(1000.0))[1]
    capped = torch.cat((vertices, pose[:, :3]), dim=0)
    local = se3.act(se3.inverse(pose)[..., None, :], capped).squeeze(0)
    expected = Inertia.from_mesh(local, faces, density=1000.0).data
    torch.testing.assert_close(actual, expected)


def test_default_mass_opt_out_and_per_part_density(mass_body: bh.SMPL) -> None:
    assert mass_body.structure.part_topology is not None
    assert mass_body.values.body_inertias is not None
    assert mass_body.structure.part_topology.num_parts == len(mass_body.structure.joint_names)
    torch.testing.assert_close(mass_body.robot.body_inertias, mass_body.values.body_inertias)
    torch.testing.assert_close(mass_body.values.density, torch.tensor(1000.0))

    opt_out = bh.SMPL(MODEL_PATH, compute_mass=False)
    assert opt_out.structure.part_topology is None
    assert opt_out.values.body_inertias is None

    densities = torch.linspace(900.0, 1100.0, len(mass_body.structure.joint_names))
    varied = body_inertias_from_shape(
        mass_body.values.v_shaped,
        mass_body.values.rest_joints,
        mass_body.structure.part_topology,
        densities,
    )
    expected_ratio = densities / mass_body.values.density
    torch.testing.assert_close(
        varied[1:, 0] / mass_body.values.body_inertias[1:, 0],
        expected_ratio,
        rtol=2e-5,
        atol=2e-5,
    )

    with pytest.raises(ValueError, match="density must be scalar or have shape"):
        bh.SMPL(MODEL_PATH, density=torch.ones(23), compute_mass=False)


@pytest.mark.parametrize(
    ("use_hands", "use_face", "expected_bodies"),
    [(True, True, 56), (False, True, 26), (True, False, 53), (False, False, 23)],
)
def test_smplx_full_and_partial_spawns_receive_mass(
    use_hands: bool,
    use_face: bool,
    expected_bodies: int,
) -> None:
    if not SMPLX_MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {SMPLX_MODEL_PATH}")
    body = bh.SMPLX(SMPLX_MODEL_PATH, use_hands=use_hands, use_face=use_face)
    assert body.values.body_inertias is not None
    assert body.values.body_inertias.shape == (expected_bodies, 10)
    assert torch.isfinite(body.values.body_inertias).all()
    assert body.values.body_inertias[1:, 0].min() > 0


def test_neutral_mass_conservation_and_com_plausibility(mass_body: bh.SMPL) -> None:
    assert mass_body.values.body_inertias is not None
    packed = mass_body.values.body_inertias[1:]
    part_mass = packed[:, 0].sum()
    whole = Inertia.from_mesh(
        mass_body.values.v_shaped,
        mass_body.structure.faces,
        density=1000.0,
    )
    relative_error = (part_mass - whole.mass).abs() / whole.mass
    assert relative_error <= CONSERVATION_RTOL
    assert 70.0 <= part_mass <= 90.0

    global_part_com = packed[:, 1:4] + mass_body.values.rest_joints
    aggregate_com = (packed[:, 0, None] * global_part_com).sum(dim=0) / part_mass
    torch.testing.assert_close(aggregate_com, whole.com, rtol=0, atol=COM_ATOL)
    lower = mass_body.values.v_shaped.amin(dim=-2)
    upper = mass_body.values.v_shaped.amax(dim=-2)
    assert bool(((aggregate_com >= lower) & (aggregate_com <= upper)).all())
    assert aggregate_com.norm() < 0.1


def test_argmax_segmentation_cross_check_against_canonical_membership(mass_body: bh.SMPL) -> None:
    if not SEGMENTATION_PATH.is_file():
        pytest.skip(f"canonical SMPL segmentation is absent: {SEGMENTATION_PATH}")
    segmentation = np.load(SEGMENTATION_PATH, allow_pickle=True).item()
    memberships = {
        part: {int(vertex) for vertex in vertices if int(vertex) < mass_body.structure.num_vertices}
        for part, vertices in segmentation["body_vertices"].items()
    }
    mismatches = sum(
        vertex not in memberships[part] for vertex, part in enumerate(mass_body.structure.part_ids.cpu().tolist())
    )
    overlap_counts = Counter(vertex for vertices in memberships.values() for vertex in vertices)
    assert mismatches == 113
    assert sum(count > 1 for count in overlap_counts.values()) == 582


def test_authored_segmentation_volume_matches_whole_mesh(mass_body: bh.SMPL) -> None:
    packed = mass_body.values.body_inertias[1:]
    part_volume = (packed[:, 0] / mass_body.values.density).sum()
    whole = Inertia.from_mesh(mass_body.values.v_shaped, mass_body.structure.faces, density=1000.0)
    whole_volume = whole.mass / 1000.0
    relative_error = (part_volume - whole_volume).abs() / whole_volume
    assert relative_error < AUTHORED_VOLUME_RTOL, f"authored volume gap {relative_error * 1e6:.1f} ppm"
    total_mass = packed[:, 0].sum()
    assert 74.0 <= total_mass <= 75.5


def test_authored_segmentation_is_batched_and_finite(mass_body: bh.SMPL) -> None:
    betas = torch.stack((torch.zeros(GRADCHECK_BETAS), torch.linspace(-0.4, 0.4, GRADCHECK_BETAS)))
    inertias = mass_body.with_shape(betas=betas).values.body_inertias
    assert inertias is not None
    assert inertias.shape == (2, len(mass_body.structure.joint_names) + 1, 10)
    assert torch.isfinite(inertias).all()
    assert inertias[:, 1:, 0].min() > 0


def test_named_density_table_feeds_per_part_density(mass_body: bh.SMPL) -> None:
    exemplar = mass_body.values.v_shaped
    num_parts = len(mass_body.structure.joint_names)
    chandler = load_density_table("body_densities.json", "Chandler", num_parts, exemplar)
    assert chandler.shape == (num_parts,)
    torch.testing.assert_close(chandler[0], exemplar.new_tensor(853.0))
    torch.testing.assert_close(chandler[16], exemplar.new_tensor(1005.0))

    body = bh.SMPL(MODEL_PATH, density="Chandler")
    torch.testing.assert_close(body.values.density, chandler)
    part_volumes = body.values.body_inertias[1:, 0] / chandler
    reference_volumes = mass_body.values.body_inertias[1:, 0] / mass_body.values.density
    torch.testing.assert_close(part_volumes, reference_volumes, rtol=2e-5, atol=2e-5)

    with pytest.raises(ValueError, match="no density table"):
        bh.SMPL(MODEL_PATH, density="Nonexistent")


def test_argmax_fallback_integrates_smplx_parts() -> None:
    if not SMPLX_MODEL_PATH.is_file():
        pytest.skip(f"licensed SMPL-X fixture is absent: {SMPLX_MODEL_PATH}")
    body = bh.SMPLX(SMPLX_MODEL_PATH)
    topology = body.structure.part_topology
    assert topology is not None
    for vertices, faces in meshes_from_topology(body.values.v_shaped, topology):
        assert Inertia.from_mesh(vertices, faces, density=1000.0).mass > 0


def test_float32_gradcheck_betas_to_total_mass(mass_body: bh.SMPL) -> None:
    betas = (torch.randn(GRADCHECK_BETAS, dtype=torch.float32) * 0.03).requires_grad_()

    def total_mass(value: torch.Tensor) -> torch.Tensor:
        inertias = mass_body.with_shape(betas=value).values.body_inertias
        assert inertias is not None
        return inertias[1:, 0].sum()

    assert torch.autograd.gradcheck(
        total_mass,
        (betas,),
        eps=1e-3,
        atol=4e-2,
        rtol=5e-2,
        fast_mode=True,
    )
